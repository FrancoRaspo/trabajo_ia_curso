"""
Siembra el golden set con CUITs REALES de la base, estratificados por banda
de riesgo.

El golden set necesita cubrir el rango de decisiones: no sirve evaluar sólo
clientes buenos. Este script consulta la población, la puntúa con el modelo
vigente y elige N clientes por banda (BAJO / MODERADO / ALTO / MUY ALTO),
imprimiendo entradas listas para pegar en evaluation/golden_set.json.

Uso:
    python evaluation/muestrear_cuits.py                 # 2 por banda
    python evaluation/muestrear_cuits.py --por-banda 3
    python evaluation/muestrear_cuits.py --cutoff 2025-04-30

Requiere SQL Server accesible (mismas envs que el pipeline) y el modelo
entrenado en models/. No consume LLM.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import datetime

from db.sqlserver_connection import get_sqlserver_engine
from ml.feature_query import FEATURE_COLUMNS, cargar_features
from ml.scoring_model import CreditScoringModel

# Cutoff por defecto = HOY, para que la banda que muestrea el sampler coincida
# EXACTAMENTE con la que calcula el eval (que puntúa a hoy). Con un cutoff viejo
# el riesgo del cliente cambia y el check banda_riesgo falla espuriamente.
HOY = datetime.date.today().isoformat()

# Motivo por defecto según la banda: uno restrictivo para riesgo alto y uno
# expansivo para riesgo bajo, para que el informe tenga que razonar distinto.
MOTIVO_POR_BANDA = {
    "BAJO":     "Solicita préstamo prendario por $15.000.000 a 48 meses.",
    "MODERADO": "Solicita préstamo personal por $6.000.000 a 36 meses.",
    "ALTO":     "Solicita renovación de acuerdo en cuenta corriente por $8.000.000.",
    "MUY ALTO": "Solicita ampliación de línea de crédito por $10.000.000.",
}

# Aserciones sugeridas por banda. El usuario las ajusta a mano después.
BANDAS_ACEPTABLES = {
    "BAJO":     ["BAJO", "MODERADO"],
    "MODERADO": ["BAJO", "MODERADO", "ALTO"],
    "ALTO":     ["MODERADO", "ALTO", "MUY ALTO"],
    "MUY ALTO": ["ALTO", "MUY ALTO"],
}


def _norm_cuit(valor) -> str:
    """La columna cliente_id llega como float64 (pandas mete NaN en la columna),
    así que 20123456789 sale como '20123456789.0'. Lo dejamos en dígitos."""
    try:
        return str(int(float(valor)))
    except (TypeError, ValueError):
        return str(valor)


def _entrada_golden(cuit: str, banda: str, idx: int) -> dict:
    return {
        "id": f"{banda.lower().replace(' ', '_')}_{idx}",
        "cuit": str(cuit),
        "motivo": MOTIVO_POR_BANDA.get(banda, "Solicita evaluación crediticia."),
        "notas": f"Muestreado automáticamente; banda del modelo al muestrear: {banda}.",
        "esperado": {
            "encontrado": True,
            "banda_riesgo_en": BANDAS_ACEPTABLES.get(banda, [banda]),
            "score_min": None,
            "score_max": None,
            "debe_contener": [],
            "no_debe_contener": [],
            "juez_puntaje_min": 4,
            "sin_alucinaciones": True,
            "cumplimiento_reglas": True,
            "ragas_faithfulness_min": 0.7,
            "ragas_answer_relevancy_min": None,  # informativa: no gatea (desajuste estructural)
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--por-banda", type=int, default=2,
                    help="cuántos CUITs muestrear por banda de riesgo")
    ap.add_argument("--cutoff", default=HOY,
                    help="fecha de corte para calcular las features (YYYY-MM-DD). "
                         "Default = hoy, para coincidir con el scoring del eval.")
    ap.add_argument("--muestra-max", type=int, default=4000,
                    help="tope de clientes a puntuar (evita puntuar toda la cartera)")
    ap.add_argument("--salida", default=None,
                    help="escribe los casos (JSON) a este archivo en vez de a stdout. "
                         "Evita que las líneas [DBG] del modelo ensucien el JSON.")
    args = ap.parse_args()

    print("Cargando modelo de scoring...", flush=True)
    model = CreditScoringModel.load()

    print(f"Consultando población al {args.cutoff}...", flush=True)
    engine = get_sqlserver_engine()
    df = cargar_features(engine, args.cutoff)
    # NO se usa filtrar_elegibles: ese filtro es de población de ENTRENAMIENTO
    # (excluye a los ya-defaulteados, que son justo los MUY ALTO que queremos en
    # el golden set). Acá sólo pedimos clientes con historial, para que el score
    # sea significativo (no thin-file).
    if "total_prestamos" in df.columns:
        df = df[df["total_prestamos"] > 0]
    if df.empty:
        print("No hay clientes con historial en ese cutoff. Probá otro --cutoff.")
        sys.exit(1)

    # Puntuar como mucho `muestra-max` clientes para no recorrer toda la cartera.
    if len(df) > args.muestra_max:
        df = df.head(args.muestra_max)
    print(f"Puntuando {len(df)} clientes...", flush=True)

    por_banda: dict[str, list] = {b: [] for b in
                                  ("BAJO", "MODERADO", "ALTO", "MUY ALTO")}
    for _, row in df.iterrows():
        feats = {c: row[c] for c in FEATURE_COLUMNS if c in row}
        res = model.predict(feats)
        banda = res.get("nivel_riesgo")
        if banda in por_banda and len(por_banda[banda]) < args.por_banda:
            por_banda[banda].append((_norm_cuit(row["cliente_id"]), res.get("score")))
        if all(len(v) >= args.por_banda for v in por_banda.values()):
            break

    casos = []
    for banda, elegidos in por_banda.items():
        if not elegidos:
            print(f"  ⚠️ banda {banda}: 0 clientes encontrados en la muestra.",
                  file=sys.stderr)
        for i, (cuit, score) in enumerate(elegidos, 1):
            print(f"  {banda:<9} cuit={cuit}  score={score}", file=sys.stderr)
            casos.append(_entrada_golden(cuit, banda, i))

    # Los CUIT salen SIEMPRE por separado, al archivo local que no se versiona:
    # son de persona física (contienen el DNI) y el repo es público. Lo que se
    # imprime o se escribe con --salida ya viene sin identificadores, así que no
    # hay forma de filtrarlos al golden set versionado copiando y pegando.
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_cuits = os.path.join(raiz, "evaluation", "golden_cuits.local.json")
    overlay = {c["id"]: c["cuit"] for c in casos}
    with open(ruta_cuits, "w", encoding="utf-8") as f:
        json.dump({
            "_descripcion": ("Identificadores reales de los casos del golden "
                             "set. NO se versiona (ver .gitignore). Lo "
                             "regenera muestrear_cuits.py."),
            "cuits": overlay,
        }, f, ensure_ascii=False, indent=2)
        f.write("\n")

    for c in casos:
        c["cuit"] = "COMPLETAR"
    payload = json.dumps(casos, ensure_ascii=False, indent=2)

    print(f"\n🔒 {len(overlay)} identificadores escritos en {ruta_cuits} "
          "(no versionado).", file=sys.stderr)
    if args.salida:
        with open(args.salida, "w", encoding="utf-8") as f:
            f.write(payload + "\n")
        print(f"✅ {len(casos)} casos escritos en {args.salida}. "
              "Pegalos en el array 'casos' de evaluation/golden_set.json.",
              file=sys.stderr)
    else:
        print("\n" + "=" * 72, file=sys.stderr)
        print("Pegá estas entradas en el array 'casos' de "
              "evaluation/golden_set.json (o usá --salida para escribir a archivo):",
              file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        print(payload)


if __name__ == "__main__":
    main()
