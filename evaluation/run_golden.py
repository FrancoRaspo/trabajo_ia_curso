"""
Evaluación automatizada sobre el golden set.

Responde a la devolución del profesor ("la evaluación conviene automatizarla:
ahora audita un CUIT por vez, sin golden set ni métricas agregadas; monta un
conjunto de casos con pass/fail y añade RAGas"): corre TODOS los casos de
evaluation/golden_set.json, y por cada uno combina tres familias de métricas:

    1. deterministas  (banda, score, substrings)          — evaluation/metricas.py
    2. juez LLM        (fidelidad, exactitud, alucinación) — evaluation/llm_judge.py
    3. RAGas           (faithfulness, answer_relevancy)    — evaluation/ragas_eval.py

Agrega un pass-rate global + medias por métrica, imprime una tabla, guarda el
reporte en evaluation/reports/ y devuelve exit code 1 si algún caso falla (para
poder colgarlo de CI).

Uso:
    python evaluation/run_golden.py
    python evaluation/run_golden.py --golden evaluation/golden_set.json
    python evaluation/run_golden.py --sin-ragas        # salta RAGas (más rápido)
    python evaluation/run_golden.py --juez openai      # proveedor del juez LLM

Requiere SQL Server + el generador (PROVEEDOR_LLM) para producir cada informe,
y una API key para el juez/RAGas (por defecto OpenAI).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from pipeline import get_pipeline, _es_no_encontrado
from evaluation.llm_judge import evaluar_informe
from evaluation import metricas
from evaluation import ragas_eval

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_REPORTES = os.path.join(RAIZ, "evaluation", "reports")


# Los identificadores reales de los casos no se versionan: son CUIT de persona
# física (contienen el DNI) etiquetados por banda de riesgo, y el repo es
# público. El golden set versionado lleva "COMPLETAR" y este archivo local los
# aporta, indexados por el id del caso. Lo genera muestrear_cuits.py --salida.
CUITS_LOCALES = os.path.join(RAIZ, "evaluation", "golden_cuits.local.json")


def _cuits_locales(path: str = CUITS_LOCALES) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("cuits", {})


def cargar_casos(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    casos = data.get("casos", [])
    overlay = _cuits_locales()
    for c in casos:
        if str(c.get("cuit", "")).strip().upper() == "COMPLETAR" and c.get("id") in overlay:
            c["cuit"] = overlay[c["id"]]
    validos, saltados = [], []
    for c in casos:
        cuit = str(c.get("cuit", "")).strip()
        # Los CUITs de ejemplo del template (COMPLETAR) no se corren.
        if not cuit or cuit.upper() == "COMPLETAR":
            saltados.append(c.get("id", "?"))
            continue
        validos.append(c)
    if saltados:
        print(f"⏭  Saltados (CUIT sin completar): {', '.join(saltados)}")
    return validos


def _generar(pipeline, cuit: str, motivo: str) -> dict:
    """Genera el informe y normaliza la salida. No propaga 'no encontrado':
    lo devuelve como encontrado=False para que sea un caso evaluable."""
    cuit = re.sub(r"\D", "", cuit)
    try:
        res = pipeline.generar(cuit, motivo)
        return {"encontrado": True, "informe": res["informe"],
                "scoring": res.get("scoring") or {},
                "contexto": res.get("contexto") or {}, "error": None}
    except Exception as exc:  # noqa: BLE001
        if _es_no_encontrado(exc):
            return {"encontrado": False, "informe": "", "scoring": {},
                    "contexto": {}, "error": None}
        return {"encontrado": None, "informe": "", "scoring": {},
                "contexto": {}, "error": f"{type(exc).__name__}: {exc}"}


def evaluar_caso(pipeline, caso: dict, juez_llm, usar_ragas: bool) -> dict:
    esperado = caso.get("esperado", {})
    print(f"\n▶ {caso['id']}  (cuit={caso['cuit']})", flush=True)

    gen = _generar(pipeline, caso["cuit"], caso.get("motivo", ""))
    if gen["error"]:
        print(f"   ✖ error generando: {gen['error']}", flush=True)
        return {"id": caso["id"], "cuit": caso["cuit"], "error": gen["error"],
                "checks": [], "resumen": {"pass": False, "ok": 0, "total": 0,
                                          "fallidos": ["error_generacion"]},
                "juez_promedio": None, "ragas": {}}

    checks: list[dict] = []
    checks += metricas.chequear_deterministas(
        esperado, gen["encontrado"], gen["informe"], gen["scoring"])

    veredicto, juez_prom, ragas_scores = {}, None, {}
    # Juez + RAGas sólo tienen sentido si hay un informe real.
    if gen["encontrado"]:
        if any(esperado.get(k) is not None for k in
               ("juez_puntaje_min", "sin_alucinaciones", "cumplimiento_reglas")):
            print("   · juez LLM...", flush=True)
            veredicto = evaluar_informe(gen["contexto"], gen["informe"], llm=juez_llm)
            juez_prom = metricas._puntaje_promedio_juez(veredicto)
            checks += metricas.chequear_juez(esperado, veredicto)

        if usar_ragas and any(esperado.get(k) is not None for k in
                              ("ragas_faithfulness_min", "ragas_answer_relevancy_min")):
            print("   · RAGas...", flush=True)
            # OJO: pasar el sub-dict "rag" (buckets politicas/normativa/bcra/…),
            # no el contexto completo — aplanar_contextos espera esos buckets.
            ragas_scores = ragas_eval.evaluar_ragas(
                caso.get("motivo", ""), gen["informe"],
                (gen["contexto"] or {}).get("rag"))
            checks += metricas.chequear_ragas(esperado, ragas_scores)

    resumen = metricas.resumen_caso(checks)
    icono = "✅" if resumen["pass"] else "❌"
    print(f"   {icono} {resumen['ok']}/{resumen['total']} checks"
          + (f" — fallan: {', '.join(resumen['fallidos'])}"
             if resumen["fallidos"] else ""), flush=True)

    return {"id": caso["id"], "cuit": caso["cuit"], "error": None,
            "encontrado": gen["encontrado"], "checks": checks,
            "resumen": resumen, "veredicto": veredicto,
            "juez_promedio": juez_prom, "ragas": ragas_scores,
            # El informe se guarda en el reporte a propósito: cuando el juez marca
            # una alucinación, lo primero que hace falta es ver EN QUÉ SECCIÓN
            # apareció. Sin esto había que re-correr el golden entero (minutos de
            # LLM) sólo para leer el texto que ya se había generado.
            "informe": gen["informe"]}


def _media(valores: list) -> float | None:
    import math
    valores = [v for v in valores
               if isinstance(v, (int, float)) and not math.isnan(v)]
    return round(sum(valores) / len(valores), 3) if valores else None


def agregar(resultados: list[dict]) -> dict:
    n = len(resultados)
    n_pass = sum(1 for r in resultados if r["resumen"]["pass"])
    total_checks = sum(r["resumen"]["total"] for r in resultados)
    ok_checks = sum(r["resumen"]["ok"] for r in resultados)
    return {
        "casos": n,
        "casos_pass": n_pass,
        "pass_rate": round(n_pass / n, 3) if n else 0.0,
        "checks_totales": total_checks,
        "checks_ok": ok_checks,
        "checks_pass_rate": round(ok_checks / total_checks, 3) if total_checks else 0.0,
        "juez_promedio_medio": _media([r.get("juez_promedio") for r in resultados]),
        "ragas_faithfulness_medio": _media(
            [r.get("ragas", {}).get("faithfulness") for r in resultados]),
        "ragas_answer_relevancy_medio": _media(
            [r.get("ragas", {}).get("answer_relevancy") for r in resultados]),
    }


def imprimir_tabla(resultados: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("RESULTADOS POR CASO")
    print("=" * 78)
    cab = f"{'caso':<26}{'pass':>6}{'checks':>9}{'juez':>7}{'faith':>8}{'a_rel':>8}"
    print(cab)
    print("-" * len(cab))

    def _fmt(val: float | None) -> str:
        return f"{val:.2f}" if isinstance(val, (int, float)) else "—"

    for r in resultados:
        rs = r["resumen"]
        rg = r.get("ragas", {})
        estado = "OK" if rs["pass"] else "FAIL"
        checks = f"{rs['ok']}/{rs['total']}"
        print(f"{r['id'][:25]:<26}{estado:>6}{checks:>9}"
              f"{_fmt(r.get('juez_promedio')):>7}"
              f"{_fmt(rg.get('faithfulness')):>8}"
              f"{_fmt(rg.get('answer_relevancy')):>8}")


def guardar_reporte(resultados: list[dict], resumen: dict,
                    etiqueta: str = "", generador: str = "", juez: str = "") -> str:
    os.makedirs(DIR_REPORTES, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", etiqueta).strip("-") or "run"
    path = os.path.join(DIR_REPORTES, f"golden_{slug}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "etiqueta": etiqueta, "generador": generador,
                   "juez": juez, "resumen": resumen, "casos": resultados},
                  f, ensure_ascii=False, indent=2, default=str)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=os.path.join(RAIZ, "evaluation", "golden_set.json"))
    ap.add_argument("--juez", default=os.environ.get("EVAL_LLM_PROVIDER", "openai"),
                    help="proveedor del juez LLM (default: openai)")
    ap.add_argument("--sin-ragas", action="store_true", help="no correr RAGas")
    ap.add_argument("--limite", type=int, default=None,
                    help="correr sólo los primeros N casos (validación rápida)")
    ap.add_argument("--solo", default=None,
                    help="correr sólo los casos cuyos id estén en esta lista separada por comas")
    ap.add_argument("--generador", default=os.environ.get("PROVEEDOR_LLM", "local"),
                    help="proveedor/modelo que GENERA los informes (default: PROVEEDOR_LLM). "
                         "Sirve para comparar generadores, ej. --generador anthropic vs gemma4:latest")
    ap.add_argument("--etiqueta", default=None,
                    help="etiqueta para el reporte (ej. 'gemma4' o 'opus'); default = generador")
    args = ap.parse_args()

    casos = cargar_casos(args.golden)
    if args.solo:
        ids = {s.strip() for s in args.solo.split(",")}
        casos = [c for c in casos if c["id"] in ids]
    if args.limite is not None:
        casos = casos[:args.limite]
    if not casos:
        print(f"No hay casos con CUIT completado. Falta {CUITS_LOCALES} "
              f"(generalo con muestrear_cuits.py --salida). "
              "Corré  python evaluation/muestrear_cuits.py  para sembrarlo.")
        return 2

    usar_ragas = not args.sin_ragas
    if usar_ragas and not ragas_eval.esta_disponible():
        print("⚠️ ragas no está instalado (pip install ragas datasets); "
              "se omiten las métricas RAGas.")
        usar_ragas = False

    etiqueta = args.etiqueta or args.generador
    print(f"Golden set: {len(casos)} casos | juez={args.juez} | "
          f"ragas={'sí' if usar_ragas else 'no'} | generador={args.generador} "
          f"| etiqueta={etiqueta}")

    from llm.generator import get_llm
    juez_llm = get_llm(args.juez)
    pipeline = get_pipeline(args.generador)

    resultados = [evaluar_caso(pipeline, c, juez_llm, usar_ragas) for c in casos]

    resumen = agregar(resultados)
    imprimir_tabla(resultados)

    print("\n" + "=" * 78)
    print("RESUMEN AGREGADO")
    print("=" * 78)
    print(f"  Casos que pasan:     {resumen['casos_pass']}/{resumen['casos']} "
          f"({resumen['pass_rate']:.0%})")
    print(f"  Checks que pasan:    {resumen['checks_ok']}/{resumen['checks_totales']} "
          f"({resumen['checks_pass_rate']:.0%})")
    print(f"  Juez (promedio/5):   {resumen['juez_promedio_medio']}")
    print(f"  RAGas faithfulness:  {resumen['ragas_faithfulness_medio']}")
    print(f"  RAGas answer_relev.: {resumen['ragas_answer_relevancy_medio']} (informativa)")

    path = guardar_reporte(resultados, resumen, etiqueta=etiqueta,
                           generador=args.generador, juez=args.juez)
    print(f"\n📄 Reporte guardado en: {path}")

    return 0 if resumen["casos_pass"] == resumen["casos"] else 1


if __name__ == "__main__":
    sys.exit(main())
