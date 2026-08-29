"""
Compara dos o más reportes del golden set lado a lado.

Sirve para la evaluación comparativa de generadores: corré run_golden.py con
distintos --generador/--etiqueta y después:

    python evaluation/comparar_reportes.py                 # los 2 reportes más recientes
    python evaluation/comparar_reportes.py a.json b.json    # reportes explícitos

Muestra, por caso, si pasó con cada generador, y un resumen agregado (pass-rate,
promedio del juez, faithfulness). Todo se lee de los JSON en evaluation/reports/;
no consume LLM ni base.
"""
from __future__ import annotations

import glob
import json
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_REPORTES = os.path.join(RAIZ, "evaluation", "reports")


def cargar(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _etiqueta(rep: dict, path: str) -> str:
    return rep.get("etiqueta") or rep.get("generador") or os.path.basename(path)


def main() -> int:
    paths = sys.argv[1:]
    if not paths:
        paths = sorted(glob.glob(os.path.join(DIR_REPORTES, "golden_*.json")))[-2:]
    if len(paths) < 2:
        print("Hacen falta al menos 2 reportes para comparar. "
              "Corré run_golden.py con distintos --generador primero.")
        return 2

    reps = [(p, cargar(p)) for p in paths]
    etiquetas = [_etiqueta(r, p) for p, r in reps]

    # Unión de ids de casos, en el orden del primer reporte.
    ids: list[str] = []
    for _, r in reps:
        for c in r.get("casos", []):
            if c["id"] not in ids:
                ids.append(c["id"])

    # Índice caso->resultado por reporte.
    idx = [{c["id"]: c for c in r.get("casos", [])} for _, r in reps]

    print("=" * (30 + 14 * len(reps)))
    print("COMPARATIVA DE GENERADORES SOBRE EL GOLDEN SET")
    print("=" * (30 + 14 * len(reps)))
    cab = f"{'caso':<28}" + "".join(f"{e[:12]:>14}" for e in etiquetas)
    print(cab)
    print("-" * len(cab))
    for cid in ids:
        fila = f"{cid[:27]:<28}"
        for m in idx:
            c = m.get(cid)
            if not c:
                fila += f"{'—':>14}"
            else:
                rs = c.get("resumen", {})
                estado = "OK" if rs.get("pass") else "FAIL"
                fila += f"{estado + ' ' + str(rs.get('ok','?')) + '/' + str(rs.get('total','?')):>14}"
        print(fila)

    print("-" * len(cab))
    # Agregados por reporte.
    def _linea(nombre, fn):
        return f"{nombre:<28}" + "".join(f"{fn(r):>14}" for _, r in reps)

    print(_linea("pass-rate casos", lambda r: f"{r['resumen'].get('pass_rate', 0):.0%}"))
    print(_linea("checks pass-rate", lambda r: f"{r['resumen'].get('checks_pass_rate', 0):.0%}"))
    print(_linea("juez prom/5", lambda r: str(r['resumen'].get('juez_promedio_medio'))))
    print(_linea("faithfulness", lambda r: str(r['resumen'].get('ragas_faithfulness_medio'))))
    print(_linea("answer_relev*", lambda r: str(r['resumen'].get('ragas_answer_relevancy_medio'))))
    print("\n* answer_relevancy es informativa (no gatea).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
