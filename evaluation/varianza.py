"""Mide la VARIANZA del harness: corre el mismo golden set N veces sin cambiar
nada y reporta cuánto se mueve cada métrica por puro azar del modelo.

Por qué existe
--------------
El golden set tiene 9 casos y el pass/fail es binario por caso, así que un solo
caso que cambia de humor mueve el resultado 11 puntos. En julio se vio la
secuencia 6/9 → 5/9 → 4/9 *mientras las métricas continuas mejoraban*, y los
casos que fallaban no eran los mismos. Sin saber cuánto de eso es ruido, no se
puede decidir si un cambio de prompt mejoró algo: se termina persiguiendo el
azar.

Este script responde tres preguntas:

1. ¿Cuánto se mueve cada métrica entre corridas idénticas? (el piso de ruido)
2. ¿Qué casos y qué checks son inestables? (dónde vive el ruido)
3. ¿Qué tamaño de mejora hace falta para que sea creíble? (el umbral de
   decisión: si un cambio mueve la métrica menos que el ruido, no probó nada)

Uso:
    python evaluation/varianza.py --etiqueta var          # todos los golden_var*.json
    python evaluation/varianza.py reports/a.json reports/b.json reports/c.json
    python evaluation/varianza.py --etiqueta var --json   # salida machine-readable
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_REPORTES = os.path.join(RAIZ, "evaluation", "reports")

# Métricas agregadas que seguimos, con su nombre legible y cuántos decimales
# valen la pena. El pass_rate va primero porque es el número que se usa para
# decidir, y es justamente el que hay que auditar.
METRICAS = [
    ("pass_rate", "Casos que pasan", 3),
    ("checks_pass_rate", "Checks que pasan", 3),
    ("juez_promedio_medio", "Juez (0-5)", 3),
    ("ragas_faithfulness_medio", "RAGas faithfulness", 3),
]


def cargar(paths: list[str]) -> list[dict]:
    corridas = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        d["_archivo"] = os.path.basename(p)
        corridas.append(d)
    return corridas


def _sd(vals: list[float]) -> float:
    """Desvío muestral. Con menos de 2 valores no está definido: devolvemos 0.0
    para no romper la tabla, pero el reporte avisa cuántas corridas hubo."""
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def resumen_agregado(corridas: list[dict]) -> dict:
    out = {}
    for clave, _titulo, _dec in METRICAS:
        vals = [c["resumen"].get(clave) for c in corridas]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        media = statistics.fmean(vals)
        out[clave] = {
            "valores": vals,
            "media": media,
            "sd": _sd(vals),
            "min": min(vals),
            "max": max(vals),
            "rango": max(vals) - min(vals),
            # Coeficiente de variación: permite comparar el ruido de métricas
            # que viven en escalas distintas (una tasa 0-1 vs el juez 0-5).
            "cv": (_sd(vals) / media) if media else 0.0,
        }
    return out


def por_caso(corridas: list[dict]) -> dict:
    """Para cada caso: en cuántas corridas pasó, y cómo se movieron sus métricas.
    Acá es donde se ve si el ruido está repartido o concentrado en dos casos."""
    casos: dict[str, dict] = {}
    for c in corridas:
        for caso in c.get("casos", []):
            e = casos.setdefault(caso["id"], {"pass": [], "juez": [], "faith": [],
                                              "ok": [], "total": []})
            e["pass"].append(bool(caso.get("resumen", {}).get("pass")))
            e["ok"].append(caso.get("resumen", {}).get("ok"))
            e["total"].append(caso.get("resumen", {}).get("total"))
            if caso.get("juez_promedio") is not None:
                e["juez"].append(caso["juez_promedio"])
            f = (caso.get("ragas") or {}).get("faithfulness")
            if f is not None:
                e["faith"].append(f)
    for cid, e in casos.items():
        n = len(e["pass"])
        n_pass = sum(e["pass"])
        e["n"] = n
        e["n_pass"] = n_pass
        # Estable = siempre pasó o nunca pasó. Inestable = cambió de veredicto
        # entre corridas idénticas, que es exactamente el ruido que buscamos.
        e["estabilidad"] = ("siempre" if n_pass == n
                            else "nunca" if n_pass == 0
                            else "INESTABLE")
        e["juez_sd"] = _sd(e["juez"])
        e["faith_sd"] = _sd(e["faith"])
    return casos


def por_check(corridas: list[dict]) -> dict:
    """Qué chequeo concreto es el que se mueve. Un check que falla siempre es
    un bug o un techo del modelo; uno que falla a veces es ruido, y es el que
    contamina el pass/fail."""
    checks: dict[tuple, list[bool]] = {}
    for c in corridas:
        for caso in c.get("casos", []):
            for ch in caso.get("checks", []):
                checks.setdefault((caso["id"], ch["nombre"]), []).append(bool(ch["ok"]))
    out = {}
    for (cid, nombre), vals in checks.items():
        n_ok = sum(vals)
        if 0 < n_ok < len(vals):          # sólo los que cambian de veredicto
            out[f"{cid} / {nombre}"] = {"ok": n_ok, "n": len(vals)}
    return out


def imprimir(corridas: list[dict], agg: dict, casos: dict, flips: dict) -> None:
    n = len(corridas)
    print("=" * 78)
    print(f"VARIANZA DEL HARNESS — {n} corridas idénticas")
    print("=" * 78)
    for c in corridas:
        r = c["resumen"]
        print(f"  {c['_archivo']}  ({c.get('generador','?')})  "
              f"{r['casos_pass']}/{r['casos']} casos")
    if n < 2:
        print("\n⚠️  Con una sola corrida no hay varianza que medir.")
        return
    if n < 3:
        print("\n⚠️  Con 2 corridas el desvío es muy poco confiable; 3 es el mínimo útil.")

    print("\n" + "=" * 78)
    print("MÉTRICAS AGREGADAS (media ± sd, y rango observado)")
    print("=" * 78)
    print(f"{'métrica':<24} {'media':>8} {'sd':>8} {'min':>8} {'max':>8} "
          f"{'rango':>8} {'cv':>7}")
    print("-" * 78)
    for clave, titulo, dec in METRICAS:
        if clave not in agg:
            continue
        a = agg[clave]
        print(f"{titulo:<24} {a['media']:>8.{dec}f} {a['sd']:>8.{dec}f} "
              f"{a['min']:>8.{dec}f} {a['max']:>8.{dec}f} {a['rango']:>8.{dec}f} "
              f"{a['cv']:>6.1%}")

    print("\n" + "=" * 78)
    print("ESTABILIDAD POR CASO")
    print("=" * 78)
    print(f"{'caso':<16} {'pasa':>7}  {'estado':<11} {'juez sd':>8} {'faith sd':>9}")
    print("-" * 78)
    for cid, e in sorted(casos.items(),
                         key=lambda kv: (kv[1]["estabilidad"] != "INESTABLE", kv[0])):
        marca = "⚠️ " if e["estabilidad"] == "INESTABLE" else "  "
        print(f"{marca}{cid:<14} {e['n_pass']}/{e['n']:<5} {e['estabilidad']:<11} "
              f"{e['juez_sd']:>8.2f} {e['faith_sd']:>9.3f}")

    inestables = [c for c, e in casos.items() if e["estabilidad"] == "INESTABLE"]
    print(f"\n  Casos inestables: {len(inestables)}/{len(casos)}"
          + (f" → {', '.join(sorted(inestables))}" if inestables else ""))

    if flips:
        print("\n" + "=" * 78)
        print("CHECKS QUE CAMBIAN DE VEREDICTO ENTRE CORRIDAS IDÉNTICAS")
        print("=" * 78)
        for nombre, e in sorted(flips.items()):
            print(f"  {nombre:<52} pasa {e['ok']}/{e['n']}")

    # ── La conclusión operativa ──────────────────────────────────────────
    print("\n" + "=" * 78)
    print("QUÉ MEJORA HACE FALTA PARA SER CREÍBLE")
    print("=" * 78)
    print("  Regla: un cambio sólo prueba algo si mueve la métrica MÁS que el")
    print("  ruido. Con n corridas por configuración, el umbral aproximado es")
    print("  2·sd·√(2/n) (dos medias independientes, ~95% de confianza).")
    print()
    print(f"{'métrica':<24} {'sd':>8} {'umbral n=1':>12} {'umbral n=3':>12}")
    print("-" * 78)
    for clave, titulo, dec in METRICAS:
        if clave not in agg:
            continue
        sd = agg[clave]["sd"]
        print(f"{titulo:<24} {sd:>8.{dec}f} {2*sd*(2/1)**0.5:>12.{dec}f} "
              f"{2*sd*(2/3)**0.5:>12.{dec}f}")
    if "pass_rate" in agg and "juez_promedio_medio" in agg:
        cv_pass = agg["pass_rate"]["cv"]
        cv_juez = agg["juez_promedio_medio"]["cv"]
        print()
        if cv_pass > cv_juez:
            print(f"  → El pass/fail es {cv_pass/cv_juez:.1f}× más ruidoso que el juez "
                  f"(cv {cv_pass:.1%} vs {cv_juez:.1%}).")
            print("    Para decidir si un cambio mejoró, mirar las métricas continuas.")
        else:
            print(f"  → El pass/fail NO es más ruidoso que el juez "
                  f"(cv {cv_pass:.1%} vs {cv_juez:.1%}).")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reportes", nargs="*", help="rutas a reportes golden_*.json")
    ap.add_argument("--etiqueta", default=None,
                    help="toma todos los reportes cuya etiqueta empiece así "
                         "(ej. --etiqueta var)")
    ap.add_argument("--json", action="store_true", help="salida machine-readable")
    args = ap.parse_args()

    paths = list(args.reportes)
    if args.etiqueta:
        paths += sorted(glob.glob(os.path.join(DIR_REPORTES,
                                               f"golden_{args.etiqueta}*.json")))
    if not paths:
        print("No hay reportes. Pasá rutas o usá --etiqueta.", file=sys.stderr)
        return 2

    corridas = cargar(sorted(set(paths)))
    agg = resumen_agregado(corridas)
    casos = por_caso(corridas)
    flips = por_check(corridas)

    if args.json:
        json.dump({"n_corridas": len(corridas), "agregado": agg,
                   "casos": casos, "checks_inestables": flips},
                  sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        imprimir(corridas, agg, casos, flips)
    return 0


if __name__ == "__main__":
    sys.exit(main())
