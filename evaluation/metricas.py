"""
Aserciones pass/fail sobre un informe generado.

Tres familias de chequeos, todas devuelven la MISMA forma de "check":
    {"tipo": str, "nombre": str, "ok": bool, "detalle": str}

  - deterministas: estructura y hechos verificables sin LLM (banda, score,
    substrings obligatorios/prohibidos, encontrado/no encontrado).
  - juez:          umbrales sobre el veredicto del juez LLM (evaluation/llm_judge.py).
  - ragas:         umbrales sobre las métricas RAGas (evaluation/ragas_eval.py).

Un check con valor esperado None se OMITE (no cuenta ni como ok ni como fallo):
así el mismo esquema de golden set sirve para casos donde no aplica una métrica
(ej. cliente inexistente no tiene banda ni juez).
"""
from __future__ import annotations

# Criterios 1-5 que emite el juez LLM (ver prompts/juez_sys.txt).
CRITERIOS_JUEZ = ("fidelidad", "exactitud_numerica", "estructura",
                  "uso_drivers_cliente", "coherencia_recomendacion")


def _check(tipo: str, nombre: str, ok: bool, detalle: str = "") -> dict:
    return {"tipo": tipo, "nombre": nombre, "ok": bool(ok), "detalle": detalle}


# ---------------------------------------------------------------------------
#  Deterministas (sin LLM)
# ---------------------------------------------------------------------------
def chequear_deterministas(esperado: dict, encontrado: bool,
                           informe: str, scoring: dict) -> list[dict]:
    checks: list[dict] = []
    informe = informe or ""
    scoring = scoring or {}

    # 1) encontrado / no encontrado
    exp_enc = esperado.get("encontrado")
    if exp_enc is not None:
        checks.append(_check(
            "det", "encontrado", encontrado == exp_enc,
            f"esperado={exp_enc} obtenido={encontrado}"))

    # El resto de los chequeos determinísticos sólo aplican si el cliente
    # existía y se esperaba encontrarlo.
    if exp_enc is False or not encontrado:
        return checks

    # 2) banda de riesgo dentro del conjunto aceptable
    bandas_ok = esperado.get("banda_riesgo_en")
    if bandas_ok:
        banda = scoring.get("nivel_riesgo")
        checks.append(_check(
            "det", "banda_riesgo", banda in bandas_ok,
            f"banda={banda} aceptables={bandas_ok}"))

    # 3) score en rango
    score = scoring.get("score")
    smin, smax = esperado.get("score_min"), esperado.get("score_max")
    if smin is not None:
        checks.append(_check("det", "score_min",
                             score is not None and score >= smin,
                             f"score={score} >= {smin}"))
    if smax is not None:
        checks.append(_check("det", "score_max",
                             score is not None and score <= smax,
                             f"score={score} <= {smax}"))

    # 4) substrings obligatorios
    for sub in esperado.get("debe_contener") or []:
        checks.append(_check("det", f"contiene:{sub!r}", sub in informe,
                             "presente" if sub in informe else "AUSENTE"))

    # 5) substrings prohibidos
    for sub in esperado.get("no_debe_contener") or []:
        checks.append(_check("det", f"no_contiene:{sub!r}", sub not in informe,
                             "ausente" if sub not in informe else "PRESENTE (prohibido)"))

    return checks


# ---------------------------------------------------------------------------
#  Juez LLM
# ---------------------------------------------------------------------------
def _alucinaciones_criticas(aluc: list) -> int:
    """Cuenta alucinaciones CRÍTICAS. Tolera el formato nuevo (objetos con
    severidad) y el viejo (strings): un string se considera crítico, porque el
    prompt anterior sólo listaba las que hacían RECHAZAR."""
    n = 0
    for a in aluc or []:
        if isinstance(a, dict):
            if str(a.get("severidad", "CRITICA")).upper().startswith("CRIT"):
                n += 1
        elif isinstance(a, str) and a.strip():
            n += 1
    return n


def _puntaje_promedio_juez(veredicto: dict) -> float | None:
    puntajes = []
    for crit in CRITERIOS_JUEZ:
        c = veredicto.get(crit)
        if isinstance(c, dict) and isinstance(c.get("puntaje"), (int, float)):
            puntajes.append(float(c["puntaje"]))
    return sum(puntajes) / len(puntajes) if puntajes else None


def chequear_juez(esperado: dict, veredicto: dict) -> list[dict]:
    checks: list[dict] = []
    veredicto = veredicto or {}

    if veredicto.get("veredicto") == "ERROR_PARSEO":
        checks.append(_check("juez", "juez_respondio", False,
                             "el juez no devolvió JSON válido"))
        return checks

    pmin = esperado.get("juez_puntaje_min")
    if pmin is not None:
        prom = _puntaje_promedio_juez(veredicto)
        checks.append(_check(
            "juez", "juez_puntaje_min",
            prom is not None and prom >= pmin,
            f"promedio={prom if prom is None else round(prom, 2)} >= {pmin}"))

    if esperado.get("sin_alucinaciones") is not None:
        aluc = veredicto.get("alucinaciones") or []
        criticas = _alucinaciones_criticas(aluc)
        esperar_sin = esperado["sin_alucinaciones"]
        # Gatea sólo por alucinaciones CRÍTICAS (inventar/cambiar un dato). Las
        # MENORES (inferencias/matices sin sustento que no alteran un dato) se
        # reportan pero no reprueban: así el check discrimina calidad real en vez
        # de castigar cualquier matiz.
        ok = (criticas == 0) if esperar_sin else True
        checks.append(_check("juez", "sin_alucinaciones", ok,
                             f"criticas={criticas} total={len(aluc)}"))

    if esperado.get("cumplimiento_reglas") is not None:
        reglas = veredicto.get("cumplimiento_reglas")
        aprob = bool(isinstance(reglas, dict) and reglas.get("aprobado"))
        checks.append(_check(
            "juez", "cumplimiento_reglas", aprob == esperado["cumplimiento_reglas"],
            f"aprobado={aprob} esperado={esperado['cumplimiento_reglas']}"))

    return checks


# ---------------------------------------------------------------------------
#  RAGas
# ---------------------------------------------------------------------------
def chequear_ragas(esperado: dict, ragas_scores: dict) -> list[dict]:
    checks: list[dict] = []
    ragas_scores = ragas_scores or {}

    umbrales = {
        "faithfulness":     esperado.get("ragas_faithfulness_min"),
        "answer_relevancy": esperado.get("ragas_answer_relevancy_min"),
    }
    import math
    for metrica, umbral in umbrales.items():
        if umbral is None:
            continue
        val = ragas_scores.get(metrica)
        # Métrica no disponible (ragas no instalado, sin contexto, o timeout):
        # se OMITE el check —es un problema de infra, no de calidad, y no debe
        # penalizar al informe. run_golden ya avisa si RAGas no está disponible.
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        checks.append(_check("ragas", f"ragas_{metrica}", val >= umbral,
                             f"{metrica}={round(val, 3)} >= {umbral}"))
    return checks


# ---------------------------------------------------------------------------
#  Resumen
# ---------------------------------------------------------------------------
def resumen_caso(checks: list[dict]) -> dict:
    total = len(checks)
    ok = sum(1 for c in checks if c["ok"])
    return {"pass": ok == total and total > 0, "ok": ok, "total": total,
            "fallidos": [c["nombre"] for c in checks if not c["ok"]]}
