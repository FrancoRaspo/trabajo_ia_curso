"""
Tests deterministas de la implementación.

Correr desde la raíz del proyecto:
    pip install pytest
    pytest -q

No necesitan SQL Server, ni Postgres, ni LLM: entrenan un modelo en memoria
y prueban la lógica pura. La evaluación generativa (juez LLM) va aparte.
"""
import numpy as np
import pandas as pd
import pytest


# ===========================================================================
#  SHAP / explicabilidad por cliente  (ml/shap_explainer.py)
# ===========================================================================
@pytest.fixture(scope="module")
def modelo():
    """Modelo XGBoost chico entrenado en memoria, con señal conocida."""
    import xgboost as xgb
    cols = ["tasa_cumplimiento", "prestamos_con_atraso", "ratio_deuda_ingresos",
            "saldo_minimo_6m", "antiguedad_meses"]
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(1500, len(cols))), columns=cols)
    logit = (-1.5 * X["tasa_cumplimiento"] + 1.4 * X["prestamos_con_atraso"]
             + 1.1 * X["ratio_deuda_ingresos"] - 0.8 * X["saldo_minimo_6m"])
    y = (rng.uniform(size=1500) < 1 / (1 + np.exp(-logit))).astype(int)
    m = xgb.XGBClassifier(n_estimators=60, max_depth=3, learning_rate=0.2,
                          eval_metric="logloss")
    m.fit(X, y)
    m._cols = cols
    return m


def test_shap_cliente_riesgoso_aumenta_riesgo(modelo):
    from ml.shap_explainer import explicar_cliente
    cliente = {c: 0.0 for c in modelo._cols}
    cliente.update({"tasa_cumplimiento": -2.0, "prestamos_con_atraso": 2.5,
                    "ratio_deuda_ingresos": 2.0})
    drivers = explicar_cliente(modelo, cliente, top_n=3, mayor_es_mas_riesgo=True)
    assert len(drivers) == 3
    assert drivers[0]["direccion"] == "aumenta el riesgo"
    for d in drivers:
        assert {"variable", "etiqueta", "valor", "contribucion", "direccion"} <= set(d)


def test_shap_cliente_bueno_reduce_riesgo(modelo):
    from ml.shap_explainer import explicar_cliente
    cliente = {c: 0.0 for c in modelo._cols}
    cliente.update({"tasa_cumplimiento": 2.0, "prestamos_con_atraso": -1.5,
                    "ratio_deuda_ingresos": -1.5})
    drivers = explicar_cliente(modelo, cliente, top_n=3, mayor_es_mas_riesgo=True)
    assert drivers[0]["direccion"] == "reduce el riesgo"


def test_shap_flag_invierte_direccion(modelo):
    from ml.shap_explainer import explicar_cliente
    cliente = {c: 0.0 for c in modelo._cols}
    cliente.update({"tasa_cumplimiento": -2.0, "prestamos_con_atraso": 2.5})
    d_true = explicar_cliente(modelo, cliente, top_n=1, mayor_es_mas_riesgo=True)[0]
    d_false = explicar_cliente(modelo, cliente, top_n=1, mayor_es_mas_riesgo=False)[0]
    assert d_true["direccion"] != d_false["direccion"]


def test_etiqueta_fallback():
    from ml.shap_explainer import _etiqueta
    assert _etiqueta("tasa_cumplimiento") == "Tasa de cumplimiento de pagos"
    assert _etiqueta("variable_que_no_existe") == "variable_que_no_existe"


# ===========================================================================
#  Limpieza de texto (rag/pgvector_indexer.py)
# ===========================================================================
def test_limpiar_texto_quita_nul_conserva_saltos():
    from rag.pgvector_indexer import limpiar_texto
    sucio = "Recibo\x00 sueldo\x07: $850\nlinea2\ttab\rcr"
    limpio = limpiar_texto(sucio)
    assert "\x00" not in limpio and "\x07" not in limpio
    assert "\n" in limpio and "\t" in limpio and "\r" in limpio
    assert "Recibo sueldo: $850" in limpio


# ===========================================================================
#  ReportValidator  (pipeline.py)
# ===========================================================================
def _informe_completo(score):
    return (
        f"## RESUMEN EJECUTIVO\nPerfil moderado, score {score}.\n"
        "## CLASIFICACIÓN DE RIESGO\nNivel MODERADO.\n"
        "## HISTORIAL CREDITICIO\nSin mora grave.\n"
        "## RECOMENDACIÓN\nAprobación condicionada.\n"
        "La decisión final corresponde al directorio."
    )


def test_validador_informe_correcto_aprueba():
    from pipeline import ReportValidator
    ctx = {"scoring": {"score": 724}}
    r = ReportValidator().validar(_informe_completo(724), ctx)
    assert r["aprobado"] is True
    assert r["advertencias"] == []


def test_validador_score_ausente_marca_alucinacion():
    from pipeline import ReportValidator
    ctx = {"scoring": {"score": 724}}
    # informe con OTRO score -> el 724 no aparece -> ALERTA -> no aprobado
    r = ReportValidator().validar(_informe_completo(999), ctx)
    assert r["aprobado"] is False
    assert any("ALERTA" in a for a in r["advertencias"])


def test_validador_seccion_faltante_advierte_pero_no_bloquea():
    from pipeline import ReportValidator
    ctx = {"scoring": {"score": 724}}
    sin_reco = _informe_completo(724).replace("## RECOMENDACIÓN", "## OTRA COSA")
    r = ReportValidator().validar(sin_reco, ctx)
    # falta sección => advertencia, pero aprobado sigue True (no es ALERTA)
    assert any("RECOMENDACIÓN" in a for a in r["advertencias"])
    assert r["aprobado"] is True


# ===========================================================================
#  Parser del juez LLM  (evaluation/llm_judge.py) — sin llamar a ningún modelo
# ===========================================================================
def test_juez_parser_json_con_fences():
    from evaluation.llm_judge import _parsear_json
    out = _parsear_json('```json\n{"veredicto": "APROBADO", "alucinaciones": []}\n```')
    assert out["veredicto"] == "APROBADO"


def test_juez_parser_json_con_texto_alrededor():
    from evaluation.llm_judge import _parsear_json
    out = _parsear_json('Mi análisis: {"veredicto": "REVISAR"} listo.')
    assert out["veredicto"] == "REVISAR"


def test_juez_parser_basura_no_rompe():
    from evaluation.llm_judge import _parsear_json
    out = _parsear_json("no devolví json")
    assert out["veredicto"] == "ERROR_PARSEO"