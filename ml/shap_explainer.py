"""
Explicabilidad por cliente (SHAP) para el modelo de scoring XGBoost.

A diferencia de la importancia global del modelo (las mismas variables para
todos), esto descompone la predicción de UN cliente puntual: cuánto empujó
cada variable el resultado de ESE caso. Ideal para justificar una decisión
individual en el informe al directorio.

Usa el TreeSHAP nativo de XGBoost (pred_contribs=True), así que NO hace falta
instalar la librería 'shap' aparte.
"""
import numpy as np


# Nombres legibles para el informe (uno por cada feature de FEATURE_COLUMNS).
# Si una variable no está acá, se usa su nombre técnico tal cual.
ETIQUETAS_FEATURES = {
    "total_prestamos":             "Cantidad total de préstamos",
    "prestamos_al_dia":            "Préstamos al día",
    "prestamos_con_atraso":        "Préstamos con atraso",
    "dias_atraso_promedio":        "Días de atraso promedio",
    "prestamos_refinanciados":     "Préstamos refinanciados",
    "cuotas_puntuales_24m":        "Cuotas pagadas en término (últimos 24 meses)",
    "atraso_leve_24m":             "Episodios de atraso leve, 1–30 días (últimos 24 meses)",
    "atraso_moderado_24m":         "Episodios de atraso moderado, 31–90 días (últimos 24 meses)",
    "saldo_promedio_ahorro_6m":    "Saldo promedio en caja de ahorro (últimos 6 meses)",
    "saldo_promedio_corriente_6m": "Saldo promedio en cuenta corriente (últimos 6 meses)",
    "saldo_minimo_6m":             "Saldo mínimo (últimos 6 meses)",
    "deuda_vigente_total":         "Deuda vigente total",
    "capacidad_pago_estimada":     "Capacidad de pago estimada",
    "antiguedad_meses":            "Antigüedad como socio (meses)",
    "productos_activos":           "Productos activos",
    "ultimo_prestamo_monto":       "Monto del último préstamo",
    "ultimo_prestamo_hace_meses":  "Meses desde el último préstamo",
    "tasa_cumplimiento":           "Tasa de cumplimiento de pagos",
    "ratio_deuda_ingresos":        "Relación deuda / ingresos",
}


def _etiqueta(nombre: str) -> str:
    return ETIQUETAS_FEATURES.get(nombre, nombre)


def _num(v):
    """Coerciona numpy/Decimal a tipos Python simples (para JSON)."""
    try:
        f = float(v)
        return int(f) if f.is_integer() else round(f, 4)
    except Exception:
        return v


def _booster(model):
    """Devuelve el xgboost.Booster subyacente, sea wrapper sklearn o Booster."""
    if hasattr(model, "get_booster"):       # XGBClassifier / XGBRegressor
        return model.get_booster()
    return model                            # ya es un Booster


def explicar_cliente(model, features, top_n: int = 5,
                     mayor_es_mas_riesgo: bool = True,
                     min_peso: float = 2.0) -> list:
    """
    Drivers SHAP del cliente: las variables que más pesaron en SU predicción.

    Parámetros
    ----------
    model : modelo XGBoost (Booster o wrapper sklearn XGBClassifier/Regressor)
    features : dict {nombre: valor}, pandas Series o DataFrame de 1 fila.
        Los nombres deben coincidir con los del entrenamiento.
    top_n : cuántos drivers devolver (los de mayor impacto absoluto).
    mayor_es_mas_riesgo : True si la salida del modelo CRECE con el riesgo
        (ej. predice probabilidad de default). Si tu modelo devuelve un "score
        bueno" donde más alto = menos riesgo, poné False para que la dirección
        ('aumenta'/'reduce el riesgo') quede bien.
    min_peso : descarta factores cuyo 'peso' (% del total) sea menor a esto.
        Evita mostrar variables que en realidad no movieron la aguja. Si tras
        filtrar no queda ninguno, devuelve igual los de mayor peso (no vacío).

    Devuelve
    --------
    list[dict] ordenada por impacto, cada uno:
        {"variable", "etiqueta", "valor", "contribucion", "peso", "direccion"}
    'peso' es el % que ese factor representa sobre la suma de TODAS las
    contribuciones (en valor absoluto): si están todos parejos, ninguno domina.
    Nota: 'contribucion' está en unidades de margen del modelo (log-odds para
    binary:logistic), sirve para ranking y dirección, no como puntos de
    probabilidad literales.
    """
    import xgboost as xgb
    import pandas as pd

    # Normalizar features a un DataFrame de 1 fila con nombres de columna
    if isinstance(features, dict):
        X = pd.DataFrame([features])
    elif isinstance(features, pd.Series):
        X = features.to_frame().T
    elif isinstance(features, pd.DataFrame):
        X = features.iloc[[0]].copy()
    else:
        raise TypeError("features debe ser dict, pandas Series o DataFrame de 1 fila")

    bst = _booster(model)
    dm  = xgb.DMatrix(X)
    # pred_contribs -> (n_filas, n_features + 1); la última columna es el bias/base.
    contribs = np.asarray(bst.predict(dm, pred_contribs=True))[0][:-1]

    # Peso relativo sobre el TOTAL de contribuciones (en valor absoluto).
    total_abs = float(np.abs(contribs).sum()) or 1.0

    filas = []
    for nombre, contrib, valor in zip(X.columns, contribs, X.iloc[0].values):
        signo = contrib if mayor_es_mas_riesgo else -contrib
        filas.append({
            "variable":     str(nombre),
            "etiqueta":     _etiqueta(str(nombre)),
            "valor":        _num(valor),
            "contribucion": round(float(contrib), 4),
            "peso":         round(float(100.0 * abs(contrib) / total_abs), 1),
            "direccion":    "aumenta el riesgo" if signo > 0 else "reduce el riesgo",
        })
    filas.sort(key=lambda d: abs(d["contribucion"]), reverse=True)

    # Filtrar factores despreciables; si quedara vacío, devolver los de mayor peso.
    significativos = [f for f in filas if f["peso"] >= min_peso]
    if not significativos:
        significativos = filas
    return significativos[:top_n]