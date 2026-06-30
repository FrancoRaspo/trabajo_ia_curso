# Sistema de Scoring Crediticio

Documentación del modelo de scoring: cómo se calcula el score, qué significa cada
parámetro y cómo se explica el resultado de cada cliente.

> **Archivos núcleo**
> - `ml/scoring_model.py` — modelo XGBoost, cálculo del score y bandas de riesgo.
> - `ml/shap_explainer.py` — explicabilidad individual (SHAP) por cliente.
> - `db/client_repository.py` — construcción de los 19 features desde SQL Server.
> - `models/scoring_model.json` — modelo entrenado.

---

## 1. Cómo nace el score (la fórmula)

El modelo **no** calcula el score directamente: predice una **probabilidad de
default** y de ahí se deriva el score (`ml/scoring_model.py:88-95`).

```python
prob_default = model.predict_proba(df)[0, 1]      # 0.0 a 1.0
score        = round((1 - prob_default) * 1000)    # se invierte y escala
score        = max(0, min(1000, score))            # se recorta a 0-1000
```

Idea central: **mayor score = menor riesgo**.

Ejemplo: si el modelo estima 20 % de probabilidad de default, el score es
`(1 - 0.20) × 1000 = 800`.

---

## 2. Bandas de riesgo

Definidas en `ml/scoring_model.py:37-42`.

| Score      | Nivel    | Semáforo    | Significado                          |
|------------|----------|-------------|--------------------------------------|
| 800–1000   | BAJO     | 🟢 VERDE    | Bajo riesgo crediticio               |
| 650–799    | MODERADO | 🟡 AMARILLO | Riesgo moderado                      |
| 500–649    | ALTO     | 🟠 NARANJA  | Riesgo alto                          |
| 0–499      | MUY ALTO | 🔴 ROJO     | Riesgo muy alto                      |

Si el score cae fuera de todo rango, se clasifica como `DESCONOCIDO` / `GRIS`.

---

## 3. Los 19 features (lo que el modelo "mira")

Son los parámetros de entrada al modelo. Se construyen en
`db/client_repository.py` y la lista canónica está en
`ml/scoring_model.py:12-35`. Agrupados por familia conceptual:

### Comportamiento de pago (historial de préstamos)

| Feature                  | Qué significa                                                       |
|--------------------------|--------------------------------------------------------------------|
| `total_prestamos`        | Cuántos préstamos tomó en total. Base de su historial.             |
| `prestamos_al_dia`       | Cuántos están vigentes con 0 días de atraso.                       |
| `prestamos_con_atraso`   | Cuántos tienen atraso actual > 0. Señal negativa.                  |
| `dias_atraso_promedio`   | Promedio de días de atraso (24m). Cuánto se demora típicamente.    |
| `prestamos_refinanciados`| Cuántos refinanció. Señal de estrés financiero pasado.            |

### Puntualidad de cuotas (últimos 24 meses)

| Feature               | Qué significa                                            |
|-----------------------|---------------------------------------------------------|
| `cuotas_puntuales_24m`| Cuotas pagadas en término. **Señal positiva fuerte.**   |
| `atraso_leve_24m`     | Episodios de atraso de 1–30 días.                       |
| `atraso_moderado_24m` | Episodios de atraso de 31–90 días. Más grave que leve.  |

### Liquidez / cuentas (últimos 6 meses)

| Feature                       | Qué significa                                            |
|-------------------------------|---------------------------------------------------------|
| `saldo_promedio_ahorro_6m`    | Plata promedio en caja de ahorro. Colchón financiero.   |
| `saldo_promedio_corriente_6m` | Plata promedio en cuenta corriente.                     |
| `saldo_minimo_6m`             | El piso más bajo que tocó. Bajo/negativo = fragilidad.  |

### Endeudamiento y capacidad

| Feature                   | Qué significa                                                                    |
|---------------------------|---------------------------------------------------------------------------------|
| `deuda_vigente_total`     | Suma de capital que todavía debe.                                               |
| `capacidad_pago_estimada` | 30 % del ingreso neto declarado (criterio estándar).                           |
| `ratio_deuda_ingresos`    | `deuda / ingreso`. Cuánto pesa la deuda sobre lo que gana (999 si no hay ingreso declarado — valor sentinel). |

### Relación con la entidad

| Feature                     | Qué significa                                                            |
|-----------------------------|-------------------------------------------------------------------------|
| `antiguedad_meses`          | Hace cuánto es socio. Más antigüedad = más confianza.                   |
| `productos_activos`         | Cuántas cuentas/productos tiene activos.                                |
| `ultimo_prestamo_monto`     | Monto del último préstamo otorgado.                                     |
| `ultimo_prestamo_hace_meses`| Hace cuánto fue ese último préstamo.                                    |
| `tasa_cumplimiento`         | `prestamos_al_dia / total_prestamos` (0 si no tiene préstamos). Resumen de cumplimiento. |

> **Features desactivados.** En `ml/scoring_model.py:16-23` están comentados
> `max_dias_atraso_historico`, `prestamos_irrecuperables` y `atraso_grave_24m`.
> El motivo anotado: en la base de datos el estado se llama `IRRECUPERABLE`, no
> `CASTIGADO`. Pendiente evaluar si conviene reactivarlos.

---

## 4. Hiperparámetros del modelo XGBoost

Configuración del algoritmo de aprendizaje (no del cliente),
`ml/scoring_model.py:47-56`.

| Parámetro          | Valor     | Qué controla                                                              |
|--------------------|-----------|--------------------------------------------------------------------------|
| `n_estimators`     | 300       | Cantidad de árboles. Más árboles = más capacidad y más riesgo de sobreajuste. |
| `max_depth`        | 5         | Profundidad de cada árbol. Limita la complejidad de cada regla.          |
| `learning_rate`    | 0.05      | Cuánto aporta cada árbol. Bajo = aprende lento pero más robusto.         |
| `subsample`        | 0.8       | Usa 80 % de las filas por árbol. Reduce sobreajuste.                     |
| `colsample_bytree` | 0.8       | Usa 80 % de los features por árbol. Diversifica los árboles.            |
| `eval_metric`      | `auc`     | Mide calidad por AUC-ROC durante el entrenamiento.                       |
| `random_state`     | 42        | Semilla para reproducibilidad.                                           |
| `scale_pos_weight` | dinámico  | Compensa el desbalance de clases: `pagadores / defaulteados` (`scoring_model.py:66`). |

**Métrica de validación:** AUC-ROC (`scoring_model.py:84`). Mide qué tan bien el
modelo separa pagadores de defaulteados, independiente del umbral de corte.

---

## 5. Explicabilidad: dos tipos de drivers

El método `predict()` devuelve dos listas distintas que conviene no confundir.

### a) `principales_drivers` — importancia GLOBAL

`ml/scoring_model.py:98-99`. Top 5 features que más pesan en el modelo *en
general*, iguales para todos los clientes. Vienen de `feature_importances_` de
XGBoost.

```python
{"feature": "dias_atraso_promedio", "importancia": 0.18}
```

### b) `drivers_cliente` — SHAP INDIVIDUAL

`ml/shap_explainer.py:60`. Top 5 features que movieron la aguja *en ESTE cliente
puntual*. Usa TreeSHAP nativo de XGBoost (`pred_contribs=True`), sin necesidad de
instalar la librería `shap` aparte. Cada driver:

```python
{
  "variable":     "dias_atraso_promedio",
  "etiqueta":     "Días de atraso promedio",   # nombre legible para el informe
  "valor":        12,                            # valor actual del cliente
  "contribucion": 0.43,                          # empuje en log-odds
  "peso":         21.5,                           # % sobre el total de contribuciones
  "direccion":    "aumenta el riesgo"            # o "reduce el riesgo"
}
```

- `contribucion`: cuánto empujó, en unidades de log-odds. Sirve para ranking y
  dirección, **no** como puntos de probabilidad literales.
- `peso`: % que ese factor representa sobre la suma de TODAS las contribuciones
  (en valor absoluto). Si están todos parejos, ninguno domina.

**Parámetros de `explicar_cliente()`** (`ml/shap_explainer.py:60-62`):

| Parámetro             | Default | Significado                                                                 |
|-----------------------|---------|-----------------------------------------------------------------------------|
| `top_n`               | 5       | Cuántos drivers devolver (los de mayor impacto absoluto).                   |
| `mayor_es_mas_riesgo` | `True`  | El modelo predice prob_default (mayor salida = más riesgo); ajusta la dirección. |
| `min_peso`            | 2.0     | Descarta factores con < 2 % de peso. Si todos quedan abajo, devuelve igual los de mayor peso (nunca vacío). |

Las etiquetas legibles de cada feature están en `ml/shap_explainer.py:17-37`.

---

## 6. Flag `sin_historial` (thin-file)

`ml/scoring_model.py:112-118`.

```python
sin_historial = not features.get("total_prestamos")  # True si 0, None o ausente
```

Si el cliente nunca tomó préstamos, los features de historial son valores de
relleno (sentinel). Un score alto en ese caso refleja **ausencia de señal
negativa, no comportamiento de pago demostrado**. Por eso se marca el flag, para
que el informe lo aclare y sugiera verificación manual.

---

## 7. Salida completa de `predict()`

`ml/scoring_model.py:120-131`.

```python
{
    "score":               int,        # 0-1000
    "prob_default":        float,      # 0.0-1.0, redondeado a 4 decimales
    "nivel_riesgo":        str,        # BAJO | MODERADO | ALTO | MUY ALTO | DESCONOCIDO
    "semaforo":            str,        # VERDE | AMARILLO | NARANJA | ROJO | GRIS
    "sin_historial":       bool,       # True si no hay préstamos previos
    "principales_drivers": list[dict], # Top 5 features globales del modelo
    "drivers_cliente":     list[dict], # Top 5 SHAP de este cliente específico
}
```

---

## 8. Constantes y ventanas temporales

| Parámetro          | Valor       | Dónde                              |
|--------------------|-------------|------------------------------------|
| Escala del score   | 0–1000      | `scoring_model.py:94`              |
| Capacidad de pago  | 30 % ingreso| `client_repository.py` (criterio)  |
| Ventana de cuotas  | 24 meses    | `client_repository.py`             |
| Ventana de saldos  | 6 meses     | `client_repository.py`             |
| Top drivers        | 5           | `scoring_model.py:99,105`          |
| Mínimo peso SHAP   | 2.0 %       | `shap_explainer.py:62,76`          |
| Umbral clasificación| 0.5 prob   | `scoring_model.py:85` (solo report)|
