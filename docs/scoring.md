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

**Las bandas se cortan sobre la PD, no sobre el score.** Los cortes reales los
calcula el entrenamiento y viven en `models/bandas_riesgo.json`; el fallback
está en `CreditScoringModel.BANDAS_RIESGO_PD`.

Cada corte es un múltiplo de la **tasa de default de la cartera** (6,42 %), que
es lo que un comité puede leer sin traducir: *"este cliente tiene 3x el riesgo
promedio de la cartera"*.

| PD (prob. de default) | Múltiplo | Nivel    | Semáforo    |
|-----------------------|----------|----------|-------------|
| > 25,7 %              | > 4x     | MUY ALTO | 🔴 ROJO     |
| 12,8 % – 25,7 %       | 2x – 4x  | ALTO     | 🟠 NARANJA  |
| 6,4 % – 12,8 %        | 1x – 2x  | MODERADO | 🟡 AMARILLO |
| < 6,4 %               | < 1x     | BAJO     | 🟢 VERDE    |

Si la PD cae fuera de todo rango, se clasifica como `DESCONOCIDO` / `GRIS`.

> **Por qué no se cortan sobre el score.** El esquema viejo cortaba el score en
> 800 / 650 / 500. Eso sólo funcionaba porque el modelo anterior usaba
> `scale_pos_weight`, que infla las probabilidades y las esparce por todo
> `[0,1]`. Con una PD real la masa vive entre 2 % y 25 %, el score entre ~750 y
> ~980, y las bandas ALTO y MUY ALTO quedaban **inalcanzables**: todo cliente
> habría salido BAJO.

### Verificación de las bandas

Una banda que promete PD 6–13 % pero cuyos clientes defaultean al 19 % no es una
banda: es una mentira con color. El entrenamiento contrasta la tasa **observada**
en cada banda contra el rango de PD que esa banda promete, usando el IC 95 % de
Wilson (para no confundir una banda rota con el ruido de pocos casos), y lo hace
en los tres splits. Si el IC entero queda fuera del rango prometido, la banda se
marca como inconsistente en `bandas_riesgo.json` y el script lo grita.

Resultado del modelo actual — las cuatro bandas cierran en TEST:

| Banda    | Rango de PD | n | defaults | tasa real | IC 95 % | |
|----------|-------------|---|----------|-----------|---------|---|
| MUY ALTO | > 25,7 %    | 90 | 35 | 38,9 % | [29,5 % – 49,2 %] | ✅ |
| ALTO     | 12,8–25,7 % | 130 | 29 | 22,3 % | [16,0 % – 30,2 %] | ✅ |
| MODERADO | 6,4–12,8 %  | 242 | 20 | 8,3 %  | [5,4 % – 12,4 %]  | ✅ |
| BAJO     | < 6,4 %     | 1746 | 39 | 2,2 % | [1,6 % – 3,0 %]   | ✅ |

---

## 2b. Calibración: por qué hoy no hay ninguna

La PD que publica el informe es **la salida cruda del booster**, sin
transformar. No es un descuido: es el resultado de medirlo.

El modelo entrena con `binary:logistic` —que optimiza log-loss, un *scoring rule*
propio— y **omite `scale_pos_weight`** justamente para no inflar las
probabilidades. Bajo esas dos condiciones, `predict_proba()` ya devuelve una
probabilidad, no un puntaje de ranking.

Una versión anterior aplicaba encima una **sigmoide de Platt**, justificada con
el argumento de que "predict_proba no es una probabilidad calibrada, es un
puntaje de ranking". Ese argumento era falso para este modelo, y la corrección
hacía daño real:

- degradaba el Brier de test de **0,0442 → 0,0452**, y
- **rompía la banda MODERADO**: prometía PD 6,4–12,8 % y sus clientes
  defaulteaban al **19,5 %** (IC 95 % [13,5 % – 27,4 %], entero por encima del
  techo de la banda). Ese número iba al comité de crédito.

La causa es que una sigmoide de 2 parámetros no puede representar la forma del
error. En la curva de confiabilidad out-of-fold, Platt **sub-predecía** en los
deciles 8–9 (ratio real/predicho ≈ 1,6) mientras **sobre-predecía** en los
deciles bajos.

Ahora la calibración **se elige, no se asume**: el entrenamiento compara *no
calibrar* contra *Platt* contra *isotónica* por Brier fuera de muestra en una CV
temporal anidada sobre las OOF (ajustar en los cortes viejos, medir en el
siguiente), y gana el mejor.

| Método    | Brier medio out-of-sample |
|-----------|---------------------------|
| **ninguna**   | **0,04135** ← elegido |
| Platt         | 0,04375 |
| isotónica     | 0,04413 |

`models/calibracion.json` guarda el método elegido y la evidencia de la
selección. `CreditScoringModel._calibrar` sabe aplicar los tres, así que si un
reentrenamiento futuro produce un modelo mal calibrado, el mecanismo lo detecta
y lo corrige solo.

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
| `saldo_promedio_corriente_6m` | Plata promedio en cuenta corriente. Hoy siempre 0: la mutual no opera cuentas corrientes. |
| `saldo_minimo_6m`             | El piso más bajo que tocó. Bajo/negativo = fragilidad.  |

### Endeudamiento y capacidad

| Feature                   | Qué significa                                                                    |
|---------------------------|---------------------------------------------------------------------------------|
| `deuda_vigente_total`     | Suma de capital que todavía debe.                                               |
| `capacidad_pago_estimada` | 30 % del ingreso neto declarado (criterio estándar).                           |
| `ratio_deuda_ingresos`    | `deuda / capacidad_pago_estimada`. Cuánto pesa la deuda sobre lo que puede afrontar (999 si no hay ingreso declarado — valor sentinel). |

### Relación con la entidad

| Feature                     | Qué significa                                                            |
|-----------------------------|-------------------------------------------------------------------------|
| `antiguedad_meses`          | Hace cuánto es socio. Más antigüedad = más confianza.                   |
| `productos_activos`         | Cuántas cuentas a la vista con movimientos en los últimos 6 meses.      |
| `ultimo_prestamo_monto`     | Monto del último préstamo otorgado.                                     |
| `ultimo_prestamo_hace_meses`| Hace cuánto fue ese último préstamo.                                    |
| `tasa_cumplimiento`         | `prestamos_al_dia / total_prestamos` (0 si no tiene préstamos). Resumen de cumplimiento. |

> **Features excluidas por fuga de datos.** `prestamos_irrecuperables` ES el
> target (clasificación de gerencia > 4) y `atraso_grave_24m` cuenta cuotas con
> +90 días de mora, que es exactamente lo que el target define. Ninguna de las
> dos puede ser feature.
>
> `max_dias_atraso_historico` SÍ se usa: medida al corte queda acotada a [0, 89]
> por la regla de elegibilidad (quien ya llegó a 90 dpd está en default y se
> excluye), así que no filtra el target y es un predictor legítimo.
>
> **Features descartadas por constantes:** `prestamos_refinanciados` (todo
> refinanciado ya estaba en mora 90+ y quedó fuera de la población elegible) y
> `saldo_promedio_corriente_6m` (la mutual no tiene ninguna línea de cuenta
> corriente; ver `LIN_CTAVISTA`). La lista real y vigente vive en
> `models/scoring_features.json`, que genera el entrenamiento.

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
