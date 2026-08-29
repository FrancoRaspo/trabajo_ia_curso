# Estado del sistema y camino a producción

> Fecha del diagnóstico: 2026-07-12
> Alcance: revisión completa del repo (modelo, evaluación, pipeline, datos, docs) + comparación con los sistemas de riesgo crediticio de referencia del mercado.

---

## 1. Resumen ejecutivo

El **modelo de scoring es la pieza fuerte** y ya está a nivel profesional: split temporal genuino, anti-fuga verificado, model card y métricas dentro del rango bueno de la industria.

Lo que está flojo es **todo lo que lo rodea**: el harness de evaluación del informe nunca pasa, la documentación describe en parte un modelo viejo que el propio código declara erróneo, y la pieza más crítica (la query de features) no tiene tests.

**Conclusión:** la credibilidad técnica ya la tenemos. Lo que hoy no resiste una pregunta incómoda es el envoltorio.

> **Cambios del 2026-07-12.** Se cerraron los dos primeros bloqueantes:
> 1. **Trazabilidad** — el log auditable declaraba 5 tablas que no existen y un `"XGBoost v1.0"` hardcodeado. Ahora publica las 27 tablas reales y la identidad del modelo desde el model card, con un test que impide que vuelva a divergir.
> 2. **Banda MODERADO** — no era la banda: era la **calibración de Platt**, que sub-predecía en la zona media-alta. Eliminada. Las 4 bandas cierran en los 3 splits y el Brier de test mejoró (0,0452 → 0,0442). El AUC no cambia: la calibración es monótona y no altera el ranking.
>
> Sobre el tercer bloqueante (golden set): se **verificó** y se cerró la mayoría de las causas (fechas, CUIT, SQL, conteo de cartera, género), y se llegó a citar **6/9 (67 %)**. Sigue **en progreso**. Ver 4.4, 4.4b, 4.4c y sobre todo **4.4d**.
>
> **Corrección del 29-ago-2026 (4.4d).** Ese 67 % no era el estado del sistema sino una corrida afortunada: medida la varianza con 3 corridas idénticas, el pass/fail por caso tiene un cv del **25 %** (3/9, 5/9, 4/9) contra **1,2 % del juez**. Dos corridas pasaron los mismos 40 checks de 49 y dieron 3/9 y 5/9. El estado honesto es **juez ≈ 4,2 y ~40/49 checks**, y las decisiones pasan a tomarse con esas métricas.

---

## 2. Qué hace el sistema

Consulta en lenguaje natural → informe crediticio ejecutivo, en una app web FastAPI con streaming SSE.

1. **Extracción** (`pipeline.py:156`): un LLM extrae CUIT + motivo de la consulta libre.
2. **Datos estructurados**: `ClientRepository` (`db/client_repository.py`) contra SQL Server — básicos, historial de préstamos, comportamiento de pagos, saldos, gestiones de mora.
3. **Scoring**: `CreditScoringModel.predict()` → score 0–1000, PD calibrada, banda de riesgo, drivers SHAP por cliente (`ml/shap_explainer.py`, TreeSHAP nativo vía `pred_contribs=True`).
4. **RAG**: `rag/pgvector_retriever.py` sobre PostgreSQL + pgvector (buckets: `politicas`, `normativa`, `notas_mora`, `informes_previos`, `documentos_cliente`, `bcra`).
5. **BCRA**: Central de Deudores por CUIT con caché TTL de 30 días (`external/`).
6. **Generación por secciones** (`llm/report_sections.py`): secciones en paralelo → Recomendación → Resumen. Prompts editables en caliente desde el modo admin.
7. **Validación** (`ReportValidator`) + **juez LLM** (`evaluation/llm_judge.py`).
8. **Log auditable** en Postgres (tabla `informes_generados`).

**Diseño anti-alucinación:** el dato numérico no lo inventa el modelo — los importes se pre-formatean antes de entrar al prompt y el validador chequea que el score aparezca literalmente en el informe.

---

## 3. Modelo de scoring — la parte sólida

### 3.1 Diseño

| | |
|---|---|
| Algoritmo | XGBClassifier (xgboost 3.2.0) |
| Target | Alcanzar 90+ días de atraso por primera vez dentro de la ventana de performance de **12 meses** |
| Validación | Panel de cortes anuales, **out-of-time** (`scripts/entrenar_modelo_temporal.py`) |
| TRAIN | Cortes 2015-12-31 … 2022-12-31 (8 cortes) |
| VALID | 2023-12-31 |
| TEST | 2024-12-31 y 2025-04-30 — **tocado una sola vez** |
| Población elegible | Préstamo abierto al corte, sin default previo, y con exposición (alguna cuota vence en la ventana) |

### 3.2 Dataset

| Split | Filas | Defaults | Prevalencia | Clientes únicos |
|---|---|---|---|---|
| TRAIN | 7.398 | 475 | 6,42 % | 2.980 |
| VALID | 744 | 26 | 3,49 % | 741 |
| TEST | 2.208 | 123 | 5,57 % | 1.360 |

- **EPV = 26,4** (475 eventos / 18 features) — cómodamente por encima del mínimo de 10.
- **Solapamiento de clientes TRAIN∩TEST = 48 %** — declarado explícitamente como limitación.

### 3.3 Features

18 features en producción (`models/scoring_features.json`), de 20 definidas en `ml/feature_query.py` (2 se descartan por constantes).

**Anti-fuga explícito:**
- `prestamos_irrecuperables` eliminada — era el target literal.
- `atraso_grave_24m` eliminada — cuotas +90d = el target.
- `max_dias_atraso_historico` se conserva con justificación válida: la elegibilidad la acota a [0, 89].

**Una sola fuente de verdad train/serve:** `construir_features_ml` (`db/client_repository.py:317`) delega en la misma query que el entrenamiento. Esto es lo que rompe la mayoría de los proyectos de ML en producción y acá está resuelto.

**Calidad de la query** (`ml/feature_query.py`): todo se calcula "a fecha `@cutoff`"; el saldo de capital se **reconstruye** sumando cuotas impagas en vez de leer el snapshot `PRES_MAESTRO.saldo_de_capital` (que sería el saldo de HOY = fuga); `fecha_baja` se compara contra `@cutoff` en vez de `IS NULL`.

### 3.4 Métricas (TEST, out-of-time)

| Métrica | TRAIN | VALID | **TEST** | Baseline* |
|---|---|---|---|---|
| AUC-ROC | 0,8917 | 0,8323 | **0,8606** | 0,6556 |
| **Gini** | 0,783 | 0,665 | **0,7212** | 0,311 |
| PR-AUC | 0,4563 | 0,2109 | **0,3378** | 0,1049 |
| KS | 0,6181 | 0,5326 | **0,5702** | 0,3125 |
| Brier | 0,046 | 0,0316 | 0,0452 | — |

\* Baseline honesto = ordenar solo por `max_dias_atraso_historico`. El modelo aporta **+0,205 de AUC** sobre la heurística obvia.

- **Lift decil 1 = 5,2**
- Los 3 deciles más riesgosos capturan el **82 %** de los defaults (101 de 123).

### 3.5 Calibración — *(corregido el 2026-07-12)*

**La PD que publica el informe es la salida cruda del booster.** No hay
calibración, y eso es el resultado de medirlo, no un descuido.

La versión anterior aplicaba una sigmoide de **Platt** justificada con el
argumento de que "predict_proba no es una probabilidad calibrada, sino un
puntaje de ranking". El argumento era falso para este modelo: entrena con
`binary:logistic` (que optimiza log-loss) y omite `scale_pos_weight` a propósito,
así que la salida cruda **ya es** una probabilidad. Platt encima la aplastaba:
degradaba el Brier y rompía la banda MODERADO (ver 3.6).

Ahora el entrenamiento **elige** la calibración por Brier fuera de muestra en una
CV temporal anidada sobre las OOF:

| Método | Brier medio out-of-sample |
|---|---|
| **ninguna** | **0,04135** ← elegido |
| Platt | 0,04375 |
| isotónica | 0,04413 |

`CreditScoringModel._calibrar` sabe aplicar los tres, así que si un
reentrenamiento futuro produce un modelo mal calibrado, el mecanismo lo detecta
y lo corrige solo. `n_estimators = 73` (mediana de folds).

### 3.6 Bandas de riesgo — *(verificadas el 2026-07-12)*

Definidas sobre la **PD** como múltiplos de la tasa de cartera (6,42 %), no sobre
el score. El entrenamiento ahora **verifica** cada banda: contrasta la tasa
observada contra el rango de PD prometido usando el IC 95 % de Wilson, en los
tres splits.

Con la calibración de Platt eliminada, **las cuatro bandas cierran en TEST**:

| Banda | PD | n (TEST) | defaults | tasa real | IC 95 % | |
|---|---|---|---|---|---|---|
| MUY ALTO | > 25,7 % | 90 | 35 | 38,9 % | [29,5–49,2 %] | ✅ |
| ALTO | 12,8–25,7 % | 130 | 29 | 22,3 % | [16,0–30,2 %] | ✅ |
| MODERADO | 6,4–12,8 % | 242 | 20 | 8,3 % | [5,4–12,4 %] | ✅ |
| BAJO | < 6,4 % | 1.746 | 39 | 2,2 % | [1,6–3,0 %] | ✅ |

También cierran las cuatro en OOF/TRAIN y en VALID.

> **Lo que estaba roto.** Con Platt, MODERADO prometía PD 6,4–12,8 % y sus
> clientes defaulteaban al **19,5 %** — con el IC 95 % ([13,5–27,4 %]) entero por
> encima del techo de la banda. No era ruido: la curva de confiabilidad OOF
> mostraba que Platt sub-predecía en los deciles 8–9 (ratio real/predicho ≈ 1,6)
> y sobre-predecía en los bajos. Una sigmoide de 2 parámetros no puede
> representar esa forma.

### 3.7 Debilidades reales del modelo

1. **Concentración de importancia**: `prestamos_al_dia` (0,448) + `prestamos_con_atraso` (0,232) = **68 %** en 2 features.
2. 2 de 18 features tienen importancia 0,0 (`saldo_minimo_6m`, `capacidad_pago_estimada`) y siguen en el vector.
3. **COVID no excluido** (`covid_excluido: false`): los cortes 2019/2020 caen en moratorias. Existe el flag `--excluir-covid` pero el modelo publicado no lo usó.
4. **Cobertura**: la población elegible es ~10 % de la cartera. Para thin-file no hay señal (flag `sin_historial`).
5. **Solapamiento TRAIN∩TEST del 48 %** de clientes (declarado, inherente al diseño de panel).

---

## 4. Harness de evaluación — bien diseñado, resultados malos

### 4.1 Qué se mide

`evaluation/run_golden.py` corre los **9 casos** de `golden_set.json` (1 CUIT inexistente + 8 reales, 2 por banda) y combina tres familias (`evaluation/metricas.py`):

1. **Deterministas**: encontrado/no, banda en conjunto aceptable, score en rango, substrings obligatorios/prohibidos.
2. **Juez LLM**: promedio ≥ 4/5 sobre 5 criterios; alucinaciones críticas = 0; cumplimiento de reglas.
3. **RAGas**: `faithfulness` ≥ 0,7 (gatea); `answer_relevancy` informativa (no gatea).

Exit code 1 si algún caso falla → listo para colgar de CI.

### 4.2 Resultados reales (8 corridas en `evaluation/reports/`)

| Generador | Casos pass | Checks | Juez /5 | Faithfulness |
|---|---|---|---|---|
| gemma4:latest (cal) | **6/9 (67 %)** | 43/49 (88 %) | **4,23** | 0,903 |
| gemma4:latest | 4/9 (44 %) | 35/49 (71 %) | 4,10 | 0,825 |
| anthropic (cal) | 3/9 (33 %) | 37/49 (76 %) | 3,80 | **0,964** |
| anthropic | 2/9 (22 %) | 33/49 (67 %) | 3,98 | — |
| anthropic | 1/9 (11 %) | 30/49 (61 %) | 3,88 | 0,865 |

**Hallazgos:**

- ❌ **Ninguna corrida pasa 9/9.** El techo era **6/9, y con gemma4** (modelo local chico), no con Opus. En CI el exit code sería `1` siempre.
- El check que más falla, de lejos, es **`sin_alucinaciones`** — falla en casi todos los casos de casi todas las corridas, incluso con el modelo fuerte.
- `answer_relevancy` es consistentemente ~0,27–0,35: artefacto estructural, por eso no gatea. Pero significa que esa métrica no aporta señal.
- El `readme.md` afirma *"Confiable"* y *"con tests"* **omitiendo que el golden set no pasa**. Es la brecha más grande entre lo que decimos y lo que hay.

### 4.3 Causa raíz de las "alucinaciones" — *(diagnosticado y corregido el 2026-07-12)*

Al agrupar las 71 alucinaciones que el juez reportó en las 6 corridas completas, **casi ninguna era el LLM inventando cosas**. Eran defectos del contexto que le dábamos:

**1. Dos promedios de días de atraso distintos, con nombres casi iguales.** *(la causa dominante)*

`pagos_resumen.promedio_dias_atraso` y `scoring.dias_atraso_promedio` medían cosas distintas y el informe los mezclaba entre secciones. El juez lo marcaba como contradicción: "usa 9,18 en el resumen pero 9,75 en pagos", "26,16 vs 37,25", "53 vs 60,75".

La causa era un `WHERE pc.vencimiento < pc.fecha_canc` en `get_resumen_comportamiento_pagos`, que **rompía el resumen de pagos de tres formas a la vez**:

- dejaba sólo las cuotas **pagadas tarde**, así que `dias_atraso = 0` era imposible por construcción → **`cuotas_puntuales` devolvía 0 para todos los clientes, siempre** (verificado contra la base en los 9 casos del golden set);
- `fecha_canc IS NULL` hace esa comparación NULL → falsa, así que **las cuotas vencidas e impagas (las peores) quedaban excluidas** del resumen de mora;
- el promedio resultante era una media condicionada a las cuotas atrasadas, sistemáticamente mayor que la del modelo.

El propio `SELECT` usaba `ISNULL(fecha_canc, GETDATE())`, lo que prueba que la intención era incluir las impagas: **el WHERE contradecía al SELECT**. Corregido alineando la definición con la de `ml/feature_query.py`, de modo que los dos números sean **el mismo número por construcción**.

**2. Aritmética de fechas.** "Antigüedad de aproximadamente 27 años" cuando eran 28,6; "hace 8 meses" cuando eran 20. Se le pedía al LLM algo en lo que es malo y que no hay razón para pedirle. Ahora el pipeline **pre-computa** antigüedades y "hace cuánto" (`_derivar_hechos`), igual que ya hacía con los importes.

**3. CUIT inventado.** El informe derivaba un CUIT del DNI (`20-04658409-1`). Lo absurdo: **el CUIT es la clave de la consulta** — lo teníamos y no se lo pasábamos formateado. Ahora va en el contexto como `cuit`.

**4. Reglas nuevas en `prompts/secciones_base.txt`**: prohibido calcular duraciones, prohibido construir identificadores, un solo promedio de atraso, y `total_cuotas = 0` significa "no venció ninguna cuota", no "falta el dato".

> **La lección:** el golden set tenía razón en fallar. Los informes efectivamente tenían números mal — pero la culpa no era del generador, sino de los datos que le pasábamos. Bajar el umbral del check habría escondido un bug real de SQL que hacía que **el resumen de comportamiento de pagos reportara cero cuotas puntuales para toda la cartera**.

### 4.4 Verificación post-fix — *(re-corrida 2026-07-12)*

Con los tres arreglos aplicados (SQL del resumen de pagos, pre-cómputo de fechas, CUIT en contexto) se volvió a correr el golden set completo con **gemma4:latest** (`evaluation/reports/golden_gemma4-fixes_20260712`):

| | gemma4 (cal, pre-fix) | **gemma4 (post-fix)** |
|---|---|---|
| Casos pass | 6/9 (67 %) | **5/9 (56 %)** |
| Checks | 43/49 (88 %) | 39/49 (80 %) |
| Juez /5 | 4,23 | 3,93 |
| Faithfulness | 0,903 | 0,823 |

**Los fixes funcionaron donde apuntaban.** Los casos que en la baseline tenían las peores alucinaciones temporales (`muy_alto_1`, `muy_alto_2`) ahora cierran **limpios: 0 alucinaciones, juez 5,0**. El error de aritmética de fechas desapareció de los veredictos.

**Pero el pass-rate no subió: quedó al descubierto otra clase de alucinación** que los fixes no tocan. El juez ya no marca fechas; marca la **lectura de la cartera de préstamos**:

- **`alto_1`** (2/6, el peor): afirma "liquidación total, saldo $0,00, todo cancelado 2009-2013" cuando hay un préstamo **vigente de $536.250** (otorgados en 2021/2023/2024) y **$7,4 M** de deuda en BCRA. Grounding grave: da por saldada una cartera con deuda viva.
- **`moderado_2`** (3/6): inventa conteos ("nueve préstamos", "12 créditos") y un **desglose por oficial de cuenta fabricado** ("ELGOZAINE, SILVIA: 5 créditos") que no está en la fuente.
- **`moderado_1`** (4/6): confunde `max_dias_atraso_historico` (14) con `dias_atraso_actual` (46) — reporta el atraso actual como si fuera el máximo histórico.
- **`bajo_1`** (5/6): correcto en los números, pero **menciona el género** ("la asociada FORCONI…") → viola la regla de características protegidas (`cumplimiento_reglas.aprobado = false`). Es el único check que lo hunde.

**Falso positivo del juez (no es bug del generador):** en `bajo_2` el juez marca el CUIT `20-04658409-1` como "no presente en la fuente". Es exactamente el fix — el CUIT es el `cliente_id` formateado — pero el juez no lo sabe. Queda como MENOR y no reprueba, pero conviene pasarle al juez que `cliente_id` == CUIT.

> **Conclusión.** El techo de 6/9 no era por fechas: era una mezcla, y las fechas eran sólo una capa. Ya están resueltas y verificadas. Lo que ahora bloquea es (a) **agregación/estado de la cartera de préstamos** — el modelo cuenta mal y confunde cancelado con vigente — y (b) una **regla de estilo** (género). Ninguna se arregla pre-formateando un dato: la primera pide **acotar/pre-agregar el historial** que entra al prompt (cuántos préstamos, cuáles vigentes, período real, saldo total vivo); la segunda, un ajuste de prompt.

### 4.4b Rondas de ajuste y techo del generador — *(2026-07-12, `golden_gemma4-fixes2..4`)*

Se atacaron las clases que quedaron, en cuatro re-corridas. Lo que movió la aguja:

- **Agregación de cartera** — nuevo `_resumen_cartera()` (pipeline.py) que agrega al contexto `cartera_resumen` (total, por estado, vigentes, saldo vivo, monto original acumulado, período). Primero se puso la regla anti-conteo solo en Historial y **no alcanzó**: el modelo contaba mal en Recomendación y en subtítulos que inventaba ("Panorama General"). Se movió la regla a `secciones_base.txt` (todas las secciones) y se pasó `cartera_resumen` también a Recomendación/Resumen. → recuperó `alto_1` y `muy_alto_1`.
- **Regla de género** (secciones_base): tratamiento neutro del sujeto. → resuelto en todos los casos.
- **CUIT** (juez_sys): se le aclara al juez que el `cuit` formateado ES el `cliente_id`. → `bajo_2` limpio.
- **Sobre-corrección** — las reglas anti-inferencia se pasaron de fuertes y el modelo empezó a decir "no es posible determinar" sobre datos presentes; se agregó "usá los datos que el contexto SÍ trae". → recuperó `bajo_1`.
- **`estado_actual`** (secciones_base): no afirmar "activo" si el estado es "Baja". → recuperó `bajo_2`.
- **Juez sobre-estricto** (juez_sys): ±1 mes de antigüedad (el informe calcula día a día, más preciso que el `meses/12` del juez) y centavos ahora son MENOR, no CRÍTICA.

**Resultado al cierre: 6/9 (67 %), juez 4,175, checks 42/49 (86 %)** — el mejor conjunto, y ahora con los arreglos atacando causas reales, no el 6/9 accidental de la baseline.

**Techo de gemma4.** Los 3 que resisten ya no son bugs de prompt ni de datos:

- `moderado_2`: el dato correcto **ya aparece** en una sección ("28 líneas, $309,8k" vía `cartera_resumen`) pero el modelo **se contradice** fabricando "10 operaciones / $45.131" en otra. Es un límite de capacidad del modelo local en carteras grandes.
- `alto_1` / `alto_2`: **alta varianza** (alto_1 salió limpio en una corrida y con error de fecha en la siguiente).

**Los dos levers que quedan son de otra naturaleza:** (1) **sacar los agregados de la cartera del alcance del LLM** — ✅ implementado (4.4c); (2) **medir el techo con un generador más fuerte** (opus) o el otro local (qwen3:14b) — pendiente.

### 4.4c Historial por plantilla + frase de cartera (13-jul-2026)

Al revisar los veredictos de los 3 casos que resistían apareció un matiz que cambia el diagnóstico: **la sección Historial no era la que mentía**. En `moderado_2` el juez señala que hay una sección "que correctamente indica 28 líneas y $309.880" y que **otras partes del informe la contradicen** ("Se registra un total de 10 operaciones", "Período Cubierto: 2001–2015"); en `alto_1`, ídem con las fechas ("préstamo más reciente 16/12/2013" cuando el último es de 2024). Los agregados los fabricaban las secciones **interpretativas** (Resumen / Recomendación), que recibían `cartera_resumen` y aun así lo recomponían a mano.

Dos cambios, mismo principio (lo verificable no lo escribe el modelo):

1. **Historial Crediticio se arma por plantilla** (`llm/plantillas.py`, sin LLM). Declarada con la clave `plantilla` en `SECCIONES`; ya no tiene prompt (se quitó `seccion_historial.txt` del catálogo, y el editor admin no la ofrece porque no hay nada que editar). Al ser código, es **testeable**: `tests/test_plantilla_historial.py` fija los casos que el juez marcaba (conteo, período, mora vigente, "sin historial" ≠ "sin atrasos").
2. **`cartera_resumen.frase`**: los agregados, **ya redactados** en una frase, viajan a todas las secciones. Los campos sueltos no alcanzaban; con la frase escrita la única acción posible es transcribirla. La regla de `secciones_base.txt` ahora apunta a esa frase y prohíbe explícitamente mencionar cualquier agregado que no esté en ella.

Efecto colateral: el informe hace **una llamada menos al LLM** (5 secciones en vez de 6) y la más larga de todas (la que listaba préstamo por préstamo) ya no gasta tokens.

**Lo que mostró la medición** (2 corridas de gemma4 sobre el golden, con los informes ya guardados en el reporte):

- **El culpable real eran las secciones fantasma.** gemma4 abría **secciones enteras inventadas** dentro del cuerpo de otra (`## Resumen de la Cartera Crediticia`, `## 📄 INFORME INTEGRAL DEL PERFIL CREDITICIO…`) y ahí adentro volvía a narrar la cartera contándola a mano: "cinco registros detallados" con 12 préstamos en la fuente, "perfil de riesgo bajo" con nivel MODERADO. Ninguna regla de prompt lo frenó en cuatro rondas — el prompt ya decía "devolvé sólo el contenido de esta sección, sin títulos". Ahora el **código** corta el cuerpo de cada sección en el primer encabezado que emita el modelo (`_recortar_secciones_fantasma`). Tras el fix, los 9 informes salen con exactamente 8 encabezados y las alucinaciones de conteo desaparecen. `moderado_2` —el caso terco— pasó de 3/6 checks a **5/6 sin ninguna alucinación**.
- **Dos definiciones de `max_dias_atraso_historico` (bug de datos abierto).** El campo de `get_historial_prestamos` sólo considera cuotas **pagadas tarde** (`fecha_canc` no nulo) e ignora las impagas vigentes; la feature ML del mismo nombre da otro número (46 vs 87 en un caso real). Es el mismo bug que ya se corrigió en `promedio_dias_atraso`: dos fuentes para el mismo concepto y el informe se contradice solo. Por eso la plantilla **no** publica un máximo histórico agregado — sólo el de `pagos_resumen`, que sí comparte definición con la feature. **Queda por unificar en el SQL.**

> **⚠️ El pass/fail sobre 9 casos es ruidoso.** Las tres corridas dieron 6/9 → 5/9 → 4/9, pero **los casos que fallan cambian en cada una** (`muy_alto_1` falla y después pasa; `bajo_2` pasa y después falla) mientras las métricas continuas mejoran (juez 3,95 → 4,03; faithfulness 0,809 → 0,855). Con 9 casos y modelos con temperatura, una diferencia de ±1 caso **no es señal**. Antes de seguir optimizando contra ese número hay que **medir la varianza del harness** (misma config, 2-3 corridas) y decidir con métricas continuas o con más casos.

### 4.4d — Varianza del harness: MEDIDA (29-ago-2026)

Se corrió el golden set **3 veces sin tocar nada** (gemma4:latest, mismo código, misma config). Resultado:

| Métrica | c1 | c2 | c3 | cv |
|---|---|---|---|---|
| **Casos que pasan** | 3/9 | 5/9 | 4/9 | **25,1 %** |
| **Checks que pasan** | 40/49 | 40/49 | 41/49 | **1,5 %** |
| **Juez** | 4,15 | 4,20 | 4,25 | **1,2 %** |
| Faithfulness | 0,832 | 0,826 | 0,774 | 3,9 % |

**El pass/fail por caso es ~21× más ruidoso que el juez, y la sospecha quedó confirmada.** Las corridas 1 y 2 pasaron **exactamente los mismos 40 checks de 49** y dieron 3/9 y 5/9 casos respectivamente. El sistema rindió idéntico; lo único que cambió fue cómo se repartieron los 9 fallos entre los casos. Es azar de agrupamiento.

**Consecuencia que obliga a corregir el relato de este documento:** la mejora de julio de **44 % → 67 %** (+0,223) **no supera el umbral de credibilidad de una sola corrida (0,315 ≈ 2,8 casos)**. Los arreglos de 4.4b eran defectos reales, verificados uno por uno contra la base, y las métricas continuas sí mejoraron — pero *el salto en el número de casos no fue lo que los demostró*. Ese 67 % era, en buena medida, una corrida afortunada. El 6/9 citado más arriba y en la tabla de la sección 8 hay que leerlo así.

**Nuevo criterio de decisión** (detalle en `docs/evaluacion.md`):
- Se decide con **el juez y `checks_pass_rate`**, no con el pass/fail por caso.
- Un cambio se acepta si mueve el juez **> 0,14** (una corrida) o **> 0,08** (promedio de 3).
- El pass/fail queda como resumen y como exit code de CI, no como medida de progreso.
- Para que el pass/fail detectara una mejora de un solo caso harían falta **~9 corridas** por configuración (~4,5 h).

**Señal vs ruido, al 29-ago:** alternan `bajo_1` y `bajo_2` → no vale optimizarlos. El check que más alterna es `sin_alucinaciones` (5 de 10 flips), lo esperable en un juicio de LLM sobre texto de otro LLM. Sobre los que fallan siempre con gemma4, ver 4.4e: correr opus mostró que **ninguno era un defecto del sistema**.

### 4.4e — Opus vs gemma4 (29-ago-2026, 2 corridas de opus vs 3 de gemma4)

`python evaluation/varianza.py --comparar var1,var2,var3 opus1,opus2`

| Métrica | gemma4 (n=3) | opus (n=2) | dif | umbral | veredicto |
|---|---|---|---|---|---|
| Casos que pasan | 0,444 | 0,667 | +0,223 | 0,373 | **no distinguible** |
| Checks que pasan | 0,823 | 0,918 | +0,096 | 0,056 | **CREÍBLE** |
| Juez | 4,200 | 4,762 | +0,562 | 0,148 | **CREÍBLE** |
| Faithfulness | 0,811 | 0,950 | +0,139 | 0,083 | **CREÍBLE** |

**Opus es netamente mejor, y las tres métricas continuas lo afirman con margen.** El detalle que vale la pena guardar: el pass/fail mejoró **+0,223 — la misma magnitud que la mejora celebrada en julio (44 %→67 %)— y sigue sin ser distinguible del ruido**. Mirando sólo ese número, hoy habríamos concluido "no hay diferencia clara" sobre un modelo claramente superior. Es la demostración más limpia de por qué 4.4d cambió el criterio.

**Esto da vuelta la conclusión de julio** (gemma4 67 % vs opus 33 %; juez 4,23 vs 3,80). No cambió opus: cambió el pipeline. Varios errores que en julio se le imputaron al modelo —aritmética de fechas, conteo de cartera— eran **defectos del contexto que le dábamos** (4.4b/4.4c). Opus, que escribe más y arriesga más, los exponía; gemma4 los tapaba escribiendo menos. Corregido el contexto, el modelo fuerte gana en todo.

**Los casos que gemma4 falla siempre, revisados uno por uno — ninguno era un defecto del generador:**

| Caso | opus | gemma4 | Qué era realmente |
|---|---|---|---|
| `alto_1` | 2/2 | 0/3 | Techo del modelo local. Confirmado |
| `moderado_2` | 1/2 | 0/3 | Techo del modelo local (el "caso terco") |
| `alto_2` | 1/2 | 0/3 | Techo del modelo local |
| `moderado_1` | 0/2 | 0/3 | **Golden set vencido**, no un defecto (ver abajo) |
| `muy_alto_2` | 0/2 | 3/3 | **Bug de datos nuestro**, no de opus (ver abajo) |

**`moderado_1`: el golden set está vencido.** El único check que falla es `banda_riesgo`: el modelo puntúa el cliente **MUY ALTO** y el set espera `["BAJO","MODERADO","ALTO"]` porque fue muestreado como MODERADO **en julio**. En seis semanas el riesgo real del cliente se movió. Es exactamente el modo de falla contra el que advierte `docs/evaluacion.md` ("con un cutoff viejo, el riesgo real del cliente cambia y `banda_riesgo` falla espuriamente"), y se pisó igual. **Ese caso viene fallando espuriamente en todas las corridas de hoy, con los dos generadores**, así que las cifras absolutas de 4.4d subestiman a ambos por igual (la comparación entre ellos no se ve afectada). **Acción: re-sembrar el golden set** con `muestrear_cuits.py` antes de la próxima medición, y tratar el re-sembrado como parte del procedimiento, no como algo que se hace una vez.

**`muy_alto_2`: opus expone un bug de datos que gemma4 tapaba.** La alucinación CRÍTICA que marca el juez en las dos corridas es el conocido **`max_dias_atraso_historico`**: el informe publica 80 días (el campo por préstamo, que sólo mira cuotas *pagadas tarde*) mientras `dias_atraso_actual` y `pagos_resumen.max_dias_atraso_periodo` dicen **166**. El informe se contradice porque **los datos se contradicen**. Es el mismo bug de doble definición ya anotado en 4.4c, el mismo que se corrigió en su momento para `promedio_dias_atraso`. gemma4 pasa el caso simplemente porque no menciona el dato.

**✅ CORREGIDO (29-ago-2026).** La subconsulta `max_atraso` de `get_historial_prestamos` filtraba `fecha_canc IS NOT NULL` —sólo cuotas pagadas tarde— e ignoraba las impagas vigentes, que son las de mayor atraso y además crece todos los días. Peor: el `ISNULL` que caía al atraso actual sólo actuaba si *no había ninguna* cuota pagada tarde, así que un atraso viejo y menor le ganaba a la mora viva. Ahora mide, por cada cuota ya vencida, los días entre su vencimiento y la fecha de pago **o hoy si sigue impaga** — la misma definición que `ml/feature_query.py`.

Verificado contra la base sobre los ~50 préstamos de los 8 clientes del golden: `muy_alto_2` pasó de 80 a **166** (consistente con `dias_atraso_actual` y con `pagos_resumen`), y la invariante **`max_dias_atraso_historico >= dias_atraso_actual`** —que antes era violable y ahora se cumple por construcción— no se rompe en ningún préstamo. `tests/test_definicion_atraso.py` compara las dos fuentes SQL y falla si vuelven a divergir; corre sin credenciales.

Efecto lateral: la plantilla de Historial evitaba publicar un máximo histórico agregado **por causa de este bug** (4.4c). El impedimento técnico ya no existe; queda fuera por una decisión de contenido (`pagos_resumen` ya informa un máximo, y dos máximos con alcances distintos confunden).

Herramienta: `evaluation/varianza.py` (`python evaluation/varianza.py --etiqueta var`).

**Pendiente:** correr opus/qwen3:14b (ahora sí con criterio para interpretarlo). Las fallas que quedan ya **no son de conteo** sino de **razonamiento de política** (recomienda aprobar con score < 700; no advierte `estado_actual = 'Baja'`) — eso es prompt de Recomendación/Cumplimiento, o techo del modelo.

> **Efecto lateral positivo (trazabilidad):** el `cartera_resumen` creado para este arreglo alimenta además una feature nueva — bajo cada sección del informe, un resumen legible de los datos fuente con los que se generó (visible para todo lector), para hacer auditable la relación dato→texto. Backend en `resumen_datos_secciones()` (report_sections.py), emitido en el evento `fin`; front colapsable por sección en `app.py`.

### 4.5 Devolución del profesor

Presentación 8, dificultad 8,5. Pidió dos cosas: (a) vigilar fuga de datos, reportar tamaño de dataset, usar split temporal; (b) automatizar la evaluación con golden set + pass/fail + RAGas. **Ambas están implementadas y bien.** Lo que falta es cerrar el loop: hacer que el golden set efectivamente pase.

---

## 5. Modelo de datos y fuentes

1. **SQL Server** (primaria, solo lectura) — core bancario/mutual real. Son **27 tablas**, declaradas en `db.client_repository.TABLAS_SQLSERVER` y verificadas contra el SQL fuente por `tests/test_trazabilidad.py`: `PERSONAS`, `PERSONA_REL`, `PERSONA_TIPO`, `MAESTROS_OPER`, `MAESTROS_REL_MAESTROS`, `PRES_MAESTRO`, `PRES_CUOTAS`, `CTACTE_MAESAL`, `LINEAS`, `LIN_CTAVISTA`, `LIN_PRESTAMO`, `PERFILES_CLIENTES`, `CLASIFICACION_CLIENTE`, `GARANTIAS_MAE/REL/TIPO/SUBTIPO`, `SITUACION_OPER`, `SITUACION_TIPO_TRAMITES`, `SITUACION_TRAMITE_PASOS`, `CUOMUTUAL_MAE`, `ACTIVIDAD`/`ACTIVIDAD_RUBRO`/`ACTIVIDAD_SUBRUB`, `DOCUMENTO_TIPO`, `ESTADO`, `MONEDA`.
2. **PostgreSQL + pgvector** — RAG cualitativo + log `informes_generados`.
3. **BCRA** — API pública de la Central de Deudores, con caché.
4. **`models/*.json`** — artefactos del modelo.

---

## 6. Comparación con el mercado

En Argentina el juego lo definen tres capas. **No competimos con ninguna: nos apoyamos en ellas.**

| Sistema | Qué es | Qué mira |
|---|---|---|
| **Central de Deudores BCRA** | Pública, gratuita | Solo deuda con entidades reguladas; situación 1–5. Es el piso normativo, no un score |
| **Veraz (Equifax)** | Privada, score 1–999 | Suma servicios, telcos, comercios, cheques. Es lo que el banco mira antes de un hipotecario |
| **Nosis** | Privada, score 0–999 | Similar; muy usada para empresas / CUIT |
| **FICO y scorecards clásicas** | Estándar global | Regresión logística sobre bins (WOE); explicable por diseño |

### 6.1 Dónde caemos contra el benchmark

Los estándares de la industria en scoring de consumo:

- **Gini** ≥ 0,50 aceptable · promedio ~0,67 · best-in-class > 0,75
- **KS** 20–70, > 50 excelente
- **AUC** > 0,80 = poder predictivo fuerte

Nuestro **Gini 0,72 / KS 0,57 / AUC 0,86** nos ubica **en el rango bueno de la industria**. No es un juguete.

### 6.2 Posicionamiento correcto

La diferencia con los burós no es la performance, es la **cobertura**: ellos ven todo el sistema financiero, nosotros solo nuestra cartera. Pero la ventaja es la inversa: tenemos **comportamiento interno de pago que el buró no ve**, y además generamos un **informe narrado con explicación SHAP y RAG de políticas**, que ninguno de ellos entrega.

> **No somos un reemplazo de Nosis. Somos scoring de comportamiento interno + informe de comité automatizado, que complementa al buró.**

---

## 7. Qué falta para producción

### 7.1 Bloqueantes

| # | Acción | Estado | Por qué |
|---|---|---|---|
| 1 | **Reparar `trazabilidad`** | ✅ **hecho** (2026-07-12) | El campo de auditoría contenía **tablas inventadas** (`clientes`, `prestamos`, `saldos_diarios`… ninguna existe) y `"modelo_scoring": "XGBoost v1.0"` hardcodeado. Ahora publica las 27 tablas reales desde `TABLAS_SQLSERVER` y la identidad del modelo desde `model_card.json`; `tests/test_trazabilidad.py` extrae las tablas del SQL fuente y falla si la lista se desincroniza |
| 2 | **Investigar la banda MODERADO** | ✅ **hecho** (2026-07-12) | No era la banda: era la **calibración de Platt**. Eliminada. Las 4 bandas cierran en los 3 splits y el Brier de test mejoró (0,0452 → 0,0442). Ver 3.5 y 3.6 |
| 3 | **Cerrar el golden set** | 🔄 en progreso | Fechas/CUIT/SQL/conteo-de-cartera/género resueltos y **verificados** (4.4b); Historial por plantilla sin LLM (4.4c). Estado medido con 3 corridas (4.4d): **juez 4,2 y ~40/49 checks**, estables; el pass/fail por caso oscila 3/9–5/9 y **no sirve para medir progreso** (cv 25 %). Fallan siempre `moderado_1`, `moderado_2`, `alto_1`, `alto_2`: ahí está el trabajo real, y ya no es conteo sino **razonamiento de política**. Hasta que eso no cierre no podemos decir que el sistema es confiable — y el readme ya lo dice |
| 4 | **Pinear `xgboost==3.2.*`** | ⬜ pendiente | `requirements.txt` dice `>=2.0.0`, pero `load()` lanza `RuntimeError` si el major no coincide → una instalación limpia **no arranca**. Ninguna dependencia está pineada |

### 7.2 Requisitos regulatorios (model risk management, tipo SR 11-7)

| Requisito | Estado |
|---|---|
| Validación out-of-time / conceptual soundness | ✅ Hecho |
| Explicabilidad por predicción (SHAP) | ✅ Hecho |
| **Motivos de rechazo en lenguaje llano (adverse action)** | ❌ Falta — el SHAP está, falta convertirlo en "los 3 principales motivos" formales. No alcanza con "el modelo es complejo" |
| **Monitoreo de drift (data / concept) con umbrales que disparen recalibración** | ❌ No existe |
| **Backtesting periódico** | ❌ No existe |
| **Bias / fair lending testing** | ❌ Ni planteado |
| Model card / documentación de gobierno | ⚠️ Existe; `docs/scoring.md` corregido en bandas y calibración, resta §4 (hiperparámetros) |

### 7.3 Deuda técnica

1. ⚠️ **`docs/scoring.md` sigue desactualizado en §4 (hiperparámetros)**: dice `n_estimators=300 / max_depth=5 / eval_metric=auc / scale_pos_weight dinámico`; los reales son **73 / 3 / aucpr / sin scale_pos_weight**. También dice "19 features" (son 18). *(§2 bandas y §2b calibración ya corregidas.)*
2. **`readme.md:98` dice AUC 0,8653**; el valor real de test es **0,8606**.
3. **`ReportValidator` es frágil**: valida "el score aparece" con `score_str not in informe` — el score `800` matchea contra `$1.800.000`.
4. **`scripts/entrenar_modelo_real.py`** sigue en el repo — es el entrenador **con fuga de datos**. Riesgo de que alguien lo corra. Borrar.
5. **Tests**: falta cobertura de `feature_query` — la pieza más crítica del sistema. La suite determinista vive en `evaluation/test_impletation.py` (nombre mal escrito, fuera de `tests/`, no la levanta `pytest tests/`). *(Ya existen `tests/test_bcra_normalizer.py` y `tests/test_trazabilidad.py`.)*
6. **No hay CI** (no existe `.github/`), pese a que `run_golden.py` está diseñado para colgarse de CI.
7. `docs/modelo-datos.md` repite las tablas ficticias en el diagrama ER.
8. ~~`__init__` muerto en `CreditScoringModel` y prints `[DBG]`~~ → ✅ eliminados en `ml/scoring_model.py` (quedan `[DBG]` en `pipeline.py`).

---

## 8. Descubrimiento con clientes

Sí, hay que hablar con clientes — pero **no para preguntar "¿qué te gustaría?"**. Eso devuelve una lista de deseos inútil. Las preguntas que mueven la aguja:

**De negocio (definen si hay producto):**
- **¿Cuál es su cutoff hoy y quién lo decide?** Si aprueban a mano, nuestro valor es la velocidad. Si ya tienen un score, el valor es la performance — y hay que ganarle en Gini a *su* modelo, no al aire.
- **¿Qué hacen hoy con la zona gris?** Ahí es donde un informe narrado con SHAP + políticas vale plata. Los extremos los resuelve cualquiera.
- **¿Cuál es su tasa de default actual y su volumen mensual?** Sin eso no se puede estimar el ahorro y no hay pitch.

**De modelo (definen si sirve tal cual o hay que reentrenar):**
- **¿Qué define "default" para ustedes?** Nosotros usamos 90+ dpd a 12 meses. Si el cliente usa 60 dpd o ventana de 6 meses, **el modelo hay que reentrenarlo, no adaptarlo**.
- ¿Qué proporción de su cartera es thin-file? (Ahí no tenemos señal.)

**De gobierno (definen el alcance del producto):**
- **¿Qué tienen que poder mostrarle al auditor / al BCRA?** Esto define si la trazabilidad es un nice-to-have o **es el producto entero**.
- ¿Necesitan motivos de rechazo formales para el cliente final?

**Técnicas (lo fácil):** latencia, API, on-prem vs cloud, quién es dueño del dato.

---

## 9. Recomendación

Antes de salir a hablar con clientes, un sprint corto sobre:

1. ✅ ~~Trazabilidad real (tablas + versión del modelo desde el model card).~~ *(hecho 2026-07-12)*
2. ✅ ~~Banda MODERADO.~~ *(hecho 2026-07-12 — era la calibración)*
3. 🔄 **Golden set en verde**. Fechas/CUIT/SQL/conteo-de-cartera/género resueltos y **verificados** — techo actual **6/9** (4.4b). El lever contra el techo del modelo local ya está aplicado (**Historial por plantilla, sin LLM** + agregados pre-redactados, 4.4c); falta **re-correr el golden** para medirlo (y opcionalmente comparar con opus/qwen3). ← lo único que falta del sprint

Con eso hay una demo que resiste una pregunta incómoda. **El modelo ya da la credibilidad técnica; lo que hoy no resiste es el envoltorio.**

### Nota sobre el efecto secundario del cambio de calibración

Al sacar Platt, el score se **despliega en un rango más amplio**. Antes la PD vivía comprimida entre 2 % y 25 %, así que el score iba de ~750 a ~980. Ahora un perfil de riesgo alto puede sacar 580. Los informes van a mostrar más dispersión entre clientes buenos y malos — que es lo correcto, pero **si alguien tiene expectativas calibradas al rango viejo, hay que avisarle**.

---

## Referencias

- [Veraz vs BCRA vs Nosis](https://liberaya.com/2026/03/26/veraz-vs-bcra-vs-nosis-diferencias-y-como-afectan-tu-historial-crediticio/)
- [Equifax Argentina — Scores de Riesgo](https://www.soluciones.equifax.com.ar/empresas/productos/scores-de-riesgo/)
- [Benchmarks de Gini / KS / AUC en credit scoring](https://ginimachine.com/academia-post/ml-model-performance-evaluation/)
- [SR 11-7 — Model Risk Management para modelos AI/ML](https://www.glacis.io/guide-sr-11-7)