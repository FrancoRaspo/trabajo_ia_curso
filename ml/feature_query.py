# ml/feature_query.py
#
# FUENTE ÚNICA DE VERDAD para las features del modelo de scoring.
#
# La misma query se usa en entrenamiento (con cutoffs históricos) y en
# inferencia (con cutoff = hoy). Eso es deliberado: antes había dos
# definiciones distintas (scripts/entrenar_modelo_real.py y
# db/client_repository.construir_features_ml) que divergían en silencio.
#
# TODAS las features se calculan "a fecha @cutoff". Ninguna columna puede
# depender del estado ACTUAL de la base, porque en entrenamiento el cutoff
# está en el pasado y el estado actual sería información del futuro.
#
# Por eso NO se usan:
#   - PRES_MAESTRO.saldo_de_capital  (snapshot sin fecha -> saldo de HOY)
#   - MAESTROS_OPER.fecha_baja       (se compara contra @cutoff, no IS NULL)
# El saldo de capital se reconstruye sumando las cuotas impagas al cutoff.
#
# Convención de "impaga al cutoff":  fecha_canc IS NULL OR fecha_canc > @cutoff
#
# Definición de default (target):
#   El cliente alcanza 90 días de mora POR PRIMERA VEZ dentro de la ventana
#   de performance (@cutoff, @cutoff + horizonte].  Una cuota "alcanza 90 dpd"
#   en la fecha vencimiento+90 si a esa altura seguía impaga.

from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

# Orden y contenido deben coincidir con CreditScoringModel.FEATURE_COLUMNS.
FEATURE_COLUMNS = [
    "total_prestamos",
    "prestamos_al_dia",
    "prestamos_con_atraso",
    "max_dias_atraso_historico",
    "dias_atraso_promedio",
    "prestamos_refinanciados",
    "cuotas_puntuales_24m",
    "atraso_leve_24m",
    "atraso_moderado_24m",
    "saldo_promedio_ahorro_6m",
    "saldo_promedio_corriente_6m",
    "saldo_minimo_6m",
    "deuda_vigente_total",
    "capacidad_pago_estimada",
    "antiguedad_meses",
    "productos_activos",
    "ultimo_prestamo_monto",
    "ultimo_prestamo_hace_meses",
    "tasa_cumplimiento",
    "ratio_deuda_ingresos",
]

# Features eliminadas respecto del modelo viejo, y por qué:
#
#   prestamos_irrecuperables  -> era el target literal (clasificacion_gerencia > 4).
#   atraso_grave_24m          -> cuotas con +90d de atraso; el target ES +90d de atraso.
#
# Ambas quedan además en CERO por construcción para toda fila elegible, ya que
# la elegibilidad excluye a quien ya estaba en default al cutoff. Serían
# constantes, no predictores.
#
# max_dias_atraso_historico SÍ se conserva: al cutoff está acotada a [0, 89] por
# la misma regla de elegibilidad, así que ya no filtra el target y es un
# predictor legítimo (cuán cerca del default estuvo el cliente en el pasado).

# ---------------------------------------------------------------------------
# La query. Parámetros: :cutoff (date), :horizonte (int, meses).
# El placeholder {filtro_cliente} permite reusarla para un solo cliente.
# ---------------------------------------------------------------------------

_SQL = """
DECLARE @T   DATE = :cutoff;
DECLARE @FIN DATE = DATEADD(MONTH, :horizonte, @T);
DECLARE @INI24 DATE = DATEADD(MONTH, -24, @T);
DECLARE @INI6  DATE = DATEADD(MONTH,  -6, @T);

WITH prestamos AS (
    -- Un renglón por préstamo abierto AL cutoff.
    SELECT
        p.id_insc            AS cliente_id,
        pr.fec_ing,
        mo.per_tipo, mo.per_codigo,
        mo.sucursal, mo.operatoria, mo.linea, mo.nro_servicio,
        mo.fecha_apert, mo.fecha_baja,
        pm.capital
    FROM PERSONAS p
    INNER JOIN PERSONA_REL pr
        ON pr.tdi_cod = p.tdi_cod AND pr.nro_doc = p.nro_doc
    INNER JOIN MAESTROS_OPER mo
        ON mo.per_tipo = pr.per_tipo AND mo.per_codigo = pr.per_codigo
    INNER JOIN PRES_MAESTRO pm
        ON  pm.sucursal     = mo.sucursal
        AND pm.operatoria   = mo.operatoria
        AND pm.linea        = mo.linea
        AND pm.nro_servicio = mo.nro_servicio
    WHERE mo.fecha_apert <= @T
      {filtro_cliente}
),

prest_estado AS (
    -- Estado de cada préstamo AL cutoff (nada de fecha_baja IS NULL).
    SELECT
        pe.cliente_id, pe.fec_ing, pe.per_tipo, pe.per_codigo,
        pe.fecha_apert, pe.capital,
        CASE WHEN pe.fecha_baja IS NULL OR pe.fecha_baja > @T
             THEN 1 ELSE 0 END                       AS vigente_T,
        ISNULL(atr.dias, 0)                          AS dias_atraso_T,
        ISNULL(sal.saldo_cap, 0)                     AS saldo_cap_T,
        CASE WHEN ISNULL(refi.n, 0) > 0 THEN 1 ELSE 0 END AS refinanciado_T
    FROM prestamos pe

    -- Atraso corriente del préstamo al cutoff: la cuota vencida e impaga
    -- más antigua.
    OUTER APPLY (
        SELECT MAX(DATEDIFF(DAY, pc.vencimiento, @T)) AS dias
        FROM PRES_CUOTAS pc
        WHERE pc.sucursal = pe.sucursal AND pc.operatoria = pe.operatoria
          AND pc.linea = pe.linea AND pc.nro_servicio = pe.nro_servicio
          AND pc.vencimiento < @T
          AND (pc.fecha_canc IS NULL OR pc.fecha_canc > @T)
    ) atr

    -- Saldo de capital reconstruido: cuotas todavía impagas al cutoff.
    OUTER APPLY (
        SELECT SUM(pc.capital) AS saldo_cap
        FROM PRES_CUOTAS pc
        WHERE pc.sucursal = pe.sucursal AND pc.operatoria = pe.operatoria
          AND pc.linea = pe.linea AND pc.nro_servicio = pe.nro_servicio
          AND (pc.fecha_canc IS NULL OR pc.fecha_canc > @T)
    ) sal

    -- Refinanciación registrada ANTES del cutoff.
    OUTER APPLY (
        SELECT COUNT(1) AS n
        FROM MAESTROS_REL_MAESTROS r
        WHERE r.sucursal_rel = pe.sucursal AND r.operatoria_rel = pe.operatoria
          AND r.linea_rel = pe.linea AND r.nro_servicio_rel = pe.nro_servicio
          AND r.id_sys_relacion = 1
          AND r.fecha_cont <= @T
    ) refi
),

agg_prestamos AS (
    SELECT
        cliente_id,
        MIN(fec_ing)                                   AS fec_ing,
        COUNT(*)                                       AS total_prestamos,
        SUM(CASE WHEN vigente_T = 1 AND dias_atraso_T = 0
                 THEN 1 ELSE 0 END)                    AS prestamos_al_dia,
        SUM(CASE WHEN dias_atraso_T > 0 THEN 1 ELSE 0 END) AS prestamos_con_atraso,
        SUM(refinanciado_T)                            AS prestamos_refinanciados,
        SUM(CASE WHEN vigente_T = 1 THEN saldo_cap_T ELSE 0 END) AS deuda_vigente_total,
        MAX(fecha_apert)                               AS ultima_apertura
    FROM prest_estado
    GROUP BY cliente_id
),

ultimo_prestamo AS (
    -- Monto del préstamo MÁS RECIENTE al cutoff (no el más grande).
    SELECT cliente_id, capital AS ultimo_prestamo_monto
    FROM (
        SELECT cliente_id, capital,
               ROW_NUMBER() OVER (PARTITION BY cliente_id
                                  ORDER BY fecha_apert DESC) AS rn
        FROM prest_estado
    ) t WHERE rn = 1
),

-- Días de mora de CADA cuota medidos al cutoff.
--   pagada antes del cutoff -> días entre vencimiento y pago
--   impaga al cutoff        -> días entre vencimiento y el cutoff
cuotas_mora AS (
    SELECT
        pe.cliente_id,
        pc.vencimiento,
        DATEDIFF(DAY, pc.vencimiento,
            CASE WHEN pc.fecha_canc IS NOT NULL AND pc.fecha_canc <= @T
                 THEN pc.fecha_canc ELSE @T END) AS dias_mora
    FROM prestamos pe
    INNER JOIN PRES_CUOTAS pc
        ON  pc.sucursal = pe.sucursal AND pc.operatoria = pe.operatoria
        AND pc.linea = pe.linea AND pc.nro_servicio = pe.nro_servicio
    WHERE pc.vencimiento <= @T
),

agg_mora AS (
    SELECT
        cliente_id,
        MAX(CASE WHEN dias_mora > 0 THEN dias_mora ELSE 0 END) AS max_dias_atraso_historico,
        AVG(CASE WHEN vencimiento > @INI24
                 THEN CAST(CASE WHEN dias_mora > 0 THEN dias_mora ELSE 0 END AS FLOAT) END)
                                                                AS dias_atraso_promedio,
        SUM(CASE WHEN vencimiento > @INI24 AND dias_mora <= 0 THEN 1 ELSE 0 END) AS cuotas_puntuales_24m,
        SUM(CASE WHEN vencimiento > @INI24 AND dias_mora BETWEEN  1 AND 30 THEN 1 ELSE 0 END) AS atraso_leve_24m,
        SUM(CASE WHEN vencimiento > @INI24 AND dias_mora BETWEEN 31 AND 90 THEN 1 ELSE 0 END) AS atraso_moderado_24m
    FROM cuotas_mora
    GROUP BY cliente_id
),

-- Saldos de cuentas a la vista, 6 meses previos al cutoff.
--
-- El tipo de cuenta NO se deduce del texto de lineas.concepto (esa tabla es
-- genérica para toda operatoria o servicio, y sus 160 descripciones son texto
-- libre). Se deduce de LIN_CTAVISTA, que sólo lista líneas de cuentas a la
-- vista y trae las propiedades que definen una cuenta corriente:
-- emite chequera, admite descubierto o admite acuerdos.
--
-- Hoy esta mutual no tiene ninguna línea con esas marcas -> todas sus cuentas
-- a la vista son cajas de ahorro. La regla queda igual escrita de forma
-- estructural para que, si mañana se habilita una cuenta corriente, se
-- clasifique sola sin tocar la query.
--
-- Los agregados monetarios se restringen a PESOS (moneda = 1). Hay líneas en
-- dólares y euros, y promediar saldos entre monedas no significa nada.
saldos AS (
    SELECT
        pe.cliente_id,
        CASE WHEN lc.chequera_emite = 1 OR lc.descubierto = 1 OR lc.acuerdos = 1
             THEN 1 ELSE 0 END          AS es_corriente,
        CASE WHEN l.moneda = 1 THEN 1 ELSE 0 END AS es_pesos,
        ct.sucursal, ct.linea, ct.nro_servicio,
        AVG(CAST(ct.saldo AS FLOAT)) AS saldo_promedio,
        MIN(CAST(ct.saldo AS FLOAT)) AS saldo_minimo
    FROM (SELECT DISTINCT cliente_id, per_tipo, per_codigo FROM prestamos) pe
    INNER JOIN MAESTROS_OPER mo
        ON mo.per_tipo = pe.per_tipo AND mo.per_codigo = pe.per_codigo
    INNER JOIN CTACTE_MAESAL ct
        ON  ct.sucursal = mo.sucursal AND ct.operatoria = mo.operatoria
        AND ct.linea = mo.linea AND ct.nro_servicio = mo.nro_servicio
    INNER JOIN lineas l
        ON l.operatoria = mo.operatoria AND l.linea = mo.linea
    INNER JOIN LIN_CTAVISTA lc
        ON lc.operatoria = mo.operatoria AND lc.linea = mo.linea
    WHERE ct.fecha > @INI6 AND ct.fecha <= @T
    GROUP BY pe.cliente_id, ct.sucursal, ct.linea, ct.nro_servicio,
             lc.chequera_emite, lc.descubierto, lc.acuerdos, l.moneda
),

agg_saldos AS (
    SELECT
        cliente_id,
        ISNULL(AVG(CASE WHEN es_pesos = 1 AND es_corriente = 0 THEN saldo_promedio END), 0) AS saldo_promedio_ahorro_6m,
        ISNULL(AVG(CASE WHEN es_pesos = 1 AND es_corriente = 1 THEN saldo_promedio END), 0) AS saldo_promedio_corriente_6m,
        ISNULL(MIN(CASE WHEN es_pesos = 1 THEN saldo_minimo END), 0)                        AS saldo_minimo_6m,
        COUNT(*)                                                                            AS productos_activos
    FROM saldos
    GROUP BY cliente_id
),

ingresos AS (
    SELECT pe.cliente_id, ISNULL(i.mensual, 0) AS ingreso_mensual
    FROM (SELECT DISTINCT cliente_id, per_tipo, per_codigo FROM prestamos) pe
    OUTER APPLY (
        SELECT TOP 1 ROUND(pf.importe_anual / 12.0, 0) AS mensual
        FROM PERFILES_CLIENTES pf
        WHERE pf.per_tipo = pe.per_tipo AND pf.per_codigo = pe.per_codigo
          AND pf.fecha_vigencia <= @T
        ORDER BY pf.fecha_vigencia DESC
    ) i
),

-- Clasificación de gerencia vigente AL cutoff (para elegibilidad, no feature).
clasif AS (
    SELECT pe.cliente_id, ISNULL(k.clasificacion_gerencia, 1) AS clasif_T
    FROM (SELECT DISTINCT cliente_id, per_tipo, per_codigo FROM prestamos) pe
    OUTER APPLY (
        SELECT TOP 1 clasificacion_gerencia
        FROM CLASIFICACION_CLIENTE c
        WHERE c.per_tipo = pe.per_tipo AND c.per_codigo = pe.per_codigo
          AND c.fecha <= @T
        ORDER BY c.fecha DESC
    ) k
),

-- Exposición: ¿tiene alguna cuota que vence dentro de la ventana? Sin
-- exposición el cliente no puede defaultear y sólo diluye la muestra.
exposicion AS (
    SELECT DISTINCT pe.cliente_id
    FROM prestamos pe
    INNER JOIN PRES_CUOTAS pc
        ON  pc.sucursal = pe.sucursal AND pc.operatoria = pe.operatoria
        AND pc.linea = pe.linea AND pc.nro_servicio = pe.nro_servicio
    WHERE pc.vencimiento > @T AND pc.vencimiento <= @FIN
),

-- TARGET: alcanza 90 dpd por primera vez dentro de (@T, @FIN].
target AS (
    SELECT DISTINCT pe.cliente_id
    FROM prestamos pe
    INNER JOIN PRES_CUOTAS pc
        ON  pc.sucursal = pe.sucursal AND pc.operatoria = pe.operatoria
        AND pc.linea = pe.linea AND pc.nro_servicio = pe.nro_servicio
    WHERE DATEADD(DAY, 90, pc.vencimiento) >  @T
      AND DATEADD(DAY, 90, pc.vencimiento) <= @FIN
      AND (pc.fecha_canc IS NULL OR pc.fecha_canc > DATEADD(DAY, 90, pc.vencimiento))
)

SELECT
    a.cliente_id,
    CAST(@T AS DATE)                                      AS cutoff,

    a.total_prestamos,
    a.prestamos_al_dia,
    a.prestamos_con_atraso,
    ISNULL(m.max_dias_atraso_historico, 0)                AS max_dias_atraso_historico,
    ISNULL(m.dias_atraso_promedio, 0)                     AS dias_atraso_promedio,
    a.prestamos_refinanciados,
    ISNULL(m.cuotas_puntuales_24m, 0)                     AS cuotas_puntuales_24m,
    ISNULL(m.atraso_leve_24m, 0)                          AS atraso_leve_24m,
    ISNULL(m.atraso_moderado_24m, 0)                      AS atraso_moderado_24m,

    ISNULL(s.saldo_promedio_ahorro_6m, 0)                 AS saldo_promedio_ahorro_6m,
    ISNULL(s.saldo_promedio_corriente_6m, 0)              AS saldo_promedio_corriente_6m,
    ISNULL(s.saldo_minimo_6m, 0)                          AS saldo_minimo_6m,

    a.deuda_vigente_total,
    ing.ingreso_mensual * 0.30                            AS capacidad_pago_estimada,
    DATEDIFF(MONTH, a.fec_ing, @T)                        AS antiguedad_meses,
    ISNULL(s.productos_activos, 0)                        AS productos_activos,
    ISNULL(u.ultimo_prestamo_monto, 0)                    AS ultimo_prestamo_monto,
    DATEDIFF(MONTH, a.ultima_apertura, @T)                AS ultimo_prestamo_hace_meses,

    -- Elegibilidad (para entrenamiento; en inferencia se ignoran).
    CASE WHEN ISNULL(m.max_dias_atraso_historico, 0) >= 90 THEN 1 ELSE 0 END AS ya_default_atraso,
    CASE WHEN c.clasif_T > 4 THEN 1 ELSE 0 END                               AS ya_default_clasif,
    CASE WHEN e.cliente_id IS NOT NULL THEN 1 ELSE 0 END                     AS expuesto,
    CASE WHEN t.cliente_id IS NOT NULL THEN 1 ELSE 0 END                     AS default_90d

FROM agg_prestamos a
LEFT JOIN agg_mora     m ON m.cliente_id = a.cliente_id
LEFT JOIN agg_saldos   s ON s.cliente_id = a.cliente_id
LEFT JOIN ultimo_prestamo u ON u.cliente_id = a.cliente_id
LEFT JOIN ingresos    ing ON ing.cliente_id = a.cliente_id
LEFT JOIN clasif        c ON c.cliente_id = a.cliente_id
LEFT JOIN exposicion    e ON e.cliente_id = a.cliente_id
LEFT JOIN target        t ON t.cliente_id = a.cliente_id
"""


def _derivar(df: pd.DataFrame) -> pd.DataFrame:
    """Features derivadas. Una sola definición, usada en train y en serve."""
    df = df.copy()
    df["tasa_cumplimiento"] = (
        df["prestamos_al_dia"] / df["total_prestamos"].where(df["total_prestamos"] > 0)
    ).fillna(0.0)

    # OJO: contra capacidad_pago_estimada (30% del ingreso), no contra el ingreso
    # bruto. El script viejo usaba la capacidad y la inferencia usaba el ingreso:
    # un factor 3,33x de diferencia entre entrenamiento y producción.
    cap = df["capacidad_pago_estimada"]
    df["ratio_deuda_ingresos"] = (df["deuda_vigente_total"] / cap.where(cap > 0)).fillna(999.0)
    return df


def cargar_features(conexion, cutoff: str, horizonte_meses: int = 12,
                    cliente_id: str | None = None) -> pd.DataFrame:
    """
    Devuelve un DataFrame con FEATURE_COLUMNS + columnas de elegibilidad/target,
    todo calculado a fecha `cutoff`.

    `conexion` puede ser un Engine o una Connection ya abierta (para poder
    reusar la sesión del repositorio en inferencia).

    En inferencia: cutoff = hoy, cliente_id = el CUIT. Las columnas de
    elegibilidad y `default_90d` no tienen sentido (la ventana futura no
    ocurrió todavía) y deben ignorarse.
    """
    filtro = "AND p.id_insc = :cliente_id" if cliente_id else ""
    sql = _SQL.format(filtro_cliente=filtro)

    params: dict = {"cutoff": cutoff, "horizonte": horizonte_meses}
    if cliente_id:
        params["cliente_id"] = cliente_id

    if isinstance(conexion, Engine):
        with conexion.connect() as conn:
            df = pd.read_sql(text(sql), conn, params=params)
    else:
        df = pd.read_sql(text(sql), conexion, params=params)

    return _derivar(df)


def features_sin_historial() -> dict:
    """
    Features para un cliente sin ningún préstamo (thin-file). La query no
    devuelve fila porque parte de MAESTROS_OPER; el modelo igual necesita un
    vector. Todo en cero, salvo el ratio deuda/ingresos, que usa el mismo
    centinela que _derivar() cuando no hay capacidad de pago declarada.

    CreditScoringModel.predict() detecta total_prestamos == 0 y marca el
    informe como thin-file: el score alto refleja AUSENCIA de señal negativa,
    no buen comportamiento demostrado.
    """
    d = {c: 0.0 for c in FEATURE_COLUMNS}
    d["ratio_deuda_ingresos"] = 999.0
    return d


def filtrar_elegibles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Población entrenable en un cutoff dado:
      - NO estaba ya en default al cutoff (ni por mora 90+ ni por clasificación),
        porque predecir un default ya ocurrido no es predecir.
      - Tiene exposición en la ventana de performance: alguna cuota vence dentro
        de ella. Sin exposición el evento es imposible por construcción.
    """
    return df[
        (df["ya_default_atraso"] == 0)
        & (df["ya_default_clasif"] == 0)
        & (df["expuesto"] == 1)
    ].copy()
