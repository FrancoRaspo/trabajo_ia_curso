# scripts/entrenar_modelo_real.py
#
# Entrena el modelo XGBoost con datos históricos reales de la base.
# Usa las tablas reales: MAESTROS_OPER, PRES_CUOTAS, PERSONAS, etc.
#
# Ejecutar con:
#   python3 scripts/entrenar_modelo_real.py

import sys
import os
# Agregar la raíz del proyecto al path para que encuentre db/, ml/, etc.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()



import os
import pandas as pd
import joblib
from sqlalchemy import text
from db.sqlserver_connection import get_sqlserver_engine
from ml.scoring_model import CreditScoringModel

os.makedirs("models", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# QUERY DE ENTRENAMIENTO
# Extrae features + target para todos los socios con historia de préstamos.
#
# TARGET (default_90d = 1) si:
#   - Tiene clasificacion_gerencia > 4  (IRRECUPERABLE en tu DB)
#   - O tuvo alguna cuota con más de 90 días de atraso
# ─────────────────────────────────────────────────────────────────────────────

QUERY_ENTRENAMIENTO = """
 SELECT 
    p.id_insc                                               AS cliente_id,

    -- Features: préstamos
    COUNT(DISTINCT mo.nro_servicio)                         AS total_prestamos,

    SUM(CASE WHEN estado_calc.estado = 'VIGENTE'
             AND ISNULL(DATEDIFF(DAY, cuotas_vencida.vencimiento, GETDATE()), 0) = 0
             THEN 1 ELSE 0 END)                             AS prestamos_al_dia,

    SUM(CASE WHEN ISNULL(DATEDIFF(DAY, cuotas_vencida.vencimiento, GETDATE()), 0) > 0
             THEN 1 ELSE 0 END)                             AS prestamos_con_atraso,

    ISNULL(MAX(ISNULL(
        max_atraso.dias,
        DATEDIFF(DAY, cuotas_vencida.vencimiento, GETDATE())
    )), 0)                                                  AS max_dias_atraso_historico,

    SUM(CASE WHEN refinanciado.cantidad > 0 THEN 1 ELSE 0 END) AS prestamos_refinanciados,

    SUM(CASE WHEN clasificacion.clasificacion_gerencia > 4
             THEN 1 ELSE 0 END)                             AS prestamos_irrecuperables,

    -- Features: comportamiento de pagos (24 meses)
    ISNULL(pagos24.cuotas_puntuales,   0)                   AS cuotas_puntuales_24m,
    ISNULL(pagos24.atraso_leve,        0)                   AS atraso_leve_24m,
    ISNULL(pagos24.atraso_moderado,    0)                   AS atraso_moderado_24m,
    ISNULL(pagos24.atraso_grave,       0)                   AS atraso_grave_24m,
    ISNULL(pagos24.promedio_atraso,    0)                   AS dias_atraso_promedio,

    -- Features: deuda y capacidad
    ISNULL(SUM(CASE WHEN estado_calc.estado NOT IN ('CANCELADO','IRRECUPERABLE')
                    THEN pm.saldo_de_capital END), 0)        AS deuda_vigente_total,

    ISNULL(MAX(ingreso.importe_neto_declarado) * 0.30, 0)   AS capacidad_pago_estimada,

    -- Features: relación con la entidad
    DATEDIFF(MONTH, pr.fec_ing, GETDATE())                  AS antiguedad_meses,
    ISNULL(MAX(pm.capital), 0)                              AS ultimo_prestamo_monto,
    ISNULL(MIN(DATEDIFF(MONTH, mo.fecha_apert, GETDATE())), 0) AS ultimo_prestamo_hace_meses,

    -- TARGET: 1 si defaulteó, 0 si pagó bien
    CASE WHEN
        MAX(CASE WHEN clasificacion.clasificacion_gerencia > 4 THEN 1 ELSE 0 END) = 1
        OR ISNULL(MAX(ISNULL(max_atraso.dias,
                      DATEDIFF(DAY, cuotas_vencida.vencimiento, GETDATE()))), 0) > 90
    THEN 1 ELSE 0 END                                       AS default_90d
    ,ISNULL(saldos_cli.saldo_promedio_6m, 0) AS saldo_promedio_ahorro_6m
    ,0.0                                     AS saldo_promedio_corriente_6m
    ,ISNULL(saldos_cli.saldo_minimo_6m,  0) AS saldo_minimo_6m
FROM PERSONAS p
INNER JOIN PERSONA_REL pr
    ON  pr.tdi_cod = p.tdi_cod
    AND pr.nro_doc = p.nro_doc

INNER JOIN MAESTROS_OPER mo
    ON  mo.per_tipo    = pr.per_tipo
    AND mo.per_codigo  = pr.per_codigo

INNER JOIN PRES_MAESTRO pm
    ON  pm.sucursal    = mo.sucursal
    AND pm.operatoria  = mo.operatoria
    AND pm.linea       = mo.linea
    AND pm.nro_servicio= mo.nro_servicio

-- Estado calculado igual que en get_historial_prestamos
OUTER APPLY (
    SELECT CASE
        WHEN clas_inner.clasificacion_gerencia > 4 THEN 'IRRECUPERABLE'
        WHEN ref_inner.cantidad > 0                THEN 'REFINANCIADO'
        WHEN mo.fecha_baja IS NULL                 THEN 'VIGENTE'
        ELSE 'CANCELADO'
    END AS estado
    FROM (
        SELECT TOP 1 clasificacion_gerencia
        FROM CLASIFICACION_CLIENTE
        WHERE per_tipo    = mo.per_tipo
          AND per_codigo  = mo.per_codigo
        ORDER BY fecha DESC
    ) clas_inner,
    (
        SELECT COUNT(1) cantidad
        FROM MAESTROS_REL_MAESTROS
        WHERE sucursal_rel    = mo.sucursal
          AND operatoria_rel  = mo.operatoria
          AND linea_rel       = mo.linea
          AND nro_servicio_rel= mo.nro_servicio
          AND id_sys_relacion = 1
    ) ref_inner
) estado_calc

-- Clasificación del cliente (para target)
OUTER APPLY (
    SELECT TOP 1 clasificacion_gerencia
    FROM CLASIFICACION_CLIENTE
    WHERE per_tipo   = mo.per_tipo
      AND per_codigo = mo.per_codigo
    ORDER BY fecha DESC
) clasificacion

-- Refinanciado (para feature)
OUTER APPLY (
    SELECT COUNT(1) cantidad
    FROM MAESTROS_REL_MAESTROS
    WHERE sucursal_rel    = mo.sucursal
      AND operatoria_rel  = mo.operatoria
      AND linea_rel       = mo.linea
      AND nro_servicio_rel= mo.nro_servicio
      AND id_sys_relacion = 1
) refinanciado

-- Cuota más vencida sin pagar (para días de atraso actual)
OUTER APPLY (
    SELECT TOP 1 pc.vencimiento
    FROM PRES_CUOTAS pc
    WHERE pc.sucursal    = mo.sucursal
      AND pc.operatoria  = mo.operatoria
      AND pc.linea       = mo.linea
      AND pc.nro_servicio= mo.nro_servicio
      AND pc.fecha_canc IS NULL
      AND pc.vencimiento < GETDATE()
    ORDER BY pc.vencimiento ASC
) cuotas_vencida

-- Máximo atraso histórico de cuotas ya pagadas
OUTER APPLY (
    SELECT TOP 1 DATEDIFF(DAY, pc.vencimiento, pc.fecha_canc) dias
    FROM PRES_CUOTAS pc
    WHERE pc.sucursal    = mo.sucursal
      AND pc.operatoria  = mo.operatoria
      AND pc.linea       = mo.linea
      AND pc.nro_servicio= mo.nro_servicio
      AND pc.fecha_canc IS NOT NULL
      AND pc.vencimiento < GETDATE()
      AND DATEDIFF(DAY, pc.vencimiento, pc.fecha_canc) > 0
    ORDER BY DATEDIFF(DAY, pc.vencimiento, pc.fecha_canc) DESC
) max_atraso

-- Ingresos declarados
OUTER APPLY (
    SELECT TOP 1 ROUND(importe_anual / 12, 0) importe_neto_declarado
    FROM PERFILES_CLIENTES
    WHERE per_tipo   = pr.per_tipo
      AND per_codigo = pr.per_codigo
    ORDER BY fecha_vigencia DESC
) ingreso
-- Saldos del cliente en CTACTE_MAESAL (últimos 6 meses)
OUTER APPLY (
    SELECT
        AVG(ctacte.saldo) AS saldo_promedio_6m,
        MIN(ctacte.saldo) AS saldo_minimo_6m
    FROM   CTACTE_MAESAL ctacte
    INNER JOIN MAESTROS_OPER mo2
        ON  mo2.nro_servicio = ctacte.nro_servicio
        AND mo2.linea        = ctacte.linea
        AND mo2.operatoria   = ctacte.operatoria
        AND mo2.sucursal     = ctacte.sucursal
    WHERE  mo2.per_tipo   = pr.per_tipo
      AND  mo2.per_codigo = pr.per_codigo
      AND  ctacte.fecha  >= DATEADD(MONTH, -6, GETDATE())
) saldos_cli
-- Resumen pagos últimos 24 meses (subconsulta agregada)
OUTER APPLY (
    SELECT
        SUM(CASE WHEN DATEDIFF(DAY, pc.vencimiento,
                               ISNULL(pc.fecha_canc, GETDATE())) = 0 THEN 1 ELSE 0 END) AS cuotas_puntuales,
        SUM(CASE WHEN DATEDIFF(DAY, pc.vencimiento,
                               ISNULL(pc.fecha_canc, GETDATE())) BETWEEN 1 AND 30 THEN 1 ELSE 0 END) AS atraso_leve,
        SUM(CASE WHEN DATEDIFF(DAY, pc.vencimiento,
                               ISNULL(pc.fecha_canc, GETDATE())) BETWEEN 31 AND 90 THEN 1 ELSE 0 END) AS atraso_moderado,
        SUM(CASE WHEN DATEDIFF(DAY, pc.vencimiento,
                               ISNULL(pc.fecha_canc, GETDATE())) > 90 THEN 1 ELSE 0 END) AS atraso_grave,
        AVG(CAST(DATEDIFF(DAY, pc.vencimiento,
                          ISNULL(pc.fecha_canc, GETDATE())) AS FLOAT)) AS promedio_atraso
    FROM PRES_CUOTAS pc
    WHERE pc.sucursal    = mo.sucursal
      AND pc.operatoria  = mo.operatoria
      AND pc.linea       = mo.linea
      AND pc.nro_servicio= mo.nro_servicio
      AND pc.vencimiento >= DATEADD(MONTH, -24, GETDATE())
      AND pc.vencimiento < GETDATE()
) pagos24

GROUP BY
    p.id_insc,
    pr.fec_ing,
    pr.per_tipo,
    pr.per_codigo,
	pagos24.cuotas_puntuales, pagos24.atraso_leve,pagos24.atraso_moderado,
	atraso_grave, promedio_atraso,saldos_cli.saldo_promedio_6m,saldos_cli.saldo_minimo_6m
HAVING COUNT(DISTINCT mo.nro_servicio) > 0
"""

# ─────────────────────────────────────────────────────────────────────────────
# EJECUCIÓN
# ─────────────────────────────────────────────────────────────────────────────

print("Conectando a SQL Server...")
engine = get_sqlserver_engine()

print("Extrayendo datos históricos (puede tardar unos segundos)...")
with engine.connect() as conn:
    df = pd.read_sql(text(QUERY_ENTRENAMIENTO), conn)

print(f"\nRegistros obtenidos: {len(df)}")
print(f"Distribución del target:")
print(df["default_90d"].value_counts().to_string())
print(f"Tasa de irrecuperables/default: {df['default_90d'].mean():.1%}")

# Verificar que hay suficientes datos para entrenar
if len(df) < 50:
    print("\n⚠️  Menos de 50 registros. El modelo no será confiable.")
    print("   Considerá usar el scoring por reglas en su lugar.")
    exit(1)

if df["default_90d"].sum() < 10:
    print("\n⚠️  Menos de 10 casos de default. El modelo no aprenderá bien.")
    print("   Considerá ampliar el período histórico o usar scoring por reglas.")
    exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURES DERIVADAS
# ─────────────────────────────────────────────────────────────────────────────

df["tasa_cumplimiento"] = df.apply(
    lambda r: r["prestamos_al_dia"] / r["total_prestamos"]
    if r["total_prestamos"] > 0 else 0,
    axis=1,
)
df["ratio_deuda_ingresos"] = df.apply(
    lambda r: r["deuda_vigente_total"] / r["capacidad_pago_estimada"]
    if r["capacidad_pago_estimada"] > 0 else 999,
    axis=1,
)
# productos_activos no viene de la query (saldos no disponibles en este contexto)
# usamos total_prestamos como proxy
df["productos_activos"] = df["total_prestamos"]

# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

print("\nEntrenando modelo XGBoost...")
model = CreditScoringModel()
auc   = model.train(df, target_col="default_90d")

# ─────────────────────────────────────────────────────────────────────────────
# TOP FEATURES
# ─────────────────────────────────────────────────────────────────────────────

importancias = sorted(
    zip(CreditScoringModel.FEATURE_COLUMNS, model.model.feature_importances_),
    key=lambda x: x[1], reverse=True,
)
print("\nTop 5 variables más importantes para predecir irrecuperabilidad:")
for feat, imp in importancias[:5]:
    print(f"  {feat:<40} {imp:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# GUARDAR
# ─────────────────────────────────────────────────────────────────────────────

model.save("models/scoring_model.json")
print("\n✅ Listo. Ahora podés correr pipeline.py")