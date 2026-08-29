from sqlalchemy import text
from datetime import datetime, date, timedelta

# Tablas base de SQL Server que este módulo y ml/feature_query realmente leen.
#
# Va acá, junto a las queries, porque es acá donde puede quedar desactualizada.
# El informe la publica en su campo de trazabilidad: si miente, el log auditable
# miente. tests/test_trazabilidad.py extrae las tablas del SQL fuente y falla si
# esta lista se desincroniza en cualquiera de las dos direcciones.
TABLAS_SQLSERVER = [
    "ACTIVIDAD",
    "ACTIVIDAD_RUBRO",
    "ACTIVIDAD_SUBRUB",
    "CLASIFICACION_CLIENTE",
    "CTACTE_MAESAL",
    "CUOMUTUAL_MAE",
    "DOCUMENTO_TIPO",
    "ESTADO",
    "GARANTIAS_MAE",
    "GARANTIAS_REL",
    "GARANTIAS_SUBTIPO",
    "GARANTIAS_TIPO",
    "LIN_CTAVISTA",
    "LIN_PRESTAMO",
    "LINEAS",
    "MAESTROS_OPER",
    "MAESTROS_REL_MAESTROS",
    "MONEDA",
    "PERFILES_CLIENTES",
    "PERSONA_REL",
    "PERSONA_TIPO",
    "PERSONAS",
    "PRES_CUOTAS",
    "PRES_MAESTRO",
    "SITUACION_OPER",
    "SITUACION_TIPO_TRAMITES",
    "SITUACION_TRAMITE_PASOS",
]


def _num(x):
    """Decimal/None de SQL -> float seguro para las features."""
    return float(x) if x is not None else 0.0

def to_date(val):
    """Normaliza datetime o date a date. Retorna None si val es None."""
    if val is None:
        return None
    return val.date() if isinstance(val, datetime) else val

class ClientRepository:
    """
    Consultas de solo lectura sobre SQL Server.
    Toda la lógica de acceso a datos del cliente vive aquí.
    """

    def __init__(self, session):
        self.session = session

    def get_datos_basicos(self, cliente_id: str) -> dict:
        result = self.session.execute(text("""
            SELECT p.id_insc id,
            p.nom_ape nombre_completo,  
            p.nro_doc nro_documento,
            DT.descripcion_red tipo_documento,
            PR.fec_ing fecha_ingreso_entidad,
            PT.descripcion    categoria,
            ingreso.importe_neto_declarado    ingreso_neto_declarado,
            CONCAT(A.descripcion,'/',t.descripcion)     actividad_economica,
            e.descripcion        estado_actual
            
            FROM PERSONAS P INNER JOIN PERSONA_REL PR ON PR.tdi_cod = P.tdi_cod 
            AND PR.nro_doc = P.nro_doc 
            INNER JOIN DOCUMENTO_TIPO DT ON DT.documento_tipo = P.tdi_cod 
            INNER JOIN ACTIVIDAD a  ON A.actividad = P.actividad 
            INNER JOIN ACTIVIDAD_RUBRO ar  ON AR.rubro = P.rubro 
            INNER JOIN ACTIVIDAD_SUBRUB t ON T.rubro =P.rubro  AND t.subrubro = p.subrubro 
            INNER JOIN PERSONA_TIPO PT ON PT.per_tipo = PR.per_tipo 
            INNER JOIN CUOMUTUAL_MAE CM ON cm.per_tipo=pr.per_tipo AND CM.per_codigo = PR.per_codigo 
            INNER JOIN ESTADO E ON E.estado = CM.estado 
            OUTER APPLY (SELECT TOP 1 round(importe_anual/12,0) importe_neto_declarado 
                        FROM PERFILES_CLIENTES PC
                         WHERE PC.per_tipo=PR.per_tipo and pc.per_codigo=pr.per_codigo
                         order by PC.fecha_vigencia DESC
            ) ingreso
            where p.id_insc = :cliente_id
        """), {"cliente_id": cliente_id}).mappings().fetchone()

        if not result:
            raise ValueError(f"Cliente {cliente_id} no encontrado en Base Financiera.")
        return dict(result)

    def get_historial_prestamos(self, cliente_id: str) -> list[dict]:
        rows = self.session.execute(text("""
            SELECT  
                   CONCAT(
                       mo.sucursal
                      ,'-'
                      ,mo.operatoria
                      ,'-'
                      ,mo.linea
                      ,'/'
                      ,mo.nro_servicio
                   )                         prestamo_id
                  ,mo.fecha_apert            fecha_otorgamiento
                  ,pm.capital                monto_original
                  ,m.descripcion_red         moneda
                  ,pm.saldo_de_capital       saldo_capital_actual
                  ,pm.tasa_liq               tasa_interes_anual
                  ,pm.nro_de_cuotas          plazo_cuotas
                  ,cuotas_pagas.cantidad     cuotas_pagadas
                  ,nro_de_cuotas - cuotas_pagas.cantidad cuotas_pendientes
                  ,ISNULL(DATEDIFF(DAY ,cuotas_vencida.vencimiento ,GETDATE()) ,0) dias_atraso_actual
                  ,ISNULL(max_atraso.dias ,0) max_dias_atraso_historico
                  ,estado.descripcion        estado -- VIGENTE / CANCELADO / IRRECUPERABLE / REFINANCIADO
                  ,ISNULL(garantias.conceptos ,'Sin garantías o garantía personal') tipo_garantia
                  ,ppu.nom_ape               oficial_credito
                  ,pm.vencimiento            fecha_vencimiento
            FROM   MAESTROS_OPER mo
                   inner JOIN persona_rel pu
                        ON  mo.per_tipo_sol = pu.per_tipo
                        AND mo.usr_solicitud = pu.per_codigo
                   inner JOIN personas ppu
                        ON  ppu.tdi_cod = pu.tdi_cod
                        AND ppu.nro_doc = pu.nro_doc
                   INNER JOIN PERSONA_REL pr
                        ON  pr.per_tipo = mo.per_tipo
                        AND pr.per_codigo = mo.per_codigo
                   INNER JOIN PERSONAS P
                        ON  p.tdi_cod = pr.tdi_cod
                        AND p.nro_doc = pr.nro_doc
                   INNER JOIN PRES_MAESTRO pm
                        ON  pm.sucursal = mo.sucursal
                        AND pm.operatoria = mo.operatoria
                        AND pm.linea = mo.linea
                        AND pm.nro_servicio = mo.nro_servicio
                   INNER JOIN LINEAS L
                        ON  l.operatoria = mo.operatoria
                        AND l.linea = mo.linea
                   inner JOIN moneda m
                        ON  m.moneda = l.moneda
                   inner JOIN lin_prestamo lp
                        ON  lp.operatoria = l.operatoria
                        AND lp.linea = l.linea
                   outer APPLY (
                SELECT COUNT(1)        cantidad
                FROM   PRES_CUOTAS     PC
                WHERE  PC.sucursal = mo.sucursal
                       AND PC.operatoria = mo.operatoria
                       AND PC.linea = mo.linea
                       AND PC.nro_servicio = mo.nro_servicio
                       AND pc.fecha_canc is NOT null
            ) cuotas_pagas
            outer APPLY (
                SELECT top 1 pc.vencimiento
                FROM   PRES_CUOTAS PC
                WHERE  PC.sucursal = mo.sucursal
                       AND PC.operatoria = mo.operatoria
                       AND PC.linea = mo.linea
                       AND PC.nro_servicio = mo.nro_servicio
                       AND pc.fecha_canc is null
                       AND pc.vencimiento < GETDATE()
                ORDER BY
                       pc.vencimiento asc
            ) cuotas_vencida
            -- Máximo atraso histórico de la operación. MISMA definición que la
            -- feature ML `max_dias_atraso_historico` (ml/feature_query.py): por
            -- cada cuota YA VENCIDA, los días entre su vencimiento y la fecha de
            -- pago si se pagó, o HOY si sigue impaga.
            --
            -- Antes filtraba `fecha_canc IS NOT NULL`, o sea SÓLO cuotas pagadas
            -- tarde, e ignoraba las impagas vigentes — justo las de mayor atraso,
            -- que además crece todos los días. Un cliente con 166 días de mora
            -- viva y un atraso viejo de 80 ya saldado publicaba "máximo histórico
            -- 80": un máximo MENOR que su atraso actual. Esa contradicción la
            -- detectó el juez del golden set como alucinación CRÍTICA en el caso
            -- muy_alto_2. Es el mismo bug que ya se corrigió en
            -- `promedio_dias_atraso`.
            --
            -- Con esta definición vale siempre max_dias_atraso_historico >=
            -- dias_atraso_actual, porque la cuota impaga más vieja entra en el MAX.
            outer APPLY (
                SELECT MAX(CASE WHEN d.dias > 0 THEN d.dias ELSE 0 END) dias
                FROM   (
                           SELECT DATEDIFF(
                                      DAY
                                     ,pc.vencimiento
                                     ,CASE
                                          WHEN pc.fecha_canc IS NOT NULL
                                               AND pc.fecha_canc <= GETDATE()
                                          THEN pc.fecha_canc
                                          ELSE GETDATE()
                                      END
                                  ) dias
                           FROM   PRES_CUOTAS PC
                           WHERE  PC.sucursal = mo.sucursal
                                  AND PC.operatoria = mo.operatoria
                                  AND PC.linea = mo.linea
                                  AND PC.nro_servicio = mo.nro_servicio
                                  AND pc.vencimiento <= GETDATE()
                       ) d
            ) max_atraso
            outer APPLY (
                SELECT COUNT(1)                  cantidad
                FROM   MAESTROS_REL_MAESTROS  AS mrm
                WHERE  mrm.sucursal_rel = mo.sucursal
                       AND mrm.operatoria_rel = mo.operatoria
                       AND mrm.linea_rel = mo.linea
                       AND mrm.nro_servicio_rel = mo.nro_servicio
                       AND mrm.id_sys_relacion = 1
            ) refinanciado
            OUTER APPLY (
                SELECT TOP 1 CC.clasificacion_gerencia
                FROM   CLASIFICACION_CLIENTE AS cc
                WHERE  CC.per_tipo = MO.per_tipo
                       AND cc.per_codigo = mo.per_codigo
                ORDER BY
                       CC.fecha DESC
            ) clasificacion 
            OUTER APPLY (
                SELECT CASE 
                            WHEN clasificacion.clasificacion_gerencia > 4 THEN 'IRRECUPERABLE'
                            WHEN refinanciado.cantidad > 0 THEN 'REFINANCIADO'
                            WHEN mo.fecha_baja IS NULL THEN 'VIGENTE'
                            WHEN mo.fecha_baja IS NOT NULL THEN 'CANCELADO'
                       END descripcion
            ) estado
            OUTER APPLY (
                SELECT STRING_AGG(
                           CONCAT(GT.concepto ,'-' ,GS.concepto ,' importe ' ,gm.importe)
                          ,', '
                       )                                conceptos
                FROM   GARANTIAS_REL                 AS gr
                       INNER JOIN GARANTIAS_TIPO     AS gt
                            ON  GT.tipo = GR.tipo
                       INNER JOIN GARANTIAS_SUBTIPO  AS gs
                            ON  gr.tipo = GS.tipo
                            AND gs.subtipo = gr.subtipo
                       INNER JOIN GARANTIAS_MAE      AS gm
                            ON  gm.tipo = gr.tipo
                            AND gm.subtipo = gr.subtipo
                            AND gm.sucursal = gr.sucursal
                            AND gm.numero = gr.numero
                WHERE  GR.sucursal_op = MO.sucursal
                       AND gr.operatoria = mo.operatoria
                       AND gr.linea = mo.linea
                       AND gr.nro_servicio = mo.nro_servicio
            )                                garantias
            WHERE  p.id_insc =   :cliente_id
            AND mo.estado_ini = 6
            ORDER BY mo.fecha_apert DESC
        """), {"cliente_id": cliente_id}).mappings().fetchall()
        return [dict(r) for r in rows]

    def get_resumen_comportamiento_pagos(self, cliente_id: str,
                                          meses: int = 24) -> dict:
        """
        Resumen del comportamiento de pagos sobre las cuotas VENCIDAS en el
        período (default: 24 meses).

        La definición de `dias_atraso` es deliberadamente la MISMA que la de
        ml/feature_query.py (CTE cuotas_mora): días entre el vencimiento y la
        cancelación —o hoy, si sigue impaga—, con piso en 0. Así el promedio que
        narra el informe y el `dias_atraso_promedio` que ve el modelo son EL
        MISMO NÚMERO. Si divergen, el informe se contradice solo.

        La versión anterior filtraba `AND pc.vencimiento < pc.fecha_canc`, lo que
        rompía el resumen de tres formas a la vez:

          - dejaba sólo las cuotas PAGADAS TARDE, así que `dias_atraso = 0` era
            imposible: `cuotas_puntuales` daba 0 para TODOS los clientes, siempre;
          - `fecha_canc IS NULL` hace que esa comparación sea NULL -> falsa, así
            que las cuotas vencidas e IMPAGAS (las peores) quedaban excluidas del
            resumen de mora;
          - el promedio resultante era una media condicionada a las cuotas
            atrasadas, sistemáticamente mayor que `dias_atraso_promedio` del
            modelo (ej. 60,75 vs 54,0). El informe recibía dos números distintos
            con nombres casi iguales y los mezclaba entre secciones: era la causa
            principal de las "alucinaciones" que marcaba el juez.

        El propio SELECT usaba ISNULL(fecha_canc, GETDATE()), lo que prueba que
        la intención era incluir las impagas: el WHERE contradecía al SELECT.
        """
        resultado = self.session.execute(text("""
            -- Los conteos van con ISNULL: sin filas, SUM() devuelve NULL y el
            -- informe lo narra como "no disponible" cuando la verdad es "no
            -- venció ninguna cuota en el período". Cero es un dato real.
            -- El promedio y el máximo SÍ quedan en NULL: sobre cero cuotas no
            -- existen, y un 0 ahí sí mentiría.
            SELECT COUNT(*)                         AS total_cuotas
                  ,ISNULL(SUM(CASE WHEN dias_atraso = 0 THEN 1 ELSE 0 END), 0) AS cuotas_puntuales
                  ,ISNULL(SUM(CASE WHEN dias_atraso BETWEEN 1 AND 30 THEN 1 ELSE 0 END), 0) AS atraso_leve_1_30d
                  ,ISNULL(SUM(CASE WHEN dias_atraso BETWEEN 31 AND 90 THEN 1 ELSE 0 END), 0) AS atraso_moderado_31_90d
                  ,ISNULL(SUM(CASE WHEN dias_atraso > 90 THEN 1 ELSE 0 END), 0) AS atraso_grave_mas90d
                  ,ISNULL(SUM(CASE WHEN impaga = 1 THEN 1 ELSE 0 END), 0)      AS cuotas_impagas_vencidas
                  ,AVG(CAST(dias_atraso AS FLOAT))  AS promedio_dias_atraso
                  ,MAX(dias_atraso)                 AS max_dias_atraso_periodo
        FROM   PRES_CUOTAS PC
               INNER JOIN MAESTROS_OPER         AS mo
                    ON  mo.sucursal = PC.sucursal
                    AND mo.operatoria = PC.operatoria
                    AND mo.linea = PC.linea
                    AND mo.nro_servicio = PC.nro_servicio
               inner JOIN persona_rel pr
                    ON  pr.per_tipo = mo.per_tipo
                    AND pr.per_codigo = mo.per_codigo
               inner JOIN personas p
                    ON  p.tdi_cod = pr.tdi_cod
                    AND p.nro_doc = pr.nro_doc
               outer APPLY (
            SELECT CASE WHEN DATEDIFF(DAY, pc.vencimiento,
                                      ISNULL(pc.fecha_canc, GETDATE())) > 0
                        THEN DATEDIFF(DAY, pc.vencimiento,
                                      ISNULL(pc.fecha_canc, GETDATE()))
                        ELSE 0 END                          AS dias_atraso
                  ,CASE WHEN pc.fecha_canc IS NULL THEN 1 ELSE 0 END AS impaga
        )                                          cuotas
        WHERE  p.id_insc = :cliente_id
              AND pc.vencimiento >= DATEADD(MONTH, :meses_neg, GETDATE())
              AND pc.vencimiento < GETDATE()
        """), {"cliente_id": cliente_id, "meses_neg": -meses}).mappings().fetchone()
        return dict(resultado) if resultado else {}

    def get_saldos_promedio_cuentas(self, cliente_id: str,
                                     meses: int = 6) -> list[dict]:
        """Saldos promedio por tipo de cuenta en los últimos N meses."""
        rows = self.session.execute(text("""
            SELECT l.concepto            tipo_cuenta
              ,CONCAT(sd.sucursal ,'-' ,sd.linea ,'/' ,sd.nro_servicio) nro_cuenta
              ,m.descripcion_red     moneda
              ,AVG(sd.saldo)      AS saldo_promedio
              ,MIN(sd.saldo)      AS saldo_minimo
              ,MAX(sd.saldo)      AS saldo_maximo -- Último saldo disponible
              ,ultimo.saldo       AS saldo_ultimo
        FROM   maestros_oper mo
               outer APPLY (
            SELECT TOP 1             saldo
            FROM   CTACTE_MAESAL     SD
            WHERE  sd.nro_servicio = mo.nro_servicio
                   AND sd.linea = mo.linea
                   AND sd.operatoria = mo.operatoria
                   AND sd.sucursal = mo.sucursal
            ORDER BY
                   SD.fecha DESC
        ) ultimo 
        inner JOIN persona_rel pr
            ON  pr.per_tipo = mo.per_tipo
            AND pr.per_codigo = mo.per_codigo
           inner JOIN personas p
                ON  p.tdi_cod = pr.tdi_cod
                AND p.nro_doc = pr.nro_doc
           JOIN ctacte_maesal sd
                ON  sd.nro_servicio = mo.nro_servicio
                AND sd.linea = mo.linea
                AND sd.operatoria = mo.operatoria
                AND sd.sucursal = mo.sucursal
           inner JOIN lineas l
                ON  l.operatoria = mo.operatoria
                AND l.linea = mo.linea
       inner JOIN MONEDA  AS m
            ON  m.moneda = l.moneda
            WHERE p.id_insc  = :cliente_id
              AND sd.fecha >= DATEADD(MONTH, :meses_neg, GETDATE())
            GROUP BY
               l.concepto
              ,sd.sucursal
              ,sd.linea
              ,sd.nro_servicio
              ,m.descripcion_red
              ,ultimo.saldo 
        """), {"cliente_id": cliente_id, "meses_neg": -meses}).mappings().fetchall()
        return [dict(r) for r in rows]

    def get_gestiones_mora_estructuradas(self, cliente_id: str) -> list[dict]:
        """
        Gestiones de mora cargadas como datos estructurados en SQL Server.
        (Campos cortos: tipo_gestion, fecha, resultado_contacto)
        Las notas largas de texto libre van a pgvector.
        """
        rows = self.session.execute(text("""
           select
                stp.id,
                stp.fecha_tramite fecha_gestion,
                st.descripcion tipo_gestion,
                STP.detalle resultado_contacto,
                so.total_adeudado_act monto_comprometido,
                stp.fecha_probable fecha_compromiso_pago
            from
                SITUACION_TRAMITE_PASOS stp
            INNER JOIN SITUACION_OPER SO ON
                SO.sucursal = STP.sucursal
                and
              so.operatoria = stp.operatoria
                and so.linea = stp.linea
                and so.nro_servicio = stp.nro_servicio
                and stp.fecha = so.fecha
                and so.situacion = stp.situacion
            INNER JOIN PERSONAS P ON
                P.tdi_cod = STP.tdi_cod
                and p.nro_doc = stp.nro_doc
            inner join SITUACION_TIPO_TRAMITES st ON
                ST.situacion = STP.situacion
                AND ST.tramite = STP.tipo_tramite
            WHERE p.id_insc  = :cliente_id
            ORDER BY stp.fecha_tramite DESC
        """), {"cliente_id": cliente_id}).mappings().fetchall()
        return [dict(r) for r in rows]

    def construir_features_ml(self, cliente_id: str, cutoff: str | None = None) -> dict:
        """
        Vector de features para el modelo de scoring, a fecha `cutoff`
        (por defecto, hoy).

        Delega en ml.feature_query, que es la MISMA query que usa el
        entrenamiento. Antes esta función reimplementaba las features a mano y
        divergía del entrenamiento en silencio:

          - ratio_deuda_ingresos dividía por el ingreso, no por la capacidad de
            pago (30% del ingreso): un factor 3,33x de diferencia.
          - productos_activos contaba cuentas; el entrenamiento contaba préstamos.
          - saldo_promedio_ahorro_6m filtraba tipo_cuenta == 'CAJA_AHORRO', un
            literal que no existe en lineas.concepto -> daba 0 para TODOS los
            clientes, siempre, mientras el entrenamiento veía el valor real.

        No reintroducir cálculos de features acá. Si falta una, va en la query.
        """
        from ml.feature_query import (FEATURE_COLUMNS, cargar_features,
                                      features_sin_historial)

        cutoff = cutoff or date.today().isoformat()
        df = cargar_features(self.session.connection(), cutoff, cliente_id=cliente_id)

        if df.empty:          # cliente sin préstamos: thin-file
            return features_sin_historial()

        fila = df.iloc[0]
        return {c: float(fila[c]) for c in FEATURE_COLUMNS}
