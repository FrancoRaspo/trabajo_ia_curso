"""Normaliza el payload crudo del BCRA a dos formas usables:

  - `resumir()`  -> dict de métricas estructuradas (para reglas y para la
                    sección del informe; cero alucinación, datos directos).
  - `a_texto()`  -> resumen en prosa para indexar en el RAG (pgvector).

Opera sobre el dict que devuelve bcra_client.consultar_todo() /
bcra_cache.obtener_bcra():
    {"deudas": <payload|None>,
     "deudas_historicas": <payload|None>,
     "cheques_rechazados": <payload|None>}
"""

# Escala de clasificación del deudor (Central de Deudores del BCRA).
SITUACIONES = {
    1: "Normal",
    2: "Riesgo bajo / seguimiento especial",
    3: "Con problemas / riesgo medio",
    4: "Alto riesgo de insolvencia",
    5: "Irrecuperable",
    6: "Irrecuperable por disposición técnica",
}


# El API de la Central de Deudores del BCRA expresa todos los importes
# (montos de deuda y de cheques rechazados) en MILES de pesos. Para informar
# en pesos hay que multiplicar por esta escala.
ESCALA_BCRA = 1000


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _monto(x) -> float:
    """Convierte un importe crudo del BCRA (en miles de pesos) a pesos."""
    return _num(x) * ESCALA_BCRA


def _entidades_periodo_vigente(deudas):
    """(periodo, [entidades]) del período más reciente del payload Deudas."""
    if not deudas:
        return None, []
    res = deudas.get("results") or {}
    periodos = res.get("periodos") or []
    if not periodos:
        return None, []
    # El período vigente suele venir primero; tomamos el de mayor 'periodo' (AAAAMM).
    p = max(periodos, key=lambda x: str(x.get("periodo", "")))
    return p.get("periodo"), (p.get("entidades") or [])


def resumir(crudo: dict) -> dict:
    """Construye el resumen estructurado a partir del payload crudo del BCRA."""
    crudo   = crudo or {}
    deudas  = crudo.get("deudas")
    cheques = crudo.get("cheques_rechazados")

    periodo, entidades = _entidades_periodo_vigente(deudas)

    situaciones    = [int(e.get("situacion") or 0) for e in entidades if e.get("situacion")]
    peor_situacion = max(situaciones) if situaciones else None
    deuda_total    = sum(_monto(e.get("monto")) for e in entidades)

    detalle_entidades = [
        {
            "entidad":     e.get("entidad"),
            "situacion":   int(e.get("situacion") or 0),
            "monto":       _monto(e.get("monto")),
            "dias_atraso": int(e.get("diasAtrasoPago") or 0),
        }
        for e in entidades
    ]

    # --- cheques rechazados ---
    ch_detalle = []
    if cheques:
        res = cheques.get("results") or {}
        for causal in (res.get("causales") or []):
            for ent in (causal.get("entidades") or []):
                for d in (ent.get("detalle") or []):
                    ch_detalle.append({
                        "causal":           causal.get("causal"),
                        "entidad":          ent.get("entidad"),
                        "nro_cheque":       d.get("nroCheque"),
                        "fecha_rechazo":    d.get("fechaRechazo"),
                        "monto":            _monto(d.get("monto")),
                        "pagado":           bool(d.get("fechaPago")),
                        "proceso_judicial": bool(d.get("procesoJud")),
                    })
    ch_impagos = [c for c in ch_detalle if not c["pagado"]]

    return {
        "tiene_datos":               bool(entidades) or bool(ch_detalle),
        "periodo_informado":         periodo,
        "cant_entidades":            len(entidades),
        "peor_situacion":            peor_situacion,
        "peor_situacion_desc":       SITUACIONES.get(peor_situacion) if peor_situacion else None,
        "tiene_situacion_irregular": bool(peor_situacion and peor_situacion >= 3),
        "deuda_total_sistema":       round(deuda_total, 2),
        "en_revision":               any(e.get("enRevision") for e in entidades),
        "proceso_judicial":          any(e.get("procesoJud") for e in entidades),
        "refinanciaciones":          any(e.get("refinanciaciones") for e in entidades),
        "entidades":                 detalle_entidades,
        "cheques_rechazados_total":  len(ch_detalle),
        "cheques_impagos":           len(ch_impagos),
        "monto_cheques_rechazados":  round(sum(c["monto"] for c in ch_detalle), 2),
        "monto_cheques_impagos":     round(sum(c["monto"] for c in ch_impagos), 2),
        # Recorte defensivo: el detalle individual no debe inflar el prompt/RAG.
        "cheques_detalle":           ch_detalle[:20],
    }


def formato_pesos(x) -> str:
    """Importe en formato argentino: punto como separador de miles y coma para
    los decimales (ej. $1.234.567,89)."""
    entero, dec = f"{abs(x):,.2f}".split(".")
    entero = entero.replace(",", ".")          # 1,234,567 -> 1.234.567
    signo = "-" if x < 0 else ""
    return f"{signo}${entero},{dec}"


_money = formato_pesos   # alias interno (usado en a_texto)


def a_texto(resumen: dict, cuit: str = "") -> str:
    """Resumen en prosa del perfil BCRA, para indexar en el RAG."""
    ident = f" (CUIT {cuit})" if cuit else ""

    if not resumen.get("tiene_datos"):
        return (
            f"Central de Deudores del BCRA{ident}: sin registros de deuda ni "
            "cheques rechazados en el sistema financiero al momento de la consulta."
        )

    lineas = [f"Situación en el sistema financiero según la Central de Deudores "
              f"del BCRA{ident}."]
    if resumen.get("periodo_informado"):
        lineas.append(f"Período informado: {resumen['periodo_informado']}.")

    if resumen.get("cant_entidades"):
        lineas.append(
            f"Registra deuda en {resumen['cant_entidades']} entidad(es) por un total "
            f"de {_money(resumen['deuda_total_sistema'])}. Peor situación: "
            f"{resumen.get('peor_situacion')} ({resumen.get('peor_situacion_desc')})."
        )
        for e in resumen.get("entidades", []):
            lineas.append(
                f"- {e['entidad']}: situación {e['situacion']} "
                f"({SITUACIONES.get(e['situacion'], 's/d')}), monto "
                f"{_money(e['monto'])}, {e['dias_atraso']} días de atraso."
            )
        if resumen.get("proceso_judicial"):
            lineas.append("Registra deuda en proceso judicial.")
        if resumen.get("refinanciaciones"):
            lineas.append("Registra refinanciaciones.")
        if resumen.get("en_revision"):
            lineas.append("Tiene registros en revisión.")
    else:
        lineas.append("No registra deuda vigente en entidades del sistema financiero.")

    if resumen.get("cheques_rechazados_total"):
        lineas.append(
            f"Cheques rechazados: {resumen['cheques_rechazados_total']} en total "
            f"({_money(resumen['monto_cheques_rechazados'])}), de los cuales "
            f"{resumen['cheques_impagos']} permanecen impagos "
            f"({_money(resumen['monto_cheques_impagos'])})."
        )
    else:
        lineas.append("No registra cheques rechazados.")

    return "\n".join(lineas)
