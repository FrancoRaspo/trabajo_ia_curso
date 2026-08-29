# llm/plantillas.py
"""
Secciones armadas por PLANTILLA (sin LLM).

Una sección factual —que sólo transcribe datos ya calculados— no necesita un
modelo: pedirle que "sólo formatee" le da igualmente la oportunidad de contar
mal, inventar un período o contradecir a otra sección. El juez marcó
exactamente eso en la cartera (conteos y fechas fabricados), y el techo no era
del prompt sino del modelo.

Acá la sección se arma con código: los números salen de los mismos agregados
que ya calcula el pipeline (`cartera_resumen`) y de las filas crudas, así que
son verificables y no pueden divergir entre secciones.

El estilo (títulos en negrita, listas, formato AR) imita al que venía
produciendo el LLM, para que el informe se lea igual que antes.
"""
from external.bcra_normalizer import formato_pesos


def _num(v) -> str:
    """Número a formato argentino (miles con punto, coma decimal). Hasta dos
    decimales, sin ceros de relleno: 24 -> '24', 12.4 -> '12,4'."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f == int(f):
        return f"{int(f):,}".replace(",", ".")
    txt = f"{f:,.2f}".replace(",", "@").replace(".", ",").replace("@", ".")
    return txt.rstrip("0").rstrip(",")


def _monto(v) -> str:
    """Los montos del contexto ya vienen formateados a pesos (strings '$…').
    Si llega un número crudo (uso fuera del pipeline), se formatea acá."""
    if v is None or v == "":
        return "s/d"
    if isinstance(v, str):
        return v
    try:
        return formato_pesos(float(v))
    except (TypeError, ValueError):
        return str(v)


def _entero(v, defecto=0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return defecto


def _plural(n: int, singular: str, plural: str) -> str:
    return singular if n == 1 else plural


def _estado_plural(estado: str, n: int) -> str:
    """'CANCELADO' x2 -> 'cancelados'; x1 -> 'cancelado'."""
    e = str(estado).lower()
    return e if n == 1 or e.endswith("s") else e + "s"


def _linea_prestamo(p: dict) -> str:
    """Una línea por préstamo. El atraso actual sólo tiene sentido en los
    vigentes: en los cancelados el dato es residual y confundiría."""
    vigente = str(p.get("estado") or "").upper() == "VIGENTE"
    partes = [
        f"**{p.get('prestamo_id', 's/id')}**",
        f"otorgado el {p.get('fecha_otorgamiento_texto') or 's/d'}",
        f"monto original {_monto(p.get('monto_original'))}",
    ]
    if vigente:
        partes.append(f"saldo actual {_monto(p.get('saldo_capital_actual'))}")
        pend = _entero(p.get("cuotas_pendientes"), -1)
        if pend >= 0:
            partes.append(f"{_num(pend)} {_plural(pend, 'cuota pendiente', 'cuotas pendientes')}")
        atraso = _entero(p.get("dias_atraso_actual"))
        if atraso > 0:
            gravedad = "**MORA VIGENTE GRAVE**" if atraso > 90 else "**mora vigente**"
            partes.append(f"atraso actual {_num(atraso)} días ({gravedad})")
        else:
            partes.append("sin atraso actual")
    else:
        # En un préstamo no vigente el estado ES la información; el atraso actual
        # es un residuo del cálculo y no se informa (confundiría con mora viva).
        partes.append(f"estado: {str(p.get('estado') or 's/estado').capitalize()}")
    return "- " + ", ".join(partes) + "."


def cartera_frase(car: dict) -> str:
    """Los agregados de la cartera en UNA frase ya redactada, lista para copiar.

    Se pasa dentro de `cartera_resumen` a TODAS las secciones que hablan de la
    cartera. Darle los campos sueltos no alcanzó: el modelo los tenía y aun así
    los recomponía mal en el Resumen y en la Recomendación (contaba 10 préstamos
    donde había 28, o cerraba el período en 2015 cuando el último es de 2024).
    Con la frase ya escrita, la única acción posible es transcribirla.
    """
    car = car or {}
    total = _entero(car.get("total"))
    estados = car.get("por_estado") or {}
    detalle = ", ".join(f"{_num(n)} {_estado_plural(e, n)}" for e, n in estados.items())
    f = f"{_num(total)} {_plural(total, 'préstamo registrado', 'préstamos registrados')}"
    if detalle:
        f += f" ({detalle})"
    desde, hasta = car.get("periodo_desde_texto"), car.get("periodo_hasta_texto")
    if desde and hasta:
        f += f", otorgados entre el {desde} y el {hasta}"
    f += (f". Monto original acumulado: {car.get('monto_original_total') or 'no disponible'}. "
          f"Saldo vivo de los préstamos vigentes: {car.get('saldo_vivo_total') or '$0,00'}.")
    return f


def _bloque_cartera(car: dict) -> list[str]:
    """Encabezado con los agregados ya calculados (nunca contados a mano)."""
    return [f"**Cartera:** {cartera_frase(car)}"]


def _bloque_pagos(pagos: dict) -> list[str]:
    """Comportamiento de pago de las cuotas VENCIDAS en los últimos 24 meses.
    Es una señal distinta del atraso actual de los vigentes: se informan ambas."""
    total = pagos.get("total_cuotas")
    if total in (None, ""):
        return []
    total = _entero(total)
    if total == 0:
        return ["**Comportamiento de pago (últimos 24 meses):** no venció ninguna cuota "
                "en el período, por lo que no hay comportamiento de pago reciente que evaluar."]

    puntuales = _entero(pagos.get("cuotas_puntuales"))
    prom = pagos.get("promedio_dias_atraso")
    maximo = _entero(pagos.get("max_dias_atraso_periodo"))
    impagas = _entero(pagos.get("cuotas_impagas_vencidas"))
    tramos = [
        (_entero(pagos.get("atraso_leve_1_30d")), "con atraso leve (1 a 30 días)"),
        (_entero(pagos.get("atraso_moderado_31_90d")), "con atraso moderado (31 a 90 días)"),
        (_entero(pagos.get("atraso_grave_mas90d")), "con atraso grave (más de 90 días)"),
    ]
    l = [f"**Comportamiento de pago (últimos 24 meses):** {_num(puntuales)} de {_num(total)} "
         f"{_plural(total, 'cuota vencida fue puntual', 'cuotas vencidas fueron puntuales')}."]
    desglose = [f"{_num(n)} {txt}" for n, txt in tramos if n > 0]
    if desglose:
        l.append("- Cuotas con atraso: " + "; ".join(desglose) + ".")
    if prom not in (None, ""):
        l.append(f"- Atraso promedio: {_num(round(float(prom), 1))} días. "
                 f"Atraso máximo del período: {_num(maximo)} días"
                 + (" (**antecedente grave**)." if maximo > 90 else "."))
    if impagas > 0:
        l.append(f"- **{_num(impagas)} {_plural(impagas, 'cuota vencida sigue impaga', 'cuotas vencidas siguen impagas')}.**")
    if puntuales == total and maximo == 0:
        l.append("- Sin atrasos registrados en el período.")
    return l


def _bloque_gestiones(gestiones: list) -> list[str]:
    if not gestiones:
        return ["**Gestiones de mora:** no se registran gestiones de mora."]
    n = len(gestiones)
    l = [f"**Gestiones de mora:** se {_plural(n, 'registra', 'registran')} {_num(n)} "
         f"{_plural(n, 'gestión', 'gestiones')}."]
    for g in gestiones:
        if not isinstance(g, dict):
            continue
        # Las fechas vienen ya formateadas (DD/MM/AAAA) desde el pipeline; el
        # recorte del ISO es el respaldo si la plantilla se usa suelta.
        campos = [g.get("fecha_gestion_texto") or str(g.get("fecha_gestion") or "s/f")[:10]]
        if g.get("tipo_gestion"):
            campos.append(str(g["tipo_gestion"]))
        if g.get("resultado_contacto"):
            campos.append(str(g["resultado_contacto"]))
        if g.get("monto_comprometido") not in (None, ""):
            campos.append(f"monto adeudado {_monto(g.get('monto_comprometido'))}")
        compromiso = g.get("fecha_compromiso_pago_texto") or g.get("fecha_compromiso_pago")
        if compromiso:
            campos.append(f"compromiso de pago {str(compromiso)[:10]}")
        l.append("- " + " — ".join(campos) + ".")
    return l


def historial_markdown(ctx: dict) -> str:
    """Cuerpo (sin el título '## …') de la sección Historial Crediticio.

    `ctx` es el mismo slice que recibía el LLM: historial_prestamos,
    pagos_resumen, gestiones_mora y cartera_resumen.
    """
    ctx = ctx or {}
    prestamos = [p for p in (ctx.get("historial_prestamos") or []) if isinstance(p, dict)]
    car = ctx.get("cartera_resumen") or {}
    pagos = ctx.get("pagos_resumen") or {}
    gestiones = [g for g in (ctx.get("gestiones_mora") or []) if isinstance(g, dict)]

    if not prestamos and not _entero((pagos or {}).get("total_cuotas")):
        # Sin historial: se dice explícitamente. Nunca "no registra atrasos"
        # (la ausencia de datos no es buen comportamiento).
        return ("El asociado **no posee historial crediticio previo en la entidad**: no se "
                "registran préstamos ni cuotas vencidas, por lo que su comportamiento de pago "
                "no puede evaluarse con datos internos.\n\n"
                + "\n".join(_bloque_gestiones(gestiones)))

    lineas: list[str] = []
    if car:
        lineas += _bloque_cartera(car)
        lineas.append("")

    # Detalle por préstamo: los vigentes primero (son los que exponen riesgo hoy),
    # y dentro de cada grupo, del más reciente al más antiguo.
    vigentes = [p for p in prestamos if str(p.get("estado") or "").upper() == "VIGENTE"]
    resto = [p for p in prestamos if p not in vigentes]
    orden = lambda p: str(p.get("fecha_otorgamiento") or "")

    if vigentes:
        en_mora = [p for p in vigentes if _entero(p.get("dias_atraso_actual")) > 0]
        cab = f"**Préstamos vigentes ({_num(len(vigentes))}):**"
        if en_mora:
            peor = max(_entero(p.get("dias_atraso_actual")) for p in en_mora)
            # "el mayor atraso" a secas se leía como el máximo HISTÓRICO (y el juez
            # lo contrastaba contra otro número). Es el atraso ACTUAL: se dice así.
            cab += (f" **{_num(len(en_mora))} con mora vigente**; el mayor atraso ACTUAL "
                    f"es de {_num(peor)} días.")
        lineas.append(cab)
        lineas += [_linea_prestamo(p) for p in sorted(vigentes, key=orden, reverse=True)]
        lineas.append("")

    if resto:
        lineas.append(f"**Préstamos no vigentes ({_num(len(resto))}):**")
        lineas += [_linea_prestamo(p) for p in sorted(resto, key=orden, reverse=True)]
        lineas.append("")

    # Acá NO se publica un "atraso máximo histórico de la cartera" agregando los
    # `max_dias_atraso_historico` de cada préstamo. El motivo original era un BUG,
    # ya corregido (29-ago-2026): ese campo sólo miraba cuotas PAGADAS TARDE e
    # ignoraba las impagas vigentes, así que daba un número distinto del de la
    # feature ML homónima y el informe se contradecía solo. Hoy las dos
    # definiciones coinciden (ver el comentario en la query de
    # `client_repository.get_historial_prestamos` y `tests/test_definicion_atraso.py`),
    # así que el impedimento técnico ya no existe.
    #
    # Se mantiene fuera por una decisión de contenido, no de datos: `pagos_resumen`
    # ya informa un máximo, y publicar dos máximos con alcances distintos (toda la
    # historia vs. el período del resumen) invita a la confusión que se acaba de
    # eliminar. Si se decide agregarlo, hay que decir explícitamente el alcance de
    # cada uno.
    lineas += _bloque_pagos(pagos)
    lineas.append("")
    lineas += _bloque_gestiones(gestiones)
    return "\n".join(lineas).strip()