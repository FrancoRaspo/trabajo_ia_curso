# llm/report_sections.py
"""
Generación del informe POR SECCIONES.

En vez de un único prompt monolítico, cada sección tiene su prompt especializado
y recibe SOLO el contexto que necesita. El orquestador las corre en un orden con
dependencias (las factuales primero, la Recomendación después, el Resumen último)
y arma el documento final.

Ventajas frente al prompt único:
  - Reglas enfocadas por sección (no se pisan entre sí).
  - Cada llamada es corta -> no se trunca con cli
  entes de historial largo.
  - Cada sección recibe contexto acotado -> menos confusión / alucinación.

Secciones por PLANTILLA (sin LLM): una sección que sólo transcribe datos ya
calculados no necesita modelo. Historial Crediticio —la más densa en números y
la que el juez marcaba por conteos y períodos fabricados— se arma con código
(llm/plantillas.py). Se declara con la clave "plantilla" en SECCIONES y no tiene
prompt: no hay nada que editar ni que alucinar.
"""
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .plantillas import historial_markdown

# Los TEXTOS de los prompts viven en archivos .txt (carpeta prompts/), para que
# un experto de dominio pueda leerlos/editarlos sin tocar este código. Acá solo
# queda la ORQUESTACIÓN (qué sección usa qué prompt y en qué orden).
from prompts import load


def _j(data) -> str:
    """Serializa un slice de contexto a JSON legible (tolerante a Decimal/fechas)."""
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


# Reglas comunes a TODAS las secciones -> prompts/secciones_base.txt
# Instrucción de cada sección                -> prompts/seccion_<id>.txt
# Recomendación / Resumen Ejecutivo          -> prompts/recomendacion.txt / resumen.txt


# Secciones independientes (dependen solo de los datos del cliente).
# Cada una: id (= nombre del prompt: prompts/seccion_<id>.txt), título visible y
# las claves del contexto que recibe. El texto de la instrucción está en el .txt.
SECCIONES = [
    {
        "id": "clasificacion",
        "titulo": "Clasificación de Riesgo",
        "claves": ["scoring"],
    },
    {
        "id": "historial",
        "titulo": "Historial Crediticio",
        "claves": ["historial_prestamos", "pagos_resumen", "gestiones_mora",
                   "cartera_resumen"],
        # Sección determinista: la arma el código, no el LLM (cero alucinación).
        "plantilla": historial_markdown,
    },
    {
        "id": "bcra",
        "titulo": "Situación en el Sistema Financiero (BCRA)",
        "claves": ["bcra"],
        "omitir_si_vacio": True,   # si no se consultó el BCRA, no se incluye la sección
    },
    {
        "id": "financiera",
        "titulo": "Información Financiera",
        "claves": ["datos_cliente", "saldos_cuentas"],
    },
    {
        "id": "cumplimiento",
        "titulo": "Cumplimiento de Políticas",
        # cartera_resumen va también acá: sin él, la sección contaba los préstamos
        # a mano para chequear políticas de exposición (y los contaba mal).
        "claves": ["scoring", "datos_cliente", "rag", "historial_prestamos",
                   "cartera_resumen"],
    },
]

# Índice por id para lookups (regeneración de una sola sección).
SECCIONES_POR_ID = {s["id"]: s for s in SECCIONES}


def especificacion_seccion(section_id: str, contexto: dict, bodies: dict | None = None):
    """Devuelve (titulo, prompt_name, ctx_dict) de UNA sección, con el MISMO
    contexto que usa la generación completa. Centraliza el armado de contexto
    para poder regenerar una sección aislada (ver regenerar_seccion_stream).

    `bodies` es {section_id: texto} de secciones ya redactadas; lo necesitan las
    interpretativas: Recomendación depende de Cumplimiento y Resumen de Recomendación.
    """
    bodies = bodies or {}
    if section_id in SECCIONES_POR_ID:
        s = SECCIONES_POR_ID[section_id]
        # Las secciones por plantilla no tienen prompt (prompt_name = None).
        nombre = None if s.get("plantilla") else f"seccion_{section_id}"
        return s["titulo"], nombre, _slice(contexto, s["claves"])
    if section_id == "recomendacion":
        return "Recomendación", "recomendacion", {
            "scoring":             contexto.get("scoring"),
            "tipo_decision":       contexto.get("tipo_decision"),
            "cumplimiento":        bodies.get("cumplimiento", ""),
            "historial_prestamos": contexto.get("historial_prestamos"),
            "gestiones_mora":      contexto.get("gestiones_mora"),
            # Agregados de cartera ya resueltos: la Recomendación mencionaba
            # cantidades y períodos contándolos a mano y erraba (5/9/10 préstamos,
            # "todo cancelado"). Con esto copia los números correctos.
            "cartera_resumen":     contexto.get("cartera_resumen"),
        }
    if section_id == "resumen":
        return "Resumen Ejecutivo", "resumen", {
            "scoring":       contexto.get("scoring"),
            "tipo_decision": contexto.get("tipo_decision"),
            "recomendacion": bodies.get("recomendacion", ""),
            "cartera_resumen": contexto.get("cartera_resumen"),
        }
    raise ValueError(f"Sección desconocida o no regenerable: {section_id}")


def regenerar_seccion_stream(section_id: str, contexto: dict, bodies: dict, llm):
    """Genera de nuevo SOLO una sección (streaming). Relee el prompt del .txt en
    fresco (así toma la edición del admin) y usa el contexto retenido de la sesión.
    Emite eventos {"tipo":"token","texto":...} con el CUERPO (sin el título ##) y
    devuelve (return) el cuerpo completo ya redactado.

    Si la sección es por plantilla no hay prompt ni modelo: se vuelve a armar con
    el código (el resultado es el mismo, salvo que hayan cambiado los datos)."""
    _titulo, prompt_name, ctx = especificacion_seccion(section_id, contexto, bodies)
    if prompt_name is None:
        texto = SECCIONES_POR_ID[section_id]["plantilla"](ctx)
        yield {"tipo": "token", "texto": texto}
        return texto
    partes = []
    for chunk in _cadena(llm, load(prompt_name)).stream({"ctx": _j(ctx)}):
        partes.append(chunk)
        yield {"tipo": "token", "texto": chunk}
    return _recortar_secciones_fantasma("".join(partes))


def _cadena(llm, instruccion):
    """Arma la cadena LCEL para una sección: system (base + instrucción) | llm | str.
    `instruccion` es el texto ya cargado del .txt de la sección."""
    system = load("secciones_base") + "\n\nInstrucción de ESTA sección:\n" + instruccion
    human  = load("seccion_human")
    return (
        ChatPromptTemplate.from_messages([("system", system), ("human", human)])
        | llm
        | StrOutputParser()
    )


def _recortar_secciones_fantasma(texto: str) -> str:
    """Corta el cuerpo de una sección en el primer encabezado Markdown que emita
    el modelo.

    El prompt le pide devolver SOLO el contenido de su sección, sin títulos, y
    aun así a veces abre una sección propia ('## Resumen de la Cartera
    Crediticia', '## INFORME INTEGRAL DEL PERFIL…') y ahí adentro vuelve a
    narrar la cartera — recontándola a mano y contradiciendo al resto del
    informe. Era el origen de las alucinaciones CRÍTICAS de conteo que
    sobrevivieron a todas las rondas de prompt.

    Lo que viene después de ese encabezado es, por definición, contenido fuera
    de la sección: se descarta. Como la sección ya dijo lo suyo antes del
    encabezado, no se pierde información de la sección.
    """
    lineas = texto.splitlines()
    for i, linea in enumerate(lineas):
        if linea.lstrip().startswith("#"):
            return "\n".join(lineas[:i]).strip()
    return texto.strip()


def _slice(contexto: dict, claves: list) -> dict:
    """Devuelve solo las claves pedidas del contexto (contexto acotado por sección)."""
    return {k: contexto.get(k) for k in claves}


def _slice_vacio(sl: dict) -> bool:
    """True si todas las claves del slice están vacías (None/''/[]/{}).
    Se usa para omitir secciones opcionales (ej. BCRA si no se consultó)."""
    return all(v in (None, "", [], {}) for v in sl.values())


def _encabezado(contexto: dict) -> str:
    """Encabezado por plantilla (sin LLM), con datos top-level conocidos."""
    return (
        "# INFORME CREDITICIO — DIRECTORIO\n"
        f"**Solicitud:** {contexto.get('tipo_decision', '')}\n"
        f"**ID interno:** {contexto.get('cliente_id', '')}"
    )


def _stream_seccion(llm, titulo: str, instruccion: str, ctx_dict: dict):
    """Generador de UNA sección. Emite su título (## ...) y luego sus tokens en
    vivo. Devuelve (return) el cuerpo ya redactado, sin el título —para poder
    reensamblar el documento en orden de lectura al final.

    Los eventos que produce:
      {"tipo": "seccion", "titulo": <str>}   una vez, al empezar la sección
      {"tipo": "token",   "texto":  <str>}   repetido (incluye el encabezado ##)
    """
    yield {"tipo": "seccion", "titulo": titulo}
    yield {"tipo": "token", "texto": f"\n\n## {titulo}\n"}
    partes = []
    for chunk in _cadena(llm, instruccion).stream({"ctx": _j(ctx_dict)}):
        partes.append(chunk)
        yield {"tipo": "token", "texto": chunk}
    # El documento final se arma con el cuerpo YA saneado: si el modelo abrió una
    # sección propia, lo que escribió ahí no entra al informe (ver la función).
    return _recortar_secciones_fantasma("".join(partes))


def _stream_plantilla(titulo: str, plantilla, ctx_dict: dict):
    """Igual que _stream_seccion pero SIN LLM: el cuerpo lo arma el código.
    Emite los mismos eventos (la UI no distingue) y devuelve el cuerpo."""
    yield {"tipo": "seccion", "titulo": titulo}
    yield {"tipo": "token", "texto": f"\n\n## {titulo}\n"}
    texto = plantilla(ctx_dict).strip()
    # Se emite por líneas para que la vista tipo chat la pinte progresivamente,
    # como cualquier otra sección.
    for linea in texto.splitlines(keepends=True):
        yield {"tipo": "token", "texto": linea}
    return texto


def armar_informe_stream(contexto: dict, llm):
    """
    Versión STREAMING de armar_informe: genera el informe por secciones emitiendo
    eventos a medida que el LLM produce cada token, para la vista tipo chat.

    Orden de EJECUCIÓN/transmisión en vivo (respeta dependencias):
      1) secciones independientes (Clasificación, Historial, Información, Cumplimiento)
      2) Recomendación      (usa el cumplimiento ya redactado)
      3) Resumen Ejecutivo  (usa la recomendación ya redactada)
      4) Decisión Final     (texto fijo)

    Eventos emitidos:
      {"tipo": "seccion", "titulo": <str>}    al empezar cada sección
      {"tipo": "token",   "texto":  <str>}    tokens en vivo
      {"tipo": "informe", "texto":  <str>}    UNA vez al final: el documento
                                              completo en ORDEN DE LECTURA
                                              (Resumen arriba), para el render final.
    """
    encabezado = _encabezado(contexto)
    yield {"tipo": "token", "texto": encabezado}

    generadas = {}
    for s in SECCIONES:
        sl = _slice(contexto, s["claves"])
        if s.get("omitir_si_vacio") and _slice_vacio(sl):
            continue                      # sección opcional sin datos -> no se genera
        if s.get("plantilla"):            # sección determinista: la arma el código
            texto = yield from _stream_plantilla(s["titulo"], s["plantilla"], sl)
        else:
            texto = yield from _stream_seccion(
                llm, s["titulo"], load(f"seccion_{s['id']}"), sl)
        generadas[s["id"]] = (s["titulo"], texto)

    # Recomendación: depende del cumplimiento ya redactado y necesita ver las
    # señales operativas (atraso actual de préstamos vigentes y gestiones de mora)
    # para poder ponderarlas aunque el score sea favorable. El contexto se arma
    # con especificacion_seccion() —el MISMO que usa la regeneración aislada—.
    tit_r, prompt_r, ctx_reco = especificacion_seccion(
        "recomendacion", contexto,
        {"cumplimiento": generadas.get("cumplimiento", ("", ""))[1]})
    recomendacion = yield from _stream_seccion(llm, tit_r, load(prompt_r), ctx_reco)

    # Resumen Ejecutivo: último, depende de la recomendación.
    tit_s, prompt_s, ctx_res = especificacion_seccion(
        "resumen", contexto, {"recomendacion": recomendacion})
    resumen = yield from _stream_seccion(llm, tit_s, load(prompt_s), ctx_res)

    # Decisión Final (texto fijo) — se transmite también en vivo para cerrar.
    decision = "La decisión final corresponde al directorio."
    yield {"tipo": "seccion", "titulo": "Decisión Final"}
    yield {"tipo": "token", "texto": f"\n\n## Decisión Final\n{decision}"}

    # Ensamble final en ORDEN DE LECTURA (Resumen arriba), para el render final.
    partes = [encabezado, f"## Resumen Ejecutivo\n{resumen}"]
    for s in SECCIONES:
        if s["id"] not in generadas:      # se omitió (sección opcional sin datos)
            continue
        titulo, texto = generadas[s["id"]]
        partes.append(f"## {titulo}\n{texto}")
    partes.append(f"## Recomendación\n{recomendacion}")
    partes.append(f"## Decisión Final\n{decision}")

    # bodies: cuerpo (str) de cada sección por id -> permite regenerar una sola
    # sección después (incluidas las interpretativas, que dependen de otras).
    bodies = {sid: texto for sid, (_t, texto) in generadas.items()}
    bodies["recomendacion"] = recomendacion
    bodies["resumen"] = resumen
    yield {"tipo": "informe", "texto": "\n\n".join(partes), "bodies": bodies}


def armar_informe(contexto: dict, llm) -> str:
    """
    Orquesta la generación por secciones y devuelve el informe completo en Markdown.
    Reusa armar_informe_stream() y se queda con el documento final ya ensamblado
    (en orden de lectura: encabezado, Resumen, las 4 secciones, Recomendación, Decisión).
    """
    informe = ""
    for ev in armar_informe_stream(contexto, llm):
        if ev["tipo"] == "informe":
            informe = ev["texto"]
    return informe


# ===========================================================================
#  TRAZABILIDAD: resumen legible de los datos fuente por sección
# ===========================================================================
# Para cada sección FACTUAL, un puñado de hechos derivados del MISMO contexto que
# alimentó al LLM, presentados en lenguaje llano. Permite que un humano (no solo
# el admin) vea la relación entre lo que dice el informe y los datos, sin leer el
# JSON crudo. La clave del dict es el TÍTULO de la sección, para que el front lo
# cruce directamente con los encabezados '## ...' del markdown.

def _num_ar(v) -> str | None:
    """Número a formato argentino (coma decimal). Enteros sin decimales."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f == int(f):
        return f"{int(f):,}".replace(",", ".")
    return f"{f:,.1f}".replace(",", "@").replace(".", ",").replace("@", ".")


def _hecho(etiqueta: str, valor, alerta: bool = False) -> dict | None:
    """Un hecho {etiqueta, valor, alerta}. Devuelve None si el valor está vacío
    (así el llamador lo filtra y no se muestran filas sin dato)."""
    if valor is None or valor == "":
        return None
    return {"etiqueta": etiqueta, "valor": str(valor), "alerta": bool(alerta)}


def resumen_datos_secciones(contexto: dict) -> dict:
    """{titulo_seccion: [ {etiqueta, valor, alerta}, ... ]} con los datos fuente
    ya resumidos. Solo secciones factuales; las interpretativas (Resumen,
    Recomendación) sintetizan a las demás y no tienen datos propios."""
    contexto = contexto or {}
    out: dict[str, list] = {}

    def poner(titulo, hechos):
        hs = [h for h in hechos if h]
        if hs:
            out[titulo] = hs

    # --- Clasificación de Riesgo (scoring) ---
    sc = contexto.get("scoring") or {}
    if sc:
        pd = sc.get("prob_default")
        pd_txt = f"{round(float(pd) * 100, 1):.1f} %".replace(".", ",") if isinstance(pd, (int, float)) else None
        poner("Clasificación de Riesgo", [
            _hecho("Score", sc.get("score")),
            _hecho("Probabilidad de incumplimiento", pd_txt),
            _hecho("Nivel de riesgo", sc.get("nivel_riesgo"),
                   alerta=str(sc.get("nivel_riesgo") or "").upper() in ("ALTO", "MUY ALTO")),
            _hecho("Sin historial para scorear", "sí" if sc.get("sin_historial") else None,
                   alerta=True),
        ])

    # --- Historial Crediticio (cartera_resumen + pagos_resumen + gestiones) ---
    car = contexto.get("cartera_resumen") or {}
    pagos = contexto.get("pagos_resumen") or {}
    gest = contexto.get("gestiones_mora") or []
    if car or pagos or gest:
        estados = car.get("por_estado") or {}
        estados_txt = ", ".join(f"{n} {e.lower()}" for e, n in estados.items()) or None
        periodo = None
        if car.get("periodo_desde_texto") and car.get("periodo_hasta_texto"):
            periodo = f"{car['periodo_desde_texto']} – {car['periodo_hasta_texto']}"
        atraso_prom = pagos.get("promedio_dias_atraso")
        max_hist = pagos.get("max_dias_atraso_periodo")
        total_cuotas = pagos.get("total_cuotas")
        puntuales = pagos.get("cuotas_puntuales")
        cuotas_txt = None
        if total_cuotas not in (None, ""):
            cuotas_txt = f"{_num_ar(puntuales)} de {_num_ar(total_cuotas)} puntuales" \
                if total_cuotas else "no venció ninguna cuota en 24 meses"
        poner("Historial Crediticio", [
            _hecho("Préstamos registrados", car.get("total")),
            _hecho("Por estado", estados_txt),
            _hecho("Vigentes", f"{car.get('vigentes')} · saldo vivo {car.get('saldo_vivo_total')}"
                   if car.get("vigentes") else None,
                   alerta=bool(car.get("vigentes"))),
            _hecho("Monto original acumulado", car.get("monto_original_total")),
            _hecho("Período de otorgamiento", periodo),
            _hecho("Comportamiento de pago (24 m)", cuotas_txt),
            _hecho("Atraso promedio", f"{_num_ar(atraso_prom)} días" if atraso_prom not in (None, "") else None),
            _hecho("Atraso máximo del período", f"{_num_ar(max_hist)} días" if max_hist not in (None, "") else None,
                   alerta=bool(max_hist and float(max_hist) > 90)),
            _hecho("Gestiones de mora", len(gest) if gest else "0", alerta=bool(gest)),
        ])

    # --- Situación en el Sistema Financiero (BCRA) ---
    bcra = contexto.get("bcra") or {}
    if bcra and bcra.get("tiene_datos"):
        ps = bcra.get("peor_situacion")
        ps_txt = None
        if ps is not None:
            ps_txt = f"Situación {ps}"
            if bcra.get("peor_situacion_desc"):
                ps_txt += f" ({bcra['peor_situacion_desc']})"
        ents = bcra.get("entidades") or []
        poner("Situación en el Sistema Financiero (BCRA)", [
            _hecho("Peor situación", ps_txt,
                   alerta=bool(bcra.get("tiene_situacion_irregular"))),
            _hecho("Entidades informantes", bcra.get("cant_entidades") or (len(ents) or None)),
            _hecho("Deuda total en el sistema", bcra.get("deuda_total_sistema")),
            _hecho("Cheques impagos", bcra.get("monto_cheques_impagos"),
                   alerta=bool(bcra.get("monto_cheques_impagos"))),
        ])

    # --- Información Financiera (datos_cliente + saldos_cuentas) ---
    dc = contexto.get("datos_cliente") or {}
    saldos = contexto.get("saldos_cuentas") or []
    if dc or saldos:
        cuentas_txt = None
        if saldos:
            cuentas_txt = "; ".join(
                f"{s.get('tipo_cuenta', 'cuenta')}: prom. {s.get('saldo_promedio', 's/d')}"
                for s in saldos if isinstance(s, dict))
        poner("Información Financiera", [
            _hecho("Estado en la entidad", dc.get("estado_actual"),
                   alerta=str(dc.get("estado_actual") or "").strip().lower() not in ("alta", "activo", "")),
            _hecho("Antigüedad en la entidad", dc.get("antiguedad_texto")),
            _hecho("Ingreso neto declarado", dc.get("ingreso_neto_declarado")),
            _hecho("Actividad", dc.get("actividad_economica")),
            _hecho("Cuentas", cuentas_txt),
        ])

    # --- Cumplimiento de Políticas (rag + scoring) ---
    rag = contexto.get("rag") or {}
    if rag or sc:
        politicas = rag.get("politicas") if isinstance(rag, dict) else None
        n_pol = len(politicas) if isinstance(politicas, list) else None
        poner("Cumplimiento de Políticas", [
            _hecho("Políticas internas recuperadas", n_pol),
            _hecho("Nivel de riesgo del modelo", sc.get("nivel_riesgo")),
        ])

    return out