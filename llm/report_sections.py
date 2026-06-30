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

Nota: las secciones puramente factuales (Clasificación, Historial, Información)
hoy siguen pasando por el LLM con instrucción de "solo formatear". El próximo
paso natural es convertirlas en plantillas de código (cero alucinación), una vez
que tengamos los nombres exactos de los campos del repositorio.
"""
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def _j(data) -> str:
    """Serializa un slice de contexto a JSON legible (tolerante a Decimal/fechas)."""
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


# Reglas comunes a TODAS las secciones (mini system prompt compartido).
BASE = (
    "Sos un analista crediticio de una institución financiera. Redactás UNA "
    "sección de un informe en Markdown, en español rioplatense, con tono sobrio.\n"
    "Reglas generales (valen para toda sección):\n"
    "- No inventes datos: usá SOLO lo que está en el contexto.\n"
    "- Si un dato es null, escribí 'no disponible'. Pero un 0 o una lista vacía es "
    "un dato REAL (ej. '0 préstamos', 'no posee préstamos previos'), NUNCA "
    "'no disponible'.\n"
    "- No agregues calificativos sin respaldo en los datos (nada de 'estable', "
    "'sólido', 'excelente').\n"
    "- NO declares 'sin antecedentes negativos', 'ausencia de historial negativo', "
    "'no registra atrasos' ni equivalentes si en el contexto hay cuotas no "
    "puntuales, días de atraso (históricos o actuales) o gestiones de mora. Un "
    "score favorable NO implica un historial sin atrasos.\n"
    "- NO afirmes que el asociado 'no posee' o 'carece de' historial crediticio "
    "salvo que el contexto lo indique explícitamente (scoring.sin_historial = true "
    "o la lista de préstamos vacía). Si hay préstamos registrados, SÍ posee "
    "historial.\n"
    "- NO afirmes que un préstamo está 'al día', 'vigente' o 'activo' si NINGÚN "
    "préstamo está en estado VIGENTE (p. ej. todos figuran CANCELADO o "
    "IRRECUPERABLE). No inventes un préstamo en curso que no existe.\n"
    "- Los importes en pesos YA VIENEN formateados en el contexto como texto "
    "(ej. \"$1.234.567,89\"): copialos TAL CUAL, sin reformatearlos, redondearlos "
    "ni recalcularlos. Para cualquier otro número que debas mostrar, usá el formato "
    "argentino: punto como separador de miles y coma para los decimales; NUNCA el "
    "formato anglosajón ($1,234,567.89).\n"
    "- NUNCA menciones en el texto nombres internos de campos, claves ni "
    "estructuras de datos (p. ej. scoring, drivers, políticas, documentos como si "
    "fueran campos). El lector es un directivo y no ve el JSON: redactá en prosa "
    "natural. Si una categoría no tiene datos, decilo en prosa (ej. 'no se "
    "registran políticas internas relevantes'), sin nombrar el campo de origen.\n"
    "- Devolvé SOLO el contenido de esta sección (sin su título, sin encabezados "
    "de otras secciones, sin bloques de código)."
)


# Secciones independientes (dependen solo de los datos del cliente).
# Cada una: id, título visible, claves del contexto que recibe, instrucción propia.
SECCIONES = [
    {
        "id": "clasificacion",
        "titulo": "Clasificación de Riesgo",
        "claves": ["scoring"],
        "instruccion": (
            "Presentá la clasificación de riesgo: score, nivel de riesgo, "
            "semáforo y probabilidad de incumplimiento. Después listá los factores "
            "que más influyeron en el score de este asociado SIEMPRE de mayor a "
            "menor peso (%), indicando para cada uno si aumenta o reduce el riesgo. "
            "El factor que más influyó es el de mayor peso; no reordenes ni eleves "
            "uno de menor peso. Si de los datos surge que el asociado no posee "
            "historial crediticio previo, aclará que estos factores derivan de "
            "campos sin historial real y no son interpretables; no destaques "
            "ninguno como determinante."
        ),
    },
    {
        "id": "historial",
        "titulo": "Historial Crediticio",
        "claves": ["historial_prestamos", "pagos_resumen", "gestiones_mora"],
        "instruccion": (
            "Resumí el historial de préstamos: cada préstamo en UNA línea con "
            "monto original, saldo actual, cuotas pendientes, días de atraso y "
            "estado. "
            "IMPORTANTE: si algún préstamo VIGENTE tiene 'dias_atraso_actual' "
            "mayor a 0, destacalo EXPLÍCITAMENTE como mora vigente (un atraso "
            "actual superior a 90 días es GRAVE), AUNQUE el resumen de pagos de los "
            "últimos 24 meses no muestre atrasos graves. No confundas el resumen de "
            "comportamiento de los últimos 24 meses (cuotas ya vencidas) con el "
            "atraso actual de los préstamos vigentes: son señales distintas e "
            "informás AMBAS; nunca afirmes 'sin atrasos' si hay atraso actual. "
            "Resumí también el comportamiento de pagos del asociado, indicando "
            "EXPLÍCITAMENTE las cuotas puntuales sobre el total y el atraso máximo "
            "histórico: si hubo cuotas no puntuales o atrasos máximos mayores a 0, "
            "son antecedentes negativos y no podés describir el historial como "
            "limpio. "
            "Si el contexto incluye gestiones de mora ('gestiones_mora'), "
            "informalas (fecha, tipo de gestión, resultado y monto comprometido); "
            "si la lista está vacía, indicá que no se registran gestiones de mora. "
            "No vuelques todos los campos ni el JSON crudo. Si no hay "
            "préstamos ni cuotas (0 o listas vacías), indicá EXPLÍCITAMENTE que el "
            "asociado no posee historial crediticio previo en la entidad y que su "
            "comportamiento de pago no puede evaluarse; no digas 'no registra "
            "atrasos' ni infieras buen comportamiento."
        ),
    },
    {
        "id": "bcra",
        "titulo": "Situación en el Sistema Financiero (BCRA)",
        "claves": ["bcra"],
        "omitir_si_vacio": True,   # si no se consultó el BCRA, no se incluye la sección
        "instruccion": (
            "Presentá la situación del asociado en el sistema financiero según la "
            "Central de Deudores del BCRA, usando SOLO el contexto provisto. Indicá: "
            "cantidad de entidades en las que registra deuda, deuda total, y la PEOR "
            "situación crediticia (escala BCRA del 1=normal al 5=irrecuperable) con su "
            "descripción; mencioná si hay deuda en proceso judicial, refinanciaciones o "
            "registros en revisión. Informá los cheques rechazados: cantidad total, "
            "cuántos siguen impagos y los montos. Si el contexto indica que no hay datos "
            "(tiene_datos en falso), escribí en prosa que no se obtuvieron registros de "
            "la Central de Deudores del BCRA al momento del informe, sin inferir nada. "
            "Una situación 1 o la ausencia de deuda es información POSITIVA, decilo como "
            "tal. No listes números de cheque individuales."
        ),
    },
    {
        "id": "financiera",
        "titulo": "Información Financiera",
        "claves": ["datos_cliente", "saldos_cuentas"],
        "instruccion": (
            "Presentá la identificación e información financiera del cliente: "
            "nombre/identificación, ingreso declarado, actividad económica, "
            "antigüedad, estado y saldos promedio. Solo hechos, sin adjetivar."
        ),
    },
    {
        "id": "cumplimiento",
        "titulo": "Cumplimiento de Políticas",
        "claves": ["scoring", "datos_cliente", "rag", "historial_prestamos"],
        "instruccion": (
            "Listá las políticas internas relevantes provistas y, para cada una, "
            "indicá su cumplimiento basándote SOLO en hechos documentados: el "
            "score del asociado, los documentos que aportó y el historial de "
            "préstamos. Si no se provee ninguna política, indicá en prosa que no se "
            "registran políticas internas relevantes para evaluar. "
            "Para políticas que dependan del tiempo desde el último crédito (ej. "
            "'más de 3 meses del último crédito'), usá la antigüedad del último "
            "préstamo —la fecha de otorgamiento del historial o "
            "'ultimo_prestamo_hace_meses' del scoring— y la vigencia de la "
            "documentación (comparando fechas) para determinar el cumplimiento "
            "cuando los datos lo permitan; NO seas ambiguo si el dato está "
            "disponible. Solo abstenete de evaluar si el asociado NO posee "
            "historial (sin préstamos previos) o si realmente falta el dato. "
            "Cuando una política exija un documento con fecha de vigencia (ej. "
            "constancia de inscripción ARCA), BUSCÁ esa fecha en el contenido de "
            "los documentos del contexto y compará su vigencia contra la fecha del "
            "informe; NO declares el dato 'no disponible' si el documento lo "
            "contiene."
        ),
    },
]


# Instrucciones de las dos secciones interpretativas (con dependencias).
INSTR_RECOMENDACION = (
    "Redactá la recomendación crediticia a partir del score (nivel, semáforo y si "
    "el asociado posee o no historial crediticio) y del resumen de cumplimiento de "
    "políticas ya redactado. La recomendación debe ser coherente con el nivel de "
    "riesgo y el semáforo. "
    "Si en el contexto hay préstamos vigentes con atraso actual "
    "('dias_atraso_actual' > 0) o gestiones de mora registradas, la recomendación "
    "DEBE ponderarlas de forma explícita y condicionar o alertar, AUNQUE el score "
    "sea favorable: un score de bajo riesgo NO anula una mora vigente. Nunca "
    "afirmes 'ausencia de atrasos' si existe atraso actual en un préstamo vigente. "
    "Si una política interna fija un umbral (ej. score mínimo) y el asociado NO lo "
    "cumple, la recomendación debe ser de NO aprobación (no meramente "
    "'condicionar la aprobación'), explicando el incumplimiento. "
    "Si el asociado no posee historial crediticio, "
    "condicioná SIEMPRE la aprobación a verificación manual, aclarando que el "
    "score refleja ausencia de señal negativa y NO comportamiento de pago "
    "demostrado. No cierres con disclaimers (eso va en otra sección)."
)

INSTR_RESUMEN = (
    "Redactá un Resumen Ejecutivo de UN párrafo a partir del score y de la "
    "recomendación ya redactada. Mencioná el perfil de riesgo, el score y la "
    "decisión recomendada. Si la recomendación señala mora vigente o atraso actual "
    "en algún préstamo, el resumen DEBE reflejarlo y no describir el perfil como "
    "favorable sin matizar esa señal. Si el asociado no posee historial crediticio "
    "previo, dejalo claro y no infieras buen comportamiento. No declares cuál "
    "factor 'influyó más' (eso ya está en Clasificación de Riesgo)."
)


def _cadena(llm, instruccion):
    """Arma la cadena LCEL para una sección: system (base + instrucción) | llm | str."""
    system = BASE + "\n\nInstrucción de ESTA sección:\n" + instruccion
    human  = "Contexto (JSON):\n{ctx}\n\nRedactá la sección pedida."
    return (
        ChatPromptTemplate.from_messages([("system", system), ("human", human)])
        | llm
        | StrOutputParser()
    )


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
    return "".join(partes).strip()


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
        texto = yield from _stream_seccion(llm, s["titulo"], s["instruccion"], sl)
        generadas[s["id"]] = (s["titulo"], texto)

    # Recomendación: depende del cumplimiento ya redactado y necesita ver las
    # señales operativas (atraso actual de préstamos vigentes y gestiones de mora)
    # para poder ponderarlas aunque el score sea favorable.
    ctx_reco = {
        "scoring":             contexto.get("scoring"),
        "tipo_decision":       contexto.get("tipo_decision"),
        "cumplimiento":        generadas.get("cumplimiento", ("", ""))[1],
        "historial_prestamos": contexto.get("historial_prestamos"),
        "gestiones_mora":      contexto.get("gestiones_mora"),
    }
    recomendacion = yield from _stream_seccion(
        llm, "Recomendación", INSTR_RECOMENDACION, ctx_reco
    )

    # Resumen Ejecutivo: último, depende de la recomendación.
    ctx_res = {
        "scoring":       contexto.get("scoring"),
        "tipo_decision": contexto.get("tipo_decision"),
        "recomendacion": recomendacion,
    }
    resumen = yield from _stream_seccion(
        llm, "Resumen Ejecutivo", INSTR_RESUMEN, ctx_res
    )

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
    yield {"tipo": "informe", "texto": "\n\n".join(partes)}


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