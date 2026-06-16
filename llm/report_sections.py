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
        "claves": ["historial_prestamos", "pagos_resumen"],
        "instruccion": (
            "Resumí el historial de préstamos: cada préstamo en UNA línea con "
            "monto original, saldo actual, cuotas pendientes, días de atraso y "
            "estado. No vuelques todos los campos ni el JSON crudo. Resumí también "
            "el comportamiento de pagos del asociado. Si no hay "
            "préstamos ni cuotas (0 o listas vacías), indicá EXPLÍCITAMENTE que el "
            "asociado no posee historial crediticio previo en la entidad y que su "
            "comportamiento de pago no puede evaluarse; no digas 'no registra "
            "atrasos' ni infieras buen comportamiento."
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
        "claves": ["scoring", "datos_cliente", "rag"],
        "instruccion": (
            "Listá las políticas internas relevantes provistas y, para cada una, "
            "indicá su cumplimiento basándote SOLO en hechos documentados: el "
            "score del asociado y los documentos que aportó. Si no se provee "
            "ninguna política, indicá en prosa que no se registran políticas "
            "internas relevantes para evaluar. NO infieras si una política que "
            "presupone créditos previos (ej. 'más de 3 meses del último crédito') "
            "'se aplica' o 'no se aplica' a un cliente sin historial; limitate a "
            "constatar la documentación efectivamente presentada."
        ),
    },
]


# Instrucciones de las dos secciones interpretativas (con dependencias).
INSTR_RECOMENDACION = (
    "Redactá la recomendación crediticia a partir del score (nivel, semáforo y si "
    "el asociado posee o no historial crediticio) y del resumen de cumplimiento de "
    "políticas ya redactado. La recomendación debe ser coherente con el nivel de "
    "riesgo y el semáforo. Si el asociado no posee historial crediticio, "
    "condicioná SIEMPRE la aprobación a verificación manual, aclarando que el "
    "score refleja ausencia de señal negativa y NO comportamiento de pago "
    "demostrado. No cierres con disclaimers (eso va en otra sección)."
)

INSTR_RESUMEN = (
    "Redactá un Resumen Ejecutivo de UN párrafo a partir del score y de la "
    "recomendación ya redactada. Mencioná el perfil de riesgo, el score y la "
    "decisión recomendada. Si el asociado no posee historial crediticio previo, "
    "dejalo claro y no infieras buen comportamiento. No declares cuál factor "
    "'influyó más' (eso ya está en Clasificación de Riesgo)."
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
        texto = yield from _stream_seccion(
            llm, s["titulo"], s["instruccion"], _slice(contexto, s["claves"])
        )
        generadas[s["id"]] = (s["titulo"], texto)

    # Recomendación: depende del cumplimiento ya redactado.
    ctx_reco = {
        "scoring":       contexto.get("scoring"),
        "tipo_decision": contexto.get("tipo_decision"),
        "cumplimiento":  generadas.get("cumplimiento", ("", ""))[1],
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