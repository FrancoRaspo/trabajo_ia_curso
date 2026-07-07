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
        "claves": ["historial_prestamos", "pagos_resumen", "gestiones_mora"],
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
        "claves": ["scoring", "datos_cliente", "rag", "historial_prestamos"],
    },
]


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
        texto = yield from _stream_seccion(
            llm, s["titulo"], load(f"seccion_{s['id']}"), sl)
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
        llm, "Recomendación", load("recomendacion"), ctx_reco
    )

    # Resumen Ejecutivo: último, depende de la recomendación.
    ctx_res = {
        "scoring":       contexto.get("scoring"),
        "tipo_decision": contexto.get("tipo_decision"),
        "recomendacion": recomendacion,
    }
    resumen = yield from _stream_seccion(
        llm, "Resumen Ejecutivo", load("resumen"), ctx_res
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