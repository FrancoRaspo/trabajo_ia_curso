"""
Prompts del sistema, externalizados en archivos .txt (uno por prompt).

La idea es que un experto de dominio (analista de crédito) pueda LEER y EDITAR
los prompts sin tocar código Python. Cada .txt de esta carpeta es el texto que se
le envía al modelo. Los `{marcadores}` entre llaves se rellenan en tiempo de
ejecución con datos del informe (ver prompts/README.md).

Uso desde el código:
    from prompts import load
    texto = load("recomendacion")            # lee prompts/recomendacion.txt

No se cachea a propósito: cada informe relee el .txt, así un cambio en el archivo
impacta en el próximo informe SIN reiniciar el proceso.

Además del `load()`, este módulo expone el CATÁLOGO de prompts (con título y
descripción legibles) y un `guardar()` con validación, que usa el "modo
administrador" de la app para editar los prompts desde la web sin poder romper
las plantillas (las llaves `{...}` se validan contra las variables permitidas).
"""
import re
from pathlib import Path

_DIR = Path(__file__).resolve().parent


def load(nombre: str) -> str:
    """Devuelve el contenido de prompts/<nombre>.txt (sin el salto de línea final
    que suelen agregar los editores). Lanza FileNotFoundError si no existe."""
    ruta = _DIR / f"{nombre}.txt"
    return ruta.read_text(encoding="utf-8").rstrip("\n")


# ===========================================================================
#  CATÁLOGO
# ===========================================================================
# Cada prompt declara:
#   titulo/descripcion -> textos legibles para la UI del editor
#   grupo              -> agrupación en la UI
#   modo               -> cómo lo consume el código:
#                         "template" (LangChain ChatPromptTemplate) o
#                         "format"   (str.format de Python) -> en ambos, las
#                           ÚNICAS llaves permitidas son `placeholders`;
#                         "literal"  -> texto crudo (p. ej. un ejemplo JSON):
#                           las llaves NO se interpretan, no se validan.
#   placeholders       -> variables `{...}` que el texto DEBE conservar (y las
#                         únicas permitidas) en modo template/format.
CATALOGO: dict[str, dict] = {
    # --- Secciones del informe (instrucción propia de cada una) ---
    "seccion_clasificacion": {
        "titulo": "Clasificación de Riesgo",
        "descripcion": "Instrucción de la sección de score, nivel de riesgo y factores.",
        "grupo": "Secciones del informe", "modo": "template", "placeholders": [],
    },
    # Historial Crediticio NO tiene prompt: se arma por plantilla determinista
    # (llm/plantillas.py). No pasa por el modelo, así que no hay nada que editar.
    "seccion_bcra": {
        "titulo": "Situación en el Sistema Financiero (BCRA)",
        "descripcion": "Instrucción de la sección de Central de Deudores y cheques.",
        "grupo": "Secciones del informe", "modo": "template", "placeholders": [],
    },
    "seccion_financiera": {
        "titulo": "Información Financiera",
        "descripcion": "Instrucción de identificación, ingresos y saldos del cliente.",
        "grupo": "Secciones del informe", "modo": "template", "placeholders": [],
    },
    "seccion_cumplimiento": {
        "titulo": "Cumplimiento de Políticas",
        "descripcion": "Instrucción del chequeo de políticas internas contra los hechos.",
        "grupo": "Secciones del informe", "modo": "template", "placeholders": [],
    },
    # --- Reglas y plantilla comunes a todas las secciones ---
    "secciones_base": {
        "titulo": "Reglas comunes a todas las secciones",
        "descripcion": "Mini system prompt compartido (anti-alucinación, formato AR). "
                       "Afecta a TODAS las secciones.",
        "grupo": "Reglas y plantilla comunes", "modo": "template", "placeholders": [],
    },
    "seccion_human": {
        "titulo": "Plantilla del mensaje (contexto)",
        "descripcion": "Mensaje 'human' con el que se pasa el contexto a cada sección. "
                       "Debe conservar {ctx}.",
        "grupo": "Reglas y plantilla comunes", "modo": "template", "placeholders": ["ctx"],
    },
    # --- Secciones interpretativas (con dependencias) ---
    "recomendacion": {
        "titulo": "Recomendación",
        "descripcion": "Recomendación crediticia a partir del score y el cumplimiento.",
        "grupo": "Secciones interpretativas", "modo": "template", "placeholders": [],
    },
    "resumen": {
        "titulo": "Resumen Ejecutivo",
        "descripcion": "Resumen de un párrafo a partir del score y la recomendación.",
        "grupo": "Secciones interpretativas", "modo": "template", "placeholders": [],
    },
    # --- Rama 'sin datos internos' (solicitante fuera de la base) ---
    "sin_datos_sys": {
        "titulo": "Sin datos — system",
        "descripcion": "System del informe cuando el cliente no está en la base interna. "
                       "Debe conservar {regla_bcra} y {secciones}.",
        "grupo": "Sin datos internos", "modo": "format",
        "placeholders": ["regla_bcra", "secciones"],
    },
    "sin_datos_hum": {
        "titulo": "Sin datos — human",
        "descripcion": "Mensaje 'human' del informe sin datos. Debe conservar "
                       "{razon_social}, {cuit}, {tipo_decision} y {bcra}.",
        "grupo": "Sin datos internos", "modo": "template",
        "placeholders": ["razon_social", "cuit", "tipo_decision", "bcra"],
    },
    "sin_datos_regla_bcra_con_datos": {
        "titulo": "Sin datos — regla BCRA (con datos)",
        "descripcion": "Fragmento que se inserta cuando el BCRA sí devolvió registros.",
        "grupo": "Sin datos internos", "modo": "template", "placeholders": [],
    },
    "sin_datos_regla_bcra_sin_datos": {
        "titulo": "Sin datos — regla BCRA (sin registros)",
        "descripcion": "Fragmento cuando se consultó el BCRA pero no hay registros.",
        "grupo": "Sin datos internos", "modo": "template", "placeholders": [],
    },
    "sin_datos_regla_bcra_no_consultado": {
        "titulo": "Sin datos — regla BCRA (no consultado)",
        "descripcion": "Fragmento cuando no se consultó el BCRA.",
        "grupo": "Sin datos internos", "modo": "template", "placeholders": [],
    },
    # --- Extractor de la consulta en lenguaje natural ---
    "extractor_sys": {
        "titulo": "Extractor — system",
        "descripcion": "Instruye al modelo a extraer CUIT/razón social/motivo de la consulta.",
        "grupo": "Extractor de la consulta", "modo": "template", "placeholders": [],
    },
    "extractor_hum": {
        "titulo": "Extractor — human",
        "descripcion": "Mensaje 'human' del extractor. Debe conservar {consulta}.",
        "grupo": "Extractor de la consulta", "modo": "template", "placeholders": ["consulta"],
    },
    # --- Juez LLM (evaluación de informes) ---
    "juez_sys": {
        "titulo": "Juez — system",
        "descripcion": "Rúbrica de evaluación del juez LLM (incluye un ejemplo JSON).",
        "grupo": "Juez (evaluación)", "modo": "literal", "placeholders": [],
    },
    "juez_hum": {
        "titulo": "Juez — human",
        "descripcion": "Mensaje 'human' del juez. Debe conservar {datos_fuente} y {informe}.",
        "grupo": "Juez (evaluación)", "modo": "format",
        "placeholders": ["datos_fuente", "informe"],
    },
}


# Prompts COMPARTIDOS por todas las secciones (reglas comunes + plantilla del
# mensaje). Editarlos afecta a TODO el informe, por eso el editor contextual de
# una sección NO los muestra: se editan desde el panel global "Prompts".
PROMPTS_COMPARTIDOS: list[str] = ["secciones_base", "seccion_human"]


# Prompts que afectan a cada sección del informe interactivo (acceso contextual).
# Toda sección usa las reglas comunes + su prompt propio + la plantilla del mensaje.
PROMPTS_POR_SECCION: dict[str, list[str]] = {
    "resumen":       ["secciones_base", "resumen", "seccion_human"],
    "clasificacion": ["secciones_base", "seccion_clasificacion", "seccion_human"],
    "bcra":          ["secciones_base", "seccion_bcra", "seccion_human"],
    "financiera":    ["secciones_base", "seccion_financiera", "seccion_human"],
    "cumplimiento":  ["secciones_base", "seccion_cumplimiento", "seccion_human"],
    "recomendacion": ["secciones_base", "recomendacion", "seccion_human"],
}


def catalogo() -> list[dict]:
    """Lista de prompts (para la UI del editor), en el orden del CATÁLOGO.
    No incluye el `modo` ni los detalles internos, solo lo visible."""
    return [
        {"name": name, "titulo": meta["titulo"],
         "descripcion": meta["descripcion"], "grupo": meta["grupo"]}
        for name, meta in CATALOGO.items()
    ]


_TOKEN = re.compile(r"\{([^{}]*)\}")


def _validar_llaves(name: str, texto: str, permitidos: list[str]) -> None:
    """Valida que el texto use SOLO las variables `{...}` permitidas y conserve
    las requeridas. Evita romper las plantillas (LangChain / str.format) al editar.
    Lanza ValueError con un mensaje claro para mostrar en la UI."""
    encontrados = _TOKEN.findall(texto)
    for t in encontrados:
        if t not in permitidos:
            perm = ", ".join("{%s}" % p for p in permitidos) or "ninguna"
            raise ValueError(
                f"El prompt no permite la variable «{{{t}}}». "
                f"Variables permitidas: {perm}."
            )
    # Llaves sueltas (sin cerrar / sin abrir) tras quitar los pares válidos.
    resto = _TOKEN.sub("", texto)
    if "{" in resto or "}" in resto:
        raise ValueError(
            "Hay una llave suelta «{» o «}». Las variables se escriben como "
            "{nombre}; no uses llaves para otra cosa."
        )
    faltan = [p for p in permitidos if p not in encontrados]
    if faltan:
        raise ValueError(
            f"El prompt debe conservar la variable «{{{faltan[0]}}}»: se "
            "reemplaza con datos en tiempo de ejecución y sin ella el informe falla."
        )


def guardar(name: str, contenido: str) -> None:
    """Escribe prompts/<name>.txt validando nombre y variables.

    - `name` debe estar en el CATÁLOGO (whitelist; sin path traversal posible).
    - En modo template/format valida las llaves `{...}` (ver _validar_llaves).
    - En modo literal (p. ej. juez_sys) no se validan las llaves.
    Lanza ValueError si algo no valida."""
    meta = CATALOGO.get(name)
    if meta is None:
        raise ValueError(f"Prompt desconocido: «{name}».")
    if contenido is None or not contenido.strip():
        raise ValueError("El prompt no puede quedar vacío.")
    if meta["modo"] != "literal":
        _validar_llaves(name, contenido, meta["placeholders"])
    # Normaliza a un único salto final (load() lo vuelve a quitar al leer).
    ruta = _DIR / f"{name}.txt"
    ruta.write_text(contenido.rstrip("\n") + "\n", encoding="utf-8")
