"""
Supervisor / juez LLM.

Audita un informe crediticio generado, comparándolo CONTRA los datos fuente
(la única verdad). Detecta alucinaciones, números cambiados, secciones
faltantes y violaciones de reglas.

Pensado como EVAL PERIÓDICA y con un modelo MÁS fuerte que
el generador.
Se asigna el modelo en la variable JUEZ_LLM_PROVIDER (default 'anthropic').
"""
import json
import os
import re

RUBRICA = """{
  "fidelidad":                {"puntaje": 0a5, "comentario": "..."},
  "exactitud_numerica":       {"puntaje": 0a5, "comentario": "..."},
  "estructura":               {"puntaje": 0a5, "comentario": "..."},
  "uso_drivers_cliente":      {"puntaje": 0a5, "comentario": "..."},
  "coherencia_recomendacion": {"puntaje": 0a5, "comentario": "..."},
  "cumplimiento_reglas":      {"aprobado": true, "comentario": "..."},
  "alucinaciones":            ["afirmacion no respaldada por los datos", "..."],
  "veredicto":                "APROBADO",
  "resumen":                  "2-3 frases con la conclusión"
}"""

SYS_JUEZ = (
    "Sos un AUDITOR SENIOR de informes crediticios de una entidad financiera.\n"
    "Evaluás un informe generado por otro sistema, comparándolo CONTRA los datos "
    "fuente, que son la ÚNICA verdad. Sos estricto y objetivo.\n\n"
    "Tu trabajo es encontrar:\n"
    "- Alucinaciones: afirmaciones o números que NO se desprenden de los datos.\n"
    "- Números cambiados, redondeados o estimados respecto de la fuente.\n"
    "- Secciones faltantes (Resumen Ejecutivo, Clasificación de Riesgo, "
    "Historial Crediticio, Recomendación).\n"
    "- Violaciones de reglas: mención de edad/género/etnia u otras "
    "características protegidas; mezclar hecho con inferencia sin marcarla; "
    "no cerrar con que la decisión final corresponde al directorio.\n"
    "- Si los 'drivers' del riesgo son los de ESTE cliente o genéricos/inventados.\n"
    "- Si la recomendación se condice con el score y el nivel de riesgo.\n\n"
    "Devolvé SOLAMENTE un JSON válido con esta forma exacta (sin texto fuera "
    "del JSON, sin ```):\n" + RUBRICA + "\n\n"
    "Escala de puntaje: 5 = impecable, 3 = aceptable con observaciones, "
    "1 = grave.\n"
    "veredicto: 'RECHAZADO' si hay alucinaciones o números inventados/cambiados; "
    "'REVISAR' si hay dudas menores; 'APROBADO' si está limpio.\n"
    "'alucinaciones' es la lista de afirmaciones puntuales sin respaldo (vacía si "
    "no hay)."
)

HUM_JUEZ = (
    "## DATOS FUENTE (única verdad)\n{datos_fuente}\n\n"
    "## INFORME A EVALUAR\n{informe}\n\n"
    "Auditá el informe contra los datos fuente y devolvé el JSON."
)


def _parsear_json(txt: str) -> dict:
    """Parser tolerante: quita fences y toma el primer objeto {...}."""
    if not txt:
        return {"veredicto": "ERROR_PARSEO", "raw": ""}
    txt = re.sub(r"```(?:json)?", "", txt).strip()
    ini, fin = txt.find("{"), txt.rfind("}")
    if ini == -1 or fin == -1:
        return {"veredicto": "ERROR_PARSEO", "raw": txt}
    try:
        return json.loads(txt[ini:fin + 1])
    except Exception:
        return {"veredicto": "ERROR_PARSEO", "raw": txt}


def _construir_juez_llm():
    """Arma el LLM del juez vía get_llm. Debería ser MÁS fuerte que el generador."""
    from llm.generator import get_llm
    provider = os.environ.get("JUEZ_LLM_PROVIDER", "anthropic")
    return get_llm(provider)


def evaluar_informe(datos_fuente, informe: str, llm=None) -> dict:
    """
    Parámetros
    ----------
    datos_fuente : dict o str con los hechos que el informe debía respetar
        (ej. {'datos_cliente':..., 'scoring':..., 'rag':...}). Es lo que el juez
        usa como verdad para detectar alucinaciones.
    informe : str — el informe markdown generado.
    llm : modelo de chat LangChain. Si es None, se arma con JUEZ_LLM_PROVIDER.

    Devuelve
    --------
    dict con la rúbrica completada (ver RUBRICA). Si el modelo no devolvió JSON
    válido, 'veredicto' == 'ERROR_PARSEO' y 'raw' trae la salida cruda.
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    if isinstance(datos_fuente, dict):
        datos_fuente = json.dumps(datos_fuente, ensure_ascii=False,
                                  indent=2, default=str)

    llm = llm or _construir_juez_llm()
    human = HUM_JUEZ.format(datos_fuente=datos_fuente, informe=informe)
    resp = llm.invoke([SystemMessage(content=SYS_JUEZ),
                       HumanMessage(content=human)])
    salida = getattr(resp, "content", str(resp))
    return _parsear_json(salida)


def aprobado(veredicto: dict) -> bool:
    """Helper para tests: True si el juez aprobó sin alucinaciones."""
    return (veredicto.get("veredicto") == "APROBADO"
            and not veredicto.get("alucinaciones"))