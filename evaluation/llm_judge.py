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

# Los prompts del juez viven en prompts/juez_sys.txt y prompts/juez_hum.txt
# (la rúbrica JSON esperada está incluida dentro de juez_sys.txt).
from prompts import load as load_prompt


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
    dict con la rúbrica completada (ver prompts/juez_sys.txt). Si el modelo no devolvió JSON
    válido, 'veredicto' == 'ERROR_PARSEO' y 'raw' trae la salida cruda.
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    if isinstance(datos_fuente, dict):
        datos_fuente = json.dumps(datos_fuente, ensure_ascii=False,
                                  indent=2, default=str)

    llm = llm or _construir_juez_llm()
    human = load_prompt("juez_hum").format(datos_fuente=datos_fuente, informe=informe)
    resp = llm.invoke([SystemMessage(content=load_prompt("juez_sys")),
                       HumanMessage(content=human)])
    salida = getattr(resp, "content", str(resp))
    return _parsear_json(salida)


def aprobado(veredicto: dict) -> bool:
    """Helper para tests: True si el juez aprobó sin alucinaciones."""
    return (veredicto.get("veredicto") == "APROBADO"
            and not veredicto.get("alucinaciones"))