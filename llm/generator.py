from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os


def get_llm(provider: str = "anthropic"):
    """
    Prototipo/Curso:    API de Anthropic, OpenAI o Gemini.
    Producción real:    Modelo local (Ollama) para no enviar datos de
                        clientes a servicios externos.
    """
    if provider == "anthropic":
        return ChatAnthropic(
            model=os.environ["ANTHROPIC_LLM"],
            temperature=0.1,
            max_tokens=4096,
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=os.environ["GOOGLE_LLM"],
            temperature=0.1,
            max_output_tokens=4096,
            google_api_key=os.environ["GOOGLE_API_KEY"],
        )
    elif provider == "openai":
        model = os.environ["OPENAI_LLM"]
        # Los modelos de razonamiento (gpt-5*, o1/o3/o4*) NO aceptan max_tokens ni
        # temperature!=1, y su límite de salida (max_completion_tokens) incluye los
        # tokens de razonamiento + los visibles. Por eso hay que darle margen amplio
        # y omitir temperature; si no, el informe se trunca (ej. clientes con
        # historial largo) antes de llegar a la recomendación.
        es_razonador = model.startswith(("gpt-5", "o1", "o3", "o4"))
        if es_razonador:
            return ChatOpenAI(
                model=model,
                max_completion_tokens=8192,   # razonamiento + salida visible
                reasoning_effort="low",       # menos tokens "pensando" -> más para el informe
                api_key=os.environ["OPENAI_API_KEY"],
            )
        return ChatOpenAI(
            model=model,
            temperature=0.1,
            max_tokens=4096,
            api_key=os.environ["OPENAI_API_KEY"],
        )
    elif provider == "local":
        # Producción con datos reales — sin envío a servicios externos.
        # ChatOllama (langchain-ollama) reemplaza a la clase Ollama deprecada.
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.environ["LOCAL_LLM"],
            temperature=0.1,
            num_ctx=8192,
            num_predict=4096,     # techo de tokens de salida (informe completo)
            keep_alive="30m",     # no descargar el modelo entre pedidos
            reasoning=False,      # apaga el "thinking" de qwen3 (más rápido)
        )
    else:
        raise ValueError(f"Provider desconocido: {provider}")