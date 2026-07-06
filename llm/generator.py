from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os


def _ollama_local(modelo: str):
    """Construye el cliente Ollama para un modelo local (por nombre)."""
    # ChatOllama (langchain-ollama) reemplaza a la clase Ollama deprecada.
    from langchain_ollama import ChatOllama
    kwargs = dict(
        model=modelo,
        temperature=0.1,
        num_ctx=8192,
        num_predict=4096,     # techo de tokens de salida (informe completo)
        keep_alive="30m",     # no descargar el modelo entre pedidos
    )
    # `reasoning=False` apaga el "thinking" de qwen3 (más rápido). Modelos que
    # no soportan thinking (gemma, etc.) devuelven error si se les pasa el flag.
    if modelo.startswith(("qwen", "deepseek", "magistral")):
        kwargs["reasoning"] = False
    return ChatOllama(**kwargs)


def get_llm(provider: str = "anthropic"):
    """
    Prototipo/Curso:    API de Anthropic, OpenAI o Gemini.
    Producción real:    Modelo local (Ollama) para no enviar datos de
                        clientes a servicios externos.

    `provider` puede ser:
      - un proveedor conocido: "anthropic" / "gemini" / "openai" / "local"
        ("local" toma el nombre del modelo de la env LOCAL_LLM), o
      - directamente el NOMBRE de un modelo local de Ollama (ej. "gemma4:latest",
        "qwen3:14b"), en cuyo caso se usa Ollama con ese modelo.
    """
    provider = (provider or "").strip()
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
        # El nombre del modelo viene de la env LOCAL_LLM.
        return _ollama_local(os.environ["LOCAL_LLM"])
    elif provider:
        # No es un proveedor conocido -> se interpreta como el NOMBRE de un modelo
        # local de Ollama (ej. "gemma4:latest", "qwen3:14b").
        return _ollama_local(provider)
    else:
        raise ValueError(f"Provider desconocido: {provider!r}")