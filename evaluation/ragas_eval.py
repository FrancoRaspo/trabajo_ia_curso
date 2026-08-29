"""
Métricas RAGas sobre un informe generado.

RAGas evalúa la calidad del pipeline RAG sin ground truth escrito a mano:
  - faithfulness      : ¿cada afirmación del informe se sostiene en el contexto
                        recuperado? (mide alucinación respecto del RAG)
  - answer_relevancy  : ¿la respuesta responde al motivo de la consulta?

El LLM y los embeddings evaluadores usan OpenAI (el juez del curso). Si `ragas`
no está instalado o algo falla, devuelve {} y el runner marca las métricas RAGas
como no disponibles en lugar de romper toda la evaluación.

    pip install ragas datasets
"""
from __future__ import annotations

import os
import sys
import types


def _shim_vertexai() -> None:
    """RAGas importa `langchain_community.chat_models.vertexai.ChatVertexAI` en
    el top-level de ragas.llms.base, pero ese módulo fue REMOVIDO en
    langchain-community 0.4.x (el proyecto usa 0.4.x). Sin él, `import ragas`
    revienta con ModuleNotFoundError aunque nunca usemos Vertex.

    Inyectamos un stub del módulo: ragas sólo usa ChatVertexAI en checks de
    `isinstance` para decidir si el backend soporta multiple completions; como
    nuestro LLM evaluador es OpenAI, ese isinstance da False y el placeholder
    alcanza. Es idempotente y no toca el entorno del usuario ni la app.
    """
    mod_name = "langchain_community.chat_models.vertexai"
    if mod_name in sys.modules:
        return
    try:
        __import__(mod_name)
        return  # si algún día vuelve a existir, no shimeamos
    except Exception:
        pass
    stub = types.ModuleType(mod_name)

    class ChatVertexAI:  # placeholder; nunca se instancia
        pass

    stub.ChatVertexAI = ChatVertexAI
    sys.modules[mod_name] = stub


def esta_disponible() -> bool:
    _shim_vertexai()
    try:
        import ragas  # noqa: F401
        return True
    except Exception:
        return False


def aplanar_contextos(rag_context) -> list[str]:
    """Convierte el dict de buckets de recuperar_contexto_completo() en una
    lista plana de textos, que es lo que RAGas espera como retrieved_contexts."""
    textos: list[str] = []
    if isinstance(rag_context, dict):
        # Defensa: si nos pasan el contexto completo en vez del sub-dict "rag",
        # bajamos a la clave "rag".
        if isinstance(rag_context.get("rag"), dict):
            rag_context = rag_context["rag"]
        for bucket in rag_context.values():
            if isinstance(bucket, list):
                for item in bucket:
                    if isinstance(item, dict) and item.get("texto"):
                        textos.append(str(item["texto"]))
                    elif isinstance(item, str) and item.strip():
                        textos.append(item)
    return textos


def _construir_evaluadores():
    """LLM + embeddings evaluadores (OpenAI), envueltos para RAGas.

    OJO: RAGas FIJA la temperature al invocar el modelo (ej. 0.3). Los modelos
    de razonamiento (gpt-5*/o1/o3/o4*) rechazan cualquier temperature != 1, así
    que NO se puede reusar get_llm('openai') si OPENAI_LLM es un razonador (da
    BadRequestError y las métricas salen nan). Usamos un modelo NO-razonador
    dedicado (gpt-4o-mini por defecto): alcanza de sobra para las métricas y es
    barato. Override con RAGAS_LLM_MODEL."""
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    chat_model = os.environ.get("RAGAS_LLM_MODEL", "gpt-4o-mini")
    llm = LangchainLLMWrapper(
        ChatOpenAI(model=chat_model, api_key=os.environ["OPENAI_API_KEY"]))

    emb_model = os.environ.get("RAGAS_EMBED_MODEL", "text-embedding-3-small")
    emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=emb_model,
                         api_key=os.environ["OPENAI_API_KEY"]))
    return llm, emb


def _normalizar_scores(fila: dict) -> dict:
    """Mapea las columnas del resultado RAGas a nombres estables, tolerando
    los renombres entre versiones (answer_relevancy / response_relevancy).

    RAGas devuelve NaN cuando una métrica no se pudo calcular (típicamente un
    TimeoutError en informes largos). NaN NO es un 0 de calidad: lo tratamos
    como 'no disponible' (se omite) para no penalizar injustamente."""
    import math
    out: dict[str, float] = {}
    for col, val in fila.items():
        if not isinstance(val, (int, float)) or math.isnan(val):
            continue
        low = str(col).lower()
        if "faith" in low:
            out["faithfulness"] = float(val)
        elif "relevan" in low:
            out["answer_relevancy"] = float(val)
    return out


def evaluar_ragas(motivo: str, informe: str, rag_context) -> dict:
    """
    Devuelve {"faithfulness": float, "answer_relevancy": float} o {} si RAGas
    no está disponible / no hay contexto recuperado / falla la evaluación.
    """
    if not esta_disponible():
        return {}

    contextos = aplanar_contextos(rag_context)
    if not contextos:
        print("[ragas] sin contexto RAG recuperado; se omiten las métricas.",
              flush=True)
        return {}

    try:
        from ragas import evaluate
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        from ragas.metrics import Faithfulness, ResponseRelevancy
        from ragas.run_config import RunConfig

        llm, emb = _construir_evaluadores()
        muestra = SingleTurnSample(
            user_input=motivo or "",
            response=informe or "",
            retrieved_contexts=contextos,
        )
        ds = EvaluationDataset(samples=[muestra])
        # Timeout amplio: faithfulness descompone el informe en afirmaciones y
        # verifica cada una con el LLM; en informes largos (ej. un modelo fuerte)
        # el default (180s) se queda corto y devuelve NaN. RAGAS_TIMEOUT lo ajusta.
        timeout = int(os.environ.get("RAGAS_TIMEOUT", "600"))
        resultado = evaluate(
            dataset=ds,
            metrics=[Faithfulness(), ResponseRelevancy()],
            llm=llm,
            embeddings=emb,
            show_progress=False,
            run_config=RunConfig(timeout=timeout),
        )
        fila = resultado.to_pandas().iloc[0].to_dict()
        return _normalizar_scores(fila)
    except Exception as e:  # noqa: BLE001 — eval auxiliar: degradar, no romper
        print(f"[ragas] falló la evaluación: {type(e).__name__}: {e}", flush=True)
        return {}
