# Documentación técnica

Documentación del **Generador de Informes Crediticios con IA**. Los diagramas
están escritos en [Mermaid](https://mermaid.js.org/) embebido en Markdown: se
renderizan solos en GitHub y se versionan junto al código (se actualizan en el
mismo commit que el cambio que documentan).

| Documento | Contenido |
|---|---|
| [arquitectura.md](arquitectura.md) | Visión general del sistema y flujo de datos de punta a punta (consulta → informe). |
| [flujo-generacion.md](flujo-generacion.md) | Secuencia detallada de la generación en streaming (SSE): navegador ↔ FastAPI ↔ pipeline ↔ LLM. |
| [modo-admin.md](modo-admin.md) | Modo administrador: editar prompts, regenerar una sección y ver los datos fuente. |
| [modelo-datos.md](modelo-datos.md) | Fuentes de datos: SQL Server, RAG en pgvector y el log de informes en Postgres. |
| [scoring.md](scoring.md) | Modelo de scoring crediticio (XGBoost + SHAP), split temporal y control de fuga de datos. |
| [evaluacion.md](evaluacion.md) | Evaluación automatizada: golden set con pass/fail, juez LLM y RAGas, con métricas agregadas. |
| [estado-y-produccion.md](estado-y-produccion.md) | Diagnóstico del estado actual, comparación con el mercado (BCRA/Veraz/Nosis) y qué falta para producción. |

## Mapa rápido de módulos

| Módulo | Responsabilidad |
|---|---|
| `app.py` | App web FastAPI: formulario, streaming SSE del informe, endpoints del modo admin. |
| `pipeline.py` | Orquestador: SQL Server → ML → RAG/BCRA → LLM → validación → log. Sesiones en memoria para regenerar secciones. |
| `llm/report_sections.py` | Generación del informe **por secciones** (orquestación + regen de una sección). |
| `llm/generator.py` | Selección del proveedor de LLM (local Ollama / Anthropic / OpenAI / Gemini). |
| `prompts/` | Textos de todos los prompts en `.txt` (editables sin tocar código) + catálogo y validación. |
| `ml/scoring_model.py`, `ml/shap_explainer.py` | Score XGBoost y explicabilidad por cliente. |
| `rag/pgvector_indexer.py`, `rag/pgvector_retriever.py` | Indexación y recuperación de contexto cualitativo (embeddings). |
| `external/` | Integración con el BCRA (Central de Deudores) con caché. |
| `db/` | Conexiones y `ClientRepository` (consultas a SQL Server). |
| `evaluation/llm_judge.py` | Juez LLM que audita el informe contra los datos fuente. |
| `evaluation/run_golden.py` | Evaluación automatizada sobre el golden set: deterministas + juez + RAGas, con métricas agregadas y exit code para CI. |
