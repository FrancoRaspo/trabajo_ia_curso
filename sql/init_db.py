"""
Inicializa la base PostgreSQL para el generador de informes crediticios.

Crea / asegura SOLO lo que el código realmente usa:
  - extensión pgvector
  - tablas de langchain_postgres: langchain_pg_collection y langchain_pg_embedding
    (las crea PGVector; ahí vive todo el RAG: políticas, notas, documentos, etc.)
  - columna 'embedding' como vector(1024) + índice ANN para búsqueda semántica
  - índice GIN sobre 'cmetadata' para filtrar rápido por tipo / cliente_id
  - tabla 'informes_generados' (log de trazabilidad que escribe _guardar_log)

Uso:
    python init_db.py

Nota: la primera vez descarga el modelo de embeddings (bge-m3 ~2 GB o el que
tengas en EMBED_MODEL), así que puede tardar.
"""
import os

from sqlalchemy import create_engine, text

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from db.postgres_connection import get_postgres_connstring
from rag.pgvector_indexer import get_embeddings, PGVectorIndexer, EMBED_DIM


def main():
    conn   = get_postgres_connstring()
    engine = create_engine(conn)

    # 1) Extensión pgvector ---------------------------------------------------
    with engine.begin() as c:
        c.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    print("[init] extensión vector OK")

    # 2) Tablas de langchain_postgres ----------------------------------------
    #    Instanciar PGVector crea langchain_pg_collection + langchain_pg_embedding.
    from langchain_postgres.vectorstores import PGVector
    PGVector(
        embeddings=get_embeddings(),
        collection_name=PGVectorIndexer.COLLECTION_NAME,
        connection=conn,
        use_jsonb=True,
    )
    print("[init] tablas langchain_pg_collection / langchain_pg_embedding OK")

    # 3) Dimensión de la columna embedding + índices -------------------------
    #    langchain crea la columna sin dimensión fija; para poder indexarla
    #    (HNSW/IVFFlat) hay que fijarla. La fijamos a EMBED_DIM (1024).
    with engine.begin() as c:
        try:
            c.execute(text(
                f"ALTER TABLE langchain_pg_embedding "
                f"ALTER COLUMN embedding TYPE vector({EMBED_DIM});"
            ))
            print(f"[init] columna embedding fijada a vector({EMBED_DIM})")
        except Exception as e:
            # Falla si ya hay vectores de OTRA dimensión cargados.
            print(f"[init] AVISO: no se pudo fijar la dimensión a {EMBED_DIM}: {e}")
            print("       Si venís de otro modelo (ej. 768), corré primero "
                  "migracion_embeddings_1024.sql (hace TRUNCATE + ALTER).")

    with engine.begin() as c:
        # Índice ANN para búsqueda semántica (coseno, va con normalize_embeddings=True).
        # HNSW requiere pgvector >= 0.5. Si tu versión es vieja, usá IVFFlat (abajo).
        c.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_lpe_embedding_cos "
            "ON langchain_pg_embedding USING hnsw (embedding vector_cosine_ops);"
        ))
        # Alternativa IVFFlat (descomentar si no tenés HNSW):
        # c.execute(text(
        #     "CREATE INDEX IF NOT EXISTS idx_lpe_embedding_cos "
        #     "ON langchain_pg_embedding USING ivfflat (embedding vector_cosine_ops) "
        #     "WITH (lists = 100);"
        # ))

        # Índice para los filtros por metadata (tipo, cliente_id, vigente).
        c.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_lpe_cmetadata "
            "ON langchain_pg_embedding USING gin (cmetadata jsonb_path_ops);"
        ))
    print("[init] índices ANN + cmetadata OK")

    # 4) Tabla de trazabilidad (la usa _guardar_log) -------------------------
    with engine.begin() as c:
        c.execute(text("""
            CREATE TABLE IF NOT EXISTS informes_generados (
                id              SERIAL PRIMARY KEY,
                cliente_id      VARCHAR(50) NOT NULL,
                tipo_decision   TEXT        NOT NULL,
                score_calculado INTEGER,
                nivel_riesgo    VARCHAR(20),
                informe_texto   TEXT,
                validacion_ok   BOOLEAN,
                advertencias    JSONB,
                trazabilidad    JSONB,
                generado_en     TIMESTAMPTZ DEFAULT NOW(),
                revisado_por    VARCHAR(100),
                aprobado        BOOLEAN DEFAULT FALSE
            );
        """))
    print("[init] tabla informes_generados OK")

    # 5) (Opcional) borrar la tabla huérfana del esquema viejo ---------------
    #    Descomentar si querés limpiar 'doc_embeddings' (langchain no la usa).
    # with engine.begin() as c:
    #     c.execute(text("DROP TABLE IF EXISTS doc_embeddings CASCADE;"))
    # print("[init] doc_embeddings (huérfana) eliminada")

    print("\n✅ Base inicializada. Ya podés levantar la app.")


if __name__ == "__main__":
    main()