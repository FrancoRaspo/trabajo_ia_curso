from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import create_engine, text
from rag.pgvector_indexer import get_embeddings, PGVectorIndexer, limpiar_texto
from db.postgres_connection import get_postgres_connstring

class PGVectorRetriever:
    """
    Recupera contexto cualitativo relevante desde pgvector.
    Filtra por cliente_id cuando corresponde (notas, informes previos, documentos).
    Busca sin filtro cuando corresponde (políticas, normativa).
    """

    def __init__(self):
        self.embeddings  = get_embeddings()
        self.conn_string = get_postgres_connstring()
        self.store = PGVector(
            embeddings=self.embeddings,
            collection_name=PGVectorIndexer.COLLECTION_NAME,
            connection=self.conn_string,
            use_jsonb=True,
        )
        # Engine para chequeos rápidos (ej. saber si hay documentos indexados).
        self.engine = create_engine(self.conn_string)

    def recuperar_notas_mora(self, cliente_id: str, k: int = 5) -> list[dict]:
        """Busca notas de gestión de mora del cliente específico."""
        docs = self.store.similarity_search(
            query=f"gestión mora cliente {cliente_id} compromiso pago",
            k=k,
            filter={"tipo": "nota_mora", "cliente_id": cliente_id},
        )
        return self._serializar_docs(docs)

    def recuperar_politicas(self, tipo_decision: str, k: int = 3) -> list[dict]:
        """Busca políticas internas relevantes según el tipo de decisión."""
        query = f"política {tipo_decision} crédito aprobación condiciones requisitos"
        docs  = self.store.similarity_search(
            query=query,
            k=k,
            filter={"tipo": "politica", "vigente": True},
        )
        return self._serializar_docs(docs)

    def recuperar_normativa(self, tipo_decision: str, k: int = 2) -> list[dict]:
        """Busca normativa regulatoria relevante."""
        query = f"normativa regulación {tipo_decision} entidades crediticias"
        docs  = self.store.similarity_search(
            query=query,
            k=k,
            filter={"tipo": "normativa"},
        )
        return self._serializar_docs(docs)

    def recuperar_informes_anteriores(self, cliente_id: str,
                                       k: int = 3) -> list[dict]:
        """Recupera fragmentos de informes anteriores del cliente."""
        docs = self.store.similarity_search(
            query=f"informe crediticio cliente {cliente_id} historial recomendación",
            k=k,
            filter={"tipo": "informe_anterior", "cliente_id": cliente_id},
        )
        return self._serializar_docs(docs)

    def recuperar_documentos_cliente(self, cliente_id: str,
                                     tipo_decision: str, k: int = 4) -> list[dict]:
        """Recupera fragmentos de documentos subidos para ese cliente."""
        docs = self.store.similarity_search(
            query=f"documento del cliente {cliente_id} {tipo_decision}",
            k=k,
            filter={"tipo": "documento_cliente", "cliente_id": cliente_id},
        )
        return self._serializar_docs(docs)

    def indexar_documento_cliente(self, cliente_id: str, texto: str,
                                  nombre: str = "") -> int:
        """Indexa (fragmentado) un documento subido para un cliente.
        Devuelve la cantidad de fragmentos indexados."""
        texto = limpiar_texto(texto)        # quita NUL/control chars (PDFs)
        if not texto.strip():
            return 0
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.create_documents(
            [texto],
            metadatas=[{
                "tipo":       "documento_cliente",
                "cliente_id": cliente_id,
                "archivo":    nombre,
            }],
        )
        if chunks:
            self.store.add_documents(chunks)
        return len(chunks)

    def _hay_documentos(self) -> bool:
        """True si hay al menos un documento indexado (evita 4 encodings en vano)."""
        try:
            with self.engine.connect() as conn:
                row = conn.execute(
                    text("SELECT 1 FROM langchain_pg_embedding LIMIT 1")
                ).fetchone()
                return row is not None
        except Exception:
            return False

    def recuperar_contexto_completo(self, cliente_id: str,
                                     tipo_decision: str) -> dict:
        """Punto de entrada: recupera todo el contexto cualitativo de una vez."""
        vacio = {
            "notas_mora": [], "politicas": [], "normativa": [],
            "informes_previos": [], "documentos_cliente": [],
        }
        # Fast-path: si no hay nada indexado, no gastamos encodings.
        if not self._hay_documentos():
            return vacio

        return {
            "notas_mora":         self.recuperar_notas_mora(cliente_id),
            "politicas":          self.recuperar_politicas(tipo_decision),
            "normativa":          self.recuperar_normativa(tipo_decision),
            "informes_previos":   self.recuperar_informes_anteriores(cliente_id),
            "documentos_cliente": self.recuperar_documentos_cliente(cliente_id, tipo_decision),
        }

    def _serializar_docs(self, docs) -> list[dict]:
        return [
            {
                "texto":    doc.page_content,
                "metadata": doc.metadata,
                "fuente":   doc.metadata.get("tipo", "desconocido"),
            }
            for doc in docs
        ]