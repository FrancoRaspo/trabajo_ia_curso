from langchain_postgres.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from sqlalchemy import create_engine, text
from db.postgres_connection import get_postgres_connstring
import os
import re
import json
from datetime import date


def limpiar_texto(t: str) -> str:
    """Quita NUL (0x00) y otros caracteres de control que PostgreSQL no acepta
    en columnas de texto. Conserva tab, salto de línea y retorno de carro."""
    if not t:
        return ""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", t)

# Modelos de embeddings disponibles — AMBOS de 1024 dimensiones.
# Se elige con la env EMBED_MODEL: "bge-m3" (default) o "qwen3".
# También se acepta un nombre de modelo HF completo si querés otro.
MODELOS_EMBED = {
    "bge-m3": "BAAI/bge-m3",                 # híbrido denso+sparse, contexto largo
    "qwen3":  "Qwen/Qwen3-Embedding-0.6B",   # liviano, top multilingüe
}
EMBED_DIM = 1024   # la columna vector(...) en la tabla debe coincidir con esto

def _modelo_embed() -> str:
    clave = os.environ.get("EMBED_MODEL", "").strip() or "bge-m3"
    # clave corta conocida -> nombre HF; si no, se usa tal cual (modelo HF completo)
    return MODELOS_EMBED.get(clave.lower(), clave)

def _detectar_device() -> str:
    """Elige el dispositivo para los embeddings.
    En Apple Silicon (M-series) la GPU es 'mps' (Metal), no 'cuda'."""
    device = os.environ.get("EMBED_DEVICE")   # permite forzar: cpu / mps / cuda
    if device:
        return device
    try:
        import torch
        if torch.backends.mps.is_available():   # Apple Silicon (M1/M2/M3/M4)
            return "mps"
        if torch.cuda.is_available():           # GPU NVIDIA
            return "cuda"
    except Exception:
        pass
    return "cpu"

def get_embeddings():
    modelo = _modelo_embed()
    device = _detectar_device()
    print(f"[EMBEDDINGS] modelo = {modelo} | device = {device}", flush=True)
    return HuggingFaceEmbeddings(
        model_name=modelo,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

class PGVectorIndexer:
    """
    Indexa documentos no estructurados en PostgreSQL usando pgvector.
    Colección única 'credito_documentos' (ver COLLECTION_NAME), con el campo
    metadata 'tipo' como discriminador. Físicamente todo queda en la tabla
    langchain_pg_embedding que crea langchain_postgres.
    """

    COLLECTION_NAME = "credito_documentos"

    def __init__(self):
        self.embeddings   = get_embeddings()
        self.conn_string  = get_postgres_connstring()

        # Instancia del vector store de LangChain conectado a pgvector
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name=self.COLLECTION_NAME,
            connection=self.conn_string,
            use_jsonb=True,          # metadata en JSONB — más eficiente
        )

    # ------------------------------------------------------------------
    # Métodos de indexación por tipo de documento
    # ------------------------------------------------------------------

    def indexar_nota_mora(self, cliente_id: str, texto_nota: str,
                           fecha: str, oficial: str, tipo_gestion: str):
        """
        Indexa una nota de gestión de mora (texto libre).
        Complementa los datos estructurados de gestiones_mora en SQL Server.
        """
        doc = Document(
            page_content=texto_nota,
            metadata={
                "tipo":         "nota_mora",
                "cliente_id":   cliente_id,
                "fecha":        fecha,
                "oficial":      oficial,
                "tipo_gestion": tipo_gestion,
            }
        )
        self.vector_store.add_documents([doc])

    def indexar_politica(self, titulo: str, contenido: str,
                          categoria: str, version: str, vigente: bool = True):
        """
        Indexa un fragmento de política interna de la entidad.
        Ej: política de refinanciación, criterios de aprobación de montos, etc.
        """
        doc = Document(
            page_content=f"POLÍTICA: {titulo}\n\n{contenido}",
            metadata={
                "tipo":      "politica",
                "titulo":    titulo,
                "categoria": categoria,
                "version":   version,
                "vigente":   vigente,
            }
        )
        self.vector_store.add_documents([doc])

    def indexar_normativa(self, titulo: str, contenido: str,
                           organismo: str, nro_norma: str):
        """
        Indexa un fragmento de normativa regulatoria.
        Ej: comunicaciones BCRA, resoluciones del INAES, circulares, etc.
        """
        doc = Document(
            page_content=f"NORMATIVA — {organismo} — {nro_norma}: {titulo}\n\n{contenido}",
            metadata={
                "tipo":      "normativa",
                "titulo":    titulo,
                "organismo": organismo,
                "nro_norma": nro_norma,
            }
        )
        self.vector_store.add_documents([doc])

    def indexar_desde_pdf(self, pdf_path: str, tipo: str,
                           metadata_extra: dict = None):
        """
        Carga un PDF completo, lo fragmenta y lo indexa en pgvector.
        Útil para reglamentos, manuales, circulares en PDF.
        """
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        loader   = PyPDFLoader(pdf_path)
        pages    = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " "],
        )
        chunks = splitter.split_documents(pages)

        # Agregar metadata de tipo a cada chunk + limpiar NUL/control chars
        for chunk in chunks:
            chunk.page_content = limpiar_texto(chunk.page_content)
            chunk.metadata["tipo"] = tipo
            if metadata_extra:
                chunk.metadata.update(metadata_extra)

        chunks = [c for c in chunks if c.page_content.strip()]
        self.vector_store.add_documents(chunks)
        print(f"Indexados {len(chunks)} fragmentos de '{pdf_path}' (tipo: {tipo})")

    def indexar_informe_anterior(self, cliente_id: str, informe_texto: str,
                                  fecha_informe: str, tipo_decision: str):
        """
        Indexa un informe generado anteriormente para ese cliente.
        Sirve como contexto histórico en futuros informes.
        """
        # Fragmentar el informe para no exceder el contexto del embedding
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
        chunks = splitter.create_documents(
            [informe_texto],
            metadatas=[{
                "tipo":          "informe_anterior",
                "cliente_id":    cliente_id,
                "fecha":         fecha_informe,
                "tipo_decision": tipo_decision,
            }]
        )
        self.vector_store.add_documents(chunks)

    # ------------------------------------------------------------------
    # Políticas actualizables (se reindexan al iniciar la aplicación)
    # ------------------------------------------------------------------

    def reindexar_politicas(self, carpeta: str = "politicas") -> int:
        """
        Borra las políticas indexadas y vuelve a indexar TODO lo que haya en
        `carpeta` (PDF / TXT / MD). Pensado para llamarse al iniciar la app:
        editás los archivos de políticas, reiniciás, y quedan actualizadas.
        Devuelve cuántos archivos se indexaron.
        """
        engine = create_engine(self.conn_string)

        # 1) Borrar las políticas anteriores (refresco limpio, sin duplicados).
        try:
            with engine.begin() as conn:
                conn.execute(text(
                    "DELETE FROM langchain_pg_embedding "
                    "WHERE cmetadata->>'tipo' = 'politica'"
                ))
        except Exception as e:
            print(f"[POLITICAS] no se pudieron borrar las previas: {e}", flush=True)

        if not os.path.isdir(carpeta):
            print(f"[POLITICAS] la carpeta '{carpeta}' no existe; nada que indexar", flush=True)
            return 0

        total = 0
        for nombre in sorted(os.listdir(carpeta)):
            ruta = os.path.join(carpeta, nombre)
            if not os.path.isfile(ruta):
                continue
            ext = nombre.lower().rsplit(".", 1)[-1] if "." in nombre else ""
            try:
                if ext == "pdf":
                    self.indexar_desde_pdf(
                        ruta, tipo="politica",
                        metadata_extra={"vigente": True, "titulo": nombre, "categoria": "general"},
                    )
                    total += 1
                elif ext in ("txt", "md"):
                    with open(ruta, encoding="utf-8") as f:
                        contenido = f.read()
                    self.indexar_politica(
                        titulo=nombre, contenido=contenido,
                        categoria="general", version="startup", vigente=True,
                    )
                    total += 1
                else:
                    print(f"[POLITICAS] ignorado (formato no soportado): {nombre}", flush=True)
            except Exception as e:
                print(f"[POLITICAS] error indexando {nombre}: {e}", flush=True)

        print(f"[POLITICAS] {total} archivo(s) reindexados desde '{carpeta}'", flush=True)
        return total