# rag/pgvector_indexer.py
from langchain_postgres import PGEngine, PGVectorStore, Column
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from sqlalchemy import create_engine, text
from db.postgres_connection import get_postgres_connstring
import os
import re


def limpiar_texto(t: str) -> str:
    """Quita NUL (0x00) y otros caracteres de control que PostgreSQL no acepta
    en columnas de texto. Conserva tab, salto de línea y retorno de carro."""
    if not t:
        return ""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", t)


# Modelos de embeddings disponibles — AMBOS de 1024 dimensiones.
MODELOS_EMBED = {
    "bge-m3": "BAAI/bge-m3",
    "qwen3":  "Qwen/Qwen3-Embedding-0.6B",
}
EMBED_DIM  = 1024                 # debe coincidir con vector_size de la tabla
TABLE_NAME = "credito_vectores"   # tabla nueva (esquema PGVectorStore)
# Columnas de metadata declaradas, sobre las que se puede FILTRAR.
# El resto (titulo, fecha, categoria, etc.) va a la columna JSON automática.
META_COLS  = ["tipo", "cliente_id", "vigente"]


def _modelo_embed() -> str:
    clave = os.environ.get("EMBED_MODEL", "").strip() or "bge-m3"
    return MODELOS_EMBED.get(clave.lower(), clave)


def _detectar_device() -> str:
    """En Apple Silicon la GPU es 'mps' (Metal), no 'cuda'."""
    device = os.environ.get("EMBED_DEVICE")
    if device:
        return device
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
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


def crear_tabla(overwrite: bool = False) -> None:
    """Crea la tabla de vectores con el esquema nuevo de PGVectorStore.
    Llamar UNA vez (desde init_db.py o a mano). Si ya existe y overwrite=False,
    no hace nada (capturamos el error)."""
    engine = PGEngine.from_connection_string(get_postgres_connstring())
    try:
        engine.init_vectorstore_table(
            table_name=TABLE_NAME,
            vector_size=EMBED_DIM,
            metadata_columns=[
                Column("tipo",       "TEXT"),
                Column("cliente_id", "TEXT"),
                Column("vigente",    "BOOLEAN"),
            ],
            overwrite_existing=overwrite,
        )
        print(f"[VECTORES] tabla '{TABLE_NAME}' creada (vector_size={EMBED_DIM})", flush=True)
    except Exception as e:
        print(f"[VECTORES] no se creó la tabla (¿ya existe?): {e}", flush=True)


def _get_store(embeddings) -> PGVectorStore:
    """Instancia el PGVectorStore (sync) conectado a la tabla nueva."""
    engine = PGEngine.from_connection_string(get_postgres_connstring())
    return PGVectorStore.create_sync(
        engine=engine,
        embedding_service=embeddings,
        table_name=TABLE_NAME,
        metadata_columns=META_COLS,
    )


class PGVectorIndexer:
    """
    Indexa documentos no estructurados en PostgreSQL usando PGVectorStore.
    El campo 'tipo' (columna declarada) discrimina el tipo de documento.
    """

    TABLE_NAME = TABLE_NAME   # compatibilidad con código que lo referenciaba

    def __init__(self):
        self.embeddings   = get_embeddings()
        self.conn_string  = get_postgres_connstring()
        self.vector_store = _get_store(self.embeddings)

    # ------------------------------------------------------------------
    def indexar_nota_mora(self, cliente_id: str, texto_nota: str,
                           fecha: str, oficial: str, tipo_gestion: str):
        doc = Document(
            page_content=limpiar_texto(texto_nota),
            metadata={
                "tipo":         "nota_mora",
                "cliente_id":   cliente_id,
                "fecha":        fecha,
                "oficial":      oficial,
                "tipo_gestion": tipo_gestion,
            },
        )
        self.vector_store.add_documents([doc])

    def indexar_politica(self, titulo: str, contenido: str,
                          categoria: str, version: str, vigente: bool = True):
        doc = Document(
            page_content=limpiar_texto(f"POLÍTICA: {titulo}\n\n{contenido}"),
            metadata={
                "tipo":      "politica",
                "titulo":    titulo,
                "categoria": categoria,
                "version":   version,
                "vigente":   vigente,
            },
        )
        self.vector_store.add_documents([doc])

    def indexar_normativa(self, titulo: str, contenido: str,
                           organismo: str, nro_norma: str):
        doc = Document(
            page_content=limpiar_texto(
                f"NORMATIVA — {organismo} — {nro_norma}: {titulo}\n\n{contenido}"),
            metadata={
                "tipo":      "normativa",
                "titulo":    titulo,
                "organismo": organismo,
                "nro_norma": nro_norma,
            },
        )
        self.vector_store.add_documents([doc])

    def indexar_desde_pdf(self, pdf_path: str, tipo: str,
                           metadata_extra: dict = None):
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        loader   = PyPDFLoader(pdf_path)
        pages    = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " "],
        )
        chunks = splitter.split_documents(pages)
        for chunk in chunks:
            chunk.page_content = limpiar_texto(chunk.page_content)
            chunk.metadata["tipo"] = tipo
            if metadata_extra:
                chunk.metadata.update(metadata_extra)
        chunks = [c for c in chunks if c.page_content.strip()]
        if chunks:
            self.vector_store.add_documents(chunks)
        print(f"Indexados {len(chunks)} fragmentos de '{pdf_path}' (tipo: {tipo})")

    def indexar_informe_anterior(self, cliente_id: str, informe_texto: str,
                                  fecha_informe: str, tipo_decision: str):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
        chunks = splitter.create_documents(
            [limpiar_texto(informe_texto)],
            metadatas=[{
                "tipo":          "informe_anterior",
                "cliente_id":    cliente_id,
                "fecha":         fecha_informe,
                "tipo_decision": tipo_decision,
            }],
        )
        if chunks:
            self.vector_store.add_documents(chunks)

    # ------------------------------------------------------------------
    def reindexar_politicas(self, carpeta: str = "politicas") -> int:
        """Borra las políticas indexadas y reindexar TODO lo que haya en `carpeta`."""
        # 'tipo' ahora es una COLUMNA real -> borrado directo.
        engine = create_engine(self.conn_string)
        try:
            with engine.begin() as conn:
                conn.execute(text(f"DELETE FROM {TABLE_NAME} WHERE tipo = 'politica'"))
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