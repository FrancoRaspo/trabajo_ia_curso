"""
App web local para generar informes crediticios, con progreso en vivo.

Flujo:
  Formulario (CUIT + razón social + motivo)
    -> stream de eventos (SSE): muestra el progreso por etapas
       (SQL Server -> ML -> RAG -> LLM) y va escribiendo el informe tipo chat.

Variables de entorno (.env):
    EMPRESA_NOMBRE = nombre de la institución mostrado en la interfaz

Requisitos:
    pip install fastapi uvicorn python-multipart

Para correr (parado en la carpeta del proyecto, junto a pipeline.py):
    uvicorn app:app --host 127.0.0.1 --port 8000
Luego abrir: http://127.0.0.1:8000
"""

import hashlib
import hmac
import html
import io
import json
import os
import re
import traceback
from contextlib import asynccontextmanager
from datetime import datetime

from typing import Any

from fastapi import FastAPI, Form, File, UploadFile, Header, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, Response, JSONResponse
from pydantic import BaseModel

# Cargar variables de entorno (.env). pipeline.py también lo hace al importarse,
# pero lo dejamos acá para que app.py sea autosuficiente.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ===========================================================================
#  INTEGRACIÓN CON EL PIPELINE
# ===========================================================================
# get_pipeline() construye ReportPipeline una sola vez (modelo cargado una vez)
# y lo reutiliza en cada request.
from pipeline import get_pipeline

PROVEEDOR_LLM = os.environ.get("PROVEEDOR_LLM", "local")   # "local" / "anthropic" / "gemini" / "openai"

# Nombre de la institución, tomado del entorno (.env -> EMPRESA_NOMBRE).
EMPRESA = os.environ.get("EMPRESA_NOMBRE", "Mi Institución")

# ---------------------------------------------------------------------------
#  Modo administrador (editar prompts + regenerar secciones)
# ---------------------------------------------------------------------------
# La autenticación es una contraseña compartida en .env (ADMIN_PASSWORD). Si está
# vacía, el modo admin queda DESHABILITADO (default seguro). El token que se
# entrega al iniciar sesión es un HMAC de la contraseña: se puede verificar sin
# guardar estado en el server, y no revela la contraseña.
ADMIN_PASSWORD = (os.environ.get("ADMIN_PASSWORD") or "").strip()


def _admin_token() -> str:
    """Token de sesión admin: HMAC-SHA256(ADMIN_PASSWORD, 'trabajo_ia_admin')."""
    return hmac.new(ADMIN_PASSWORD.encode("utf-8"),
                    b"trabajo_ia_admin", hashlib.sha256).hexdigest()


def _es_admin(token: str) -> bool:
    """True si el token corresponde a la contraseña admin (comparación en tiempo
    constante). Siempre False si ADMIN_PASSWORD no está configurada."""
    if not ADMIN_PASSWORD or not token:
        return False
    return hmac.compare_digest(token, _admin_token())


def _requerir_admin(token: str) -> None:
    """Corta con 401 si el token no es de admin. Usar al inicio de cada endpoint admin."""
    if not _es_admin(token):
        raise HTTPException(status_code=401, detail="No autorizado (modo administrador).")

def nuevo_numero_informe() -> str:
    """Genera un número de informe único y legible: INF-AAAAMMDD-HHMMSS."""
    return "INF-" + datetime.now().strftime("%Y%m%d-%H%M%S")


def extraer_texto_archivo(nombre: str, datos: bytes) -> str:
    """Extrae texto de un archivo subido (PDF / TXT / MD). Otros formatos: vacío."""
    ext = (nombre or "").lower().rsplit(".", 1)[-1] if "." in (nombre or "") else ""
    texto = ""
    if ext in ("txt", "md", "csv"):
        texto = datos.decode("utf-8", errors="ignore")
    elif ext == "pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(datos))
            texto = "\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception:
            traceback.print_exc()
            texto = ""
    # Quitar NUL/control chars que PostgreSQL rechaza (frecuentes en PDFs).
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", texto)


def reindexar_politicas_startup():
    """Reindexa las políticas de la entidad desde POLITICAS_DIR al iniciar la app."""
    carpeta = os.environ.get("POLITICAS_DIR", "politicas")
    try:
        from rag.pgvector_indexer import PGVectorIndexer
        n = PGVectorIndexer().reindexar_politicas(carpeta)
        print(f"[STARTUP] políticas reindexadas: {n}", flush=True)
    except Exception as e:
        print(f"[STARTUP] no se pudieron reindexar políticas: {e}", flush=True)

# ===========================================================================
#  Validación de CUIT (formato + dígito verificador argentino)
# ===========================================================================

def normalizar_cuit(cuit: str) -> str:
    return re.sub(r"\D", "", cuit or "")

def cuit_valido(cuit: str) -> bool:
    d = normalizar_cuit(cuit)
    if len(d) != 11:
        return False
    mult = [5, 4, 3, 2, 7, 6, 5, 4, 3, 2]
    resto = sum(int(c) * m for c, m in zip(d[:10], mult)) % 11
    ver = 11 - resto
    if ver == 11:
        ver = 0
    elif ver == 10:
        return False
    return ver == int(d[10])

# ===========================================================================
#  Estilos y armazón de página
# ===========================================================================

ESTILOS = """
:root {
  --tinta: #15263b; --tinta-suave: #3d4f63;
  --papel: #f7f5f0; --tarjeta: #ffffff; --linea: #d9d3c7;
  --acento: #b67a23; --error: #9a2b2b; --ok: #2f6b3f;
  --btn-bg: #15263b; --btn-fg: #ffffff;
}
[data-tema="dark"] {
  --tinta: #e8e6e1; --tinta-suave: #9aa6b2;
  --papel: #131922; --tarjeta: #1b242e; --linea: #2e3a46;
  --acento: #d99a3f; --error: #e08a86; --ok: #7fc08e;
  --btn-bg: #d99a3f; --btn-fg: #131922;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--papel); color: var(--tinta);
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; line-height: 1.55;
  transition: background .2s, color .2s;
}
.envoltura { max-width: 820px; margin: 0 auto; padding: 28px 24px 80px; }
.barra-sup { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
.btn-tema { font: inherit; font-size: 13px; font-weight: 600; cursor: pointer;
  background: transparent; color: var(--tinta-suave); border: 1px solid var(--linea);
  border-radius: 999px; padding: 7px 14px; display: inline-flex; align-items: center; gap: 7px; }
.btn-tema:hover { color: var(--tinta); border-color: var(--acento); }
.marca { font-size: 13px; letter-spacing: .18em; text-transform: uppercase;
  color: var(--acento); font-weight: 600; margin-bottom: 6px; }
h1 { font-family: Georgia, "Times New Roman", serif; font-weight: 600;
  font-size: 34px; margin: 0 0 4px; letter-spacing: -.01em; }
.subtitulo { color: var(--tinta-suave); margin: 0 0 36px; }
.tarjeta { background: var(--tarjeta); border: 1px solid var(--linea);
  border-radius: 10px; padding: 32px; }
label { display: block; font-weight: 600; font-size: 14px; margin: 0 0 6px; }
.ayuda { font-weight: 400; color: var(--tinta-suave); font-size: 13px; }
input, textarea { width: 100%; padding: 12px 14px; font: inherit; color: var(--tinta);
  background: var(--papel); border: 1px solid var(--linea); border-radius: 7px; margin-bottom: 22px; }
input:focus, textarea:focus { outline: 2px solid var(--acento); outline-offset: 1px; border-color: var(--acento); }
textarea { min-height: 90px; resize: vertical; }
input[type="file"] { padding: 10px 12px; cursor: pointer; }
input[type="file"]::file-selector-button {
  font: inherit; font-weight: 600; cursor: pointer; margin-right: 10px;
  background: var(--papel); color: var(--tinta);
  border: 1px solid var(--linea); border-radius: 6px; padding: 6px 12px; }
button { font: inherit; font-weight: 600; cursor: pointer; background: var(--btn-bg);
  color: var(--btn-fg); border: 0; padding: 13px 26px; border-radius: 7px; }
button:disabled { opacity: .6; cursor: progress; }
button.secundario { background: transparent; color: var(--tinta); border: 1px solid var(--linea); }
.barra-acciones { display: flex; gap: 12px; margin-top: 8px; }
.aviso { border-left: 3px solid var(--error); background: #fbedec; color: var(--error);
  padding: 12px 16px; border-radius: 6px; margin-bottom: 24px; font-size: 14px; }

/* progreso en un solo renglón (spinner + texto que cambia, con brillo) */
.progreso { display: flex; align-items: center; gap: 10px; margin: 0 0 26px;
  font-size: 14px; min-height: 20px; }
.progreso.oculto { display: none; }
.spinner { width: 14px; height: 14px; border-radius: 50%; flex: none;
  border: 2px solid var(--linea); border-top-color: var(--acento);
  animation: girar .7s linear infinite; }
@keyframes girar { to { transform: rotate(360deg); } }
.progreso-texto {
  background: linear-gradient(90deg, var(--tinta-suave) 35%, var(--tinta) 50%, var(--tinta-suave) 65%);
  background-size: 200% auto; background-clip: text; -webkit-background-clip: text;
  -webkit-text-fill-color: transparent; color: transparent;
  animation: brillo 1.6s linear infinite;
}
@keyframes brillo { to { background-position: -200% center; } }

/* informe en vivo */
.informe { background: var(--tarjeta); border: 1px solid var(--linea);
  border-radius: 10px; padding: 40px; }
.doc-encabezado { border-bottom: 2px solid var(--tinta); padding-bottom: 14px; margin-bottom: 20px; }
.doc-empresa { font-family: Georgia, "Times New Roman", serif; font-size: 22px; font-weight: 600; line-height: 1.2; }
.doc-sub { color: var(--tinta-suave); font-size: 13px; letter-spacing: .04em; margin-top: 3px; }
.meta { color: var(--tinta-suave); font-size: 14px; margin-bottom: 24px; }
.meta strong { color: var(--tinta); }
.informe-texto h1 { font-family: Georgia, serif; font-size: 25px; margin: 0 0 18px; }
.informe-texto h2 { font-family: Georgia, serif; font-size: 19px; margin: 28px 0 10px;
  padding-bottom: 6px; border-bottom: 1px solid var(--linea); }
.informe-texto h3 { font-size: 16px; margin: 20px 0 8px; }
.informe-texto p { margin: 0 0 12px; }
.informe-texto ul, .informe-texto ol { margin: 0 0 14px; padding-left: 22px; }
.informe-texto li { margin: 3px 0; }
.informe-texto strong { color: var(--tinta); }
.informe-texto code { background: var(--papel); padding: 1px 5px; border-radius: 4px; font-size: 13px; }
.informe-texto hr { border: 0; border-top: 1px solid var(--linea); margin: 18px 0; }
.informe-texto table { border-collapse: collapse; width: 100%; margin: 6px 0 18px; font-size: 14px; }
.informe-texto th, .informe-texto td { border: 1px solid var(--linea); padding: 7px 10px; text-align: left; }
.informe-texto th { background: var(--papel); font-weight: 600; }
.caret { display: inline-block; width: 7px; height: 1.05em; background: var(--acento);
  vertical-align: text-bottom; margin-left: 1px; animation: parpadeo 1s steps(2) infinite; }
@keyframes parpadeo { 0%,100% { opacity: 1; } 50% { opacity: 0; } }

/* validación */
.validacion { border-radius: 8px; padding: 14px 18px; margin: 24px 0 0; font-size: 14px;
  border: 1px solid var(--linea); }
.validacion.ok     { background: #eef5ef; border-color: #c5ddc9; color: var(--ok); }
.validacion.alerta { background: #fbedec; border-color: #e7c6c4; color: var(--error); }
.validacion h3 { margin: 0 0 8px; font-size: 14px; letter-spacing: .03em; }
.validacion ul { margin: 0; padding-left: 18px; }

/* ---- Modo administrador: barra, editor de prompts, edición por sección ---- */
.acciones-sup { display: flex; gap: 8px; align-items: center; }
.sec { position: relative; }
.sec-editar { position: absolute; top: 22px; right: 0; font: inherit; font-size: 12px;
  font-weight: 600; cursor: pointer; background: transparent; color: var(--acento);
  border: 1px solid var(--linea); border-radius: 999px; padding: 3px 10px; }
.sec-editar:hover { background: var(--papel); border-color: var(--acento); }
.sec.regenerando .sec-cuerpo { opacity: .7; }
body:not([data-admin]) .sec-editar { display: none; }

/* Trazabilidad: resumen de datos fuente por sección (visible para todo lector) */
.sec-datos { margin: 6px 0 18px; border: 1px solid var(--linea); border-radius: 8px;
  background: var(--papel); font-size: 13px; }
.sec-datos > summary { cursor: pointer; padding: 8px 12px; font-weight: 600;
  color: var(--acento); list-style: none; user-select: none; }
.sec-datos > summary::-webkit-details-marker { display: none; }
.sec-datos > summary::before { content: '▸ '; }
.sec-datos[open] > summary::before { content: '▾ '; }
.sec-datos-tabla { width: 100%; border-collapse: collapse; margin: 0 0 4px; }
.sec-datos-tabla th, .sec-datos-tabla td { text-align: left; padding: 5px 12px;
  vertical-align: top; border-top: 1px solid var(--linea); }
.sec-datos-tabla th { font-weight: 600; color: var(--tinta-2, #667); width: 45%; }
.sec-datos-tabla tr.dato-alerta td, .sec-datos-tabla tr.dato-alerta th { color: var(--error); }

.modal { position: fixed; inset: 0; background: rgba(10,14,20,.55); display: flex;
  align-items: flex-start; justify-content: center; padding: 40px 16px; z-index: 50; overflow: auto; }
.modal.oculto { display: none; }
.modal-caja { background: var(--tarjeta); border: 1px solid var(--linea); border-radius: 12px;
  width: 100%; max-width: 760px; box-shadow: 0 12px 40px rgba(0,0,0,.3); }
.modal-cab { display: flex; justify-content: space-between; align-items: center;
  padding: 16px 20px; border-bottom: 1px solid var(--linea); }
.modal-cab strong { font-family: Georgia, serif; font-size: 18px; }
.modal-cab .cerrar { background: transparent; color: var(--tinta-suave); border: 0;
  font-size: 18px; padding: 4px 8px; cursor: pointer; }
.modal-cuerpo { padding: 18px 20px 24px; }
.editor-ayuda { color: var(--tinta-suave); font-size: 13px; margin: 0 0 16px; }
.p-grupo { font-size: 12px; letter-spacing: .12em; text-transform: uppercase; color: var(--acento);
  margin: 18px 0 8px; border-bottom: 1px solid var(--linea); padding-bottom: 4px; }
.p-item { border: 1px solid var(--linea); border-radius: 8px; padding: 12px 14px;
  margin: 0 0 14px; background: var(--papel); }
.p-tit { font-weight: 600; font-size: 14px; }
.p-name { font-weight: 400; font-size: 12px; color: var(--tinta-suave); margin-left: 6px; }
.p-desc { color: var(--tinta-suave); font-size: 12.5px; margin: 4px 0 8px; }
.p-ta { width: 100%; min-height: 120px; margin: 0 0 8px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px; line-height: 1.5; }
.p-acc { display: flex; align-items: center; gap: 10px; }
.p-acc button { padding: 7px 14px; font-size: 13px; }
.p-msg { font-size: 12.5px; }
.p-msg.ok { color: var(--ok); }
.p-msg.err { color: var(--error); }
.adm-pass { max-width: 280px; }
.regen-bar { margin-top: 6px; display: flex; align-items: center; gap: 10px; }
.btn-regen { background: var(--acento); color: #fff; }

.datos-sec { margin: 0 0 16px; border: 1px solid var(--linea); border-radius: 8px; background: var(--papel); }
.datos-sec > summary { cursor: pointer; padding: 10px 14px; font-size: 13px; font-weight: 600;
  color: var(--tinta-suave); list-style: none; user-select: none; }
.datos-sec > summary:hover { color: var(--tinta); }
.datos-sec[open] > summary { border-bottom: 1px solid var(--linea); }
.datos-json { margin: 0; max-height: 320px; overflow: auto; padding: 12px 14px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; line-height: 1.5;
  white-space: pre-wrap; word-break: break-word; color: var(--tinta); }

@media print {
  /* Forzar hoja blanca con letras negras, sin importar el tema activo. */
  :root, [data-tema="dark"] {
    --tinta: #000000; --tinta-suave: #333333;
    --papel: #ffffff; --tarjeta: #ffffff; --linea: #999999;
    --acento: #000000; --error: #000000; --ok: #000000;
    --btn-bg: #ffffff; --btn-fg: #000000;
  }
  body { background: #ffffff; color: #000000; }
  .barra-acciones, .marca, .progreso, .barra-sup, .modal, .sec-editar, .sec-datos { display: none; }
  .informe { border: 0; padding: 0; }
  .doc-empresa, .meta strong, .informe-texto strong,
  .informe-texto h1, .informe-texto h2, .informe-texto h3 { color: #000000; }
}
"""

def pagina(titulo: str, contenido: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="es"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(titulo)}</title>
<script>
  // Aplica el tema guardado lo antes posible para evitar parpadeo.
  (function () {{
    try {{
      var t = localStorage.getItem('tema');
      if (t === 'dark') document.documentElement.setAttribute('data-tema', 'dark');
    }} catch (e) {{}}
  }})();
</script>
<style>{ESTILOS}</style>
</head><body><div class="envoltura">{contenido}</div></body></html>"""

# El contenido es estático: todo lo dinámico ocurre en el navegador vía streaming.
CONTENIDO = """
  <div class="barra-sup">
    <div class="marca">__EMPRESA__</div>
    <div class="acciones-sup">
      <button type="button" id="btnDatos" class="btn-tema" onclick="abrirDatosGlobal()" style="display:none;">Datos</button>
      <button type="button" id="btnPrompts" class="btn-tema" onclick="abrirPromptsGlobal()" style="display:none;">Prompts</button>
      <button type="button" id="btnAdmin" class="btn-tema" onclick="toggleAdmin()">🔒 Admin</button>
      <button type="button" id="btnTema" class="btn-tema" onclick="alternarTema()">Modo oscuro</button>
    </div>
  </div>
  <h1>Generador de informes</h1>
  <p class="subtitulo">Describí la consulta en lenguaje natural. El sistema identifica al asociado y genera el informe.</p>

  <div id="aviso" class="aviso" style="display:none;"></div>

  <div id="tarjetaForm" class="tarjeta">
    <form id="form">
      <label for="consulta">Consulta</label>
      <textarea id="consulta" name="consulta" rows="4"
                placeholder="Ej.: El asociado Juan Pérez, CUIT 23-25079318-9, quiere sacar un crédito en la entidad por 10 millones."
                required></textarea>

      <label for="docs">Documentos del asociado
        <span class="ayuda">— opcional (PDF, TXT). Se suman al análisis de este informe.</span></label>
      <input type="file" id="docs" name="docs" multiple accept=".pdf,.txt,.md,.csv">

      <div class="barra-acciones">
        <button id="btn" type="submit">Generar informe</button>
      </div>
    </form>
  </div>

  <div id="proceso" style="display:none;">
    <div class="progreso" id="progreso">
      <span class="spinner"></span>
      <span class="progreso-texto" id="progresoTexto">Iniciando…</span>
    </div>

    <div id="tarjetaInforme" class="informe" style="display:none;">
      <div class="doc-encabezado" id="docEncabezado"></div>
      <div class="meta" id="meta"></div>
      <div class="informe-texto" id="texto"></div>
    </div>

    <div id="panelValidacion"></div>

    <div class="barra-acciones" id="accionesFin" style="display:none;">
      <button type="button" class="secundario" onclick="reiniciar()">Nuevo informe</button>
      <button type="button" id="btnPdf" onclick="descargarPDF()">Descargar PDF</button>
    </div>
  </div>

  <!-- Modal del modo administrador (login / editor de prompts) -->
  <div id="modalAdmin" class="modal oculto">
    <div class="modal-caja">
      <div class="modal-cab">
        <strong id="modalTitulo">Prompts</strong>
        <button type="button" class="cerrar" onclick="cerrarModal()" aria-label="Cerrar">✕</button>
      </div>
      <div class="modal-cuerpo" id="modalCuerpo"></div>
    </div>
  </div>

  <script>
    const form = document.getElementById('form');
    const aviso = document.getElementById('aviso');
    const tarjetaForm = document.getElementById('tarjetaForm');
    const proceso = document.getElementById('proceso');
    const progreso = document.getElementById('progreso');
    const progresoTexto = document.getElementById('progresoTexto');
    const tarjetaInforme = document.getElementById('tarjetaInforme');
    const docEncabezado = document.getElementById('docEncabezado');
    const elTexto = document.getElementById('texto');
    const elMeta = document.getElementById('meta');
    const panelVal = document.getElementById('panelValidacion');
    const accionesFin = document.getElementById('accionesFin');
    let mdCrudo = '';
    let rafRender = null;   // id de requestAnimationFrame para agrupar renders
    let datosInforme = {};  // datos del membrete (del evento meta) para el PDF

    // --- Estado del modo administrador ---
    let sesionId = '';        // id de la sesión del informe (para regenerar secciones)
    let adminToken = '';      // token de admin (si hay sesión iniciada)
    let adminConfig = null;   // catálogo de prompts + mapa de secciones (de /admin/config)
    let secModelo = [];       // [{titulo, cuerpo}] del informe renderizado por secciones
    let datosSecciones = {};  // {titulo: [{etiqueta,valor,alerta}]} datos fuente por sección (trazabilidad)
    let encabezadoMd = '';    // markdown previo a la 1ª sección (membrete del documento)
    const promptOriginal = {}; // contenido original de cada prompt (para detectar cambios)
    const ADMIN_HABILITADO = ('__ADMIN_HABILITADO__' === '1');

    const ETIQUETAS = {
      extraccion: 'Interpretando la consulta…',
      documentos: 'Procesando documentos adjuntos…',
      sql: 'Leyendo datos financieros…',
      ml:  'Calculando score crediticio…',
      rag: 'Recuperando contexto cualitativo…',
      llm: 'Redactando informe…',
    };

    form.addEventListener('submit', function (e) { e.preventDefault(); generar(); });

    // --- Modo oscuro ---
    const btnTema = document.getElementById('btnTema');
    function sincronizarBotonTema() {
      const oscuro = document.documentElement.getAttribute('data-tema') === 'dark';
      btnTema.textContent = oscuro ? 'Modo claro' : 'Modo oscuro';
    }
    function alternarTema() {
      const oscuro = document.documentElement.getAttribute('data-tema') === 'dark';
      if (oscuro) { document.documentElement.removeAttribute('data-tema'); }
      else { document.documentElement.setAttribute('data-tema', 'dark'); }
      try { localStorage.setItem('tema', oscuro ? 'light' : 'dark'); } catch (e) {}
      sincronizarBotonTema();
    }
    sincronizarBotonTema();

    function escapar(s) { const d = document.createElement('div'); d.textContent = s || ''; return d.innerHTML; }
    function fmtCuit(d) { d = (d || '').replace(/\\D/g, ''); return d.length === 11 ? d.slice(0,2)+'-'+d.slice(2,10)+'-'+d.slice(10) : d; }
    function mostrarAviso(m) { aviso.textContent = m; aviso.style.display = 'block'; }
    function ocultarAviso() { aviso.style.display = 'none'; }

    function inline(t) {
      t = t.replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>');
      t = t.replace(/\\*([^*]+)\\*/g, '<em>$1</em>');
      t = t.replace(/`([^`]+)`/g, '<code>$1</code>');
      return t;
    }

    function mdToHtml(md) {
      const lineas = escapar(md).split('\\n');
      let html = '', i = 0, enUl = false, enOl = false, parr = [];
      function cerrarParr() { if (parr.length) { html += '<p>' + inline(parr.join(' ')) + '</p>'; parr = []; } }
      function cerrarListas() { if (enUl) { html += '</ul>'; enUl = false; } if (enOl) { html += '</ol>'; enOl = false; } }
      while (i < lineas.length) {
        const t = lineas[i].trim();
        // tabla (encabezado + fila separadora |---|---|)
        if (/^\\|.*\\|$/.test(t) && i + 1 < lineas.length && /^\\|[\\s:|-]+\\|$/.test(lineas[i + 1].trim())) {
          cerrarParr(); cerrarListas();
          const heads = t.split('|').slice(1, -1).map(c => c.trim());
          html += '<table><thead><tr>' + heads.map(h => '<th>' + inline(h) + '</th>').join('') + '</tr></thead><tbody>';
          i += 2;
          while (i < lineas.length && /^\\|.*\\|$/.test(lineas[i].trim())) {
            const celdas = lineas[i].trim().split('|').slice(1, -1).map(c => c.trim());
            html += '<tr>' + celdas.map(c => '<td>' + inline(c) + '</td>').join('') + '</tr>';
            i++;
          }
          html += '</tbody></table>'; continue;
        }
        let m = /^(#{1,4})\\s+(.*)$/.exec(t);
        if (m) { cerrarParr(); cerrarListas(); const n = m[1].length; html += '<h' + n + '>' + inline(m[2]) + '</h' + n + '>'; i++; continue; }
        if (/^(-{3,}|\\*{3,})$/.test(t)) { cerrarParr(); cerrarListas(); html += '<hr>'; i++; continue; }
        m = /^[-*]\\s+(.*)$/.exec(t);
        if (m) { cerrarParr(); if (enOl) cerrarListas(); if (!enUl) { html += '<ul>'; enUl = true; } html += '<li>' + inline(m[1]) + '</li>'; i++; continue; }
        m = /^\\d+\\.\\s+(.*)$/.exec(t);
        if (m) { cerrarParr(); if (enUl) cerrarListas(); if (!enOl) { html += '<ol>'; enOl = true; } html += '<li>' + inline(m[1]) + '</li>'; i++; continue; }
        if (t === '') { cerrarParr(); cerrarListas(); i++; continue; }
        cerrarListas(); parr.push(t); i++;
      }
      cerrarParr(); cerrarListas();
      return html;
    }

    function setProgreso(texto) {
      progreso.classList.remove('oculto');
      progresoTexto.textContent = texto;
    }
    function ocultarProgreso() { progreso.classList.add('oculto'); }

    // Render agrupado: re-parsea el Markdown como mucho 1 vez por frame, no por token.
    function cercaDelFondo() {
      return (window.innerHeight + window.scrollY) >= (document.body.scrollHeight - 140);
    }
    function programarRender() {
      if (rafRender) return;
      rafRender = requestAnimationFrame(function () {
        rafRender = null;
        const abajo = cercaDelFondo();
        elTexto.innerHTML = mdToHtml(mdCrudo) + '<span class="caret"></span>';
        if (abajo) window.scrollTo(0, document.body.scrollHeight);
      });
    }

    function reiniciar() {
      proceso.style.display = 'none';
      tarjetaForm.style.display = 'block';
      ocultarAviso();
    }

    // Descarga el PDF desde el servidor (mismo renderizador que el endpoint
    // app-to-app), enviando el informe YA generado -> el PDF es idéntico a lo
    // que se ve en pantalla.
    async function descargarPDF() {
      if (!datosInforme.informe) { return; }
      const btn = document.getElementById('btnPdf');
      const txt = btn.textContent;
      btn.disabled = true; btn.textContent = 'Generando PDF…';
      try {
        const resp = await fetch('/api/informe/pdf-render', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(datosInforme),
        });
        if (!resp.ok) {
          let msg = 'No se pudo generar el PDF.';
          try { msg = (await resp.json()).error || msg; } catch (e) {}
          mostrarAviso(msg); return;
        }
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'informe-' + (datosInforme.cuit || 'sd') + '-' + (datosInforme.numero || '') + '.pdf';
        document.body.appendChild(a); a.click(); a.remove();
        URL.revokeObjectURL(url);
      } catch (e) {
        mostrarAviso('No se pudo conectar con el servidor para generar el PDF.');
      } finally {
        btn.disabled = false; btn.textContent = txt;
      }
    }
    function volverConError(msg) {
      proceso.style.display = 'none';
      tarjetaForm.style.display = 'block';
      mostrarAviso(msg);
    }

    function panelValidacionHTML(ev) {
      const adv = ev.advertencias || [];
      if (adv.length === 0)
        return '<div class="validacion ok"><h3>Validación correcta</h3><p style="margin:0;">Sin observaciones.</p></div>';
      const items = adv.map(a => '<li>' + escapar(a) + '</li>').join('');
      const clase = ev.aprobado ? 'ok' : 'alerta';
      const titulo = ev.aprobado ? 'Validación con observaciones' : 'Validación con alertas';
      return '<div class="validacion ' + clase + '"><h3>' + titulo + '</h3><ul>' + items + '</ul></div>';
    }

    async function generar() {
      ocultarAviso();
      const consulta = document.getElementById('consulta').value;
      if (!consulta.trim()) { mostrarAviso('Escribí la consulta.'); return; }

      // pasar a la vista de proceso, limpiar estado previo
      tarjetaForm.style.display = 'none';
      proceso.style.display = 'block';
      tarjetaInforme.style.display = 'none';
      elTexto.innerHTML = ''; elMeta.textContent = ''; docEncabezado.innerHTML = ''; panelVal.innerHTML = ''; mdCrudo = '';
      if (rafRender) { cancelAnimationFrame(rafRender); rafRender = null; }
      accionesFin.style.display = 'none';
      setProgreso('Interpretando la consulta…');

      const fd = new FormData();
      fd.append('consulta', consulta);
      const docs = document.getElementById('docs').files;
      for (const f of docs) fd.append('docs', f);

      let resp;
      try {
        resp = await fetch('/informe/stream', { method: 'POST', body: fd });
      } catch (err) { volverConError('No se pudo conectar con el servidor.'); return; }

      const reader = resp.body.getReader();
      const dec = new TextDecoder();
      let buf = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf('\\n\\n')) >= 0) {
          const bloque = buf.slice(0, i); buf = buf.slice(i + 2);
          const linea = bloque.split('\\n').find(l => l.startsWith('data:'));
          if (!linea) continue;
          let ev; try { ev = JSON.parse(linea.slice(5).trim()); } catch (e) { continue; }
          manejar(ev);
        }
      }
    }

    function manejar(ev) {
      if (ev.tipo === 'etapa') {
        setProgreso(ETIQUETAS[ev.clave] || 'Procesando…');
        if (ev.clave === 'llm') { tarjetaInforme.style.display = 'block'; }
      } else if (ev.tipo === 'seccion') {
        setProgreso('Redactando: ' + (ev.titulo || 'informe') + '…');
      } else if (ev.tipo === 'meta') {
        // Guardar los datos del membrete para poder generar el PDF idéntico al
        // informe mostrado (se envían al endpoint /api/informe/pdf-render).
        datosInforme = {
          numero: ev.numero || '', cuit: ev.cuit || '', nombre: ev.razon || '',
          motivo: ev.motivo || '', modelo: ev.modelo || '',
          score: (ev.score === undefined ? null : ev.score),
          nivel: ev.nivel || '', bcra: ev.bcra || null,
          encontrado: ev.encontrado !== false,
        };
        // Membrete del documento: institución + N° de informe + fecha (datos fijos)
        const fecha = new Date().toLocaleDateString('es-AR', { day: '2-digit', month: '2-digit', year: 'numeric' });
        let cab = '';
        if (ev.empresa) cab += '<div class="doc-empresa">' + escapar(String(ev.empresa)) + '</div>';
        const sub = [];
        sub.push('Informe crediticio');
        if (ev.numero) sub.push('N° ' + escapar(String(ev.numero)));
        sub.push(fecha);
        if (ev.modelo) sub.push('Modelo: ' + escapar(String(ev.modelo)));
        cab += '<div class="doc-sub">' + sub.join(' &nbsp;·&nbsp; ') + '</div>';
        docEncabezado.innerHTML = cab;

        let h = '';
        h += '<strong>CUIT:</strong> ' + escapar(fmtCuit(ev.cuit || ''));
        // La razón social extraída se muestra solo si NO se hallaron datos del cliente.
        if (ev.encontrado === false) {
          h += ' &nbsp;·&nbsp; <strong>Razón social:</strong> ' + escapar(ev.razon || '') +
               ' <span style="color:var(--error);">(sin datos en la base)</span>';
        }
        if (ev.score !== undefined && ev.score !== null) h += '<br><strong>Score:</strong> ' + escapar(String(ev.score));
        if (ev.nivel) h += ' &nbsp;·&nbsp; <strong>Nivel:</strong> ' + escapar(String(ev.nivel));
        // BCRA (Central de Deudores): se muestra si el pipeline lo consultó.
        if (ev.bcra) {
          if (ev.bcra.tiene_datos) {
            const irregular = Number(ev.bcra.peor_situacion) >= 3;
            let sit = 'Situación ' + escapar(String(ev.bcra.peor_situacion));
            if (ev.bcra.peor_situacion_desc) sit += ' (' + escapar(String(ev.bcra.peor_situacion_desc)) + ')';
            let b = '<strong>BCRA:</strong> ' +
                    (irregular ? '<span style="color:var(--error);">' + sit + '</span>' : sit);
            if (ev.bcra.cant_entidades) b += ' &nbsp;·&nbsp; ' + escapar(String(ev.bcra.cant_entidades)) + ' entidad(es)';
            if (ev.bcra.cheques_impagos) b += ' &nbsp;·&nbsp; <span style="color:var(--error);">' +
                    escapar(String(ev.bcra.cheques_impagos)) + ' cheque(s) impago(s)</span>';
            h += '<br>' + b;
          } else {
            h += '<br><strong>BCRA:</strong> sin registros en el sistema financiero';
          }
        }
        h += '<br><strong>Motivo:</strong> ' + escapar(ev.motivo || '');
        elMeta.innerHTML = h;
      } else if (ev.tipo === 'token') {
        mdCrudo += ev.texto;
        programarRender();
      } else if (ev.tipo === 'validacion') {
        panelVal.innerHTML = panelValidacionHTML(ev);
      } else if (ev.tipo === 'fin') {
        ocultarProgreso();
        if (rafRender) { cancelAnimationFrame(rafRender); rafRender = null; }
        // Durante el stream las secciones llegan en orden de ejecución; el evento
        // 'fin' trae el documento en orden de lectura (Resumen arriba). Si viene,
        // se usa para el render final; si no, se cae al texto acumulado.
        datosInforme.informe = ev.informe || mdCrudo;   // markdown final para el PDF
        datosSecciones = ev.datos_secciones || {};      // datos fuente por sección (trazabilidad)
        sesionId = ev.sesion_id || '';                  // habilita regenerar secciones
        // Render por secciones (contenedores independientes): permite, en modo
        // admin, regenerar UNA sección in situ sin tocar el resto.
        renderInformePorSecciones(datosInforme.informe);
        if (typeof actualizarBotonesAdmin === 'function') actualizarBotonesAdmin();
        accionesFin.style.display = 'flex';
      } else if (ev.tipo === 'error') {
        volverConError(ev.mensaje || 'Ocurrió un error.');
      }
    }

    // =====================================================================
    //  MODO ADMINISTRADOR
    // =====================================================================
    const modal = document.getElementById('modalAdmin');
    const modalTitulo = document.getElementById('modalTitulo');
    const modalCuerpo = document.getElementById('modalCuerpo');

    function esAdmin() { return !!adminToken; }
    function mostrarModal() { modal.classList.remove('oculto'); }
    function cerrarModal() { modal.classList.add('oculto'); }

    // Llamada genérica a la API admin (agrega el header del token).
    async function apiAdmin(url, metodo, body) {
      const opts = { method: metodo, headers: { 'X-Admin-Token': adminToken } };
      if (body !== undefined) { opts.headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body); }
      let resp;
      try { resp = await fetch(url, opts); }
      catch (e) { return { ok: false, error: 'Sin conexión con el servidor.' }; }
      if (resp.status === 401) return { __noauth: true, ok: false, error: 'No autorizado.' };
      let data = {}; try { data = await resp.json(); } catch (e) {}
      if (metodo === 'GET') { data.ok = resp.ok; return data; }
      if (!resp.ok) return { ok: false, error: data.error || ('Error ' + resp.status) };
      return Object.assign({ ok: true }, data);
    }

    // --- Login / logout ---
    function toggleAdmin() { if (esAdmin()) salirAdmin(); else abrirLogin(); }

    function abrirLogin() {
      modalTitulo.textContent = 'Modo administrador';
      modalCuerpo.innerHTML =
        '<p class="editor-ayuda">Ingresá la contraseña de administrador para ver y editar los prompts del sistema.</p>' +
        '<input type="password" id="admPass" class="adm-pass" placeholder="Contraseña" autocomplete="current-password">' +
        '<div class="p-acc"><button id="btnLogin" type="button">Ingresar</button><span class="p-msg" id="loginMsg"></span></div>';
      mostrarModal();
      const inp = document.getElementById('admPass');
      inp.focus();
      document.getElementById('btnLogin').addEventListener('click', hacerLogin);
      inp.addEventListener('keydown', e => { if (e.key === 'Enter') hacerLogin(); });
    }

    async function hacerLogin() {
      const inp = document.getElementById('admPass');
      const msg = document.getElementById('loginMsg');
      let resp, data = {};
      try {
        resp = await fetch('/admin/login', { method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ password: inp.value }) });
        data = await resp.json().catch(() => ({}));
      } catch (e) { msg.textContent = 'Sin conexión con el servidor.'; msg.className = 'p-msg err'; return; }
      if (!resp.ok || !data.token) { msg.textContent = data.error || 'No se pudo iniciar sesión.'; msg.className = 'p-msg err'; return; }
      adminToken = data.token;
      try { localStorage.setItem('admin_token', adminToken); } catch (e) {}
      const ok = await activarAdmin();
      if (ok) cerrarModal();
    }

    // Muestra/oculta los botones admin (Prompts siempre en admin; Datos solo si
    // hay un informe generado en pantalla).
    function actualizarBotonesAdmin() {
      const admin = esAdmin() && !!adminConfig;
      document.getElementById('btnPrompts').style.display = admin ? '' : 'none';
      document.getElementById('btnDatos').style.display = (admin && sesionId) ? '' : 'none';
    }

    async function activarAdmin() {
      const cfg = await apiAdmin('/admin/config', 'GET');
      if (!cfg || cfg.__noauth || !cfg.prompts) { salirAdmin(); return false; }
      adminConfig = cfg;
      document.body.setAttribute('data-admin', '1');
      document.getElementById('btnAdmin').textContent = '🔓 Salir admin';
      actualizarBotonesAdmin();
      // Si ya hay un informe en pantalla, re-render para mostrar los botones de edición.
      if (secModelo.length) renderInformePorSecciones(datosInforme.informe);
      return true;
    }

    function salirAdmin() {
      adminToken = ''; adminConfig = null;
      try { localStorage.removeItem('admin_token'); } catch (e) {}
      document.body.removeAttribute('data-admin');
      document.getElementById('btnAdmin').textContent = '🔒 Admin';
      actualizarBotonesAdmin();
      if (secModelo.length) renderInformePorSecciones(datosInforme.informe);
    }

    // Visor global: todos los datos fuente del informe (solo lectura).
    async function abrirDatosGlobal() {
      if (!sesionId) { mostrarAviso('Generá un informe primero para ver sus datos.'); return; }
      modalTitulo.textContent = 'Datos del informe';
      modalCuerpo.innerHTML = '<p class="editor-ayuda">Cargando datos…</p>';
      mostrarModal();
      const d = await apiAdmin('/admin/datos/' + encodeURIComponent(sesionId), 'GET');
      if (!d || !d.ok || d.__noauth) {
        modalCuerpo.innerHTML = '<p class="p-msg err">' + escapar((d && d.error) || 'No se pudieron cargar los datos.') + '</p>';
        return;
      }
      modalCuerpo.innerHTML =
        '<p class="editor-ayuda">Datos fuente con los que se generó el informe (cliente, historial, ' +
        'scoring, BCRA y contexto cualitativo). Es lo que recibe el modelo. Solo lectura.</p>' +
        '<pre class="datos-json">' + escapar(d.json) + '</pre>';
    }

    // --- Render del informe por secciones (contenedores independientes) ---
    function parseSecciones(md) {
      const lineas = (md || '').split('\\n');
      let encab = [], secs = [], actual = null;
      for (const ln of lineas) {
        if (ln.slice(0, 3) === '## ') {
          if (actual) secs.push(actual);
          actual = { titulo: ln.slice(3).trim(), cuerpo: [] };
        } else if (actual) { actual.cuerpo.push(ln); }
        else { encab.push(ln); }
      }
      if (actual) secs.push(actual);
      return {
        encabezado: encab.join('\\n').trim(),
        secciones: secs.map(s => ({ titulo: s.titulo, cuerpo: s.cuerpo.join('\\n').trim() })),
      };
    }

    function idDeTitulo(titulo) {
      if (adminConfig && adminConfig.titulos && adminConfig.titulos[titulo]) return adminConfig.titulos[titulo];
      return '';
    }

    // Colapsable con el resumen de datos fuente de la sección (trazabilidad).
    // Deja ver a cualquier lector la relación entre lo que dice el informe y los
    // datos con los que se generó, sin exponer el JSON crudo.
    function bloqueDatosHTML(titulo) {
      const hechos = datosSecciones[titulo];
      if (!hechos || !hechos.length) return '';
      let filas = '';
      for (const h of hechos) {
        const cls = h.alerta ? ' class="dato-alerta"' : '';
        filas += '<tr' + cls + '><th>' + escapar(h.etiqueta) + '</th><td>' + escapar(h.valor) + '</td></tr>';
      }
      return '<details class="sec-datos"><summary>📊 Datos con los que se generó esta sección</summary>' +
             '<table class="sec-datos-tabla">' + filas + '</table></details>';
    }

    function bloqueSeccionHTML(s, idx) {
      const sid = idDeTitulo(s.titulo);
      const editable = esAdmin() && sesionId && sid;
      const btn = editable
        ? '<button class="sec-editar" data-sid="' + sid + '" data-idx="' + idx +
          '" title="Ver/editar los prompts que afectan a esta sección">✎ prompts</button>'
        : '';
      return '<div class="sec" data-sec-idx="' + idx + '" data-sec-id="' + sid + '">' + btn +
             '<div class="sec-cuerpo">' + mdToHtml('## ' + s.titulo + '\\n' + s.cuerpo) + '</div>' +
             bloqueDatosHTML(s.titulo) + '</div>';
    }

    function renderInformePorSecciones(md) {
      const p = parseSecciones(md);
      encabezadoMd = p.encabezado;
      secModelo = p.secciones;
      let h = '';
      if (encabezadoMd) h += '<div class="sec sec-encab">' + mdToHtml(encabezadoMd) + '</div>';
      secModelo.forEach((s, idx) => { h += bloqueSeccionHTML(s, idx); });
      elTexto.innerHTML = h;
    }

    // Reconstruye datosInforme.informe desde el modelo de secciones (para el PDF).
    function rebuildInforme() {
      const partes = [];
      if (encabezadoMd) partes.push(encabezadoMd);
      for (const s of secModelo) partes.push('## ' + s.titulo + '\\n' + s.cuerpo);
      datosInforme.informe = partes.join('\\n\\n');
    }

    // --- Editor de prompts (global o por sección) ---
    async function cargarPrompt(name) {
      const d = await apiAdmin('/admin/prompt/' + encodeURIComponent(name), 'GET');
      const contenido = (d && d.contenido) || '';
      promptOriginal[name] = contenido;
      return { name: name, titulo: (d && d.titulo) || name, contenido: contenido };
    }

    function metaPrompt(name) {
      return (adminConfig && adminConfig.prompts.find(p => p.name === name)) || {};
    }

    function itemPromptHTML(it) {
      const meta = metaPrompt(it.name);
      return '<div class="p-item">' +
        '<div class="p-tit">' + escapar(it.titulo) + '<span class="p-name">' + escapar(it.name) + '</span></div>' +
        (meta.descripcion ? '<div class="p-desc">' + escapar(meta.descripcion) + '</div>' : '') +
        '<textarea class="p-ta" data-name="' + escapar(it.name) + '" spellcheck="false">' + escapar(it.contenido) + '</textarea>' +
        '<div class="p-acc"><button type="button" class="secundario btn-guardar-prompt" data-name="' + escapar(it.name) +
        '">Guardar</button><span class="p-msg" id="msg-' + escapar(it.name) + '"></span></div>' +
        '</div>';
    }

    async function abrirEditor(titulo, nombres, regenSid, regenIdx, conGrupos) {
      modalTitulo.textContent = titulo;
      modalCuerpo.innerHTML = '<p class="editor-ayuda">Cargando prompts…</p>';
      mostrarModal();
      // En el editor de sección, además de los prompts traemos los datos que
      // recibe esa sección (para revisarlos junto al texto y decidir mejor).
      const [items, datos] = await Promise.all([
        Promise.all(nombres.map(cargarPrompt)),
        regenSid ? apiAdmin('/admin/seccion/datos', 'POST', { sesion_id: sesionId, section_id: regenSid })
                 : Promise.resolve(null),
      ]);
      let h = '';
      if (regenSid) h += '<p class="editor-ayuda">Revisá el texto y, si hace falta, mejorá el prompt de esta sección. ' +
        'Si hacés cambios, al regenerar se reescribe SOLO esta sección con los datos ya cargados (no se vuelve a consultar la base). ' +
        'Los cambios valen también para los próximos informes. Las reglas comunes a todas las secciones se editan desde el botón «Prompts».</p>';
      if (regenSid) {
        const cuerpo = (datos && datos.ok && datos.json)
          ? escapar(datos.json)
          : escapar((datos && datos.error) || 'No se pudieron cargar los datos.');
        h += '<details class="datos-sec"><summary>📊 Ver los datos que recibe esta sección</summary>' +
             '<pre class="datos-json">' + cuerpo + '</pre></details>';
      }
      let grupoActual = '';
      for (const it of items) {
        if (conGrupos) {
          const g = metaPrompt(it.name).grupo || '';
          if (g && g !== grupoActual) { h += '<h4 class="p-grupo">' + escapar(g) + '</h4>'; grupoActual = g; }
        }
        h += itemPromptHTML(it);
      }
      if (regenSid) h += '<div class="regen-bar"><button type="button" id="btnRegen" class="btn-regen">' +
        'Guardar y regenerar sección</button><span class="p-msg" id="regenMsg"></span></div>';
      modalCuerpo.innerHTML = h;
      modalCuerpo.querySelectorAll('.btn-guardar-prompt').forEach(b => {
        b.addEventListener('click', () => guardarPrompt(b.dataset.name));
      });
      if (regenSid) document.getElementById('btnRegen').addEventListener('click',
        () => guardarYRegenerar(nombres, regenSid, regenIdx));
    }

    function abrirPromptsGlobal() {
      if (!adminConfig) return;
      abrirEditor('Prompts del sistema', adminConfig.prompts.map(p => p.name), null, null, true);
    }

    function abrirPromptsSeccion(sid, idx) {
      if (!adminConfig || !adminConfig.secciones[sid]) return;
      const sec = adminConfig.secciones[sid];
      // Editor ENFOCADO: solo el/los prompt(s) propios de esta sección. Las
      // reglas comunes (compartidas por todas) se editan en el panel "Prompts".
      const nombres = (sec.propios && sec.propios.length) ? sec.propios : sec.prompts;
      abrirEditor('Prompt de: ' + sec.titulo, nombres, sid, idx, false);
    }

    async function guardarPrompt(name) {
      const ta = modalCuerpo.querySelector('.p-ta[data-name="' + name + '"]');
      const msg = document.getElementById('msg-' + name);
      if (!ta) return false;
      const r = await apiAdmin('/admin/prompt', 'POST', { name: name, contenido: ta.value });
      if (r.ok) {
        promptOriginal[name] = ta.value;
        if (msg) { msg.textContent = 'Guardado ✓'; msg.className = 'p-msg ok'; }
        return true;
      }
      if (msg) { msg.textContent = r.error || 'No se pudo guardar.'; msg.className = 'p-msg err'; }
      return false;
    }

    async function guardarYRegenerar(nombres, sid, idx) {
      const btn = document.getElementById('btnRegen');
      const rmsg = document.getElementById('regenMsg');
      btn.disabled = true; rmsg.textContent = 'Guardando cambios…'; rmsg.className = 'p-msg';
      for (const n of nombres) {
        const ta = modalCuerpo.querySelector('.p-ta[data-name="' + n + '"]');
        if (ta && ta.value !== promptOriginal[n]) {
          const ok = await guardarPrompt(n);
          if (!ok) { rmsg.textContent = 'Corregí el prompt marcado en rojo y reintentá.'; rmsg.className = 'p-msg err'; btn.disabled = false; return; }
        }
      }
      cerrarModal();
      await regenerarSeccion(sid, idx);
    }

    // --- Regeneración en vivo de UNA sección (SSE) ---
    async function regenerarSeccion(sid, idx) {
      const cont = elTexto.querySelector('.sec[data-sec-idx="' + idx + '"]');
      const cuerpoEl = cont ? cont.querySelector('.sec-cuerpo') : null;
      if (!cont || !cuerpoEl || !secModelo[idx]) return;
      const titulo = secModelo[idx].titulo;
      cont.classList.add('regenerando');
      let acc = '', raf = null, nuevoCuerpo = '';
      function pintar() {
        if (raf) return;
        raf = requestAnimationFrame(() => {
          raf = null;
          cuerpoEl.innerHTML = mdToHtml('## ' + titulo + '\\n' + acc) + '<span class="caret"></span>';
        });
      }
      let resp;
      try {
        resp = await fetch('/admin/seccion/regenerar', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'X-Admin-Token': adminToken },
          body: JSON.stringify({ sesion_id: sesionId, section_id: sid }),
        });
      } catch (e) { cont.classList.remove('regenerando'); mostrarAviso('No se pudo conectar para regenerar.'); return; }
      if (!resp.ok) { cont.classList.remove('regenerando'); mostrarAviso('No autorizado o error al regenerar la sección.'); return; }

      const reader = resp.body.getReader(); const dec = new TextDecoder(); let buf = '';
      while (true) {
        const r = await reader.read(); if (r.done) break;
        buf += dec.decode(r.value, { stream: true });
        let i;
        while ((i = buf.indexOf('\\n\\n')) >= 0) {
          const bloque = buf.slice(0, i); buf = buf.slice(i + 2);
          const linea = bloque.split('\\n').find(l => l.slice(0, 5) === 'data:');
          if (!linea) continue;
          let ev; try { ev = JSON.parse(linea.slice(5).trim()); } catch (e) { continue; }
          if (ev.tipo === 'token') { acc += ev.texto; pintar(); }
          else if (ev.tipo === 'fin_seccion') { nuevoCuerpo = ev.texto || acc; }
          else if (ev.tipo === 'error') { if (raf) cancelAnimationFrame(raf); cont.classList.remove('regenerando'); mostrarAviso(ev.mensaje || 'Error al regenerar.'); return; }
        }
      }
      if (raf) cancelAnimationFrame(raf);
      const cuerpoFinal = (nuevoCuerpo || acc).trim();
      secModelo[idx].cuerpo = cuerpoFinal;
      cuerpoEl.innerHTML = mdToHtml('## ' + titulo + '\\n' + cuerpoFinal);
      cont.classList.remove('regenerando');
      rebuildInforme();   // el PDF descargable refleja la sección regenerada
    }

    // --- Inicialización del modo admin ---
    (function inicializarAdmin() {
      const btnAdmin = document.getElementById('btnAdmin');
      if (!ADMIN_HABILITADO) {
        btnAdmin.style.display = 'none';
        document.getElementById('btnPrompts').style.display = 'none';
        return;
      }
      // Delegación: botón "✎ prompts" de cada sección.
      elTexto.addEventListener('click', function (e) {
        const b = e.target.closest('.sec-editar');
        if (b) abrirPromptsSeccion(b.dataset.sid, Number(b.dataset.idx));
      });
      // Cerrar el modal al hacer click en el fondo.
      modal.addEventListener('click', e => { if (e.target === modal) cerrarModal(); });
      // Restaurar sesión admin previa (token en localStorage).
      let t = ''; try { t = localStorage.getItem('admin_token') || ''; } catch (e) {}
      if (t) { adminToken = t; activarAdmin(); }
    })();
  </script>
"""

# ===========================================================================
#  Rutas
# ===========================================================================

@asynccontextmanager
async def lifespan(app):
    # Al iniciar: refrescar las políticas de la entidad desde la carpeta.
    reindexar_politicas_startup()
    yield

app = FastAPI(title="Generador de informes crediticios", lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
def formulario():
    contenido = (CONTENIDO
                 .replace("__EMPRESA__", html.escape(EMPRESA))
                 .replace("__ADMIN_HABILITADO__", "1" if ADMIN_PASSWORD else "0"))
    return pagina("Generador de informes — " + EMPRESA, contenido)

def _sse(ev: dict) -> str:
    return f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

@app.post("/informe/stream")
def informe_stream(consulta: str = Form(...),
                   docs: list[UploadFile] = File(default=[])):
    # Leer los archivos ahora (el generador corre fuera del contexto del request).
    adjuntos = []
    for f in (docs or []):
        try:
            contenido = f.file.read()
            texto = extraer_texto_archivo(f.filename, contenido)
            if texto.strip():
                adjuntos.append((f.filename, texto))
        except Exception:
            traceback.print_exc()

    def eventos():
        texto = (consulta or "").strip()
        if not texto:
            yield _sse({"tipo": "error", "mensaje": "Escribí la consulta."})
            return

        numero = nuevo_numero_informe()
        try:
            pipeline = get_pipeline(PROVEEDOR_LLM)

            # 1) Extracción + clasificación de la consulta
            yield _sse({"tipo": "etapa", "clave": "extraccion"})
            datos  = pipeline.extraer_datos(texto)

            # La consulta debe ser una solicitud crediticia
            if not datos.get("es_credito", True):
                yield _sse({"tipo": "error",
                            "mensaje": "La consulta no parece una solicitud crediticia. "
                                       "Describí un pedido de crédito o financiación de un "
                                       "asociado e incluí su CUIT."})
                return

            cuit   = normalizar_cuit(datos.get("cuit", ""))
            razon  = datos.get("razon_social", "")
            motivo = datos.get("tipo_decision", "") or texto

            # Anti-alucinación: el CUIT extraído debe aparecer en la consulta.
            if cuit and cuit not in normalizar_cuit(texto):
                cuit = ""
            if not cuit_valido(cuit):
                yield _sse({"tipo": "error",
                            "mensaje": "No identifiqué un CUIT válido en la consulta. "
                                       "Asegurate de incluir los 11 dígitos del CUIT."})
                return

            # 2) Indexar documentos adjuntos para este cliente (si hay).
            #    Con embeddings en GPU (MPS) esto es rápido y además los persiste.
            if adjuntos:
                yield _sse({"tipo": "etapa", "clave": "documentos"})
                try:
                    pipeline.indexar_documentos_cliente(cuit, adjuntos)
                except Exception:
                    traceback.print_exc()

            # 3) Flujo normal con los datos extraídos
            for ev in pipeline.generar_stream(cuit, motivo, razon):
                if ev.get("tipo") == "meta":
                    ev = {**ev, "numero": numero, "empresa": EMPRESA,
                          "cuit": cuit, "razon": razon, "motivo": motivo}
                yield _sse(ev)
        except Exception as exc:
            traceback.print_exc()   # traceback completo en la consola del server
            yield _sse({"tipo": "error",
                        "mensaje": str(exc).strip() or "No se pudo generar el informe."})

    return StreamingResponse(
        eventos(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ===========================================================================
#  API: generar el informe y devolver el PDF directamente (app-to-app)
# ===========================================================================
# A diferencia de la web (que extrae los datos de una consulta en lenguaje
# natural y transmite el informe por SSE), este endpoint recibe los campos ya
# estructurados —CUIT, nombre y objetivo del crédito—, genera el informe y
# devuelve el PDF listo para descargar. La aplicación web sigue igual.

class InformePDFRequest(BaseModel):
    cuit: str
    nombre: str = ""
    objetivo: str


def _generar_informe(pipeline, cuit: str, motivo: str, razon: str):
    """Corre el pipeline de generación y devuelve (informe_markdown, meta).

    Reutiliza generar_stream() —igual que la web— para cubrir también el caso
    del solicitante sin datos internos. Acumula los tokens y, al cerrar, usa el
    documento en orden de lectura si vino ('fin'.informe) o el acumulado."""
    md_acc, meta, informe = [], {}, ""
    for ev in pipeline.generar_stream(cuit, motivo, razon):
        tipo = ev.get("tipo")
        if tipo == "token":
            md_acc.append(ev.get("texto", ""))
        elif tipo == "meta":
            meta = ev
        elif tipo == "fin":
            informe = ev.get("informe") or "".join(md_acc)
    return (informe or "".join(md_acc)), meta


@app.post("/api/informe/pdf")
def api_informe_pdf(req: InformePDFRequest):
    """Genera el informe crediticio y lo devuelve como PDF descargable.

    Body JSON: {"cuit": "...", "nombre": "...", "objetivo": "..."}
    Respuesta: application/pdf (attachment) o JSON con {"error": ...} si falla.
    """
    cuit = normalizar_cuit(req.cuit)
    if not cuit_valido(cuit):
        return JSONResponse(status_code=422, content={
            "error": "CUIT inválido: deben ser 11 dígitos con dígito verificador correcto."})
    objetivo = (req.objetivo or "").strip()
    if not objetivo:
        return JSONResponse(status_code=422, content={
            "error": "Falta el objetivo del crédito."})
    razon = (req.nombre or "").strip()

    numero = nuevo_numero_informe()
    try:
        pipeline = get_pipeline(PROVEEDOR_LLM)
        informe_md, meta = _generar_informe(pipeline, cuit, objetivo, razon)
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={
            "error": str(exc).strip() or "No se pudo generar el informe."})

    from pdf_export import informe_a_pdf
    try:
        pdf = informe_a_pdf(
            informe_md,
            empresa=EMPRESA, numero=numero,
            fecha=datetime.now().strftime("%d/%m/%Y"),
            modelo=meta.get("modelo", ""),
            cuit=cuit, razon=razon, motivo=objetivo,
            score=meta.get("score"), nivel=meta.get("nivel"),
            bcra=meta.get("bcra"), encontrado=meta.get("encontrado", True),
        )
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={
            "error": str(exc).strip() or "No se pudo generar el PDF."})

    filename = f"informe-{cuit}-{numero}.pdf"
    return Response(content=pdf, media_type="application/pdf", headers={
        "Content-Disposition": f'attachment; filename="{filename}"'})


class InformePDFRenderRequest(BaseModel):
    """Render del informe YA generado (lo usa el botón 'Descargar PDF' de la web).
    No corre el pipeline: rinde el mismo Markdown que se ve en pantalla, para que
    el PDF sea idéntico al informe mostrado y al del endpoint app-to-app."""
    informe: str
    numero: str = ""
    cuit: str = ""
    nombre: str = ""
    motivo: str = ""
    modelo: str = ""
    score: Any = None
    nivel: str = ""
    bcra: Any = None
    encontrado: bool = True


@app.post("/api/informe/pdf-render")
def api_informe_pdf_render(req: InformePDFRenderRequest):
    """Renderiza a PDF un informe ya producido (Markdown + datos del membrete).
    Comparte el renderizador con /api/informe/pdf -> ambos PDF salen iguales."""
    if not (req.informe or "").strip():
        return JSONResponse(status_code=422, content={"error": "Falta el informe a renderizar."})

    cuit = normalizar_cuit(req.cuit)
    from pdf_export import informe_a_pdf
    try:
        pdf = informe_a_pdf(
            req.informe,
            empresa=EMPRESA, numero=req.numero,
            fecha=datetime.now().strftime("%d/%m/%Y"),
            modelo=req.modelo,
            cuit=cuit, razon=req.nombre, motivo=req.motivo,
            score=req.score, nivel=req.nivel,
            bcra=req.bcra, encontrado=req.encontrado,
        )
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={
            "error": str(exc).strip() or "No se pudo generar el PDF."})

    filename = f"informe-{cuit or 's/d'}-{req.numero or 's-n'}.pdf"
    return Response(content=pdf, media_type="application/pdf", headers={
        "Content-Disposition": f'attachment; filename="{filename}"'})


# ===========================================================================
#  API: modo administrador — ver/editar prompts y regenerar una sección
# ===========================================================================

class LoginRequest(BaseModel):
    password: str = ""


class PromptGuardarRequest(BaseModel):
    name: str
    contenido: str


class RegenerarSeccionRequest(BaseModel):
    sesion_id: str
    section_id: str


class SeccionDatosRequest(BaseModel):
    sesion_id: str
    section_id: str


@app.post("/admin/login")
def admin_login(req: LoginRequest):
    """Valida la contraseña admin y devuelve el token de sesión."""
    if not ADMIN_PASSWORD:
        return JSONResponse(status_code=403, content={
            "error": "El modo administrador no está habilitado (falta ADMIN_PASSWORD en .env)."})
    if not req.password or not hmac.compare_digest(req.password, ADMIN_PASSWORD):
        return JSONResponse(status_code=401, content={"error": "Contraseña incorrecta."})
    return {"token": _admin_token()}


@app.get("/admin/config")
def admin_config(x_admin_token: str = Header(default="")):
    """Catálogo de prompts + mapa de secciones para la UI del editor."""
    _requerir_admin(x_admin_token)
    from prompts import catalogo, PROMPTS_POR_SECCION, PROMPTS_COMPARTIDOS
    from llm.report_sections import especificacion_seccion
    secciones, titulos = {}, {}
    for sid, nombres in PROMPTS_POR_SECCION.items():
        # el título no depende de los valores del contexto, solo de sus claves
        titulo, _p, _c = especificacion_seccion(sid, {}, {})
        # 'propios' = prompts específicos de esta sección (los que ve el editor
        # contextual); 'compartidos' = reglas comunes (se editan en el panel global).
        propios = [n for n in nombres if n not in PROMPTS_COMPARTIDOS]
        compartidos = [n for n in nombres if n in PROMPTS_COMPARTIDOS]
        secciones[sid] = {"titulo": titulo, "prompts": nombres,
                          "propios": propios, "compartidos": compartidos}
        titulos[titulo] = sid
    return {"prompts": catalogo(), "secciones": secciones, "titulos": titulos}


@app.get("/admin/prompt/{name}")
def admin_get_prompt(name: str, x_admin_token: str = Header(default="")):
    """Devuelve el contenido actual de un prompt del catálogo."""
    _requerir_admin(x_admin_token)
    from prompts import CATALOGO, load
    if name not in CATALOGO:
        return JSONResponse(status_code=404, content={"error": "Prompt desconocido."})
    return {"name": name, "titulo": CATALOGO[name]["titulo"], "contenido": load(name)}


@app.post("/admin/prompt")
def admin_guardar_prompt(req: PromptGuardarRequest, x_admin_token: str = Header(default="")):
    """Guarda un prompt editado (validando nombre y variables `{...}`)."""
    _requerir_admin(x_admin_token)
    from prompts import guardar
    try:
        guardar(req.name, req.contenido)
    except ValueError as exc:
        return JSONResponse(status_code=422, content={"error": str(exc)})
    return {"ok": True}


@app.get("/admin/datos/{sesion_id}")
def admin_datos(sesion_id: str, x_admin_token: str = Header(default="")):
    """Devuelve, como JSON legible, TODOS los datos que generaron el informe
    (contexto: cliente, historial, scoring, BCRA, RAG). Para que el admin vea la
    fuente y decida mejor al editar los prompts."""
    _requerir_admin(x_admin_token)
    try:
        ctx = get_pipeline(PROVEEDOR_LLM).contexto_sesion(sesion_id)
    except KeyError as exc:
        return JSONResponse(status_code=404, content={
            "error": exc.args[0] if exc.args else "Sesión no encontrada."})
    from llm.report_sections import _j
    return {"json": _j(ctx)}


@app.post("/admin/seccion/datos")
def admin_seccion_datos(req: SeccionDatosRequest, x_admin_token: str = Header(default="")):
    """Devuelve los datos EXACTOS que recibe una sección (el mismo JSON que se le
    pasa al modelo), para revisarlos junto al prompt de esa sección."""
    _requerir_admin(x_admin_token)
    try:
        titulo, ctx = get_pipeline(PROVEEDOR_LLM).datos_seccion(req.sesion_id, req.section_id)
    except KeyError as exc:
        return JSONResponse(status_code=404, content={
            "error": exc.args[0] if exc.args else "Sesión no encontrada."})
    except ValueError as exc:
        return JSONResponse(status_code=422, content={"error": str(exc)})
    from llm.report_sections import _j
    return {"titulo": titulo, "json": _j(ctx)}


@app.post("/admin/seccion/regenerar")
def admin_regenerar_seccion(req: RegenerarSeccionRequest, x_admin_token: str = Header(default="")):
    """Regenera SOLO una sección de un informe ya generado (SSE), usando el
    contexto retenido en memoria y el prompt (posiblemente editado) del .txt."""
    _requerir_admin(x_admin_token)

    def eventos():
        try:
            pipeline = get_pipeline(PROVEEDOR_LLM)
            for ev in pipeline.regenerar_seccion_stream(req.sesion_id, req.section_id):
                yield _sse(ev)
        except KeyError as exc:
            # str(KeyError) viene con comillas; usar exc.args[0] para el mensaje limpio.
            msg = exc.args[0] if exc.args else "Sesión no encontrada."
            yield _sse({"tipo": "error", "mensaje": str(msg)})
        except Exception as exc:
            traceback.print_exc()
            yield _sse({"tipo": "error",
                        "mensaje": str(exc).strip() or "No se pudo regenerar la sección."})

    return StreamingResponse(
        eventos(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)