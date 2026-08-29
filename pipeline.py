from dotenv import load_dotenv
load_dotenv()
print("[DBG] dotenv loaded", flush=True)

import copy
import os
import uuid
from collections import OrderedDict
from datetime import datetime
from sqlalchemy import text
print("[DBG] core libs imported", flush=True)

from external.bcra_normalizer import formato_pesos

from db.sqlserver_connection import get_sqlserver_session
from db.postgres_connection import get_postgres_session
from db.client_repository import ClientRepository, TABLAS_SQLSERVER
from ml.scoring_model import CreditScoringModel
from prompts import load as load_prompt   # textos de prompts en prompts/*.txt
print("[DBG] project modules imported", flush=True)


def descripcion_modelo_scoring(path: str = "models/model_card.json") -> str:
    """Identifica el modelo que produjo el score, para el log auditable.

    Se lee del model card, no de un literal: antes decía "XGBoost v1.0", que no
    correspondía a ningún modelo entrenado y no permitía saber cuál se había
    usado. Si un informe se cuestiona, esto es lo que dice qué lo generó.
    """
    import json
    try:
        with open(path, "r", encoding="utf-8") as f:
            card = json.load(f)
    except (OSError, ValueError):
        return "desconocido (falta models/model_card.json)"

    return (f"{card.get('modelo', 'desconocido')} "
            f"(xgboost {card.get('xgboost_version', '?')}, "
            f"entrenado {card.get('generado', '?')}, "
            f"target {card.get('target', {}).get('horizonte_meses', '?')}m)")


class ClienteNoEncontrado(Exception):
    """Se levanta cuando un cliente/CUIT no existe en la Base Financiera.

    Para una detección limpia, conviene que el repositorio levante ESTA
    excepción. Ejemplo en db/client_repository.py:
        from pipeline import ClienteNoEncontrado   # o un módulo común
        raise ClienteNoEncontrado(f"Cliente {cliente_id} no encontrado en Base Financiera")
    """
    pass


def _es_no_encontrado(exc: Exception) -> bool:
    """True si la excepción representa 'cliente no encontrado'.
    Reconoce ClienteNoEncontrado y, como respaldo, el mensaje típico."""
    if isinstance(exc, ClienteNoEncontrado):
        return True
    msg = str(exc).lower()
    return ("no encontrado" in msg) or ("no existe" in msg) or ("not found" in msg)


# Campos monetarios de cada sección del contexto. Se pre-formatean a pesos
# argentinos (string) ANTES de pasarlos al LLM para que los transcriba tal cual
# y no reinterprete el punto decimal (ej. 89.492781 -> "$89,49", nunca $89.492,78).
_CAMPOS_MONTO = {
    "datos_cliente":       ["ingreso_neto_declarado"],
    "historial_prestamos": ["monto_original", "saldo_capital_actual"],
    "saldos_cuentas":      ["saldo_promedio", "saldo_minimo", "saldo_maximo", "saldo_ultimo"],
    "gestiones_mora":      ["monto_comprometido"],
}


def _fmt_campos(d: dict, claves) -> None:
    """Reemplaza in-place los campos numéricos de `d` por su formato en pesos."""
    for k in claves:
        v = d.get(k)
        if v is None:
            continue
        try:
            d[k] = formato_pesos(float(v))
        except (TypeError, ValueError):
            pass            # no es numérico: se deja como está


# ===========================================================================
#  SESIONES EN MEMORIA (para regenerar UNA sección desde el modo admin)
# ===========================================================================
# Tras generar un informe se retiene su contexto (ya formateado) y el cuerpo de
# cada sección, indexados por un sesion_id. Así el admin puede editar un prompt y
# regenerar SOLO esa sección sin re-consultar SQL Server / ML / RAG / BCRA.
# Acotado a las últimas _SESIONES_MAX; vive en RAM y se limpia al reiniciar.
_SESIONES: "OrderedDict[str, dict]" = OrderedDict()
_SESIONES_MAX = 20


def _guardar_sesion(sesion_id: str, contexto_fmt: dict, bodies: dict) -> None:
    _SESIONES[sesion_id] = {"contexto": contexto_fmt, "bodies": dict(bodies or {})}
    _SESIONES.move_to_end(sesion_id)
    while len(_SESIONES) > _SESIONES_MAX:
        _SESIONES.popitem(last=False)          # descarta la sesión más vieja


# ---------------------------------------------------------------------------
#  Hechos derivados: identificadores y aritmética de fechas
#
# El LLM no debe DERIVAR nada verificable, sólo narrar lo que se le da. Los dos
# errores más frecuentes que marcaba el juez venían de pedirle exactamente eso:
#
#   - Antigüedades y "hace cuánto" mal calculados ("aproximadamente 27 años"
#     cuando eran 28,6; "hace 8 meses" cuando eran 20). Los LLM son malos en
#     aritmética de fechas y no hay razón para pedírsela: la calculamos acá.
#   - CUIT "inventado": el modelo tomaba el DNI y lo formateaba como CUIT
#     (20-04658409-1). El CUIT ES la clave de la consulta; simplemente no se lo
#     estábamos pasando ya formateado.
#
# Mismo criterio que con los importes: el dato llega listo para copiar.
# ---------------------------------------------------------------------------

def _fmt_cuit(cuit: str) -> str:
    """20123456789 -> 20-12345678-9. Si no tiene 11 dígitos, se devuelve igual."""
    d = "".join(ch for ch in str(cuit or "") if ch.isdigit())
    return f"{d[:2]}-{d[2:10]}-{d[10]}" if len(d) == 11 else str(cuit or "")


def _fmt_fecha(v) -> str | None:
    """ISO / datetime -> DD/MM/AAAA."""
    d = _a_fecha(v)
    return d.strftime("%d/%m/%Y") if d else None


def _a_fecha(v):
    if isinstance(v, datetime):
        return v.date()
    if hasattr(v, "year") and not isinstance(v, str):
        return v
    try:
        return datetime.fromisoformat(str(v).strip().replace(" ", "T")).date()
    except (TypeError, ValueError):
        return None


def _antiguedad(desde, hasta) -> tuple[int, str] | tuple[None, None]:
    """(meses, 'X años y Y meses') entre dos fechas. Redondeo a mes cumplido."""
    a, b = _a_fecha(desde), _a_fecha(hasta)
    if not a or not b or b < a:
        return None, None
    meses = (b.year - a.year) * 12 + (b.month - a.month) - (1 if b.day < a.day else 0)
    meses = max(meses, 0)
    anios, resto = divmod(meses, 12)
    if anios and resto:
        txt = f"{anios} años y {resto} meses"
    elif anios:
        txt = f"{anios} años"
    else:
        txt = f"{resto} meses"
    return meses, txt


def _resumen_cartera(prestamos) -> dict | None:
    """Agrega la cartera de préstamos en hechos que el LLM NO debe derivar solo.

    El juez detectó que el modelo contaba mal los préstamos, confundía el período
    (año del último otorgamiento) e inventaba desgloses por oficial de cuenta. Nada
    de eso hay que pedírselo: se calcula acá, sobre los montos CRUDOS (antes de
    formatear), y el saldo vivo total se formatea a pesos en el momento. Igual
    criterio que con fechas e importes: el dato llega listo para copiar.
    """
    items = [p for p in (prestamos or []) if isinstance(p, dict)]
    if not items:
        return None
    from collections import Counter
    estados = Counter((str(p.get("estado") or "SIN ESTADO").upper()) for p in items)
    vigentes = [p for p in items if str(p.get("estado") or "").upper() == "VIGENTE"]

    def _suma(campo, filas):
        t = 0.0
        for p in filas:
            try:
                t += float(p.get(campo) or 0)
            except (TypeError, ValueError):
                pass
        return t

    fechas = [f for f in (_a_fecha(p.get("fecha_otorgamiento")) for p in items) if f]
    resumen = {
        "total":                len(items),
        "por_estado":           dict(estados),
        "vigentes":             len(vigentes),
        "saldo_vivo_total":     formato_pesos(_suma("saldo_capital_actual", vigentes)),
        "monto_original_total": formato_pesos(_suma("monto_original", items)),
        "periodo_desde_texto":  min(fechas).strftime("%d/%m/%Y") if fechas else None,
        "periodo_hasta_texto":  max(fechas).strftime("%d/%m/%Y") if fechas else None,
    }
    # Los campos sueltos no alcanzaron: el modelo los recibía y aun así recomponía
    # mal los agregados en el Resumen y la Recomendación. `frase` es esa misma
    # información YA REDACTADA, para que sólo tenga que transcribirla.
    from llm.plantillas import cartera_frase
    resumen["frase"] = cartera_frase(resumen)
    return resumen


def _derivar_hechos(ctx: dict) -> dict:
    """Agrega al contexto (ya copiado) los hechos que el LLM NO debe calcular."""
    hoy = ctx.get("fecha_informe")

    ctx["cuit"] = _fmt_cuit(ctx.get("cliente_id"))
    ctx["fecha_informe_texto"] = _fmt_fecha(hoy)

    datos = ctx.get("datos_cliente")
    if isinstance(datos, dict):
        ing = datos.get("fecha_ingreso_entidad")
        meses, txt = _antiguedad(ing, hoy)
        datos["cuit"] = ctx["cuit"]
        datos["fecha_ingreso_entidad_texto"] = _fmt_fecha(ing)
        datos["antiguedad_meses"] = meses
        datos["antiguedad_texto"] = txt

    for p in (ctx.get("historial_prestamos") or []):
        if not isinstance(p, dict):
            continue
        otorg = p.get("fecha_otorgamiento")
        meses, txt = _antiguedad(otorg, hoy)
        p["fecha_otorgamiento_texto"] = _fmt_fecha(otorg)
        p["fecha_vencimiento_texto"] = _fmt_fecha(p.get("fecha_vencimiento"))
        p["otorgado_hace_meses"] = meses
        p["otorgado_hace_texto"] = f"hace {txt}" if txt else None

    for g in (ctx.get("gestiones_mora") or []):
        if isinstance(g, dict):
            g["fecha_gestion_texto"] = _fmt_fecha(g.get("fecha_gestion"))
            g["fecha_compromiso_pago_texto"] = _fmt_fecha(g.get("fecha_compromiso_pago"))

    # Agregados de la cartera (conteo, estados, saldo vivo, período) ya resueltos,
    # para que la sección Historial no los derive a mano (los contaba mal).
    ctx["cartera_resumen"] = _resumen_cartera(ctx.get("historial_prestamos"))

    return ctx


def _fmt_montos_contexto(contexto: dict) -> dict:
    """Copia del contexto con los importes ya formateados en pesos argentinos y
    los hechos temporales / identificadores ya derivados.
    No muta el original: ese sigue siendo la fuente de verdad (validador/juez)."""
    ctx = _derivar_hechos(copy.deepcopy(contexto))

    datos = ctx.get("datos_cliente")
    if isinstance(datos, dict):
        _fmt_campos(datos, _CAMPOS_MONTO["datos_cliente"])

    for clave in ("historial_prestamos", "saldos_cuentas", "gestiones_mora"):
        for item in (ctx.get(clave) or []):
            if isinstance(item, dict):
                _fmt_campos(item, _CAMPOS_MONTO[clave])

    bcra = ctx.get("bcra")
    if isinstance(bcra, dict):
        _fmt_campos(bcra, ["deuda_total_sistema", "monto_cheques_rechazados",
                           "monto_cheques_impagos"])
        for e in (bcra.get("entidades") or []):
            if isinstance(e, dict):
                _fmt_campos(e, ["monto"])
        for c in (bcra.get("cheques_detalle") or []):
            if isinstance(c, dict):
                _fmt_campos(c, ["monto"])

    return ctx


class ReportPipeline:
    """
    Orquestador principal del sistema.
    SQL Server → features + scoring ML → pgvector RAG → LLM → validación → log.
    """

    def __init__(self, llm_provider: str = "anthropic"):
        print("[DBG] ReportPipeline.__init__ start", flush=True)
        print("[DBG] loading scoring model...", flush=True)
        self.scorer     = CreditScoringModel.load("models/scoring_model.json")
        print("[DBG] scoring model loaded", flush=True)
        print("[DBG] initializing RAG retriever...", flush=True)
        from rag.pgvector_retriever import PGVectorRetriever
        self.rag        = PGVectorRetriever()
        print("[DBG] RAG retriever initialized", flush=True)

        print("[DBG] creating validator...", flush=True)
        self.validator  = ReportValidator()
        print("[DBG] validator created", flush=True)
        print(f"[DBG] creating LLM provider={llm_provider}...", flush=True)
        from llm.generator import get_llm
        self.llm        = get_llm(llm_provider)
        print("[DBG] LLM client created", flush=True)
        # El informe (batch y streaming) se arma POR SECCIONES en llm/report_sections.py.

        # Extractor de datos. Por defecto reusa la MISMA LLM del informe.
        # Si EXTRACTOR_MODEL es un NOMBRE DE MODELO real (ej. "qwen3:1.7b"),
        # arma un extractor aparte (más chico/rápido). Si está vacío o es un
        # nombre de proveedor (local/anthropic/...), reusa la principal.
        self.extractor = self.llm
        modelo_extractor = (os.environ.get("EXTRACTOR_MODEL") or "").strip()
        _proveedores = {"local", "anthropic", "gemini", "openai", "same"}
        if modelo_extractor and modelo_extractor.lower() not in _proveedores:
            try:
                from langchain_ollama import ChatOllama
                self.extractor = ChatOllama(model=modelo_extractor, temperature=0,
                                            num_predict=300, keep_alive="30m")
                print(f"[DBG] extractor model = {modelo_extractor}", flush=True)
            except Exception as e:
                print(f"[DBG] no se pudo crear extractor ({e}); uso la LLM principal", flush=True)
                self.extractor = self.llm
        else:
            print("[DBG] extractor = misma LLM del informe", flush=True)

    def extraer_datos(self, consulta: str) -> dict:
        """Extrae {cuit, razon_social, tipo_decision} desde un texto libre,
        usando la LLM. Devuelve cuit solo con dígitos."""
        import json, re
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        SYS = load_prompt("extractor_sys")
        HUM = load_prompt("extractor_hum")
        chain = (
            ChatPromptTemplate.from_messages([("system", SYS), ("human", HUM)])
            | self.extractor
            | StrOutputParser()
        )
        salida = chain.invoke({"consulta": consulta})

        # Parseo tolerante: tomar el primer objeto {...} de la salida.
        datos = {}
        m = re.search(r"\{.*\}", salida, re.S)
        if m:
            try:
                datos = json.loads(m.group(0))
            except Exception:
                datos = {}

        # es_credito puede venir como bool o como string ("true"/"false")
        ec = datos.get("es_credito", True)
        if isinstance(ec, str):
            ec = ec.strip().lower() not in ("false", "no", "0", "")
        es_credito = bool(ec)

        cuit   = re.sub(r"\D", "", str(datos.get("cuit", "")))
        razon  = str(datos.get("razon_social", "") or "").strip()
        motivo = str(datos.get("motivo", "") or "").strip() or consulta.strip()
        return {"es_credito": es_credito, "cuit": cuit,
                "razon_social": razon, "tipo_decision": motivo}

    def indexar_documentos_cliente(self, cliente_id: str, items: list) -> int:
        """Indexa documentos subidos para un cliente. items = [(nombre, texto), ...].
        Devuelve la cantidad total de fragmentos indexados."""
        total = 0
        for nombre, texto in items:
            if texto and texto.strip():
                total += self.rag.indexar_documento_cliente(cliente_id, texto, nombre)
        return total

    def _perfil_bcra(self, cliente_id: str) -> dict | None:
        """Obtiene el perfil BCRA del asociado (con caché) y lo indexa en el RAG.
        Devuelve el perfil completo {"resumen":.., "texto":.., "crudo":..}, o None
        si la integración está deshabilitada / el CUIT es inválido / el API falla
        sin caché. Nunca corta el informe: cualquier error se traga y devuelve None."""
        try:
            from external import perfil_bcra
            perfil = perfil_bcra(cliente_id)
        except Exception as e:
            print(f"[BCRA] error obteniendo perfil: {e}", flush=True)
            return None
        if not perfil:
            return None
        try:
            self.rag.indexar_bcra(
                cliente_id, perfil["texto"],
                perfil["resumen"].get("periodo_informado") or "",
            )
        except Exception as e:
            print(f"[BCRA] no se pudo indexar el perfil en el RAG: {e}", flush=True)
        return perfil

    @staticmethod
    def _bcra_meta(resumen: dict | None) -> dict | None:
        """Resumen mínimo del perfil BCRA para el badge de la UI (evento meta).
        Devuelve None si no se consultó el BCRA (no se muestra nada en el badge)."""
        if not resumen:
            return None
        return {
            "tiene_datos":         bool(resumen.get("tiene_datos")),
            "peor_situacion":      resumen.get("peor_situacion"),
            "peor_situacion_desc": resumen.get("peor_situacion_desc"),
            "cant_entidades":      resumen.get("cant_entidades"),
            "deuda_total":         resumen.get("deuda_total_sistema"),
            "cheques_impagos":     resumen.get("cheques_impagos"),
        }

    def _nombre_llm(self) -> str:
        """Nombre legible del modelo que genera el informe (no el extractor)."""
        llm = self.llm
        for attr in ("model", "model_name", "model_id"):
            v = getattr(llm, attr, None)
            if v:
                return str(v)
        return type(llm).__name__

    def _nombre_embedding(self) -> str:
        """Nombre del modelo de embeddings realmente cargado en el retriever."""
        emb = getattr(self.rag, "embeddings", None)
        for attr in ("model_name", "model"):
            v = getattr(emb, attr, None)
            if v:
                return str(v)
        try:
            from rag.pgvector_indexer import _modelo_embed
            return _modelo_embed()
        except Exception:
            return "desconocido"

    def generar(self, cliente_id: str, tipo_decision: str,
                oficial_solicitante: str = "") -> dict:

        print(f"\n[1/5] Recuperando datos de SQL Server para cliente {cliente_id}...")
        with get_sqlserver_session() as ss:
            repo     = ClientRepository(ss)
            datos    = repo.get_datos_basicos(cliente_id)
            prestamos= repo.get_historial_prestamos(cliente_id)
            pagos    = repo.get_resumen_comportamiento_pagos(cliente_id)
            saldos   = repo.get_saldos_promedio_cuentas(cliente_id)
            gestiones= repo.get_gestiones_mora_estructuradas(cliente_id)
            features = repo.construir_features_ml(cliente_id)

        print("[2/5] Calculando score crediticio (ML)...")
        scoring = self.scorer.predict(features)

        print("[3/5] Recuperando contexto cualitativo (pgvector) + BCRA...")
        bcra_perfil = self._perfil_bcra(cliente_id)
        rag_context = self.rag.recuperar_contexto_completo(cliente_id, tipo_decision)

        contexto = {
            "cliente_id":          cliente_id,
            "tipo_decision":       tipo_decision,
            "fecha_informe":       datetime.now().isoformat(),
            "datos_cliente":       datos,
            "historial_prestamos": prestamos,
            "pagos_resumen":       pagos,
            "saldos_cuentas":      saldos,
            "gestiones_mora":      gestiones,
            "scoring":             scoring,
            "bcra":                (bcra_perfil or {}).get("resumen"),
            "rag":                 rag_context,
        }

        print("[4/5] Generando informe por secciones...")
        from llm.report_sections import armar_informe
        # El LLM recibe los importes ya formateados (string); el contexto crudo
        # se conserva para validación, log y juez.
        informe_texto = armar_informe(_fmt_montos_contexto(contexto), self.llm)

        print("[5/5] Validando y guardando en PostgreSQL...")
        validacion = self.validator.validar(informe_texto, contexto)
        self._guardar_log(contexto, informe_texto, validacion, oficial_solicitante)

        return {
            "informe":        informe_texto,
            "validacion":     validacion,
            "scoring":        scoring,
            "cliente_id":     cliente_id,
            "tipo_decision":  tipo_decision,
            "fecha":          contexto["fecha_informe"],
            "trazabilidad":   self._trazabilidad(contexto),
            "contexto":       contexto,   # hechos fuente (para el juez LLM)
        }

    def generar_stream(self, cliente_id: str, tipo_decision: str,
                       razon_social: str = ""):
        """Versión generadora de generar(): emite eventos de progreso y entrega
        el informe en chunks, para streaming tipo chat en la web.

        Eventos (dicts) que produce:
          {"tipo":"etapa","clave":"sql"|"ml"|"rag"|"llm"}
          {"tipo":"meta","encontrado":bool,"score":..,"nivel":..}
          {"tipo":"token","texto":"..."}                (repetido, streaming)
          {"tipo":"validacion","aprobado":bool,"advertencias":[...]}
          {"tipo":"fin"}

        Si el cliente NO está en la Base Financiera, NO corta: genera igual un
        informe "sin datos" usando la razón social provista.
        Otras excepciones (DB caída, etc.) se propagan al llamador.
        """
        yield {"tipo": "etapa", "clave": "sql"}
        encontrado = True
        datos = prestamos = pagos = saldos = gestiones = features = None
        try:
            with get_sqlserver_session() as ss:
                repo      = ClientRepository(ss)
                datos     = repo.get_datos_basicos(cliente_id)
                prestamos = repo.get_historial_prestamos(cliente_id)
                pagos     = repo.get_resumen_comportamiento_pagos(cliente_id)
                saldos    = repo.get_saldos_promedio_cuentas(cliente_id)
                gestiones = repo.get_gestiones_mora_estructuradas(cliente_id)
                features  = repo.construir_features_ml(cliente_id)
        except Exception as exc:
            if _es_no_encontrado(exc):
                encontrado = False          # rama "sin datos", no es un error
            else:
                raise                       # cualquier otro error sí se propaga

        # ---- Rama: cliente NO encontrado -> informe sin datos ----
        if not encontrado:
            yield from self._stream_informe_sin_datos(cliente_id, tipo_decision, razon_social)
            return

        # ---- Rama normal: cliente encontrado ----
        yield {"tipo": "etapa", "clave": "ml"}
        scoring = self.scorer.predict(features)

        yield {"tipo": "etapa", "clave": "rag"}
        bcra_perfil = self._perfil_bcra(cliente_id)
        rag_context = self.rag.recuperar_contexto_completo(cliente_id, tipo_decision)

        contexto = {
            "cliente_id":          cliente_id,
            "tipo_decision":       tipo_decision,
            "razon_social":        razon_social,   # se pasa al pipeline
            "fecha_informe":       datetime.now().isoformat(),
            "datos_cliente":       datos,
            "historial_prestamos": prestamos,
            "pagos_resumen":       pagos,
            "saldos_cuentas":      saldos,
            "gestiones_mora":      gestiones,
            "scoring":             scoring,
            "bcra":                (bcra_perfil or {}).get("resumen"),
            "rag":                 rag_context,
        }

        yield {"tipo": "meta",
               "encontrado": True,
               "modelo": self._nombre_llm(),
               "score": scoring.get("score"),
               "nivel": scoring.get("nivel_riesgo"),
               "bcra":  self._bcra_meta((bcra_perfil or {}).get("resumen"))}

        yield {"tipo": "etapa", "clave": "llm"}
        # Generación por secciones, en streaming. armar_informe_stream emite
        # eventos {seccion|token} en vivo y, al final, {informe: <doc completo>}
        # ya ordenado (Resumen arriba) para el render final.
        from llm.report_sections import armar_informe_stream, resumen_datos_secciones
        informe_texto = ""
        bodies = {}
        # El LLM recibe los importes ya formateados (string); el contexto crudo
        # se conserva para validación, log y juez. Guardamos el MISMO contexto
        # formateado para poder regenerar una sección idéntica más tarde.
        contexto_fmt = _fmt_montos_contexto(contexto)
        for ev in armar_informe_stream(contexto_fmt, self.llm):
            if ev["tipo"] == "informe":
                informe_texto = ev["texto"]
                bodies = ev.get("bodies", {})
            else:                       # "seccion" / "token" -> se reenvían a la UI
                yield ev

        validacion = self.validator.validar(informe_texto, contexto)
        self._guardar_log(contexto, informe_texto, validacion, "")
        yield {"tipo": "validacion",
               "aprobado":     validacion.get("aprobado", True),
               "advertencias": validacion.get("advertencias", [])}
        # Retener la sesión para permitir regenerar una sección desde el admin.
        sesion_id = uuid.uuid4().hex
        _guardar_sesion(sesion_id, contexto_fmt, bodies)
        # 'informe' lleva el documento en orden de lectura para el render final
        # (durante el stream las secciones llegan en orden de ejecución).
        # 'datos_secciones': resumen legible de los datos fuente por sección, para
        # que el lector vea la relación entre el texto del informe y los datos.
        yield {"tipo": "fin", "informe": informe_texto, "sesion_id": sesion_id,
               "datos_secciones": resumen_datos_secciones(contexto_fmt)}

    def contexto_sesion(self, sesion_id: str) -> dict:
        """Devuelve el contexto (ya formateado) que generó el informe de la sesión,
        para que el admin vea los datos fuente. Lanza KeyError si no existe."""
        ses = _SESIONES.get(sesion_id)
        if ses is None:
            raise KeyError("La sesión del informe expiró (o el servidor se reinició). "
                           "Generá el informe de nuevo para ver sus datos.")
        return ses["contexto"]

    def datos_seccion(self, sesion_id: str, section_id: str):
        """Devuelve (titulo, ctx_dict) con los datos EXACTOS que recibe una sección
        (el mismo JSON que se le pasa al modelo). Lanza KeyError/ValueError."""
        ses = _SESIONES.get(sesion_id)
        if ses is None:
            raise KeyError("La sesión del informe expiró (o el servidor se reinició). "
                           "Generá el informe de nuevo para ver sus datos.")
        from llm.report_sections import especificacion_seccion
        titulo, _prompt, ctx = especificacion_seccion(
            section_id, ses["contexto"], ses["bodies"])
        return titulo, ctx

    def regenerar_seccion_stream(self, sesion_id: str, section_id: str):
        """Regenera UNA sección de un informe ya generado, en streaming, usando el
        contexto retenido en memoria (sin re-consultar SQL/ML/RAG/BCRA). Relee el
        prompt del .txt, así toma la última edición del admin. Actualiza el cuerpo
        guardado (para que una regeneración posterior de Recomendación/Resumen use
        la versión vigente).

        Eventos: {"tipo":"seccion",...}, {"tipo":"token",...},
                 {"tipo":"fin_seccion","texto": <cuerpo nuevo>, "section_id":...}
        Lanza KeyError si la sesión no existe (p. ej. el server se reinició)."""
        ses = _SESIONES.get(sesion_id)
        if ses is None:
            raise KeyError("La sesión del informe expiró (o el servidor se reinició). "
                           "Generá el informe de nuevo para volver a editar sus secciones.")
        from llm.report_sections import (especificacion_seccion,
                                          regenerar_seccion_stream as _regen)
        titulo, _prompt, _ctx = especificacion_seccion(
            section_id, ses["contexto"], ses["bodies"])
        yield {"tipo": "seccion", "titulo": titulo, "section_id": section_id}
        nuevo = yield from _regen(section_id, ses["contexto"], ses["bodies"], self.llm)
        ses["bodies"][section_id] = nuevo          # el cuerpo vigente pasa a ser este
        _SESIONES.move_to_end(sesion_id)
        yield {"tipo": "fin_seccion", "texto": nuevo,
               "section_id": section_id, "titulo": titulo}

    def _stream_informe_sin_datos(self, cliente_id: str, tipo_decision: str,
                                  razon_social: str):
        """Genera un informe para un solicitante que NO está en la Base
        Financiera. No corre ML ni RAG interno, pero SÍ consulta el BCRA: aunque
        no sea socio, puede registrar deuda en otras entidades o cheques
        rechazados. Si hay datos del BCRA, se incorporan al informe."""
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        # Aunque no esté en la base interna, consultamos el BCRA por el CUIT.
        bcra_perfil  = self._perfil_bcra(cliente_id)
        bcra_resumen = (bcra_perfil or {}).get("resumen") or {}
        bcra_texto   = (bcra_perfil or {}).get("texto") or ""
        tiene_bcra   = bool(bcra_resumen.get("tiene_datos"))

        # La sección BCRA solo se incluye si efectivamente se pudo consultar.
        secciones = "Resumen Ejecutivo, Clasificación de Riesgo, Historial Crediticio"
        if bcra_perfil:
            secciones += ", Situación en el Sistema Financiero (BCRA)"
        secciones += ", Recomendación"

        # Fragmento de regla BCRA según lo que se obtuvo (textos en prompts/*.txt).
        # El .txt guarda el bullet sin el salto final; el esqueleto lo necesita, se agrega acá.
        if tiene_bcra:
            regla_bcra = load_prompt("sin_datos_regla_bcra_con_datos") + "\n"
        elif bcra_perfil:
            regla_bcra = load_prompt("sin_datos_regla_bcra_sin_datos") + "\n"
        else:
            regla_bcra = load_prompt("sin_datos_regla_bcra_no_consultado") + "\n"

        SYS = load_prompt("sin_datos_sys").format(regla_bcra=regla_bcra,
                                                   secciones=secciones)
        HUM = load_prompt("sin_datos_hum")

        # meta sin score; marca que NO se encontró (la UI mostrará la razón social).
        # Si el BCRA trajo algo, se adjunta para que el badge lo muestre.
        yield {"tipo": "meta", "encontrado": False, "nivel": "SIN DATOS INTERNOS",
               "modelo": self._nombre_llm(),
               "bcra": self._bcra_meta(bcra_resumen if bcra_perfil else None)}
        yield {"tipo": "etapa", "clave": "llm"}

        chain = (
            ChatPromptTemplate.from_messages([("system", SYS), ("human", HUM)])
            | self.llm
            | StrOutputParser()
        )
        partes = []
        for chunk in chain.stream({
            "razon_social": razon_social or "(no informada)",
            "cuit":         cliente_id,
            "tipo_decision": tipo_decision,
            "bcra":         bcra_texto or "No se obtuvieron datos del BCRA.",
        }):
            partes.append(chunk)
            yield {"tipo": "token", "texto": chunk}
        informe_texto = "".join(partes)

        # Log defensivo: si falla (ej. columna score NOT NULL), no rompe el informe.
        try:
            ctx_log = {
                "cliente_id":    cliente_id,
                "tipo_decision": tipo_decision,
                "scoring":       {"score": None, "nivel": "SIN DATOS"},
                "trazabilidad":  {
                    "sin_datos_internos": True,
                    "razon_social":       razon_social,
                    "bcra": {
                        "consultado":      bool(bcra_perfil),
                        "tiene_datos":     tiene_bcra,
                        "peor_situacion":  bcra_resumen.get("peor_situacion"),
                        "cheques_impagos": bcra_resumen.get("cheques_impagos"),
                    },
                },
            }
            validacion = {"aprobado": True, "advertencias": []}
            self._guardar_log(ctx_log, informe_texto, validacion, "")
        except Exception:
            import traceback as _tb
            _tb.print_exc()

        if tiene_bcra:
            advertencia = (
                "No se encontraron datos internos del cliente. El informe se apoya "
                "en la identificación provista y en la información del BCRA (Central "
                "de Deudores); requiere verificación manual de identidad."
            )
        else:
            advertencia = (
                "No se encontraron datos financieros del cliente en la base interna "
                "ni registros en el BCRA. El informe se generó en base a la "
                "identificación provista y requiere verificación manual."
                if bcra_perfil else
                "No se encontraron datos financieros del cliente en la base. "
                "El informe se generó en base a la identificación provista y "
                "requiere verificación manual."
            )
        yield {"tipo": "validacion", "aprobado": True, "advertencias": [advertencia]}
        yield {"tipo": "fin"}

    def _guardar_log(self, ctx, informe, validacion, oficial):
        import json
        with get_postgres_session() as ps:
            ps.execute(text("""
                INSERT INTO informes_generados
                    (cliente_id, tipo_decision, score_calculado, nivel_riesgo,
                     informe_texto, validacion_ok, advertencias, trazabilidad)
                VALUES
                    (:cliente_id, :tipo_decision, :score, :nivel,
                     :informe, :validacion_ok,
                     cast(:advertencias as jsonb), cast(:trazabilidad as jsonb))
            """), {
                'cliente_id': ctx['cliente_id'],
                'tipo_decision': ctx['tipo_decision'],
                'score': ctx['scoring'].get('score', 0),
                # El scoring del modelo usa la clave 'nivel_riesgo'; el camino
                # "sin datos" usa 'nivel'. Se contemplan ambas.
                'nivel': (ctx['scoring'].get('nivel_riesgo')
                          or ctx['scoring'].get('nivel') or 'DESCONOCIDO'),
                'informe': informe,
                # El validador devuelve 'aprobado' (no 'ok').
                'validacion_ok': validacion.get('aprobado', True),
                'advertencias': json.dumps(validacion.get('advertencias', [])),
                'trazabilidad': json.dumps(ctx.get('trazabilidad', {})),
            })
            ps.commit()

    def _trazabilidad(self, ctx) -> dict:
        bcra = ctx.get("bcra") or {}
        scoring = ctx.get("scoring") or {}
        return {
            "fuentes_sql_server": list(TABLAS_SQLSERVER),
            "fuentes_pgvector": {
                "notas_mora":     len(ctx["rag"]["notas_mora"]),
                "politicas":      len(ctx["rag"]["politicas"]),
                "normativa":      len(ctx["rag"]["normativa"]),
                "informes_prev":  len(ctx["rag"]["informes_previos"]),
            },
            "bcra": {
                "consultado":      bool(ctx.get("bcra")),
                "peor_situacion":  bcra.get("peor_situacion"),
                "cant_entidades":  bcra.get("cant_entidades"),
                "cheques_impagos": bcra.get("cheques_impagos"),
            },
            "modelo_scoring": descripcion_modelo_scoring(),
            # PD cruda vs PD publicada: si algún día vuelve a haber calibración,
            # el auditor tiene que poder ver ambas y cuál se usó.
            "scoring_pd": {
                "prob_default":       scoring.get("prob_default"),
                "prob_default_cruda": scoring.get("prob_default_cruda"),
                "calibracion":        scoring.get("calibracion"),
            },
            "modelo_llm":     self._nombre_llm(),
            "modelo_embedding": self._nombre_embedding(),
        }


class ReportValidator:
    """Valida completitud y detecta posibles alucinaciones."""

    SECCIONES = ["RESUMEN EJECUTIVO", "CLASIFICACIÓN DE RIESGO",
                 "HISTORIAL CREDITICIO", "RECOMENDACIÓN"]

    def validar(self, informe: str, ctx: dict) -> dict:
        advertencias = []
        informe_up   = informe.upper()

        for s in self.SECCIONES:
            if s not in informe_up:
                advertencias.append(f"Sección faltante: {s}")

        score_str = str(ctx["scoring"]["score"])
        if score_str not in informe:
            advertencias.append(
                f"ALERTA: Score {score_str} no aparece en el informe — "
                "posible alucinación numérica."
            )

        if "decisión final corresponde" not in informe.lower():
            advertencias.append("Falta el disclaimer de decisión humana.")

        alertas = [a for a in advertencias if "ALERTA" in a]
        return {
            "aprobado":    len(alertas) == 0,
            "advertencias": advertencias,
        }


# ---------------------------------------------------------------------------
#  Acceso para apps externas (ej. app.py)
#  Construye el pipeline UNA sola vez y lo reutiliza, para no recargar el
#  modelo de scoring / RAG / LLM en cada request.
# ---------------------------------------------------------------------------
_pipeline_singleton = None

def get_pipeline(llm_provider: str = "local") -> "ReportPipeline":
    global _pipeline_singleton
    if _pipeline_singleton is None:
        print("[DBG] construyendo pipeline (primera vez)...", flush=True)
        _pipeline_singleton = ReportPipeline(llm_provider=llm_provider)
    return _pipeline_singleton


# Uso
if __name__ == "__main__":
    pipeline = ReportPipeline(llm_provider="local")
    resultado = pipeline.generar(
        cliente_id="20123456789",
        tipo_decision="Aprobación préstamo personal $1.000.000 — 24 cuotas",
        oficial_solicitante="Lic. Franco S RASPO",
    )
    print("[DBG] pipeline.generar() finished", flush=True)
    print(resultado["informe"])