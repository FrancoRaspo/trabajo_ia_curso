from dotenv import load_dotenv
load_dotenv()
print("[DBG] dotenv loaded", flush=True)

import copy
import os
from datetime import datetime
from sqlalchemy import text
print("[DBG] core libs imported", flush=True)

from external.bcra_normalizer import formato_pesos

from db.sqlserver_connection import get_sqlserver_session
from db.postgres_connection import get_postgres_session
from db.client_repository import ClientRepository
from ml.scoring_model import CreditScoringModel
print("[DBG] project modules imported", flush=True)


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


def _fmt_montos_contexto(contexto: dict) -> dict:
    """Copia del contexto con los importes ya formateados en pesos argentinos.
    No muta el original: ese sigue siendo la fuente de verdad (validador/juez)."""
    ctx = copy.deepcopy(contexto)

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

        SYS = (
            "Analizás una consulta y devolvés SOLO un objeto JSON válido, sin "
            "texto adicional, sin explicaciones y sin bloques de código.\n"
            "Primero determiná si la consulta es una SOLICITUD CREDITICIA "
            "(pedido de crédito, préstamo, financiación o evaluación crediticia "
            "de un asociado/empresa). Si NO lo es (por ejemplo un tema personal "
            "o ajeno a lo financiero), poné es_credito en false y el resto vacío.\n"
            "Claves exactas:\n"
            '  "es_credito": true o false.\n'
            '  "cuit": solo los 11 dígitos, sin guiones ni espacios.\n'
            '  "razon_social": nombre de la persona o empresa.\n'
            '  "motivo": resumen breve de lo que solicita.\n'
            "Si un dato no aparece, usá cadena vacía. No inventes el CUIT: "
            "copialo tal como aparece en la consulta. /no_think"
        )
        HUM = "Consulta: {consulta}"
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
        from llm.report_sections import armar_informe_stream
        informe_texto = ""
        # El LLM recibe los importes ya formateados (string); el contexto crudo
        # se conserva para validación, log y juez.
        for ev in armar_informe_stream(_fmt_montos_contexto(contexto), self.llm):
            if ev["tipo"] == "informe":
                informe_texto = ev["texto"]
            else:                       # "seccion" / "token" -> se reenvían a la UI
                yield ev

        validacion = self.validator.validar(informe_texto, contexto)
        self._guardar_log(contexto, informe_texto, validacion, "")
        yield {"tipo": "validacion",
               "aprobado":     validacion.get("aprobado", True),
               "advertencias": validacion.get("advertencias", [])}
        # 'informe' lleva el documento en orden de lectura para el render final
        # (durante el stream las secciones llegan en orden de ejecución).
        yield {"tipo": "fin", "informe": informe_texto}

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

        if tiene_bcra:
            regla_bcra = (
                "- SÍ disponés de información del BCRA (Central de Deudores) en el "
                "contexto. Reportala con precisión en la sección 'Situación en el "
                "Sistema Financiero (BCRA)': entidades donde registra deuda, peor "
                "situación (escala 1=normal a 5=irrecuperable), deuda total y cheques "
                "rechazados/impagos. No inventes nada fuera de ese contexto. Esta "
                "información externa SÍ permite una evaluación crediticia parcial.\n"
            )
        elif bcra_perfil:
            regla_bcra = (
                "- Se consultó el BCRA y no se hallaron registros externos de deuda "
                "ni cheques rechazados; indicalo en prosa (es una señal favorable), "
                "sin inferir nada más.\n"
            )
        else:
            regla_bcra = (
                "- No se pudo consultar el BCRA; no menciones la Central de Deudores.\n"
            )

        SYS = (
            "Sos un analista crediticio de una institución financiera. Vas a "
            "redactar un informe en Markdown para un solicitante del cual NO se "
            "encontraron datos en la base financiera interna.\n"
            "Reglas:\n"
            "- No inventes historial interno, montos ni score de la entidad: no hay "
            "información financiera interna disponible.\n"
            "- Dejá explícito que no se hallaron registros internos; la evaluación se "
            "apoya únicamente en información externa (BCRA), si la hay.\n"
            + regla_bcra +
            "- En Historial Crediticio aclará que no hay historial interno en la "
            "entidad.\n"
            "- Recomendá verificación manual de identidad y alta de datos antes "
            "de avanzar.\n"
            f"- Estructurá con títulos Markdown (##): {secciones}.\n"
            "- En Clasificación de Riesgo indicá **Nivel:** SIN DATOS INTERNOS.\n"
            "- No uses bloques de código.\n"
            "- Terminá con: La decisión final corresponde al oficial de crédito."
        )
        HUM = (
            "Solicitante NO encontrado en la base financiera interna.\n"
            "- Razón social informada: {razon_social}\n"
            "- CUIT: {cuit}\n"
            "- Motivo de la consulta: {tipo_decision}\n\n"
            "Información del BCRA (Central de Deudores):\n{bcra}\n\n"
            "Redactá el informe correspondiente."
        )

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
        return {
            "fuentes_sql_server": ["clientes", "prestamos", "cuotas",
                                   "saldos_diarios", "gestiones_mora"],
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
            "modelo_scoring": "XGBoost v1.0",
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
        cliente_id="27299772417",
        tipo_decision="Aprobación préstamo personal $1.000.000 — 24 cuotas",
        oficial_solicitante="Lic. Franco S RASPO",
    )
    print("[DBG] pipeline.generar() finished", flush=True)
    print(resultado["informe"])