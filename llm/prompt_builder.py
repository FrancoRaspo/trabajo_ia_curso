import json
from datetime import datetime

class PromptBuilder:
    MAX_PRESTAMOS = 10  # evitar prompts gigantes

    def build(self, contexto: dict) -> dict:
        return {
            "tipo_decision":          contexto["tipo_decision"],
            "fecha_informe":          self._fmt_fecha(contexto["fecha_informe"]),
            "datos_cliente_json":     self._s(contexto["datos_cliente"]),
            "historial_prestamos_json": self._s(
                contexto["historial_prestamos"][:self.MAX_PRESTAMOS]
            ),
            "pagos_resumen_json":     self._s(contexto["pagos_resumen"]),
            "saldos_cuentas_json":    self._s(contexto["saldos_cuentas"]),
            "gestiones_mora_json":    self._s(contexto["gestiones_mora"]),
            "scoring_json":           self._s(contexto["scoring"]),
            "notas_mora":             self._fmt_fragmentos(contexto["rag"]["notas_mora"]),
            "politicas_relevantes":   self._fmt_fragmentos(contexto["rag"]["politicas"]),
            "normativa":              self._fmt_fragmentos(contexto["rag"]["normativa"]),
            "informes_anteriores":    self._fmt_fragmentos(contexto["rag"]["informes_previos"]),
            "documentos_cliente":     self._fmt_fragmentos(contexto["rag"].get("documentos_cliente", [])),
        }

    def _s(self, data) -> str:
        if not data:
            return "[SIN DATOS]"
        return json.dumps(data, ensure_ascii=False, indent=2, default=str)

    def _fmt_fecha(self, iso: str) -> str:
        try:
            return datetime.fromisoformat(iso).strftime("%d/%m/%Y %H:%M")
        except Exception:
            return iso

    def _fmt_fragmentos(self, fragmentos: list) -> str:
        if not fragmentos:
            return "[SIN INFORMACIÓN DISPONIBLE EN ESTA CATEGORÍA]"
        partes = []
        for i, f in enumerate(fragmentos, 1):
            meta = json.dumps(f.get("metadata", {}), ensure_ascii=False)
            partes.append(f"[Fragmento {i} | Fuente: {f.get('fuente','?')}]\n"
                          f"Metadata: {meta}\n{f['texto']}")
        return "\n\n".join(partes)