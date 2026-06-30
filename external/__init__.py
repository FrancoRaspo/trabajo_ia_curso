"""Integraciones con fuentes de datos externas.

Por ahora: Central de Deudores del BCRA (situación crediticia en todas las
entidades del sistema financiero + cheques rechazados), consultada por CUIT.

Punto de entrada de alto nivel para el resto del sistema: `perfil_bcra(cuit)`.
"""
import os


def bcra_habilitado() -> bool:
    """True salvo que BCRA_ENABLED esté explícitamente apagado.

    Consultar el BCRA envía el CUIT a un API externo (rompe el invariante
    'nada sale del on-premise'), por eso se puede desactivar por entorno.
    """
    return (os.environ.get("BCRA_ENABLED", "true").strip().lower()
            not in ("false", "0", "no"))


def perfil_bcra(cuit: str, forzar: bool = False) -> dict | None:
    """Perfil BCRA de un CUIT, listo para el pipeline.

    Devuelve un dict:
        {"resumen": <métricas estructuradas>,
         "texto":   <resumen narrativo para el RAG>,
         "crudo":   <payload original del BCRA>}
    o None si la integración está deshabilitada, el CUIT es inválido o el API
    falla sin caché disponible. NUNCA levanta: ante cualquier error devuelve
    None y deja un log, para no cortar la generación del informe.
    """
    if not bcra_habilitado():
        return None
    cuit = "".join(ch for ch in str(cuit or "") if ch.isdigit())
    if len(cuit) != 11:
        return None
    try:
        from external.bcra_cache import obtener_bcra
        from external import bcra_normalizer as norm
        crudo   = obtener_bcra(cuit, forzar=forzar)
        resumen = norm.resumir(crudo)
        texto   = norm.a_texto(resumen, cuit)
        return {"resumen": resumen, "texto": texto, "crudo": crudo}
    except Exception as e:
        print(f"[BCRA] no se pudo obtener el perfil de {cuit}: {e}", flush=True)
        return None