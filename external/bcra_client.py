"""Cliente del API público de la Central de Deudores del BCRA.

Consulta, por CUIT/CUIL, tres recursos:
  - Deudas vigentes en todas las entidades del sistema financiero.
  - Deudas históricas (evolución mensual de los últimos ~24 períodos).
  - Cheques rechazados.

Es un API público y gratuito (no requiere API key). Documentación:
  https://www.bcra.gob.ar/Catalogo/apis.asp

El recurso devuelve HTTP 404 cuando el CUIT no tiene registros: eso NO es un
error, es "sin deudas / sin cheques". Se traduce a None (payload vacío).

Sin dependencias externas: usa urllib de la stdlib (no agrega requests/httpx).
"""
import json
import os
import ssl
import time
import urllib.error
import urllib.request


BASE_DEFAULT = "https://api.bcra.gob.ar/centraldedeudores/v1.0"


class BCRAError(Exception):
    """Falla consultando el API del BCRA (red, 5xx o payload inválido)."""


def _base() -> str:
    return (os.environ.get("BCRA_API_BASE") or BASE_DEFAULT).rstrip("/")


def _timeout() -> float:
    try:
        return float(os.environ.get("BCRA_TIMEOUT", "15"))
    except ValueError:
        return 15.0


def _verify_ssl() -> bool:
    """El BCRA suele requerir TLS válido, pero en algunas redes corporativas la
    cadena de certificados no resuelve. BCRA_VERIFY_SSL=false lo desactiva."""
    return (os.environ.get("BCRA_VERIFY_SSL", "true").strip().lower()
            not in ("false", "0", "no"))


def _ssl_context():
    if _verify_ssl():
        return None
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def solo_digitos(cuit: str) -> str:
    return "".join(ch for ch in str(cuit) if ch.isdigit())


def _get(url: str, reintentos: int = 2) -> dict | None:
    """GET con reintentos. Devuelve el JSON parseado, o None si el CUIT no tiene
    registros (HTTP 404). Levanta BCRAError ante fallas de red/servidor."""
    headers = {
        "User-Agent": "trabajo_ia-bcra/1.0",
        "Accept":     "application/json",
    }
    ultimo_error = None
    for intento in range(reintentos + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(
                req, timeout=_timeout(), context=_ssl_context()
            ) as resp:
                raw = resp.read().decode("utf-8")
            return json.loads(raw)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None                 # CUIT sin registros: no es error
            ultimo_error = e
            if e.code < 500:
                break                       # 4xx (no 404) no se reintenta
        except (urllib.error.URLError, TimeoutError, ssl.SSLError,
                json.JSONDecodeError) as e:
            ultimo_error = e
        if intento < reintentos:
            time.sleep(1.5 * (intento + 1))
    raise BCRAError(f"Fallo consultando {url}: {ultimo_error}")


def get_deudas(cuit: str) -> dict | None:
    """Situación crediticia vigente en todas las entidades (período actual)."""
    return _get(f"{_base()}/Deudas/{solo_digitos(cuit)}")


def get_deudas_historicas(cuit: str) -> dict | None:
    """Evolución mensual de la situación crediticia (histórico)."""
    return _get(f"{_base()}/Deudas/Historicas/{solo_digitos(cuit)}")


def get_cheques_rechazados(cuit: str) -> dict | None:
    """Cheques rechazados por causal (sin fondos, defectos formales, etc.)."""
    return _get(f"{_base()}/Deudas/ChequesRechazados/{solo_digitos(cuit)}")


def consultar_todo(cuit: str) -> dict:
    """Consulta los tres recursos y los devuelve unificados (sin normalizar).
    Cada clave puede ser None (sin registros) o el payload crudo del BCRA."""
    return {
        "deudas":             get_deudas(cuit),
        "deudas_historicas":  get_deudas_historicas(cuit),
        "cheques_rechazados": get_cheques_rechazados(cuit),
    }
