"""Caché en PostgreSQL de las consultas al BCRA.

El BCRA actualiza la Central de Deudores una vez al mes, así que no tiene
sentido pegarle al API en cada informe. Guardamos el último payload crudo por
CUIT y lo reusamos mientras esté fresco (TTL configurable, default 30 días).

Tabla: bcra_cache(cuit, payload jsonb, fecha_consulta timestamptz).
Se crea sola la primera vez (asegurar_tabla), no requiere correr init_db.
"""
import json
import os
from datetime import datetime, timedelta, timezone

from sqlalchemy import text

from db.postgres_connection import get_postgres_engine
from external import bcra_client


def _ttl_dias() -> int:
    try:
        return int(os.environ.get("BCRA_TTL_DIAS", "30"))
    except ValueError:
        return 30


def asegurar_tabla(engine=None) -> None:
    """Crea la tabla de caché si no existe (idempotente)."""
    engine = engine or get_postgres_engine()
    with engine.begin() as c:
        c.execute(text("""
            CREATE TABLE IF NOT EXISTS bcra_cache (
                cuit           VARCHAR(11) PRIMARY KEY,
                payload        JSONB       NOT NULL,
                fecha_consulta TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """))


def _leer_cache(engine, cuit: str):
    with engine.connect() as c:
        return c.execute(text(
            "SELECT payload, fecha_consulta FROM bcra_cache WHERE cuit = :cuit"
        ), {"cuit": cuit}).mappings().fetchone()


def _guardar_cache(engine, cuit: str, payload: dict) -> None:
    with engine.begin() as c:
        c.execute(text("""
            INSERT INTO bcra_cache (cuit, payload, fecha_consulta)
            VALUES (:cuit, cast(:payload as jsonb), NOW())
            ON CONFLICT (cuit) DO UPDATE
                SET payload        = EXCLUDED.payload,
                    fecha_consulta = NOW()
        """), {"cuit": cuit, "payload": json.dumps(payload, default=str)})


def obtener_bcra(cuit: str, forzar: bool = False,
                 ttl_dias: int | None = None) -> dict:
    """Payload CRUDO del BCRA para un CUIT, usando caché.

    - Caché fresca (< TTL) y sin forzar -> se devuelve sin pegarle al API.
    - Caché vencida/ausente o forzar=True -> consulta el API y actualiza.
    - API falla pero hay caché vieja -> devuelve la vieja (degradación amable).
    """
    cuit = bcra_client.solo_digitos(cuit)
    ttl  = ttl_dias if ttl_dias is not None else _ttl_dias()
    engine = get_postgres_engine()
    asegurar_tabla(engine)

    fila = _leer_cache(engine, cuit)
    if fila and not forzar:
        edad = datetime.now(timezone.utc) - fila["fecha_consulta"]
        if edad <= timedelta(days=ttl):
            return fila["payload"]

    try:
        payload = bcra_client.consultar_todo(cuit)
        _guardar_cache(engine, cuit, payload)
        return payload
    except bcra_client.BCRAError:
        if fila:                      # mejor un dato viejo que ninguno
            print(f"[BCRA] API no disponible; uso caché vieja de {cuit}", flush=True)
            return fila["payload"]
        raise
