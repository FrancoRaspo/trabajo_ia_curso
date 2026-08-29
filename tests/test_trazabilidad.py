"""La trazabilidad tiene que nombrar las tablas que el sistema REALMENTE lee.

Este test existe por un bug concreto: `pipeline._trazabilidad` declaraba haber
consultado `clientes`, `prestamos`, `cuotas`, `saldos_diarios` y
`gestiones_mora` — cinco tablas que no existen en la base. Esa lista se escribía
en el campo `trazabilidad` del log auditable de cada informe.

Una constante escrita a mano vuelve a divergir apenas alguien toque una query.
Acá se extraen las tablas del SQL fuente y se comparan contra la constante.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from db.client_repository import TABLAS_SQLSERVER

RAIZ = Path(__file__).resolve().parent.parent
FUENTES = [RAIZ / "db" / "client_repository.py", RAIZ / "ml" / "feature_query.py"]

# El SQL vive en strings triple-quoted. Se extraen primero para no confundir
# `from __future__ import ...` de Python con un FROM de SQL.
_SQL = re.compile(r'"""(.*?)"""', re.S)
# `FROM x` / `JOIN x`, con o sin el prefijo dbo.
_REF = re.compile(r"\b(?:from|join)\s+(?:dbo\.)?([a-z_][a-z0-9_]*)", re.I)
# Nombres de CTE. Se declaran como `WITH x AS (` y, las siguientes, con el
# nombre al inicio de su propia línea (`x AS (`) — a veces con un comentario
# entre medio, así que no alcanza con buscar la coma. No son tablas.
_CTE = re.compile(r"^\s*(?:with\s+)?([a-z_][a-z0-9_]*)\s+as\s*\(\s*$", re.I | re.M)
# Palabras que siguen a FROM/JOIN pero no son tablas.
_NO_TABLAS = {"select"}


def tablas_referenciadas(fuente: Path) -> set[str]:
    sql = "\n".join(_SQL.findall(fuente.read_text(encoding="utf-8")))
    ctes = {c.lower() for c in _CTE.findall(sql)}
    refs = {r.lower() for r in _REF.findall(sql)}
    return refs - ctes - _NO_TABLAS


def test_trazabilidad_declara_las_tablas_que_se_consultan():
    """TABLAS_SQLSERVER == las tablas base que aparecen en las queries."""
    reales = set()
    for f in FUENTES:
        reales |= tablas_referenciadas(f)

    declaradas = {t.lower() for t in TABLAS_SQLSERVER}

    faltantes = reales - declaradas
    sobrantes = declaradas - reales

    assert not faltantes, (
        f"Estas tablas se consultan pero no están en TABLAS_SQLSERVER: "
        f"{sorted(faltantes)}. La trazabilidad del informe las estaría omitiendo."
    )
    assert not sobrantes, (
        f"TABLAS_SQLSERVER declara tablas que ninguna query lee: "
        f"{sorted(sobrantes)}. La trazabilidad estaría mintiendo."
    )


def test_no_reaparecen_las_tablas_ficticias():
    """Guard explícito contra el bug original."""
    inventadas = {"clientes", "prestamos", "cuotas", "saldos_diarios", "gestiones_mora"}
    declaradas = {t.lower() for t in TABLAS_SQLSERVER}
    assert not (inventadas & declaradas), (
        f"Volvieron las tablas inventadas: {sorted(inventadas & declaradas)}"
    )


def test_trazabilidad_reporta_la_version_real_del_modelo(tmp_path, monkeypatch):
    """El campo modelo_scoring sale del model card, no de un literal."""
    from pipeline import descripcion_modelo_scoring

    desc = descripcion_modelo_scoring()
    assert "v1.0" not in desc, "modelo_scoring sigue hardcodeado como 'XGBoost v1.0'"
    # El model card real declara xgboost 3.2.0 y una fecha de generación.
    assert "XGBClassifier" in desc or "desconocido" in desc