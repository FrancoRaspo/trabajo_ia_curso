"""`max_dias_atraso_historico` tiene que significar lo mismo en los dos lugares
que lo calculan.

El bug que motiva este test: la query del informe
(`db/client_repository.get_historial_prestamos`) calculaba el máximo atraso
mirando **sólo las cuotas pagadas tarde** (`fecha_canc IS NOT NULL`), así que
ignoraba las cuotas impagas vigentes — justo las de mayor atraso, que además
crece todos los días. La feature del modelo (`ml/feature_query.py`), en cambio,
mide cada cuota vencida contra su fecha de pago **o contra hoy si sigue impaga**.

Resultado: para un cliente real con 166 días de mora viva, el informe publicaba
"máximo histórico 80" — un máximo MENOR que el atraso actual del mismo préstamo.
El juez del golden set lo marcó como alucinación CRÍTICA (caso `muy_alto_2`), y
tenía razón: el informe se contradecía porque los datos se contradecían.

Es el mismo bug que ya se había corregido en `promedio_dias_atraso`, y volvió a
aparecer en el campo de al lado. Una definición duplicada en dos SQL distintos
diverge apenas alguien toca una; por eso el test mira el SQL fuente y no la base
(así corre en CI sin credenciales).
"""
from __future__ import annotations

import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
REPO = RAIZ / "db" / "client_repository.py"
FEATURES = RAIZ / "ml" / "feature_query.py"


def _subconsulta_max_atraso(fuente: Path) -> str:
    """El bloque `outer APPLY (...) max_atraso` del historial de préstamos."""
    texto = fuente.read_text(encoding="utf-8")
    i = texto.find("outer APPLY (\n                SELECT MAX(CASE WHEN d.dias")
    assert i != -1, "no se encontró el outer APPLY de max_atraso en el SQL"
    j = texto.find(") max_atraso", i)
    assert j != -1, "no se encontró el cierre `) max_atraso`"
    return texto[i:j]


def test_el_maximo_atraso_no_ignora_las_cuotas_impagas():
    """El bug original: `fecha_canc IS NOT NULL` como FILTRO deja fuera la mora viva.

    Ojo con la distinción, que es exactamente el arreglo: la condición sigue
    existiendo, pero dentro de un `CASE WHEN` (para usar la fecha de pago cuando
    la cuota se pagó). Lo que no puede volver es como predicado del WHERE
    (`AND pc.fecha_canc IS NOT NULL`), que descarta filas enteras.
    """
    sub = _subconsulta_max_atraso(REPO)
    assert re.search(r"AND\s+pc\.fecha_canc\s+is\s+NOT\s+null", sub, re.I) is None, (
        "la subconsulta de max_atraso volvió a FILTRAR sólo cuotas pagadas: "
        "eso ignora las impagas vigentes y produce un máximo histórico menor "
        "que el atraso actual"
    )
    # Y la condición sí tiene que estar como rama del CASE.
    assert re.search(r"WHEN\s+pc\.fecha_canc\s+IS\s+NOT\s+NULL", sub, re.I), (
        "falta la rama que usa la fecha de pago para las cuotas ya canceladas"
    )


def test_el_maximo_atraso_mide_las_impagas_contra_hoy():
    """La cuota impaga se mide contra GETDATE(), igual que la feature ML mide
    contra su cutoff. Sin esa rama, la mora viva no entra en el máximo."""
    sub = _subconsulta_max_atraso(REPO)
    assert "ELSE GETDATE()" in sub, (
        "falta la rama que mide la cuota impaga contra hoy"
    )
    assert re.search(r"pc\.vencimiento\s*<=\s*GETDATE\(\)", sub, re.I), (
        "falta acotar a cuotas ya vencidas"
    )


def test_las_dos_definiciones_tienen_la_misma_forma():
    """Ambas fuentes miden `DATEDIFF(vencimiento -> fecha de pago o corte)` sobre
    las cuotas ya vencidas, y pisan los negativos a 0. Si alguien cambia una sola,
    esto falla."""
    sub = _subconsulta_max_atraso(REPO)
    feat = FEATURES.read_text(encoding="utf-8")

    # La feature usa el parámetro @T como corte; el informe usa GETDATE().
    for nombre, sql, corte in (("informe", sub, r"GETDATE\(\)"), ("feature", feat, r"@T")):
        assert re.search(rf"fecha_canc\s+IS\s+NOT\s+NULL\s+AND\s+pc\.fecha_canc\s*<=\s*{corte}",
                         sql, re.I | re.S), f"{nombre}: no usa la fecha de pago acotada al corte"
        assert re.search(rf"ELSE\s+{corte}", sql, re.I), \
            f"{nombre}: no mide la cuota impaga contra el corte"

    # Los atrasos negativos (pagó antes de vencer) cuentan como 0, no restan.
    assert re.search(r"CASE WHEN d\.dias > 0 THEN d\.dias ELSE 0 END", sub)
    assert re.search(r"CASE WHEN dias_mora > 0 THEN dias_mora ELSE 0 END", feat)
