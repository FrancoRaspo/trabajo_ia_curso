"""La sección Historial se arma con código, así que sus números se pueden testear.

Ese es el punto de la plantilla: cuando la escribía el LLM, la única forma de
saber si contaba bien la cartera era generar un informe y preguntarle al juez.
Los casos de acá son los que el juez marcaba como alucinación CRÍTICA: conteos
que no coincidían con la lista, el período cerrado en un año anterior al último
préstamo, y "sin atrasos" con mora vigente sobre la mesa.
"""
from __future__ import annotations

import pytest

from llm.plantillas import cartera_frase, historial_markdown
from llm.report_sections import _recortar_secciones_fantasma
from pipeline import _resumen_cartera


def contexto(prestamos, pagos=None, gestiones=None) -> dict:
    """Slice de Historial armado como lo arma el pipeline (mismo `cartera_resumen`)."""
    return {
        "historial_prestamos": prestamos,
        "pagos_resumen": pagos or {},
        "gestiones_mora": gestiones or [],
        "cartera_resumen": _resumen_cartera(prestamos),
    }


CARTERA = [
    {"prestamo_id": "1-2-3/4", "fecha_otorgamiento": "2001-03-12",
     "fecha_otorgamiento_texto": "12/03/2001", "monto_original": 100_000,
     "saldo_capital_actual": 0, "estado": "CANCELADO", "dias_atraso_actual": 0},
    {"prestamo_id": "1-2-3/5", "fecha_otorgamiento": "2024-10-03",
     "fecha_otorgamiento_texto": "03/10/2024", "monto_original": 209_880,
     "saldo_capital_actual": 45_131, "estado": "VIGENTE", "cuotas_pendientes": 12,
     "dias_atraso_actual": 120},
]


def test_agregados_salen_del_resumen_no_de_un_conteo_a_mano():
    md = historial_markdown(contexto(CARTERA))
    assert "2 préstamos registrados" in md
    assert "1 cancelado, 1 vigente" in md
    # El período llega hasta el ÚLTIMO préstamo: el modelo lo cerraba antes.
    assert "entre el 12/03/2001 y el 03/10/2024" in md
    assert "$309.880,00" in md          # monto original acumulado
    assert "$45.131,00" in md           # saldo vivo de los vigentes


def test_mora_vigente_grave_queda_explicita():
    md = historial_markdown(contexto(CARTERA))
    assert "MORA VIGENTE GRAVE" in md   # 120 días > 90
    assert "el mayor atraso ACTUAL es de 120 días" in md


def test_cuotas_puntuales_y_atraso_maximo():
    pagos = {"total_cuotas": 24, "cuotas_puntuales": 20, "atraso_leve_1_30d": 3,
             "atraso_moderado_31_90d": 0, "atraso_grave_mas90d": 1,
             "cuotas_impagas_vencidas": 1, "promedio_dias_atraso": 12.4,
             "max_dias_atraso_periodo": 95}
    md = historial_markdown(contexto(CARTERA, pagos=pagos))
    assert "20 de 24 cuotas vencidas fueron puntuales" in md
    assert "Atraso promedio: 12,4 días" in md
    assert "antecedente grave" in md    # máximo 95 > 90
    assert "1 cuota vencida sigue impaga" in md
    assert "Sin atrasos registrados" not in md


def test_sin_cuotas_vencidas_no_es_dato_faltante():
    md = historial_markdown(contexto(CARTERA, pagos={"total_cuotas": 0}))
    assert "no venció ninguna cuota en el período" in md
    assert "no disponible" not in md


def test_sin_historial_no_se_narra_como_buen_comportamiento():
    md = historial_markdown(contexto([], pagos={"total_cuotas": 0}))
    assert "no posee historial crediticio previo" in md
    assert "no registra atrasos" not in md
    assert "no se registran gestiones de mora" in md


def test_gestiones_de_mora_se_listan():
    gestiones = [{"fecha_gestion": "2021-01-15", "tipo_gestion": "Llamado telefónico",
                  "resultado_contacto": "Compromiso de pago", "monto_comprometido": "$45.131,00",
                  "fecha_compromiso_pago": "2021-02-01"}]
    md = historial_markdown(contexto(CARTERA, gestiones=gestiones))
    assert "se registra 1 gestión" in md
    assert "Llamado telefónico" in md
    assert "$45.131,00" in md


def test_frase_de_cartera_es_la_misma_que_ve_el_resto_del_informe():
    """Las secciones interpretativas transcriben esta frase en vez de recontar."""
    frase = _resumen_cartera(CARTERA)["frase"]
    assert frase == cartera_frase(_resumen_cartera(CARTERA))
    assert "2 préstamos registrados" in frase
    assert historial_markdown(contexto(CARTERA)).startswith(f"**Cartera:** {frase}")


def test_no_se_publica_un_maximo_historico_agregado():
    """`max_dias_atraso_historico` de los préstamos NO coincide con la feature del
    mismo nombre que ve el modelo (una ignora las cuotas impagas vigentes). Publicar
    ese agregado metía un tercer número contradictorio: el informe no lo dice."""
    cartera = [dict(CARTERA[1], max_dias_atraso_historico=107)]
    md = historial_markdown(contexto(cartera))
    assert "el mayor atraso ACTUAL es de 120 días" in md    # el actual, nombrado como tal
    assert "máximo histórico" not in md.lower()


def test_seccion_fantasma_del_modelo_se_descarta():
    """El modelo a veces abre una sección propia y ahí recuenta la cartera mal.
    Todo lo que sigue a un encabezado suyo queda fuera del informe."""
    cuerpo = ("El score es 691 y el nivel de riesgo ALTO.\n\n"
              "## Resumen de la Cartera Crediticia\n"
              "El portafolio está compuesto por cinco registros detallados.")
    limpio = _recortar_secciones_fantasma(cuerpo)
    assert limpio == "El score es 691 y el nivel de riesgo ALTO."
    assert "cinco registros" not in limpio
    # Un cuerpo sin encabezados no se toca (más allá del strip).
    assert _recortar_secciones_fantasma("Texto normal.\nOtra línea.") == "Texto normal.\nOtra línea."


@pytest.mark.parametrize("valor", [None, [], {}])
def test_contexto_vacio_no_rompe(valor):
    historial_markdown({"historial_prestamos": valor, "pagos_resumen": valor,
                        "gestiones_mora": valor, "cartera_resumen": valor})
