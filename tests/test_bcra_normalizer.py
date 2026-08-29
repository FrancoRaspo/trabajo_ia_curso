"""Tests offline del normalizer del BCRA (sin red ni base de datos)."""
from external import bcra_normalizer as norm


CRUDO_CON_DEUDA = {
    "deudas": {
        "status": 200,
        "results": {
            "identificacion": 20123456789,
            "denominacion": "PEREZ JUAN",
            "periodos": [
                {
                    "periodo": "202405",
                    "entidades": [
                        {"entidad": "BANCO A", "situacion": 2, "monto": 150000.0,
                         "diasAtrasoPago": 25, "refinanciaciones": False,
                         "enRevision": False, "procesoJud": False},
                        {"entidad": "BANCO B", "situacion": 4, "monto": 80000.5,
                         "diasAtrasoPago": 120, "refinanciaciones": True,
                         "enRevision": False, "procesoJud": True},
                    ],
                },
                {   # período más viejo: debe ignorarse para la situación vigente
                    "periodo": "202404",
                    "entidades": [
                        {"entidad": "BANCO A", "situacion": 1, "monto": 10.0,
                         "diasAtrasoPago": 0},
                    ],
                },
            ],
        },
    },
    "deudas_historicas": None,
    "cheques_rechazados": {
        "status": 200,
        "results": {
            "causales": [
                {"causal": "SIN FONDOS", "entidades": [
                    {"entidad": "BANCO A", "detalle": [
                        {"nroCheque": 111, "fechaRechazo": "2024-01-10",
                         "monto": 5000.0, "fechaPago": "2024-02-01", "procesoJud": False},
                        {"nroCheque": 222, "fechaRechazo": "2024-03-15",
                         "monto": 7000.0, "fechaPago": None, "procesoJud": False},
                    ]},
                ]},
            ],
        },
    },
}

CRUDO_VACIO = {"deudas": None, "deudas_historicas": None, "cheques_rechazados": None}


def test_resumir_con_deuda():
    r = norm.resumir(CRUDO_CON_DEUDA)
    assert r["tiene_datos"] is True
    assert r["periodo_informado"] == "202405"        # toma el período más reciente
    assert r["cant_entidades"] == 2
    assert r["peor_situacion"] == 4                   # max(2, 4)
    assert r["tiene_situacion_irregular"] is True     # >= 3
    assert r["deuda_total_sistema"] == 230000500.0    # BCRA informa en miles de pesos
    assert r["proceso_judicial"] is True
    assert r["refinanciaciones"] is True
    assert r["cheques_rechazados_total"] == 2
    assert r["cheques_impagos"] == 1                  # el de fechaPago None
    assert r["monto_cheques_impagos"] == 7000000.0    # BCRA informa en miles de pesos


def test_resumir_vacio():
    r = norm.resumir(CRUDO_VACIO)
    assert r["tiene_datos"] is False
    assert r["peor_situacion"] is None
    assert r["cant_entidades"] == 0
    assert r["cheques_rechazados_total"] == 0


def test_a_texto_vacio_es_explicito():
    texto = norm.a_texto(norm.resumir(CRUDO_VACIO), "20123456789")
    assert "sin registros" in texto.lower()


def test_a_texto_con_deuda_menciona_peor_situacion():
    texto = norm.a_texto(norm.resumir(CRUDO_CON_DEUDA), "20123456789")
    assert "BANCO B" in texto
    assert "proceso judicial" in texto.lower()
    assert "impago" in texto.lower()
