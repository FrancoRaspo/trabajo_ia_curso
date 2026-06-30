"""Exportación del informe crediticio a PDF.

Convierte el informe en Markdown que produce el pipeline a un PDF descargable,
reusando el mismo contenido que la vista web. Dos pasos, ambos con librerías
puras de Python (sin dependencias de sistema):

    Markdown  --(python-markdown)-->  HTML  --(xhtml2pdf/pisa)-->  PDF

Se usa desde el endpoint /api/informe/pdf de app.py.
"""
import html
import io
import re

import markdown as _md
from xhtml2pdf import pisa


# CSS pensado para el subconjunto de CSS2 que soporta xhtml2pdf (pisa).
# Hoja blanca, tipografías base del motor (Helvetica/Times), pie con paginado.
_CSS = """
@page {
  size: a4 portrait;
  margin: 2.4cm 2cm 2cm 2cm;
  @frame footer {
    -pdf-frame-content: pie;
    bottom: 1cm; margin-left: 2cm; margin-right: 2cm; height: 1cm;
  }
}
body { font-family: Helvetica, sans-serif; font-size: 10.5pt; color: #15263b; line-height: 1.5; }

.membrete { border-bottom: 2px solid #15263b; padding-bottom: 8px; margin-bottom: 12px; }
.empresa { font-family: Times, serif; font-size: 17pt; font-weight: bold; }
.sub { color: #3d4f63; font-size: 8.5pt; margin-top: 2px; }

.meta { width: 100%; font-size: 9.5pt; color: #3d4f63; margin-bottom: 16px; }
.meta td { padding: 1px 0; }
.meta .k { color: #15263b; font-weight: bold; width: 22%; }
.meta .alerta { color: #9a2b2b; font-weight: bold; }

h1 { font-family: Times, serif; font-size: 17pt; margin: 0 0 12px; }
h2 { font-family: Times, serif; font-size: 13pt; margin: 16px 0 6px; padding-bottom: 3px;
     border-bottom: 1px solid #d9d3c7; }
h3 { font-size: 11pt; margin: 12px 0 5px; }
p { margin: 0 0 8px; }
ul, ol { margin: 0 0 8px; padding-left: 16px; }
li { margin: 2px 0; }
strong { color: #15263b; }
hr { border: 0; border-top: 1px solid #d9d3c7; margin: 12px 0; }
table.cuerpo-tabla { border-collapse: collapse; width: 100%; margin: 4px 0 12px; font-size: 9.5pt; }
table.cuerpo-tabla th, table.cuerpo-tabla td { border: 1px solid #d9d3c7; padding: 5px 7px; text-align: left; }
table.cuerpo-tabla th { background: #f7f5f0; font-weight: bold; }

#pie { color: #9aa6b2; font-size: 8pt; text-align: center; }
"""


def _fmt_cuit(cuit: str) -> str:
    d = re.sub(r"\D", "", cuit or "")
    return f"{d[:2]}-{d[2:10]}-{d[10:]}" if len(d) == 11 else (cuit or "")


def _esc(x) -> str:
    return html.escape("" if x is None else str(x))


def _membrete_html(*, empresa, numero, fecha, modelo, cuit, razon, motivo,
                   score, nivel, bcra, encontrado) -> str:
    sub = ["Informe crediticio"]
    if numero:
        sub.append(f"N° {_esc(numero)}")
    if fecha:
        sub.append(_esc(fecha))
    if modelo:
        sub.append(f"Modelo: {_esc(modelo)}")

    filas = [f'<tr><td class="k">CUIT</td><td>{_esc(_fmt_cuit(cuit))}</td></tr>']
    if not encontrado:
        filas.append(
            f'<tr><td class="k">Razón social</td>'
            f'<td>{_esc(razon)} <span class="alerta">(sin datos en la base)</span></td></tr>'
        )
    elif razon:
        filas.append(f'<tr><td class="k">Razón social</td><td>{_esc(razon)}</td></tr>')

    if score is not None:
        nivel_txt = f" &nbsp;·&nbsp; Nivel: {_esc(nivel)}" if nivel else ""
        filas.append(f'<tr><td class="k">Score</td><td>{_esc(score)}{nivel_txt}</td></tr>')
    elif nivel:
        filas.append(f'<tr><td class="k">Nivel</td><td>{_esc(nivel)}</td></tr>')

    if bcra:
        if bcra.get("tiene_datos"):
            sit = f"Situación {_esc(bcra.get('peor_situacion'))}"
            if bcra.get("peor_situacion_desc"):
                sit += f" ({_esc(bcra.get('peor_situacion_desc'))})"
            irregular = (bcra.get("peor_situacion") or 0) and int(bcra["peor_situacion"]) >= 3
            extra = []
            if bcra.get("cant_entidades"):
                extra.append(f"{_esc(bcra['cant_entidades'])} entidad(es)")
            if bcra.get("cheques_impagos"):
                extra.append(f"{_esc(bcra['cheques_impagos'])} cheque(s) impago(s)")
            val = f'<span class="alerta">{sit}</span>' if irregular else sit
            if extra:
                val += " &nbsp;·&nbsp; " + " &nbsp;·&nbsp; ".join(extra)
            filas.append(f'<tr><td class="k">BCRA</td><td>{val}</td></tr>')
        else:
            filas.append('<tr><td class="k">BCRA</td><td>sin registros en el sistema financiero</td></tr>')

    if motivo:
        filas.append(f'<tr><td class="k">Motivo</td><td>{_esc(motivo)}</td></tr>')

    return (
        '<div class="membrete">'
        f'<div class="empresa">{_esc(empresa)}</div>'
        f'<div class="sub">{" &nbsp;·&nbsp; ".join(sub)}</div>'
        '</div>'
        f'<table class="meta">{"".join(filas)}</table>'
    )


def _cuerpo_html(informe_md: str) -> str:
    """Markdown del informe -> HTML. Quita un posible H1/encabezado duplicado
    (el membrete ya lo provee) y marca las tablas con la clase de estilo."""
    md = informe_md or ""
    # El pipeline antepone un encabezado '# INFORME CREDITICIO...': lo sacamos,
    # porque el membrete ya cumple esa función en el PDF.
    md = re.sub(r"^\s*#\s+INFORME CREDITICIO.*?(?:\n|$)", "", md, count=1, flags=re.IGNORECASE)
    md = re.sub(r"^\s*\*\*Solicitud:\*\*.*?(?:\n|$)", "", md, count=1)
    md = re.sub(r"^\s*\*\*ID interno:\*\*.*?(?:\n|$)", "", md, count=1)

    cuerpo = _md.markdown(md, extensions=["tables", "sane_lists"])
    cuerpo = cuerpo.replace("<table>", '<table class="cuerpo-tabla">')
    return cuerpo


def informe_a_pdf(informe_md: str, *, empresa="", numero="", fecha="", modelo="",
                  cuit="", razon="", motivo="", score=None, nivel=None,
                  bcra=None, encontrado=True) -> bytes:
    """Renderiza el informe (Markdown) a un PDF (bytes) listo para descargar."""
    documento = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<style>{_CSS}</style></head><body>"
        + _membrete_html(empresa=empresa, numero=numero, fecha=fecha, modelo=modelo,
                         cuit=cuit, razon=razon, motivo=motivo, score=score,
                         nivel=nivel, bcra=bcra, encontrado=encontrado)
        + _cuerpo_html(informe_md)
        + '<div id="pie">Página <pdf:pagenumber> de <pdf:pagecount></div>'
        + "</body></html>"
    )

    salida = io.BytesIO()
    resultado = pisa.CreatePDF(src=documento, dest=salida, encoding="utf-8")
    if resultado.err:
        raise RuntimeError("No se pudo generar el PDF del informe.")
    return salida.getvalue()
