"""
Prompts del sistema, externalizados en archivos .txt (uno por prompt).

La idea es que un experto de dominio (analista de crédito) pueda LEER y EDITAR
los prompts sin tocar código Python. Cada .txt de esta carpeta es el texto que se
le envía al modelo. Los `{marcadores}` entre llaves se rellenan en tiempo de
ejecución con datos del informe (ver prompts/README.md).

Uso desde el código:
    from prompts import load
    texto = load("recomendacion")            # lee prompts/recomendacion.txt

No se cachea a propósito: cada informe relee el .txt, así un cambio en el archivo
impacta en el próximo informe SIN reiniciar el proceso.
"""
from pathlib import Path

_DIR = Path(__file__).resolve().parent


def load(nombre: str) -> str:
    """Devuelve el contenido de prompts/<nombre>.txt (sin el salto de línea final
    que suelen agregar los editores). Lanza FileNotFoundError si no existe."""
    ruta = _DIR / f"{nombre}.txt"
    return ruta.read_text(encoding="utf-8").rstrip("\n")
