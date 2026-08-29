"""Refresca la caché BCRA para una lista de CUITs (job batch / mensual).

Uso:
    python -m scripts.sync_bcra 20123456789 27123456784
    python -m scripts.sync_bcra --file cuits.txt          # un CUIT por línea
    python -m scripts.sync_bcra --file cuits.txt --forzar  # ignora el TTL

Pensado para correr una vez al mes (cron / launchd), después del cierre
mensual del BCRA. Pasale los CUITs de los asociados a refrescar.
"""
import argparse
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from external import perfil_bcra
from external.bcra_cache import asegurar_tabla
from external.bcra_client import solo_digitos


def _leer_cuits(args) -> list[str]:
    cuits = list(args.cuits)
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            cuits += [ln.strip() for ln in f if ln.strip()]
    vistos, salida = set(), []
    for c in cuits:
        d = solo_digitos(c)
        if len(d) == 11 and d not in vistos:
            vistos.add(d)
            salida.append(d)
    return salida


def main():
    ap = argparse.ArgumentParser(description="Sincroniza la caché BCRA por CUIT.")
    ap.add_argument("cuits", nargs="*", help="CUITs de 11 dígitos")
    ap.add_argument("--file", help="Archivo con un CUIT por línea")
    ap.add_argument("--forzar", action="store_true",
                    help="Ignora el TTL y reconsulta todos los CUITs")
    args = ap.parse_args()

    cuits = _leer_cuits(args)
    if not cuits:
        print("No se recibieron CUITs válidos. Pasá CUITs o --file.", file=sys.stderr)
        sys.exit(1)

    asegurar_tabla()
    con_datos = sin_datos = fallos = 0
    for i, cuit in enumerate(cuits, 1):
        perfil = perfil_bcra(cuit, forzar=args.forzar)
        if perfil is None:
            fallos += 1
            print(f"[{i}/{len(cuits)}] {cuit}: sin perfil (deshabilitado o falla de API)")
            continue
        r = perfil["resumen"]
        if r.get("tiene_datos"):
            con_datos += 1
            print(f"[{i}/{len(cuits)}] {cuit}: peor situación {r.get('peor_situacion')} "
                  f"| {r.get('cant_entidades')} entidad(es) "
                  f"| {r.get('cheques_impagos')} cheque(s) impago(s)")
        else:
            sin_datos += 1
            print(f"[{i}/{len(cuits)}] {cuit}: sin registros en BCRA")

    print(f"\nListo. Con datos: {con_datos} | Sin registros: {sin_datos} | Fallos: {fallos}")


if __name__ == "__main__":
    main()
