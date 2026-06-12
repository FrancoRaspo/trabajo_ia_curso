"""
Genera un informe para un cliente y lo audita con el juez LLM.

Va en evaluation/run_judge.py

Requiere en .env:
    JUEZ_LLM_PROVIDER y el API KEY correspondiente
"""
import sys
import os
import re
import json

# raíz del proyecto en el path. Este archivo está en evaluation/, así que la
# raíz es DOS niveles arriba (dirname dos veces).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from pipeline import get_pipeline
from evaluation.llm_judge import evaluar_informe


def main():
    if len(sys.argv) < 3:
        print('Uso: python evaluation/run_judge.py <CUIT> "<motivo>"')
        sys.exit(1)
    cuit_raw, motivo = sys.argv[1], sys.argv[2]

    # Normalizar el CUIT a SOLO dígitos: "23-25079318-9" -> "23250793189".
    # Los guiones rompen las consultas SQL.
    cuit = re.sub(r"\D", "", cuit_raw)
    if not cuit:
        print(f"CUIT inválido: {cuit_raw!r}")
        sys.exit(1)
    if cuit != cuit_raw:
        print(f"CUIT normalizado: {cuit_raw} -> {cuit}")

    # 1) Generar el informe (usa el generador configurado en PROVEEDOR_LLM)
    pipeline = get_pipeline(os.environ.get("PROVEEDOR_LLM", "local"))
    res = pipeline.generar(cuit, motivo)

    print("\n" + "=" * 72)
    print("INFORME GENERADO\n")
    print(res["informe"])
    print("=" * 72)

    # 2) Auditar con el juez (modelo de JUEZ_LLM_PROVIDER, más fuerte)
    print("\nAuditando con el juez LLM...\n")
    veredicto = evaluar_informe(res["contexto"], res["informe"])

    # 3) Mostrar el resultado
    print("VEREDICTO:", veredicto.get("veredicto"))
    print("Resumen:  ", veredicto.get("resumen"))

    for criterio in ("fidelidad", "exactitud_numerica", "estructura",
                     "uso_drivers_cliente", "coherencia_recomendacion"):
        c = veredicto.get(criterio)
        if isinstance(c, dict):
            print(f"  - {criterio}: {c.get('puntaje')}/5 — {c.get('comentario', '')}")

    reglas = veredicto.get("cumplimiento_reglas")
    if isinstance(reglas, dict):
        print(f"  - cumplimiento_reglas: {'OK' if reglas.get('aprobado') else 'FALLA'} "
              f"— {reglas.get('comentario', '')}")

    aluc = veredicto.get("alucinaciones") or []
    if aluc:
        print("\n⚠ Alucinaciones detectadas:")
        for a in aluc:
            print("   -", a)

    print("\nJSON completo del veredicto:")
    print(json.dumps(veredicto, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()