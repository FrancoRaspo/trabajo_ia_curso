# scripts/entrenar_modelo_temporal.py
#
# Entrena el modelo de scoring con separación temporal real entre las features
# y el target, y valida fuera de tiempo (out-of-time).
#
# Reemplaza a scripts/entrenar_modelo_real.py, que tenía fuga de datos: allí el
# target ("clasificación > 4 O atraso > 90d", medido HOY) se calculaba en el
# mismo instante que features derivadas de ese mismo atraso. El modelo no
# predecía el default: lo redescribía. El AUC resultante no significaba nada.
#
# Diseño (panel / stacked cutoffs, el estándar en credit scoring):
#
#   Para cada fecha de corte T:
#     - Features:  SOLO datos con fecha <= T
#     - Target:    ¿alcanza 90 días de mora en (T, T+12 meses]?
#     - Población: clientes NO defaulteados a T y CON exposición en la ventana
#
#   Entrenamiento: cortes viejos. n_estimators sale de una CV de ventana
#                  expansiva sobre esos mismos cortes, y la calibración de
#                  probabilidades se ajusta con sus predicciones out-of-fold.
#   Validación:    el corte siguiente, como chequeo out-of-time intermedio.
#   Test:          los cortes más recientes, nunca vistos.  <- el único AUC reportable
#
# Uso:
#   python3 scripts/entrenar_modelo_temporal.py
#   python3 scripts/entrenar_modelo_temporal.py --excluir-covid
#   python3 scripts/entrenar_modelo_temporal.py --sin-cache

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)

from db.sqlserver_connection import get_sqlserver_engine
from ml.feature_query import FEATURE_COLUMNS, cargar_features, filtrar_elegibles

# ─────────────────────────────────────────────────────────────────────────────
# Configuración del panel
# ─────────────────────────────────────────────────────────────────────────────

CUTOFFS_TRAIN = [f"{a}-12-31" for a in range(2015, 2023)]   # 2015..2022
CUTOFF_VALID  = "2023-12-31"                                # chequeo out-of-time
CUTOFFS_TEST  = ["2024-12-31", "2025-04-30"]                # out-of-time

# Ventanas de performance que caen dentro de las moratorias de COVID: la mora
# observada está artificialmente deprimida (ver tasas 2020/2021 vs 2018/2019).
CUTOFFS_COVID = ["2019-12-31", "2020-12-31"]

HORIZONTE_MESES = 12
CACHE_DIR = ".cache_features"


# ─────────────────────────────────────────────────────────────────────────────
# Extracción (con caché en disco: la query tarda ~1 min por corte)
# ─────────────────────────────────────────────────────────────────────────────

def extraer(engine, cutoff: str, usar_cache: bool) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"features_{cutoff}_{HORIZONTE_MESES}m.pkl")

    if usar_cache and os.path.exists(path):
        print(f"  {cutoff}  (caché)")
        return pd.read_pickle(path)

    print(f"  {cutoff}  consultando...", flush=True)
    df = cargar_features(engine, cutoff, HORIZONTE_MESES)
    df.to_pickle(path)
    return df


def construir_panel(engine, cutoffs: list[str], usar_cache: bool) -> pd.DataFrame:
    partes = []
    for T in cutoffs:
        df = filtrar_elegibles(extraer(engine, T, usar_cache))
        df["cutoff"] = T
        partes.append(df)
    return pd.concat(partes, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Métricas
# ─────────────────────────────────────────────────────────────────────────────

def ks(y_true, y_score) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def wilson(k: int, n: int) -> tuple[float, float]:
    """IC 95% de Wilson para una proporción. Con n chico el IC normal miente."""
    if n == 0:
        return (0.0, 0.0)
    z, p = 1.96, k / n
    d = 1 + z ** 2 / n
    centro = (p + z ** 2 / (2 * n)) / d
    semi = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return (max(0.0, centro - semi), min(1.0, centro + semi))


# ─────────────────────────────────────────────────────────────────────────────
# Calibración
#
# El informe hace score = (1 - PD) * 1000 y corta la PD en bandas de riesgo, así
# que la salida del booster tiene que ser una PROBABILIDAD, no un puntaje de
# ranking. La pregunta es si hace falta transformarla, y eso NO se asume: se
# mide.
#
# La versión anterior imponía Platt con este argumento: "predict_proba() no es
# una probabilidad calibrada; es un puntaje de ranking". Es falso para este
# modelo. El objetivo es binary:logistic (optimiza log-loss, que es un scoring
# rule propio) y scale_pos_weight se omite a propósito. La salida cruda YA es
# una probabilidad. Platt encima la aplastaba: degradaba el Brier de test de
# 0.0442 a 0.0452 y rompía la banda MODERADO (rango 6.4-12.8%, tasa real 19.5%).
#
# Ahora se comparan tres candidatos por CV temporal anidada sobre las OOF —
# ajustar en los cortes viejos, medir en el siguiente — y gana el de mejor Brier
# FUERA de muestra. Si un reentrenamiento futuro produce un modelo mal calibrado,
# esto lo detecta y elige la corrección; hoy elige no tocar nada.
# ─────────────────────────────────────────────────────────────────────────────

PISO_PD = 0.005     # PD mínima reportable: "menos de 0,5%" ya es el piso del informe
_EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def ajustar_calibradores(x: np.ndarray, y: np.ndarray) -> dict:
    """Devuelve {nombre: (fn, artefacto_serializable)} ajustados sobre (x, y)."""
    pl = LogisticRegression(C=1e10, solver="lbfgs").fit(x.reshape(-1, 1), y)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(x, y)

    # La isotónica se serializa como los nodos de su escalera.
    xs = np.unique(x)
    ys = iso.predict(xs)

    return {
        "ninguna": (
            lambda p: np.asarray(p, dtype=float),
            {"metodo": "ninguna"},
        ),
        "platt_sigmoid": (
            lambda p: pl.predict_proba(np.asarray(p).reshape(-1, 1))[:, 1],
            {"metodo": "platt_sigmoid",
             "w": float(pl.coef_[0][0]), "b": float(pl.intercept_[0]),
             "formula": "p_cal = 1 / (1 + exp(-(w * p_cruda + b)))"},
        ),
        "isotonica": (
            lambda p: np.clip(iso.predict(np.asarray(p)), PISO_PD, 1.0),
            {"metodo": "isotonica", "piso": PISO_PD,
             "x": [round(float(v), 6) for v in xs],
             "y": [round(float(v), 6) for v in ys],
             "formula": "interpolación lineal entre (x, y), con piso"},
        ),
    }


def elegir_calibracion(oof_raw: np.ndarray, oof_y: np.ndarray,
                       oof_cut: np.ndarray) -> tuple[str, dict, list]:
    """Elige el calibrador por Brier out-of-sample en CV temporal anidada."""
    cortes = sorted(set(oof_cut))
    brier = {n: [] for n in ("ninguna", "platt_sigmoid", "isotonica")}
    filas = []

    for j in range(2, len(cortes)):
        fit_c, val_c = cortes[:j], cortes[j]
        mf, mv = np.isin(oof_cut, fit_c), (oof_cut == val_c)
        if oof_y[mv].sum() < 5:
            continue
        cals = ajustar_calibradores(oof_raw[mf], oof_y[mf])
        fila = {"calibra_en": f"≤{fit_c[-1]}", "mide_en": val_c,
                "n": int(mv.sum()), "defaults": int(oof_y[mv].sum())}
        for nom, (fn, _) in cals.items():
            b = float(brier_score_loss(oof_y[mv], fn(oof_raw[mv])))
            brier[nom].append(b)
            fila[nom] = round(b, 5)
        filas.append(fila)

    medias = {n: float(np.mean(v)) for n, v in brier.items() if v}
    ganador = min(medias, key=medias.get) if medias else "ninguna"

    # Con el ganador decidido, se reajusta sobre TODAS las OOF.
    artefacto = ajustar_calibradores(oof_raw, oof_y)[ganador][1]
    artefacto["seleccion"] = {
        "criterio": "menor Brier out-of-sample en CV temporal anidada sobre las OOF",
        "brier_medio_por_metodo": {n: round(v, 5) for n, v in medias.items()},
        "folds": filas,
    }
    return ganador, artefacto, filas


def aplicar_calibracion(art: dict, p):
    """Aplica el artefacto elegido. Espeja ml.scoring_model._calibrar."""
    p = np.asarray(p, dtype=float)
    m = art["metodo"]
    if m == "ninguna":
        return p
    if m == "platt_sigmoid":
        return 1.0 / (1.0 + np.exp(-(art["w"] * p + art["b"])))
    if m == "isotonica":
        return np.clip(np.interp(p, art["x"], art["y"]), art["piso"], 1.0)
    raise ValueError(f"Método desconocido: {m!r}")


def metricas(y_true, y_score, es_probabilidad: bool = True) -> dict:
    prevalencia = float(np.mean(y_true))
    m = {
        "n": int(len(y_true)),
        "defaults": int(np.sum(y_true)),
        "prevalencia": round(prevalencia, 4),
        "auc_roc": round(float(roc_auc_score(y_true, y_score)), 4),
        # PR-AUC importa más que AUC-ROC con 5% de prevalencia. El baseline de
        # un modelo inútil es la prevalencia misma, no 0.5.
        "pr_auc": round(float(average_precision_score(y_true, y_score)), 4),
        "pr_auc_baseline": round(prevalencia, 4),
        "ks": round(ks(y_true, y_score), 4),
    }
    # Brier sólo tiene sentido sobre probabilidades calibradas. El baseline es
    # un score de ranking (días de atraso), no una probabilidad.
    m["brier"] = round(float(brier_score_loss(y_true, y_score)), 4) if es_probabilidad else None
    return m


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--excluir-covid", action="store_true",
                    help="Descarta los cortes cuya ventana cae en las moratorias 2020-2021.")
    ap.add_argument("--sin-cache", action="store_true",
                    help="Ignora el caché en disco y vuelve a consultar la base.")
    ap.add_argument("--salida", default="models/scoring_model.json")
    args = ap.parse_args()

    cutoffs_train = list(CUTOFFS_TRAIN)
    if args.excluir_covid:
        cutoffs_train = [c for c in cutoffs_train if c not in CUTOFFS_COVID]
        print(f"Excluyendo cortes COVID: {CUTOFFS_COVID}")

    engine = get_sqlserver_engine()
    usar_cache = not args.sin_cache

    print(f"\nExtrayendo panel. Ventana de performance: {HORIZONTE_MESES} meses.")
    print("TRAIN:")
    tr = construir_panel(engine, cutoffs_train, usar_cache)
    print("VALID:")
    va = construir_panel(engine, [CUTOFF_VALID], usar_cache)
    print("TEST:")
    te = construir_panel(engine, CUTOFFS_TEST, usar_cache)

    # ── Tamaño del dataset. El profesor pidió explícitamente reportarlo. ──
    print("\n" + "=" * 74)
    print("TAMAÑO DEL DATASET (sólo población elegible)")
    print("=" * 74)
    print(f"{'corte':<14}{'filas':>8}{'defaults':>10}{'tasa':>9}")
    for nombre, df in (("TRAIN", tr), ("VALID", va), ("TEST", te)):
        print(f"\n  {nombre}")
        for T, g in df.groupby("cutoff", sort=True):
            print(f"  {T:<14}{len(g):>8}{int(g.default_90d.sum()):>10}{g.default_90d.mean():>8.2%}")
        print(f"  {'TOTAL':<14}{len(df):>8}{int(df.default_90d.sum()):>10}{df.default_90d.mean():>8.2%}")

    if tr.default_90d.sum() < 50:
        print("\n⚠️  Menos de 50 defaults en train. El modelo no será confiable.")
        sys.exit(1)

    # ── Solapamiento de clientes entre train y test. ──────────────────────
    # En un panel el mismo cliente aparece en varios cortes. La validación
    # out-of-time sigue siendo válida (es lo que hace la industria), pero el
    # dato hay que declararlo, no esconderlo.
    clientes_tr = set(tr.cliente_id)
    clientes_te = set(te.cliente_id)
    solap = len(clientes_tr & clientes_te) / max(len(clientes_te), 1)
    print(f"\nClientes de TEST ya vistos en TRAIN: {solap:.1%} "
          f"({len(clientes_tr & clientes_te)}/{len(clientes_te)})")

    # ── Features constantes: no aportan y ensucian el model card. ─────────
    usadas = [c for c in FEATURE_COLUMNS if tr[c].nunique() > 1]
    descartadas = [c for c in FEATURE_COLUMNS if c not in usadas]
    if descartadas:
        print(f"\nFeatures constantes en TRAIN, descartadas: {descartadas}")

    Xtr, ytr = tr[usadas].fillna(0), tr.default_90d
    Xva, yva = va[usadas].fillna(0), va.default_90d
    Xte, yte = te[usadas].fillna(0), te.default_90d

    epv = int(ytr.sum()) / max(len(usadas), 1)
    print(f"Eventos por variable (EPV): {epv:.1f}  "
          f"({int(ytr.sum())} defaults / {len(usadas)} features)"
          f"{'' if epv >= 10 else '   ⚠️ por debajo de 10'}")

    # ── Selección de n_estimators por CV de ventana expansiva ──────────────
    #
    # NO se usa early stopping contra un único corte de validación: ese corte
    # tiene ~26 eventos positivos y el AUCPR sobre 26 positivos es ruido puro.
    # Con early_stopping_rounds la primera fluctuación corta el boosting en el
    # árbol 0 y el "modelo" termina siendo un único árbol.
    #
    # En su lugar: entrenar sobre cortes [0..i) y validar sobre el corte i,
    # para cada i. Cada fold aporta su mejor iteración; se toma la mediana.
    # Así la decisión se apoya en los 475 eventos del panel, no en 26.
    def params_base(**extra) -> dict:
        p = dict(
            max_depth=3,           # el panel es chico: árboles poco profundos
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            reg_lambda=5.0,
            eval_metric="aucpr",
            random_state=42,
            nthread=1,
        )
        # scale_pos_weight se omite a propósito: infla predict_proba y la aleja
        # de una probabilidad, y el informe convierte prob -> score -> banda de
        # riesgo. El desbalance (6%) lo absorbe binary:logistic.
        p.update(extra)
        return p

    print("\nSeleccionando n_estimators con CV de ventana expansiva...")
    mejores = []
    oof_raw, oof_y, oof_cut = [], [], []   # predicciones out-of-fold, para calibrar
    for i in range(2, len(cutoffs_train)):
        c_fit, c_val = cutoffs_train[:i], cutoffs_train[i]
        f, v = tr[tr.cutoff.isin(c_fit)], tr[tr.cutoff == c_val]
        if v.default_90d.sum() < 5:
            continue
        m = xgb.XGBClassifier(n_estimators=800, early_stopping_rounds=60, **params_base())
        m.fit(f[usadas].fillna(0), f.default_90d,
              eval_set=[(v[usadas].fillna(0), v.default_90d)], verbose=False)
        mejores.append(m.best_iteration + 1)
        oof_raw.append(m.predict_proba(v[usadas].fillna(0))[:, 1])
        oof_y.append(v.default_90d.values)
        oof_cut.append(np.repeat(c_val, len(v)))
        print(f"  fit≤{c_fit[-1]}  valid={c_val}  "
              f"(pos={int(v.default_90d.sum())})  mejor_iter={m.best_iteration + 1}")

    n_est = int(np.median(mejores)) if mejores else 100
    print(f"  -> n_estimators = {n_est} (mediana de {mejores})")

    oof_raw = np.concatenate(oof_raw)
    oof_y   = np.concatenate(oof_y)
    oof_cut = np.concatenate(oof_cut)

    print("\nEntrenando modelo final sobre todo el TRAIN...")
    modelo = xgb.XGBClassifier(n_estimators=n_est, **params_base())
    modelo.fit(Xtr, ytr, verbose=False)

    # ── Calibración: se ELIGE, no se asume (ver bloque de arriba) ───────────
    print(f"\nEligiendo calibración sobre {len(oof_y)} predicciones out-of-fold "
          f"({int(oof_y.sum())} defaults, prevalencia {oof_y.mean():.2%})...")
    metodo_cal, cal_art, folds_cal = elegir_calibracion(oof_raw, oof_y, oof_cut)

    print(f"{'calibra en':<14}{'mide en':<14}{'n':>6}{'def':>5}"
          f"{'ninguna':>10}{'platt':>10}{'isotónica':>11}")
    for f_ in folds_cal:
        print(f"{f_['calibra_en']:<14}{f_['mide_en']:<14}{f_['n']:>6}{f_['defaults']:>5}"
              f"{f_['ninguna']:>10.5f}{f_['platt_sigmoid']:>10.5f}{f_['isotonica']:>11.5f}")
    medias = cal_art["seleccion"]["brier_medio_por_metodo"]
    print(f"{'PROMEDIO':<34}{'':>6}{medias['ninguna']:>10.5f}"
          f"{medias['platt_sigmoid']:>10.5f}{medias['isotonica']:>11.5f}")
    print(f"  -> calibración elegida: {metodo_cal.upper()} "
          f"(menor Brier fuera de muestra)")

    def calibrar(X):
        return aplicar_calibracion(cal_art, modelo.predict_proba(X)[:, 1])

    # ── Evaluación ─────────────────────────────────────────────────────────
    # TEST se toca UNA sola vez, acá. No se usó ni para elegir n_estimators
    # ni para elegir la calibración.
    p_tr, p_va, p_te = calibrar(Xtr), calibrar(Xva), calibrar(Xte)

    # Cuánto aportó (o costó) la calibración: Brier del modelo crudo vs elegido.
    p_te_crudo = modelo.predict_proba(Xte)[:, 1]
    brier_crudo = float(brier_score_loss(yte, p_te_crudo))

    m_tr, m_va, m_te = metricas(ytr, p_tr), metricas(yva, p_va), metricas(yte, p_te)

    # Baseline honesto: ordenar por el atraso máximo pasado, sin modelo.
    base_col = "max_dias_atraso_historico"
    m_base = metricas(yte, te[base_col].values, es_probabilidad=False)

    print("\n" + "=" * 74)
    print("MÉTRICAS")
    print("=" * 74)
    cab = f"{'':<10}{'n':>7}{'def':>6}{'prev':>8}{'AUC':>8}{'PR-AUC':>9}{'KS':>7}{'Brier':>8}"
    print(cab)
    for nom, m in (("TRAIN", m_tr), ("VALID", m_va), ("TEST", m_te)):
        print(f"{nom:<10}{m['n']:>7}{m['defaults']:>6}{m['prevalencia']:>8.2%}"
              f"{m['auc_roc']:>8.3f}{m['pr_auc']:>9.3f}{m['ks']:>7.3f}{m['brier']:>8.4f}")
    print(f"{'baseline':<10}{m_base['n']:>7}{m_base['defaults']:>6}{m_base['prevalencia']:>8.2%}"
          f"{m_base['auc_roc']:>8.3f}{m_base['pr_auc']:>9.3f}{m_base['ks']:>7.3f}{'—':>8}"
          f"   <- sólo {base_col}")
    print(f"\nPR-AUC de un modelo inútil (= prevalencia): {m_te['pr_auc_baseline']:.3f}")
    print(f"Brier en TEST: {brier_crudo:.4f} (crudo)  ->  {m_te['brier']:.4f} (calibrado). "
          f"Prob. media predicha: {p_te.mean():.3f}  vs  prevalencia real {yte.mean():.3f}")

    # Un umbral fijo de 0.5 no dice nada con 5,6% de prevalencia: el modelo
    # casi nunca cruza 0.5 y el recall queda en ~0. Lo que importa en crédito
    # es si el score ORDENA bien el riesgo. Eso se lee en la tabla de deciles.
    print("\nTEST por decil de riesgo predicho (decil 1 = mayor riesgo):")
    dec = pd.DataFrame({"p": p_te, "y": yte.values})
    dec["decil"] = pd.qcut(dec.p.rank(method="first", ascending=False),
                           10, labels=range(1, 11))
    tabla = dec.groupby("decil", observed=True).agg(
        n=("y", "size"), defaults=("y", "sum"), tasa=("y", "mean"),
        p_media=("p", "mean"))
    tabla["lift"] = tabla.tasa / yte.mean()
    print(f"{'decil':>6}{'n':>7}{'defaults':>10}{'tasa_real':>11}{'p_media':>10}{'lift':>7}")
    for d, r in tabla.iterrows():
        print(f"{d:>6}{int(r.n):>7}{int(r.defaults):>10}{r.tasa:>10.1%}"
              f"{r.p_media:>10.1%}{r.lift:>7.2f}")
    capt = tabla.defaults.iloc[:3].sum() / tabla.defaults.sum()
    print(f"\nLos 3 deciles más riesgosos capturan {capt:.0%} de los defaults.")

    # ── Bandas de riesgo ───────────────────────────────────────────────────
    #
    # Las bandas viejas cortaban el score (= (1-PD)*1000) en 800/650/500. Eso
    # sólo funcionaba porque scale_pos_weight inflaba las probabilidades y las
    # esparcía por todo el rango [0,1]. Con PD calibrada la masa vive entre
    # 0.02 y 0.25, el score entre ~750 y ~980, y las bandas ALTO / MUY ALTO
    # se vuelven inalcanzables: todo cliente saldría "BAJO".
    #
    # Se definen sobre la PD, como múltiplos de la tasa de default de la
    # cartera (la prevalencia out-of-fold). Es interpretable para el comité:
    # "este cliente tiene 3x el riesgo promedio de la cartera".
    #
    # Y se VERIFICAN: una banda que promete PD 6-13% pero cuyos clientes
    # defaultean al 19% no es una banda, es una mentira con color. El chequeo
    # compara la tasa observada contra el rango prometido usando el IC 95% de
    # Wilson, para no confundir una banda rota con el ruido de pocos casos.
    tasa_cartera = float(oof_y.mean())
    bandas = [
        ("MUY ALTO", "ROJO",     4.0 * tasa_cartera, 1.01),
        ("ALTO",     "NARANJA",  2.0 * tasa_cartera, 4.0 * tasa_cartera),
        ("MODERADO", "AMARILLO", 1.0 * tasa_cartera, 2.0 * tasa_cartera),
        ("BAJO",     "VERDE",    -0.01,              1.0 * tasa_cartera),
    ]

    def evaluar_bandas(p, y, etiqueta):
        print(f"\nBandas de riesgo en {etiqueta} "
              f"(tasa de default de la cartera = {tasa_cartera:.2%}):")
        print(f"{'banda':<11}{'PD desde':>10}{'PD hasta':>10}{'n':>7}"
              f"{'defaults':>10}{'tasa_real':>11}{'IC95%':>18}  veredicto")
        dist, rotas = {}, []
        y = np.asarray(y)
        for nombre, _color, lo, hi in bandas:
            sel = (p > lo) & (p <= hi)
            n, d = int(sel.sum()), int(y[sel].sum())
            tasa = d / n if n else 0.0
            ci_lo, ci_hi = wilson(d, n)
            if n == 0:
                veredicto = "vacía"
            elif ci_lo > hi:
                veredicto, _ = "❌ SUB-PREDICE", rotas.append(nombre)
            elif ci_hi < max(lo, 0):
                veredicto, _ = "❌ SOBRE-PREDICE", rotas.append(nombre)
            else:
                veredicto = "✅"
            dist[nombre] = {"n": n, "defaults": d, "tasa_real": round(tasa, 4),
                            "pd_desde": round(max(lo, 0), 4), "pd_hasta": round(hi, 4),
                            "ic95_tasa_real": [round(ci_lo, 4), round(ci_hi, 4)],
                            "consistente": veredicto == "✅"}
            print(f"{nombre:<11}{max(lo,0):>10.1%}{min(hi,1):>10.1%}{n:>7}{d:>10}"
                  f"{tasa:>10.1%}   [{ci_lo:>5.1%},{ci_hi:>6.1%}]  {veredicto}")
        return dist, rotas

    # Se mira en los tres splits: si una banda se rompe sólo en test, puede ser
    # ruido; si se rompe también en OOF, está rota de verdad.
    evaluar_bandas(aplicar_calibracion(cal_art, oof_raw), oof_y, "OOF/TRAIN")
    evaluar_bandas(p_va, yva.values, "VALID")
    dist_bandas, bandas_rotas = evaluar_bandas(p_te, yte.values, "TEST")

    if bandas_rotas:
        print(f"\n⚠️  Bandas inconsistentes en TEST: {bandas_rotas}. "
              f"La tasa real observada cae fuera del rango de PD que la banda "
              f"promete, con el IC 95% entero afuera. El informe estaría "
              f"reportando un nivel de riesgo equivocado al comité.")

    imp = sorted(zip(usadas, modelo.feature_importances_), key=lambda x: -x[1])
    print("Top 8 features:")
    for f, v in imp[:8]:
        print(f"  {f:<32}{v:.4f}")

    # ── Guardado: modelo + model card ──────────────────────────────────────
    os.makedirs(os.path.dirname(args.salida) or ".", exist_ok=True)
    modelo.save_model(args.salida)

    card = {
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modelo": "XGBClassifier",
        "xgboost_version": xgb.__version__,
        "target": {
            "definicion": "alcanza 90+ días de mora por primera vez dentro de la "
                          "ventana de performance",
            "horizonte_meses": HORIZONTE_MESES,
        },
        "poblacion_elegible": (
            "clientes con préstamo abierto al corte, sin default previo "
            "(ni mora 90+ ni clasificación de gerencia > 4) y con al menos una "
            "cuota venciendo dentro de la ventana de performance"
        ),
        "diseno": "panel de cortes anuales; validación out-of-time",
        "cutoffs": {"train": cutoffs_train, "valid": CUTOFF_VALID, "test": CUTOFFS_TEST},
        "covid_excluido": bool(args.excluir_covid),
        "cutoffs_covid": CUTOFFS_COVID,
        "dataset": {
            "train": {"filas": int(len(tr)), "defaults": int(ytr.sum()),
                      "clientes_unicos": int(tr.cliente_id.nunique())},
            "valid": {"filas": int(len(va)), "defaults": int(yva.sum()),
                      "clientes_unicos": int(va.cliente_id.nunique())},
            "test":  {"filas": int(len(te)), "defaults": int(yte.sum()),
                      "clientes_unicos": int(te.cliente_id.nunique())},
        },
        "solapamiento_clientes_train_test": round(solap, 4),
        "eventos_por_variable": round(epv, 2),
        "features_usadas": usadas,
        "features_descartadas_por_constantes": descartadas,
        "features_excluidas_por_fuga": {
            "prestamos_irrecuperables": "es el target (clasificación > 4)",
            "atraso_grave_24m": "cuotas con +90d de mora; el target ES +90d de mora",
        },
        "hiperparametros": modelo.get_params(),
        "n_estimators_por_cv_expansiva": {"elegido": n_est, "por_fold": mejores},
        "calibracion": {
            "metodo": metodo_cal,
            "elegida_por": "menor Brier out-of-sample en CV temporal anidada "
                           "sobre las predicciones out-of-fold",
            "brier_medio_por_metodo": cal_art["seleccion"]["brier_medio_por_metodo"],
            "ajustada_en": "predicciones out-of-fold de la CV expansiva sobre TRAIN",
            "n_oof": int(len(oof_y)), "prevalencia_oof": round(float(oof_y.mean()), 4),
            "brier_test_crudo": round(brier_crudo, 4),
            "brier_test_calibrado": m_te["brier"],
            "prob_media_predicha_test": round(float(p_te.mean()), 4),
            "prevalencia_real_test": round(float(yte.mean()), 4),
        },
        "metricas": {"train": m_tr, "valid": m_va, "test": m_te,
                     f"baseline_{base_col}_en_test": m_base},
        "bandas_riesgo": {"tasa_cartera": round(tasa_cartera, 4),
                          "distribucion_en_test": dist_bandas,
                          "bandas_inconsistentes_en_test": bandas_rotas},
        "deciles_test": {int(d): {"n": int(r.n), "defaults": int(r.defaults),
                                  "tasa_real": round(float(r.tasa), 4),
                                  "lift": round(float(r.lift), 2)}
                         for d, r in tabla.iterrows()},
        "importancia_features": {f: round(float(v), 5) for f, v in imp},
        "limitaciones": [
            "El mismo cliente aparece en varios cortes del panel; "
            f"{solap:.1%} de los clientes de TEST también están en TRAIN.",
            "La ventana de performance de los cortes 2019 y 2020 cae en las "
            "moratorias de COVID: la mora observada está deprimida.",
            "La población elegible es ~10% de la cartera: sólo quienes tienen "
            "una cuota venciendo en la ventana pueden defaultear por definición.",
            "El scoring aplica sólo a clientes con historial de préstamos; "
            "para thin-file el modelo no tiene señal.",
        ],
    }

    # Las features van también en un sidecar: la inferencia debe usar
    # exactamente esta lista y en este orden.
    card_path = "models/model_card.json"
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2, default=str)
    with open("models/scoring_features.json", "w", encoding="utf-8") as f:
        json.dump(usadas, f, ensure_ascii=False, indent=2)

    # La inferencia DEBE aplicar exactamente esta transformación sobre
    # predict_proba() crudo — ml.scoring_model._calibrar la espeja. Si el método
    # es "ninguna", la PD del informe es la salida cruda del booster, que es lo
    # que corresponde para binary:logistic sin scale_pos_weight.
    with open("models/calibracion.json", "w", encoding="utf-8") as f:
        json.dump(cal_art, f, ensure_ascii=False, indent=2)

    with open("models/bandas_riesgo.json", "w", encoding="utf-8") as f:
        json.dump({
            "criterio": "múltiplos de la tasa de default de la cartera sobre la PD",
            "tasa_cartera": round(tasa_cartera, 4),
            "bandas": [{"nivel": n, "semaforo": c, "pd_desde": lo, "pd_hasta": hi}
                       for n, c, lo, hi in bandas],
            "distribucion_en_test": dist_bandas,
            "bandas_inconsistentes_en_test": bandas_rotas,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Modelo      -> {args.salida}")
    print(f"✅ Model card  -> {card_path}")
    print(f"✅ Features    -> models/scoring_features.json")
    print(f"✅ Calibración -> models/calibracion.json")
    print(f"✅ Bandas      -> models/bandas_riesgo.json")


if __name__ == "__main__":
    main()
