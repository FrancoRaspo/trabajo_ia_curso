# ml/scoring_model.py
import bisect
import json
import math
import os

import xgboost as xgb
import pandas as pd

from ml.feature_query import FEATURE_COLUMNS as _FEATURES_QUERY

DIR_MODELOS = "models"


def _cargar_json(nombre: str):
    path = os.path.join(DIR_MODELOS, nombre)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class CreditScoringModel:
    """Scoring crediticio. Se carga entrenado; no entrena.

    Entrenamiento: python3 scripts/entrenar_modelo_temporal.py
    """

    # Lista por defecto: la de la query de features. El entrenamiento descarta
    # las que salen constantes y persiste la lista real en
    # models/scoring_features.json, que load() lee. No editar a mano.
    FEATURE_COLUMNS = list(_FEATURES_QUERY)

    # Bandas de riesgo sobre la PROBABILIDAD DE DEFAULT, no sobre el score.
    # El score (= (1-PD)*1000) sólo se usa para mostrar.
    #
    # Las bandas viejas cortaban el score en 800/650/500. Eso funcionaba sólo
    # porque el modelo anterior usaba scale_pos_weight, que infla las
    # probabilidades y las esparce por todo [0,1]. Con PD real la masa vive
    # entre 2% y 25% -> el score entre ~750 y ~980, y las bandas ALTO y MUY ALTO
    # eran inalcanzables: todo cliente habría salido "BAJO".
    #
    # Los cortes reales los calcula el entrenamiento (múltiplos de la tasa de
    # default de la cartera) y viven en models/bandas_riesgo.json. Esto es sólo
    # el fallback si el archivo no está.
    BANDAS_RIESGO_PD = [
        ("MUY ALTO", "ROJO",     0.257, 1.01),
        ("ALTO",     "NARANJA",  0.128, 0.257),
        ("MODERADO", "AMARILLO", 0.064, 0.128),
        ("BAJO",     "VERDE",   -0.01,  0.064),
    ]

    def train(self, df: pd.DataFrame, target_col: str = "default_90d"):
        raise NotImplementedError(
            "Este train() hacía un split aleatorio (train_test_split) sobre un "
            "dataset donde features y target se medían en el mismo instante: el "
            "AUC resultante no significaba nada.\n"
            "Usá: python3 scripts/entrenar_modelo_temporal.py"
        )

    def _cargar_artefactos(self) -> None:
        """
        Lee los sidecars que el entrenamiento dejó junto al modelo:
          scoring_features.json -> qué features y en qué orden
          calibracion.json      -> cómo mapear predict_proba() a una PD
          bandas_riesgo.json    -> cortes de PD para las bandas de riesgo
        """
        feats = _cargar_json("scoring_features.json")
        if feats:
            self.feature_columns = list(feats)
        else:
            print("[scoring_model] ⚠️ falta models/scoring_features.json; "
                  "uso el orden por defecto de feature_query", flush=True)

        self.calibracion = _cargar_json("calibracion.json") or {"metodo": "ninguna"}

        bandas = _cargar_json("bandas_riesgo.json")
        if bandas:
            self.bandas = [(b["nivel"], b["semaforo"], b["pd_desde"], b["pd_hasta"])
                           for b in bandas["bandas"]]

    def _calibrar(self, prob_cruda: float) -> float:
        """Mapea la salida del booster a una probabilidad de default.

        El método lo ELIGE el entrenamiento comparando, en una CV temporal
        anidada, no-calibrar contra Platt contra isotónica; gana el de mejor
        Brier fuera de muestra. Hoy gana "ninguna", y no es casualidad:

          - el objetivo es binary:logistic, que ya optimiza log-loss, y
          - scale_pos_weight se omite a propósito,

        así que predict_proba() ya devuelve una probabilidad, no un puntaje de
        ranking. Una sigmoide de Platt encima no corrige nada: la aplasta.
        Aplicarla degradaba el Brier (0.0442 -> 0.0452) y rompía la banda
        MODERADO, que sub-predecía (rango 6.4-12.8%, tasa real observada 19.5%).
        """
        metodo = (self.calibracion or {}).get("metodo", "ninguna")

        if metodo == "ninguna":
            return prob_cruda

        if metodo == "platt_sigmoid":
            z = self.calibracion["w"] * prob_cruda + self.calibracion["b"]
            return 1.0 / (1.0 + math.exp(-z))

        if metodo == "isotonica":
            # Interpolación lineal entre los nodos que dejó el entrenamiento.
            xs, ys = self.calibracion["x"], self.calibracion["y"]
            piso = self.calibracion.get("piso", 0.0)
            if prob_cruda <= xs[0]:
                return max(ys[0], piso)
            if prob_cruda >= xs[-1]:
                return max(ys[-1], piso)
            i = bisect.bisect_left(xs, prob_cruda)
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            t = 0.0 if x1 == x0 else (prob_cruda - x0) / (x1 - x0)
            return max(y0 + t * (y1 - y0), piso)

        raise ValueError(f"Método de calibración desconocido: {metodo!r}")

    def predict(self, features: dict) -> dict:
        if not self.is_trained:
            raise RuntimeError("Modelo no entrenado. Ejecutá train() primero.")

        cols = self.feature_columns
        faltantes = [c for c in cols if c not in features]
        if faltantes:
            raise ValueError(
                f"Faltan features que el modelo espera: {faltantes}. "
                "El vector debe venir de ClientRepository.construir_features_ml()."
            )

        df = pd.DataFrame([features])[cols].fillna(0)

        prob_cruda   = float(self.model.predict_proba(df)[0, 1])
        prob_default = self._calibrar(prob_cruda)

        score        = int(round((1 - prob_default) * 1000))
        score        = max(0, min(1000, score))
        nivel, color = self._clasificar(prob_default)

        importancias = dict(zip(cols, self.model.feature_importances_))
        top_drivers  = sorted(importancias.items(), key=lambda x: x[1], reverse=True)[:5]

        # Drivers SHAP de ESTE cliente (por qué sacó este score, no el promedio
        # del modelo). El modelo predice prob_default -> mayor salida = más riesgo.
        try:
            from ml.shap_explainer import explicar_cliente
            drivers_cliente = explicar_cliente(self.model, df, top_n=5,
                                               mayor_es_mas_riesgo=True,
                                               min_peso=2.0)
        except Exception as e:
            print(f"[scoring_model] SHAP no disponible: {e}", flush=True)
            drivers_cliente = []

        # Flag thin-file: cliente SIN historial crediticio previo.
        # Si no tomó préstamos, los features de historial (último préstamo, atrasos,
        # etc.) son valores de relleno (sentinel), así que los drivers de SHAP no
        # son interpretables y el score alto refleja AUSENCIA de señal negativa,
        # NO comportamiento de pago demostrado. Se marca para que el informe lo
        # aclare y sugiera verificación manual.
        sin_historial = not features.get("total_prestamos")  # True si 0, None o ausente

        return {
            "score":               score,
            "prob_default":        round(prob_default, 4),
            "prob_default_cruda":  round(prob_cruda, 4),   # trazabilidad
            "calibracion":         (self.calibracion or {}).get("metodo", "ninguna"),
            "nivel_riesgo":        nivel,
            "semaforo":            color,
            "sin_historial":       sin_historial,
            "principales_drivers": [
                {"feature": k, "importancia": round(v, 4)}
                for k, v in top_drivers
            ],
            "drivers_cliente":     drivers_cliente,
        }

    def _clasificar(self, prob_default: float) -> tuple:
        """Banda de riesgo a partir de la PD (no del score)."""
        for nivel, color, lo, hi in self.bandas:
            if lo < prob_default <= hi:
                return nivel, color
        return "DESCONOCIDO", "GRIS"

    def save(self, path: str = "models/scoring_model.json"):
        self.model.save_model(path)
        print(f"Modelo guardado en {path}")

    @classmethod
    def load(cls, path: str = "models/scoring_model.json") -> "CreditScoringModel":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modelo no encontrado: {path}")

        with open(path, "r", encoding="utf-8") as f:
            model_json = json.load(f)

        # El binario de XGBoost no es compatible entre majors. Si el runtime no
        # coincide con el que entrenó, load_model() puede fallar en silencio o
        # devolver otra cosa: mejor cortar acá.
        model_version = model_json.get("version")
        runtime_major = int(str(xgb.__version__).split(".")[0])
        model_major = (model_version[0]
                       if isinstance(model_version, list) and model_version else None)
        if model_major is not None and model_major != runtime_major:
            raise RuntimeError(
                f"Incompatibilidad de versión XGBoost: "
                f"modelo={model_version}, runtime={xgb.__version__}"
            )

        instance = cls.__new__(cls)          # sin __init__: el modelo se carga, no se construye
        instance.model = xgb.XGBClassifier(nthread=1)
        instance.model.load_model(path)
        instance.is_trained = True

        instance.feature_columns = list(cls.FEATURE_COLUMNS)
        instance.calibracion     = {"metodo": "ninguna"}
        instance.bandas          = list(cls.BANDAS_RIESGO_PD)
        instance._cargar_artefactos()

        return instance
