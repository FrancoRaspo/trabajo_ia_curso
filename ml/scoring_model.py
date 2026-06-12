# ml/scoring_model.py
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import joblib

class CreditScoringModel:

    # Deben coincidir EXACTAMENTE con las keys de construir_features_ml()
    FEATURE_COLUMNS = [
        "total_prestamos",
        "prestamos_al_dia",
        "prestamos_con_atraso",
        # "max_dias_atraso_historico",
        "dias_atraso_promedio",
        "prestamos_refinanciados",
        # "prestamos_irrecuperables",      # ← IRRECUPERABLE en tu DB, no CASTIGADO
        "cuotas_puntuales_24m",
        "atraso_leve_24m",
        "atraso_moderado_24m",
        #  "atraso_grave_24m",
        "saldo_promedio_ahorro_6m",
        "saldo_promedio_corriente_6m",
        "saldo_minimo_6m",
        "deuda_vigente_total",
        "capacidad_pago_estimada",
        "antiguedad_meses",
        "productos_activos",
        "ultimo_prestamo_monto",
        "ultimo_prestamo_hace_meses",
        "tasa_cumplimiento",
        "ratio_deuda_ingresos",
    ]

    BANDAS_RIESGO = [
        (800, 1000, "BAJO",     "VERDE"),
        (650,  799, "MODERADO", "AMARILLO"),
        (500,  649, "ALTO",     "NARANJA"),
        (0,    499, "MUY ALTO", "ROJO"),
    ]

    def __init__(self):
        print("[DBG][scoring_model] __init__ start", flush=True)
        print("[DBG][scoring_model] creating base XGBClassifier...", flush=True)
        self.model      = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            random_state=42,
            nthread=1,
        )
        print("[DBG][scoring_model] base XGBClassifier created", flush=True)
        self.is_trained = False
        print("[DBG][scoring_model] __init__ end", flush=True)

    def train(self, df: pd.DataFrame, target_col: str = "default_90d"):
        X = df[self.FEATURE_COLUMNS].fillna(0)
        y = df[target_col]

        # Compensar desbalance: muchos pagadores, pocos irrecuperables
        ratio = (y == 0).sum() / max((y == 1).sum(), 1)
        self.model.set_params(scale_pos_weight=ratio)
        print(f"  scale_pos_weight = {ratio:.1f}  "
              f"({int((y==0).sum())} pagadores / {int((y==1).sum())} defaulteados)")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50,
        )
        self.is_trained = True

        y_pred = self.model.predict_proba(X_test)[:, 1]
        auc    = roc_auc_score(y_test, y_pred)
        print(f"\nAUC-ROC en test: {auc:.4f}")
        print(classification_report(y_test, (y_pred > 0.5).astype(int)))
        return auc

    def predict(self, features: dict) -> dict:
        if not self.is_trained:
            raise RuntimeError("Modelo no entrenado. Ejecutá train() primero.")

        df          = pd.DataFrame([features])[self.FEATURE_COLUMNS].fillna(0)
        prob_default = float(self.model.predict_proba(df)[0, 1])
        score        = int(round((1 - prob_default) * 1000))
        score        = max(0, min(1000, score))
        nivel, color = self._clasificar(score)

        importancias = dict(zip(self.FEATURE_COLUMNS, self.model.feature_importances_))
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
            "nivel_riesgo":        nivel,
            "semaforo":            color,
            "sin_historial":       sin_historial,
            "principales_drivers": [
                {"feature": k, "importancia": round(v, 4)}
                for k, v in top_drivers
            ],
            "drivers_cliente":     drivers_cliente,
        }

    def _clasificar(self, score: int) -> tuple:
        for low, high, nivel, color in self.BANDAS_RIESGO:
            if low <= score <= high:
                return nivel, color
        return "DESCONOCIDO", "GRIS"

    def save(self, path: str = "models/scoring_model.json"):
        self.model.save_model(path)
        print(f"Modelo guardado en {path}")

    @classmethod
    def load(cls, path: str = "models/scoring_model.json") -> "CreditScoringModel":
        import json
        import os
        print(f"[DBG][scoring_model.load] start path={path}", flush=True)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modelo no encontrado: {path}")

        print("[DBG][scoring_model.load] validating model JSON...", flush=True)
        with open(path, "r", encoding="utf-8") as f:
            model_json = json.load(f)

        model_version = model_json.get("version")
        runtime_major = int(str(xgb.__version__).split(".")[0])
        model_major = model_version[0] if isinstance(model_version, list) and model_version else None

        print(f"[DBG][scoring_model.load] model_version={model_version}", flush=True)
        print(f"[DBG][scoring_model.load] runtime_xgboost={xgb.__version__}", flush=True)
        if model_major is not None and model_major != runtime_major:
            raise RuntimeError(
                f"Incompatibilidad de versión XGBoost: modelo={model_version}, runtime={xgb.__version__}"
            )

        print("[DBG][scoring_model.load] creating instance via __new__ (skip __init__)...", flush=True)
        instance = cls.__new__(cls)
        print("[DBG][scoring_model.load] instance created", flush=True)
        print("[DBG][scoring_model.load] creating fresh XGBClassifier for load...", flush=True)
        instance.model = xgb.XGBClassifier(nthread=1)
        print("[DBG][scoring_model.load] fresh XGBClassifier created", flush=True)
        print(f"[DBG][scoring_model.load] model path size={os.path.getsize(path)} bytes", flush=True)
        print("[DBG][scoring_model.load] calling instance.model.load_model(path)...", flush=True)
        instance.model.load_model(path)
        print("[DBG][scoring_model.load] load_model completed", flush=True)
        instance.is_trained = True
        print("[DBG][scoring_model.load] marked as trained; returning instance", flush=True)
        return instance