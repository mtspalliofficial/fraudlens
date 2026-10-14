"""Scoring logic, kept separate from FastAPI so it is easy to test and reuse
(the API and the dashboard both call this)."""
import time
from pathlib import Path

import joblib
import pandas as pd

from .config import ARTIFACT_DIR, FEATURES, RAW_COLUMNS, TOP_K_REASONS
from .data import add_features
from .explain import shap_values, top_reasons


class FraudService:
    def __init__(self, artifact_dir: Path = ARTIFACT_DIR):
        bundle = joblib.load(Path(artifact_dir) / "model.joblib")
        self.model = bundle["model"]
        self.threshold = float(bundle["threshold"])
        self.metrics = bundle["metrics"]

    def score(self, records: list[dict]) -> list[dict]:
        start = time.perf_counter()
        raw = pd.DataFrame(records)[RAW_COLUMNS]
        X = add_features(raw)[FEATURES]
        probs = self.model.predict_proba(X)[:, 1]
        contribs = shap_values(self.model, X)
        elapsed_ms = (time.perf_counter() - start) * 1000

        results = []
        for i, p in enumerate(probs):
            results.append({
                "fraud_probability": round(float(p), 6),
                "decision": "FRAUD" if p >= self.threshold else "LEGIT",
                "threshold": round(self.threshold, 6),
                "top_reasons": top_reasons(X.iloc[i], contribs[i], TOP_K_REASONS),
            })
        for r in results:
            r["latency_ms"] = round(elapsed_ms / len(results), 3)
        return results
