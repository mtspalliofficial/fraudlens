"""Per-transaction explanations with SHAP values.

XGBoost computes exact TreeSHAP values natively (pred_contribs=True), so we
get SHAP explanations with no extra dependency and in milliseconds.
Each value is a feature's push on the log-odds of fraud: positive = towards
fraud, negative = towards legitimate. Values + bias = the model's raw score.
"""
import numpy as np
import pandas as pd
import xgboost as xgb

FEATURE_LABELS = {
    "log_amount": "Transaction amount",
    "hour": "Hour of day",
}


def shap_values(model: xgb.XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """Returns an (n_rows, n_features) array; the last (bias) column is dropped."""
    contribs = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    return np.asarray(contribs)[:, :-1]


def top_reasons(row: pd.Series, contribs: np.ndarray, k: int) -> list[dict]:
    order = np.argsort(-np.abs(contribs))[:k]
    reasons = []
    for i in order:
        feature = row.index[i]
        value, impact = float(row.iloc[i]), float(contribs[i])
        label = FEATURE_LABELS.get(feature, feature)
        direction = "towards fraud" if impact > 0 else "towards legitimate"
        reasons.append({
            "feature": feature,
            "value": round(value, 4),
            "impact": round(impact, 4),
            "text": f"{label} = {value:,.2f} pushes {direction} ({impact:+.2f})",
        })
    return reasons


def global_importance(model: xgb.XGBClassifier, X: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| per feature: which features drive decisions overall."""
    vals = np.abs(shap_values(model, X)).mean(axis=0)
    return pd.Series(vals, index=X.columns).sort_values(ascending=False)
