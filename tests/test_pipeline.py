import numpy as np

from fraudlens.config import RAW_COLUMNS
from fraudlens.service import FraudService
from fraudlens.train import choose_threshold


def test_threshold_meets_precision_target():
    y = np.array([0] * 90 + [1] * 10)
    probs = np.concatenate([np.linspace(0, 0.6, 90), np.linspace(0.5, 1, 10)])
    thr, rule = choose_threshold(y, probs, 0.8)
    pred = probs >= thr
    assert pred[y == 1].sum() / pred.sum() >= 0.8
    assert "recall" in rule


def test_artifacts_written(trained):
    art, _, metrics = trained
    for f in ["model.joblib", "metrics.json", "pr_curve.png",
              "feature_importance.csv", "test_predictions.csv"]:
        assert (art / f).exists(), f
    assert 0 < metrics["threshold"] < 1
    assert metrics["xgboost"]["pr_auc"] > 0


def test_score_output(trained):
    art, df, _ = trained
    svc = FraudService(art)
    out = svc.score(df[RAW_COLUMNS].head(5).to_dict("records"))
    assert len(out) == 5
    for r in out:
        assert 0 <= r["fraud_probability"] <= 1
        assert r["decision"] in {"FRAUD", "LEGIT"}
        assert len(r["top_reasons"]) == 3
