"""Training pipeline: baseline -> XGBoost -> threshold tuning -> evaluation.

Run:  python -m fraudlens.train            (uses data/creditcard.csv)
      python -m fraudlens.train --data path/to/file.csv
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import ARTIFACT_DIR, DATA_PATH, MIN_PRECISION, RANDOM_STATE
from .data import load_data, split
from .explain import global_importance


def choose_threshold(y_true, probs, min_precision: float) -> tuple[float, str]:
    """Highest recall whose precision is still >= min_precision.
    Tuned on the VALIDATION set, never on test (that would be cheating)."""
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    precision, recall = precision[:-1], recall[:-1]   # align with thresholds
    ok = precision >= min_precision
    if ok.any():
        best = np.argmax(np.where(ok, recall, -1))
        return float(thresholds[best]), f"max recall with precision >= {min_precision:.0%}"
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1)]), "max F1 (precision target unreachable)"


def evaluate(y_true, probs, threshold, amounts) -> dict:
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    fraud = np.asarray(y_true) == 1
    caught_value = amounts[fraud & (pred == 1)].sum()
    total_value = amounts[fraud].sum()
    return {
        "pr_auc": round(average_precision_score(y_true, probs), 4),
        "roc_auc": round(roc_auc_score(y_true, probs), 4),
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, pred, zero_division=0), 4),
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "fraud_value_caught_pct": round(100 * caught_value / max(total_value, 1e-9), 2),
    }


def plot_pr_curves(curves: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (y, p) in curves.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, label=f"{name} (PR-AUC {average_precision_score(y, p):.3f})")
    ax.set_xlabel("Recall (share of fraud caught)")
    ax.set_ylabel("Precision (share of alerts that are fraud)")
    ax.set_title("Precision-recall on the test set")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(data_path: Path = DATA_PATH, out_dir: Path = ARTIFACT_DIR,
        min_precision: float = MIN_PRECISION) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test, raw_test = split(df)
    fraud_rate = y_train.mean()
    print(f"Rows: {len(df):,} | fraud rate: {fraud_rate:.4%} | "
          f"train/val/test: {len(X_train):,}/{len(X_val):,}/{len(X_test):,}")

    # 1. The accuracy trap: always predict "legit".
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    dummy_acc = accuracy_score(y_test, dummy.predict(X_test))
    print(f"Always-legit dummy accuracy: {dummy_acc:.4%} (catches 0 fraud)")

    # 2. Baseline: scaled logistic regression with balanced class weights.
    baseline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"))
    baseline.fit(X_train, y_train)
    base_val = baseline.predict_proba(X_val)[:, 1]
    base_thr, _ = choose_threshold(y_val, base_val, min_precision)
    base_test = baseline.predict_proba(X_test)[:, 1]

    # 3. XGBoost; scale_pos_weight makes each fraud count ~ (legit/fraud) times more.
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=600, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
        scale_pos_weight=spw, eval_metric="aucpr", early_stopping_rounds=50,
        tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"XGBoost best iteration: {model.best_iteration} | scale_pos_weight={spw:.0f}")

    # 4. Threshold from validation data, evaluated once on test.
    val_probs = model.predict_proba(X_val)[:, 1]
    threshold, rule = choose_threshold(y_val, val_probs, min_precision)
    test_probs = model.predict_proba(X_test)[:, 1]
    amounts = raw_test["Amount"].to_numpy()

    metrics = {
        "dataset": {"rows": len(df), "fraud_rate": round(float(fraud_rate), 6),
                    "test_rows": len(X_test), "test_fraud": int(y_test.sum())},
        "dummy_accuracy": round(dummy_acc, 4),
        "baseline_logreg": evaluate(y_test, base_test, base_thr, amounts),
        "xgboost": evaluate(y_test, test_probs, threshold, amounts),
        "threshold": round(threshold, 6),
        "threshold_rule": rule,
        "scale_pos_weight": round(float(spw), 2),
    }

    # 5. Artifacts for the API and dashboard.
    importance = global_importance(model, X_test.sample(min(5000, len(X_test)),
                                                         random_state=RANDOM_STATE))
    joblib.dump({"model": model, "threshold": threshold, "metrics": metrics},
                out_dir / "model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    importance.rename("mean_abs_shap").to_csv(out_dir / "feature_importance.csv")
    plot_pr_curves({"XGBoost": (y_test, test_probs),
                    "Logistic regression": (y_test, base_test)},
                   out_dir / "pr_curve.png")
    preds = raw_test.copy()
    preds["actual"] = y_test.to_numpy()
    preds["fraud_probability"] = test_probs
    preds["flagged"] = (test_probs >= threshold).astype(int)
    preds.to_csv(out_dir / "test_predictions.csv", index=False)

    x, b = metrics["xgboost"], metrics["baseline_logreg"]
    print(f"\nThreshold {threshold:.4f} ({rule})")
    print(f"{'':22}{'PR-AUC':>8}{'Precision':>11}{'Recall':>8}")
    print(f"{'Logistic regression':22}{b['pr_auc']:>8.3f}{b['precision']:>11.3f}{b['recall']:>8.3f}")
    print(f"{'XGBoost':22}{x['pr_auc']:>8.3f}{x['precision']:>11.3f}{x['recall']:>8.3f}")
    print(f"Fraud value caught: {x['fraud_value_caught_pct']}% | "
          f"confusion matrix: {x['confusion_matrix']}")
    print(f"Artifacts written to {out_dir}")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Train the FraudLens model")
    p.add_argument("--data", type=Path, default=DATA_PATH)
    p.add_argument("--out", type=Path, default=ARTIFACT_DIR)
    p.add_argument("--min-precision", type=float, default=MIN_PRECISION)
    a = p.parse_args()
    run(a.data, a.out, a.min_precision)


if __name__ == "__main__":
    main()
