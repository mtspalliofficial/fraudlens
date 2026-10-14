"""Analyst dashboard.  Run:  streamlit run dashboard/app.py"""
import json
import sys
from pathlib import Path
 
import altair as alt
import pandas as pd
import streamlit as st
 
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fraudlens.config import ARTIFACT_DIR, RAW_COLUMNS  # noqa: E402
from fraudlens.service import FraudService              # noqa: E402
 
st.set_page_config(page_title="FraudLens", layout="wide")
st.title("FraudLens: explainable fraud detection")
 
if not (ARTIFACT_DIR / "model.joblib").exists():
    st.error("No trained model found. Run `python -m fraudlens.train` first.")
    st.stop()
 
 
@st.cache_resource
def get_service() -> FraudService:
    return FraudService()
 
 
@st.cache_data
def load_artifacts():
    metrics = json.loads((ARTIFACT_DIR / "metrics.json").read_text())
    preds = pd.read_csv(ARTIFACT_DIR / "test_predictions.csv")
    importance = pd.read_csv(ARTIFACT_DIR / "feature_importance.csv", index_col=0)
    return metrics, preds, importance
 
 
def sorted_bars(series: pd.Series, axis_title: str) -> None:
    """Horizontal bars, largest on top (st.bar_chart sorts alphabetically).
    Orange = pushes towards fraud, blue = towards legitimate."""
    df = pd.DataFrame({"feature": series.index.astype(str), "value": series.to_numpy()})
    df["order"] = -df["value"].abs()
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("value:Q", title=axis_title),
        y=alt.Y("feature:N", sort=alt.EncodingSortField("order"), title=None),
        color=alt.condition(alt.datum.value > 0, alt.value("#e8590c"), alt.value("#1c7ed6")),
        tooltip=["feature", alt.Tooltip("value:Q", format=".3f")],
    ).properties(height=max(120, 32 * len(df)))
    st.altair_chart(chart, use_container_width=True)
 
 
def show_reasons(result: dict) -> None:
    reasons = pd.DataFrame(result["top_reasons"])
    sorted_bars(reasons.set_index("feature")["impact"], "SHAP impact (log-odds)")
    for r in result["top_reasons"]:
        st.write("- " + r["text"])
 
 
svc = get_service()
metrics, preds, importance = load_artifacts()
xgb_m, base_m = metrics["xgboost"], metrics["baseline_logreg"]
 
tab_perf, tab_flagged, tab_score = st.tabs(
    ["Model performance", "Flagged transactions", "Score a transaction"])
 
with tab_perf:
    c = st.columns(5)
    c[0].metric("PR-AUC", f"{xgb_m['pr_auc']:.3f}",
                f"{xgb_m['pr_auc'] - base_m['pr_auc']:+.3f} vs baseline")
    c[1].metric("Recall (fraud caught)", f"{xgb_m['recall']:.1%}")
    c[2].metric("Precision (alerts correct)", f"{xgb_m['precision']:.1%}")
    c[3].metric("Fraud value caught", f"{xgb_m['fraud_value_caught_pct']:.1f}%")
    c[4].metric("Decision threshold", f"{metrics['threshold']:.3f}")
    st.caption(f"Threshold rule: {metrics['threshold_rule']}. An always-legit model "
               f"scores {metrics['dummy_accuracy']:.2%} accuracy and catches no fraud, "
               "which is why accuracy is not used.")
 
    left, right = st.columns(2)
    with left:
        st.subheader("Precision-recall curve")
        st.image(str(ARTIFACT_DIR / "pr_curve.png"))
    with right:
        st.subheader("What drives decisions (mean |SHAP|)")
        sorted_bars(importance["mean_abs_shap"].head(12), "Mean |SHAP|")
 
    st.subheader("Confusion matrix (test set)")
    cm = xgb_m["confusion_matrix"]
    st.table(pd.DataFrame(
        {"Predicted legit": [cm["tn"], cm["fn"]], "Predicted fraud": [cm["fp"], cm["tp"]]},
        index=["Actual legit", "Actual fraud"]))
 
with tab_flagged:
    flagged = preds[preds["flagged"] == 1].sort_values("fraud_probability", ascending=False)
    hits = int(flagged["actual"].sum())
    st.write(f"**{len(flagged)}** test transactions flagged, **{hits}** confirmed fraud.")
    view = flagged[["Time", "Amount", "fraud_probability", "actual"]].copy()
    view["outcome"] = view["actual"].map({1: "Fraud (correct)", 0: "False alarm"})
    st.dataframe(view.drop(columns="actual"), use_container_width=True)
    if len(flagged):
        idx = st.selectbox("Explain transaction (row id)", flagged.index.tolist())
        result = svc.score([flagged.loc[idx, RAW_COLUMNS].to_dict()])[0]
        st.write(f"Fraud probability **{result['fraud_probability']:.2%}** "
                 f"→ **{result['decision']}**")
        show_reasons(result)
 
with tab_score:
    st.write("Start from a real test transaction, adjust the amount, and score it.")
    kind = st.radio("Start from", ["A fraud case", "A legitimate case"], horizontal=True)
    pool = preds[preds["actual"] == (1 if kind == "A fraud case" else 0)]
    if "pick" not in st.session_state or st.button("Pick another"):
        st.session_state.pick = int(pool.sample(1).index[0])
    pick = st.session_state.pick if st.session_state.pick in pool.index else int(pool.index[0])
    tx = preds.loc[pick, RAW_COLUMNS].to_dict()
    tx["Amount"] = st.number_input("Amount", min_value=0.0, value=float(tx["Amount"]))
    result = svc.score([tx])[0]
    colour = "red" if result["decision"] == "FRAUD" else "green"
    st.markdown(f"### :{colour}[{result['decision']}]  "
                f"probability {result['fraud_probability']:.2%}")
    show_reasons(result)
    with st.expander("Raw transaction / API request body"):
        st.json(tx)