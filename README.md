FraudLens: explainable credit-card fraud detection
Problem statement

Card fraud is rare: in the benchmark dataset only 0.17% of transactions are fraudulent. That breaks naive machine learning in two ways:

The accuracy trap. A model that calls everything "legitimate" is 99.8% accurate and catches zero fraud.
Black-box decisions. Banks and auditors must justify every blocked transaction, so a bare risk score is not enough.

Goal: detect fraudulent transactions with high recall at a controlled false-alarm rate, explain every decision, and serve predictions in real time through an API and an analyst dashboard.

Solution
Stage	What it does
Data	284,807 Kaggle transactions; engineered log_amount and hour; stratified 60/20/20 split
Baseline	Always-legit dummy (shows the accuracy trap) and balanced logistic regression
Model	XGBoost with scale_pos_weight (≈ legit/fraud ratio) and early stopping on PR-AUC
Threshold	Chosen on the validation set: highest recall with precision ≥ 80%
Evaluation	PR-AUC, precision, recall, confusion matrix and % of fraud value caught, on an untouched test set
Explainability	Exact TreeSHAP values per transaction; top 3 reasons returned with every score
API	FastAPI POST /score, POST /score/batch, GET /model-info, GET /health
Dashboard	Streamlit: metrics, PR curve, global drivers, flagged transactions, what-if scoring
creditcard.csv → data.py → train.py → artifacts/ (model, metrics, plots)
                                          │
                       service.py (scoring + SHAP reasons)
                          │                     │
                  api.py (FastAPI)      dashboard/app.py (Streamlit)
Results

On the held-out test set (20% of the data, never used for training or tuning):

Model	PR-AUC
Always-legit dummy	n/a (99.83% accuracy, 0 fraud caught)
Logistic regression (balanced)	0.728
XGBoost	0.872

At the tuned threshold (0.285):

Catches 84.2% of fraud and 81.0% of fraud value
88.9% of alerts are real fraud: 10 false alarms out of 56,651 legitimate transactions
Most influential features (mean |SHAP|): V14, V4, V12
Run it
bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Data: download creditcard.csv from
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud into data/
# (or, to try things out first: python scripts/make_sample_data.py)

python -m fraudlens.train          # trains, prints metrics, writes artifacts/
pytest -q                          # tests
uvicorn fraudlens.api:app --reload # API docs at http://127.0.0.1:8000/docs
streamlit run dashboard/app.py     # dashboard

On macOS, if importing xgboost fails mentioning libomp, run brew install libomp.

Project layout
fraudlens/config.py    paths, feature list, precision target
fraudlens/data.py      load, validate, feature engineering, stratified split
fraudlens/train.py     baselines, XGBoost, threshold tuning, evaluation, artifacts
fraudlens/explain.py   SHAP values and human-readable reasons
fraudlens/service.py   scoring logic shared by API and dashboard
fraudlens/api.py       FastAPI app
dashboard/app.py       Streamlit dashboard
scripts/make_sample_data.py   synthetic data with the same schema
tests/                 pipeline and API tests
Design decisions
PR-AUC, not accuracy or ROC-AUC. With 0.17% positives, ROC-AUC looks great even for weak models; PR-AUC focuses on how well fraud itself is ranked.
Class weighting over SMOTE. scale_pos_weight reweights errors without inventing synthetic transactions, and it is simpler to reproduce.
Threshold from a business rule, on validation data. 0.5 is arbitrary. The test set is used once, for the final numbers.
One feature function for training and serving (add_features), which prevents training/serving skew.
Native TreeSHAP via pred_contribs=True: exact SHAP values in milliseconds, no extra dependency.
Limitations
V1–V28 are anonymised PCA components, so explanations name components, not business features like merchant or country.
Two days of European transactions from 2013; a real system would need retraining, drift monitoring and time-based validation.