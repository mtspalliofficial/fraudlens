"""Loading, validation, feature engineering and splitting."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import FEATURES, RANDOM_STATE, RAW_COLUMNS, TARGET


def load_data(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Download creditcard.csv from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud into data/, "
            "or run: python scripts/make_sample_data.py"
        )
    df = pd.read_csv(path)
    missing = set(RAW_COLUMNS + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return df.drop_duplicates().reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Shared by training AND serving, so both see identical features.

    - log_amount: amounts are heavily skewed; log1p tames the long tail.
    - hour: Time is seconds since the first transaction; hour of day captures
      the night-time pattern of fraud.
    """
    out = df.copy()
    out["log_amount"] = np.log1p(out["Amount"].clip(lower=0))
    out["hour"] = (out["Time"] // 3600) % 24
    return out


def split(df: pd.DataFrame):
    """Stratified 60/20/20 split: every part keeps the same (tiny) fraud rate.

    train -> fit the model, val -> early stopping + threshold, test -> final,
    untouched evaluation.
    """
    df = add_features(df)
    X, y = df[FEATURES], df[TARGET].astype(int)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=RANDOM_STATE)
    raw_test = df.loc[X_test.index, RAW_COLUMNS]
    return X_train, X_val, X_test, y_train, y_val, y_test, raw_test
