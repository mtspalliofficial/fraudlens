import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """Train once on small synthetic data; reused by every test."""
    from fraudlens import train
    d = tmp_path_factory.mktemp("fraudlens")
    X, y = make_classification(n_samples=6000, n_features=28, n_informative=8,
                               weights=[0.97], class_sep=1.5, random_state=1)
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.arange(len(df)) * 30.0)
    df["Amount"] = np.abs(X[:, 0]) * 50 + 1
    df["Class"] = y
    df.to_csv(d / "data.csv", index=False)
    metrics = train.run(d / "data.csv", d / "artifacts", min_precision=0.8)
    return d / "artifacts", df, metrics
