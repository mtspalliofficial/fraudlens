"""Creates a SYNTHETIC dataset with the same columns as the Kaggle file, so you
can test the pipeline without downloading it. Results on it mean nothing.

Run: python scripts/make_sample_data.py  [--rows 50000]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

p = argparse.ArgumentParser()
p.add_argument("--rows", type=int, default=50_000)
p.add_argument("--out", type=Path,
               default=Path(__file__).resolve().parent.parent / "data" / "creditcard.csv")
a = p.parse_args()

rng = np.random.default_rng(0)
X, y = make_classification(n_samples=a.rows, n_features=28, n_informative=10,
                           n_redundant=4, weights=[0.9983], flip_y=0,
                           class_sep=2.5, random_state=0)
df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
df.insert(0, "Time", np.sort(rng.uniform(0, 172_800, a.rows)).round())
df["Amount"] = np.where(y == 1, rng.lognormal(4.5, 1.3, a.rows),
                        rng.lognormal(3.5, 1.2, a.rows)).round(2)
df["Class"] = y
a.out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(a.out, index=False)
print(f"Wrote {len(df):,} synthetic rows ({y.sum()} fraud) to {a.out}")
