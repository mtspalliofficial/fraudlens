"""Central settings so every module agrees on paths, columns and targets."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "creditcard.csv"
ARTIFACT_DIR = ROOT / "artifacts"

RAW_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET = "Class"

# Model features = the 28 PCA components + two engineered features.
FEATURES = [f"V{i}" for i in range(1, 29)] + ["log_amount", "hour"]

RANDOM_STATE = 42

# Business rule for the decision threshold: catch as much fraud as possible
# while at least this share of alerts are real fraud.
MIN_PRECISION = 0.80

TOP_K_REASONS = 3
