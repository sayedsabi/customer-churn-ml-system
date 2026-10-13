
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "churn_cleaned.csv"


def load_data() -> pd.DataFrame:
    """Load the cleaned customer churn dataset."""
    return pd.read_csv(DATA_PATH)


if __name__ == "__main__":
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    print(df.head())
