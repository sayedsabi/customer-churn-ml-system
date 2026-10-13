
from src.data.load_data import load_data


def test_load_data_returns_expected_dataset():
    df = load_data()

    assert not df.empty
    assert len(df) == 7043
    assert "Churn" in df.columns
    assert "customerID" in df.columns
