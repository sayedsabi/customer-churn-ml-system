
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"

REQUIRED_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges",
    "TotalCharges",
]


def predict_churn(customer: dict) -> dict:
    # Validate input before loading the model
    missing = set(REQUIRED_COLUMNS) - set(customer)

    if missing:
        raise ValueError(
            f"Missing customer fields: {sorted(missing)}"
        )

    model = joblib.load(MODEL_PATH)
    customer_df = pd.DataFrame([customer])

    prediction = int(model.predict(customer_df)[0])
    probability = float(model.predict_proba(customer_df)[0, 1])

    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4),
        "interpretation": (
            "Likely to churn" if prediction == 1
            else "Unlikely to churn"
        ),
    }


if __name__ == "__main__":
    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 65.0,
        "TotalCharges": 780.0,
    }

    print(predict_churn(customer))
