
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
    if not isinstance(customer["tenure"], (int, float)) or not 0 <= customer["tenure"] <= 100:
        raise ValueError("tenure must be between 0 and 100")

    if customer["SeniorCitizen"] not in (0, 1):
        raise ValueError("SeniorCitizen must be 0 or 1")
    allowed_values = {
    "gender": {"Female", "Male"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "PhoneService": {"Yes", "No"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
}

    for column, allowed in allowed_values.items():
        if customer[column] not in allowed:
            raise ValueError(f"Invalid value for {column}: {customer[column]}")
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
