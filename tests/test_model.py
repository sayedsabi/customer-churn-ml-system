
from src.predict import predict_churn
import pytest


def test_prediction_returns_valid_result():
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

    result = predict_churn(customer)

    assert result["churn_prediction"] in [0, 1]
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["interpretation"] in [
        "Likely to churn",
        "Unlikely to churn",
    ]



def test_prediction_rejects_missing_fields():
    with pytest.raises(ValueError, match="Missing customer fields"):
        predict_churn({"gender": "Female"})