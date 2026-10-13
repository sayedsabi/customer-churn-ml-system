
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Customer Churn API is running!"
    }


def test_predict_endpoint():
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

    response = client.post("/predict", json=customer)

    assert response.status_code == 200

    result = response.json()
    assert result["churn_prediction"] in [0, 1]
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["interpretation"] in [
        "Likely to churn",
        "Unlikely to churn",
    ]


def test_predict_rejects_missing_fields():
    response = client.post("/predict", json={
        "gender": "Female"
    })

    assert response.status_code == 422


def test_predict_rejects_invalid_contract():
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
        "Contract": "Monthly",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 65.0,
        "TotalCharges": 780.0,
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 422
