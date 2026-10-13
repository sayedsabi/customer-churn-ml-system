
from fastapi import FastAPI
from src.api.schemas import CustomerInput
from src.predict import predict_churn

app = FastAPI(title="Customer Churn Prediction API")


@app.get("/")
def home():
    return {"message": "Customer Churn API is running!"}


@app.post("/predict")
def predict(customer: CustomerInput):
    result = predict_churn(customer.model_dump())
    return result
