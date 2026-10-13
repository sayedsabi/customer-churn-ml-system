# Customer Churn Prediction System

A machine learning system that predicts customer churn using Logistic Regression, with a reusable ML pipeline, validated FastAPI endpoints, automated tests, and Docker containerization.

## Project Overview

This project explores customer churn prediction using the Telco Customer Churn dataset. It covers data cleaning, exploratory analysis, model comparison, error analysis, hyperparameter tuning, API development, testing, and containerization.

### System Architecture

```text
Customer Request (JSON)
        |
        v
   FastAPI Endpoint
        |
        v
 Pydantic Validation
        |
        v
 Trained ML Pipeline
 (Preprocessing + Logistic Regression)
        |
        v
 Prediction + Churn Probability
        |
        v
    JSON Response
```

## Technology Stack

- **Language:** Python 3.13
- **Machine Learning:** scikit-learn, pandas, NumPy
- **API:** FastAPI, Pydantic, Uvicorn
- **Model Serialization:** joblib
- **Testing:** pytest, FastAPI TestClient
- **Containerization:** Docker
- **Version Control:** Git and GitHub

## Project Structure

```text
01-ml-system/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── churn_model.joblib
├── notebooks/
├── src/
│   ├── api/
│   │   ├── main.py
│   │   └── schemas.py
│   ├── data/
│   │   └── load_data.py
│   ├── train.py
│   └── predict.py
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_api.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
├── requirements-docker.txt
└── README.md
```

## Data Cleaning and Exploration

The dataset contains 7,043 customer records.

- Converted `TotalCharges` from text to numeric.
- Identified 11 blank `TotalCharges` values, all associated with customers whose tenure was zero.
- Replaced those values with zero, preserving all customer records.
- Encoded the target as `No → 0` and `Yes → 1`.
- Excluded `customerID` from model features.
- Identified categorical and numerical features for preprocessing.
- Saved the cleaned dataset to `data/processed/churn_cleaned.csv`.

### Initial EDA Findings

| Customer segment | Churn rate |
|---|---:|
| Overall | 26.5% |
| Month-to-month contracts | 42.7% |
| One-year contracts | 11.3% |
| Two-year contracts | 2.8% |

These are descriptive patterns in the dataset and do not establish causation.

## Model Development and Evaluation

The preprocessing pipeline uses one-hot encoding for categorical features and standardization for numerical features.

### Initial Model Comparison

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 80.55% | 65.72% | 55.88% | 60.40% | 0.842 |
| Unrestricted Decision Tree | 72.25% | 47.81% | 49.73% | 48.75% | 0.650 |
| Decision Tree (depth 5) | 79.84% | 63.47% | 56.68% | 59.89% | 0.830 |
| Random Forest (200 trees) | 78.78% | 62.89% | 48.93% | 55.04% | 0.817 |

The majority-class dummy baseline achieved 73.46% accuracy but had zero churn recall.

Logistic Regression was selected as the leading candidate among the initial configurations.

### Cross-Validation and Tuning

Five-fold stratified cross-validation was used with F1-score as the selection metric.

- Original Logistic Regression mean CV F1: `0.5923`
- Tuned Logistic Regression mean CV F1: `0.5975`
- Tuned Random Forest mean CV F1: `0.5833`
- Best Logistic Regression regularization parameter: `C=10`
- Best Random Forest configuration: 200 trees, maximum depth 10, minimum leaf size 5

Tuning produced only a marginal improvement in cross-validation performance.

### Model Failure Analysis

**Class imbalance:** Using `class_weight="balanced"` increased churn recall from 55.88% to 78.34% and reduced false negatives from 165 to 81. However, false positives increased from 109 to 288.

**Prediction thresholds:** Experiments with thresholds of 0.3, 0.5, and 0.7 demonstrated the trade-off between detecting more churners and generating more false alarms.

**Data leakage:** An isolated experiment demonstrated that directly using a target-derived feature can produce misleadingly perfect predictions. This reinforced the importance of excluding information unavailable at prediction time.

**Subgroup errors:** The tuned model missed all actual churners in the one-year and two-year contract groups in the evaluated test split. This suggests that aggregate metrics alone are insufficient for understanding model behavior.

### Evaluation Caveat

The test set was reused during exploratory model comparisons and threshold experiments. Consequently, these test metrics should not be treated as an unbiased final performance estimate. A future evaluation should reserve an untouched test set until model selection and threshold decisions are complete.

## Reusable ML Pipeline

The notebook experiments were refactored into Python modules.

- `src/data/load_data.py`: loads the cleaned dataset.
- `src/train.py`: trains, evaluates, and saves the model pipeline.
- `src/predict.py`: validates customer inputs, loads the saved model, and returns predictions.
- `models/churn_model.joblib`: serialized preprocessing and trained model pipeline.

The preprocessing steps are stored with the estimator so that prediction-time transformations are consistent with training.

## FastAPI Prediction Service

The trained model is exposed through an HTTP API.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Confirm that the API is running |
| POST | `/predict` | Predict customer churn |

### Run Locally

Create and activate a virtual environment, then install the project dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-docker.txt
python -m pip install pytest
```

Ensure the cleaned dataset and trained model exist at the paths expected by the application.

Start the API from the project root:

```powershell
python -m uvicorn src.api.main:app --reload
```

Open the interactive API documentation:

`http://127.0.0.1:8000/docs`

### Example Prediction Request

Send a POST request to `/predict` with a JSON customer record:

```json
{
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
  "TotalCharges": 780.0
}
```

Example response shape:

```json
{
  "churn_prediction": 0,
  "churn_probability": 0.1835,
  "interpretation": "Unlikely to churn"
}
```

The probability is an estimate produced by the trained model, not a guarantee of customer behavior.

### Input and Output Validation

Pydantic schemas validate required fields, allowed categorical values, and numeric constraints before prediction. The response schema defines the expected output fields and probability range.

Invalid request bodies return HTTP `422` responses.

## Automated Testing

The project includes tests for data loading, prediction behavior, input validation, and API responses.

Run the complete test suite:

```powershell
python -m pytest tests/ -v
```

The API tests cover:

- Successful home endpoint responses.
- Valid prediction requests.
- Rejection of requests with missing fields.
- Rejection of invalid contract values.
- Expected prediction response fields and value ranges.

Automated tests provide a repeatable way to check behavior after code changes.

## Docker Containerization

The API is packaged in a Docker image with its runtime dependencies.

### Build the Image

Run from the project root:

```powershell
docker build -t churn-api .
```

### Start the Container

```powershell
docker run -d --name churn-api-container -p 8000:8000 churn-api
```

Open the API documentation at:

`http://localhost:8000/docs`

### Inspect and Manage the Container

```powershell
docker ps
docker logs churn-api-container
docker stop churn-api-container
docker start churn-api-container
```

The Dockerfile also defines a health check that periodically checks the application's home endpoint.

The `.dockerignore` file excludes development artifacts such as `.venv`, Git metadata, and Python cache files from the Docker build context.

## Engineering Lessons

- Why accuracy can be misleading on imbalanced datasets.
- How overfitting appears in the difference between training and test performance.
- How threshold selection changes precision and recall.
- Why target leakage can invalidate evaluation.
- How subgroup analysis reveals weaknesses hidden by aggregate metrics.
- Why cross-validation and an untouched final test set matter.
- How to separate model training, inference, and API code.
- How to validate requests and test APIs automatically.
- How to package an ML application using Docker and inspect container health.

## Current Limitations and Future Improvements

- Perform a final evaluation on an untouched test set.
- Select prediction thresholds using validation data and explicit business costs.
- Evaluate probability calibration and subgroup performance.
- Add more API tests for numeric boundaries, invalid categories, and response validation.
- Improve structured logging and error handling.
- Pin and audit runtime dependencies for reproducible builds.
- Deploy the container to a hosting platform and document deployment configuration.

## Project Status

The project currently includes a reusable churn prediction pipeline, a validated FastAPI service, automated tests, and a Dockerized application with a health check. Public deployment and final unbiased model evaluation remain future steps.
