# Customer Churn Prediction System

A machine learning system for predicting customer churn.

## Data Cleaning 

- Converted `TotalCharges` from text to numeric.
- Identified 11 blank `TotalCharges` values associated with customers whose tenure was 0.
- Replaced those values with 0, preserving all 7,043 customer records.
- Encoded the target: `No → 0`, `Yes → 1`.
- Excluded `customerID` from model features.
- Identified categorical features for later encoding.
- Saved the cleaned dataset to `data/processed/churn_cleaned.csv`.

**Initial EDA findings**
- Overall churn rate: 26.5%.
- Month-to-month churn rate: 42.7%.
- One-year contract churn rate: 11.3%.
- Two-year contract churn rate: 2.8%.

### Baseline Model Results

- **Dummy baseline accuracy:** 73.46%; churn recall: 0%.
- **Model:** Logistic Regression with one-hot encoding and standardized numerical features.
- **Accuracy:** 80.55%
- **Precision:** 65.72%
- **Recall:** 55.88%
- **F1-score:** 60.40%
- **ROC-AUC:** 0.842

The Logistic Regression pipeline improved accuracy over the majority-class baseline and identified 209 of 374 churners on the held-out test set. However, it missed 165 churners, so recall and the business cost of false negatives need further investigation.

**Note:** These are initial test-set results. Use validation data or cross-validation for subsequent model selection and threshold tuning.