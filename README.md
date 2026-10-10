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