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


## Model Comparison 

Compared Logistic Regression, an unrestricted Decision Tree, a depth-limited Decision Tree (`max_depth=5`), and Random Forest (`n_estimators=200`).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 80.55% | 65.72% | 55.88% | 60.40% | 0.842 |
| Decision Tree (unrestricted) | 72.25% | 47.81% | 49.73% | 48.75% | 0.650 |
| Decision Tree (depth 5) | 79.84% | 63.47% | 56.68% | 59.89% | 0.830 |
| Random Forest (200 trees) | 78.78% | 62.89% | 48.93% | 55.04% | 0.817 |

**Findings**
- The unrestricted Decision Tree showed overfitting: 99.80% training accuracy versus 72.25% test accuracy.
- Limiting tree depth reduced the train–test accuracy gap to approximately 0.54 percentage points.
- Logistic Regression achieved the best accuracy, F1-score and ROC-AUC among these initial configurations.
- The depth-limited Decision Tree achieved slightly higher churn recall than Logistic Regression.
- Results are preliminary; further tuning and validation are needed before selecting the final model.