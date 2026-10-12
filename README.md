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


##  Model Failure Analysis

### 1. Class Imbalance
Trained Logistic Regression with `class_weight="balanced"` to increase attention to the minority churn class.

- Original recall: 55.88%
- Balanced recall: 78.34%
- Original false negatives: 165
- Balanced false negatives: 81
- Trade-off: false positives increased from 109 to 288.

**Learning:** Improving churn detection can increase unnecessary retention outreach. Model selection should account for business costs, not accuracy alone.

### 2. Prediction Threshold Experiments
Compared thresholds of 0.3, 0.5, and 0.7 using predicted churn probabilities.

- Threshold 0.3: higher recall, more false positives.
- Threshold 0.5: intermediate trade-off.
- Threshold 0.7: higher precision, but substantially lower recall.

**Learning:** Threshold selection depends on the cost of missed churners versus unnecessary interventions.

### 3. Data Leakage
Created an isolated experiment where a feature directly copied the target column. Directly using that feature as the prediction produced 100% accuracy.

**Learning:** Leakage creates misleading evaluation results when information unavailable at prediction time enters the features. Features must be checked for target-derived and future information.

### Engineering Note
The experiments reused the test set for comparisons, so these results are exploratory rather than an unbiased final evaluation. The next stage will use cross-validation for model selection and reserve the test set for final evaluation.


## Cross-Validation and Hyperparameter Tuning

### Cross-Validation
Used 5-fold stratified cross-validation with F1 as the selection metric.

- Original Logistic Regression mean CV F1: 0.5923
- Tuned Logistic Regression mean CV F1: 0.5975
- Tuned Random Forest mean CV F1: 0.5833

### Hyperparameter Tuning
Used `GridSearchCV` to tune Logistic Regression's `C` and Random Forest's number of trees, depth, and minimum leaf size.

- Best Logistic Regression `C`: 10
- Best Random Forest configuration: 200 trees, depth 10, minimum leaf size 5

### Evaluation
The tuned Logistic Regression achieved:

- Accuracy: 80.48%
- Precision: 65.52%
- Recall: 55.88%
- F1-score: 60.32%
- ROC-AUC: 0.8412

**Conclusion:** Logistic Regression remained the leading candidate. Hyperparameter tuning produced only a marginal cross-validation improvement and did not improve the held-out test metrics.

**Limitation:** The test set was reused during earlier experiments. Its score is therefore not a fully unbiased final estimate.