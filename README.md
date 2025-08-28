# Term Deposit Investment Predictor

A complete machine learning project that predicts whether a client will **subscribe to a term deposit** based on marketing campaign attributes. 
This project covers data ingestion, exploratory data analysis (EDA), feature engineering, model training, evaluation, and (optionally) hyperparameter tuning and class imbalance handling.

## Features & Preprocessing
- Categorical encoding (one-hot encoding).
- Feature scaling (StandardScaler).
- Train/validation split with train_test_split.
- Stratified K-Fold cross-validation and scoring.
- Class imbalance handling using SMOTE.
- Modeling via sklearn/imblearn Pipelines.
- Hyperparameter tuning (GridSearchCV/RandomizedSearchCV).
- Confusion matrix analysis.
- Classification report (precision/recall/F1).
- ROC/PR curves and AUC metrics.


## Models Trained
- LogisticRegression
- RandomForest
- XGBClassifier


## Evaluation Protocol
- Stratified K-Fold cross-validation
- Cross-validation with scikit-learn scoring APIs
- Classification report (Precision/Recall/F1)
- Confusion matrix
- ROC-AUC


## Key Outcomes

### XGBoost (Base Model) Results

Cross-Validation Performance (5-Fold): 

- 5-Fold CV F1 Scores: [0.7568, 0.7504, 0.7384, 0.7442, 0.7492]

- Mean CV F1 Score: 0.7478

- Optimal Decision Threshold: 0.5512

- Confusion Matrix (Threshold = 0.5512):

6962  459
185   394
 
-Classification Report:

              precision    recall  f1-score   support

           0       0.97      0.94      0.96      7421
           1       0.46      0.68      0.55       579

    accuracy                           0.92      8000
    macro avg      0.72      0.81      0.75      8000
    weighted avg   0.94      0.92      0.93      8000

- Key Observations:
- The model achieves high accuracy (0.92), largely driven by the majority class (label 0).
- For the minority class (1), recall is 0.68 and precision is 0.46, indicating moderate success in detecting positive cases but with some false positives.
- The macro-averaged F1 score (0.75) is consistent with cross-validation results (0.7478), confirming stable generalization.

### XGBoost (Hyperparameter Tuned Model) Results

Cross-Validation Performance

- 5-Fold CV F1 Scores: [0.7614, 0.7571, 0.7613, 0.7569, 0.7624]

- Mean CV F1 Score: 0.7598

- Optimal Decision Threshold: 0.3657

- Confusion Matrix (Threshold = 0.3):
  10487   644
  255     614
 
- Classification Report:

                precision    recall  f1-score   support

           0         0.98      0.94      0.96     11131
           1         0.49      0.71      0.58       869
      accuracy                           0.93     12000
      macro avg      0.73      0.82      0.77     12000
      weighted avg   0.94      0.93      0.93     12000

- Key Observations:
- The model maintains high precision for class 0 (0.98) while achieving a recall of 0.71 for class 1 after threshold adjustment.
- Overall test accuracy is 93%, with a macro F1 of 0.77, showing improved balance compared to the untuned model.
- The tuned decision threshold (0.3657) favors recall for the minority class (1), reducing false negatives. 

## Project Workflow

1. **Load & Inspect Data**: Read the raw dataset into a DataFrame, inspect shape, missing values, and class distribution.
2. **EDA**: Univariate and bivariate analysis (histograms, bar plots, correlation heatmaps), focusing on features that influence subscription.
3. **Preprocessing**:
   - Handle missing values.
   - Encode categorical variables (one-hot encoding).
   - Optional: Scale numeric features.
   - Optional: Address class imbalance (e.g., SMOTE).
4. **Modeling**:
   - Baseline models for reference.
   - Advanced models (e.g., Random Forest, XGBoost).
   - Optional: Hyperparameter optimization (GridSearchCV/RandomizedSearchCV).
5. **Evaluation**:
   - Stratified cross-validation where appropriate.
   - Classification report, confusion matrix, ROC/PR curves.
   - Business-oriented metrics (precision at top deciles, if computed).
6. **Interpretation**:
   - Feature importance plots (tree-based models).
   - Discussion of trade-offs (precision vs recall for marketing outreach).
