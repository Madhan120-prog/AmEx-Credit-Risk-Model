# Credit Risk Modeling: AMEX Case Study

This project presents an end to end machine learning solution to predict credit default risk using the Kaggle competition dataset AMEX Default Prediction provided by American Express.

## Introduction

In consumer lending, identifying creditworthy individuals is crucial for minimizing risk and improving customer experience. This project applies machine learning to enhance default prediction so financial institutions can make better approval decisions, reduce losses, and offer credit products to the right customers at the right time.

The modeling approach leverages customer behavior using time series transactional data and anonymized profile information. The final solution is designed to be accurate and explainable, supported by feature importance and SHAP based interpretability.

Data source: Kaggle Competition AMEX Default Prediction

## Key Contributions

- Handled a large scale dataset with 5.5M plus rows using efficient chunking
- Performed missing value treatment, categorical encoding, and temporal aggregation over 12 month periods
- Engineered features such as mean, min, max, and last for numeric time series variables
- Applied one hot encoding to categorical variables, generating 45 plus new features
- Built and compared two models: XGBoost and a Neural Network using grid search
- Selected the final model based on AUC performance and variance stability across training and test splits
- Used SHAP values to explain predictions and understand feature contributions

## Project Structure

Credit_Risk_Modeling/

├── AMEX_Credit_Risk_Analysis.ipynb

├── Credit Risk Model.pptx

├── credit_risk_modeling.py

├── final_data_for_project.csv

└── README.md

## Objective

Build an accurate and explainable credit risk model using machine learning and compare two modeling strategies. The final model is selected based on AUC performance.

Models implemented:
- XGBoost
- Neural Network

## Data Overview

- Total customers: 91,783
- Time span: 13 months of historical data
- Feature groups:
  - D_: Delinquency
  - B_: Balance
  - S_: Spend
  - P_: Payment
  - R_: Risk

## Feature Engineering

- Aggregated temporal features over 12 months: mean, min, max, last
- One hot encoded 11 categorical columns resulting in 45 dummy variables
- Missing values handled by type:
  - Binary: 0
  - Categorical: mode
  - Numeric: mean

## Model Training

### XGBoost

Grid search parameters:
- Trees: 50, 100, 300
- Learning rate: 0.01, 0.1
- Row subsample: 50 percent, 80 percent
- Feature subsample: 50 percent, 100 percent
- Class weight: 1, 5, 10

Total models trained: 72  
Best AUC performance observed on Test 2  
SHAP analysis used for interpretability

### Neural Network

Preprocessing:
- Outliers capped at the 99th percentile
- Normalized using StandardScaler

Grid search parameters:
- Hidden layers: 2, 4
- Neurons: 4, 6
- Activation: ReLU, Tanh
- Dropout: 50 percent, 100 percent
- Batch size: 100, 10000

Total models trained: 32  
Did not outperform XGBoost

## SHAP Insights for XGBoost

| Feature   | Impact   |
|----------|----------|
| P_2_last | Negative |
| B_1_last | Negative |
| D_39_max | Positive |
| S_3_mean | Positive |

SHAP values help explain how each feature contributes to the model prediction.

## Final Model Selection

XGBoost outperformed the Neural Network across Train, Test1, and Test2. It achieved the highest AUC with the lowest variance across splits, so it was selected as the final model.

## Future Enhancements

- Build a Streamlit dashboard for real time scoring
- Run AutoML based model comparisons
- Add model monitoring for drift detection

## Results Summary

| Model       | AUC (Test2) | Remarks                |
|------------|-------------|------------------------|
| XGBoost     | High        | Best performer overall |
| Neural Net  | Moderate    | Underperformed         |

## Tech Stack

- Python, Pandas, NumPy
- Matplotlib, Seaborn
- XGBoost, LightGBM, Scikit Learn
- SHAP, GridSearchCV
- PowerPoint for stakeholder communication

## Contact

For questions, improvements, or collaboration, feel free to reach out or fork the repository.
