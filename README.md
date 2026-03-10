# COVID-19 Patient Mortality Prediction — End-to-End Data Science Workflow

**MSIS 522 — Analytics and Machine Learning**  
Foster School of Business | University of Washington  
Instructor: Prof. Léonard Boussioux

## Overview

This project implements a complete data science workflow for predicting COVID-19 patient mortality using a dataset of over 1 million anonymized patient records from the Mexican government's epidemiological surveillance system. The workflow covers exploratory data analysis, predictive modeling with six different algorithms, model explainability using SHAP, and deployment as an interactive Streamlit web application.

## Dataset

The dataset contains **1,021,977 patient records** with 16 features including demographics (age, sex), clinical status (hospitalization, pneumonia, COVID test result), and 10 pre-existing conditions (diabetes, hypertension, obesity, COPD, etc.). The binary target variable is **DEATH** (0 = survived, 1 = died).

**Source:** Instructor-curated COVID-19 mortality dataset (Mexican government epidemiological surveillance data).

## Project Structure

```
├── streamlit_app.py          # Streamlit web application (main entry point)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── data/
│   └── covid_data.csv        # Dataset (1M+ records)
├── models/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   ├── lightgbm.joblib
│   ├── mlp_sklearn.joblib
│   ├── scaler.joblib
│   ├── model_comparison.csv
│   ├── best_params.json
│   ├── roc_data.json
│   ├── feature_cols.json
│   └── mlp_tuning_results.csv
├── plots/
│   ├── 1_target_distribution.png
│   ├── 2_age_distribution.png
│   ├── 3_mortality_by_comorbidity.png
│   ├── 4_violin_age_sex_outcome.png
│   ├── 5_hospitalization_pneumonia.png
│   ├── 6_comorbidity_cooccurrence.png
│   ├── 7_correlation_heatmap.png
│   ├── 8_decision_tree.png
│   ├── 9_mlp_training_history.png
│   ├── 10_mlp_tuning.png
│   ├── 11_model_comparison.png
│   ├── 12_roc_curves.png
│   ├── 13_shap_summary.png
│   ├── 14_shap_bar.png
│   ├── 15_shap_waterfall.png
│   └── 16_shap_waterfall_survived.png
└── notebooks/
    ├── MSIS522_HW1_Analysis.ipynb  # Complete analysis notebook (Parts 1-3)
    ├── analysis.py                 # Analysis script (Part 1 & 2)
    └── finish.py                   # Continuation script (Part 2 cont. & Part 3)
```

## Models Trained

| Model | F1 Score | AUC-ROC |
|-------|----------|---------|
| Logistic Regression | 0.9106 | 0.9562 |
| Decision Tree (CART) | 0.9147 | 0.9455 |
| Random Forest | **0.9198** | 0.9544 |
| XGBoost | 0.9170 | **0.9571** |
| LightGBM | 0.9155 | 0.9567 |
| MLP Neural Network | 0.9150 | 0.9565 |

All tree-based models were tuned using 5-fold Stratified Cross-Validation with GridSearchCV. The MLP was tuned across hidden layer sizes, learning rates, and regularization strengths (bonus).

## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd <repo-name>
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run streamlit_app.py
   ```

4. Open `http://localhost:8501` in your browser.

## Deployed App

The app is deployed and accessible at: **[Streamlit Community Cloud Link]**

## Key Findings

- **AGE** is the single most important predictor of COVID-19 mortality, followed by **PNEUMONIA** and **HOSPITALIZED** status.
- Among comorbidities, **DIABETES** and **HYPERTENSION** have the strongest impact on mortality risk.
- All six models achieve strong performance (F1 > 0.91, AUC > 0.94), with Random Forest achieving the best F1 and XGBoost the best AUC-ROC.
- SHAP analysis reveals that the models learn medically meaningful patterns consistent with established clinical knowledge.

## Technical Details

- **Random State:** 42 (used throughout for reproducibility)
- **Train/Test Split:** 70/30 with stratification
- **Class Imbalance Handling:** Balanced subset of 10,000 patients (5,000 per class)
- **Evaluation Metrics:** F1 Score (primary), AUC-ROC, Accuracy, Precision, Recall
