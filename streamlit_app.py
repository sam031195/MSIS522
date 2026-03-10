"""
COVID-19 Mortality Prediction — Decision Cockpit
MSIS 522 · Analytics and Machine Learning · HW1
End-to-end data science workflow: EDA → Modeling → SHAP → Interactive Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="COVID-19 Mortality Prediction Cockpit",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Load artifacts
# ─────────────────────────────────────────────
BASE = os.path.dirname(__file__)

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE, "data", "covid_data.csv"))

@st.cache_data
def load_results():
    return pd.read_csv(os.path.join(BASE, "models", "model_comparison.csv"), index_col=0)

@st.cache_data
def load_feature_cols():
    with open(os.path.join(BASE, "models", "feature_cols.json")) as f:
        return json.load(f)

@st.cache_data
def load_roc_data():
    with open(os.path.join(BASE, "models", "roc_data.json")) as f:
        return json.load(f)

@st.cache_data
def load_best_params():
    with open(os.path.join(BASE, "models", "best_params.json")) as f:
        return json.load(f)

@st.cache_resource
def load_models():
    model_files = {
        "Logistic Regression": "logistic_regression",
        "Decision Tree (CART)": "decision_tree",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
        "LightGBM": "lightgbm",
        "MLP Neural Network": "mlp_sklearn",
    }
    models = {}
    for name, fname in model_files.items():
        path = os.path.join(BASE, "models", f"{fname}.joblib")
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

@st.cache_resource
def load_scaler():
    return joblib.load(os.path.join(BASE, "models", "scaler.joblib"))

# Load everything
df = load_data()
results_df = load_results()
feature_cols = load_feature_cols()
roc_data = load_roc_data()
best_params = load_best_params()
models = load_models()
scaler = load_scaler()

# Identify best model
best_model_name = results_df["AUC-ROC"].idxmax()

# Feature metadata
BINARY_FEATURES = [c for c in feature_cols if c != "AGE"]
CONTINUOUS_FEATURES = ["AGE"]

COMORBIDITY_FEATURES = ["DIABETES", "COPD", "ASTHMA", "IMMUNOSUPPRESSION",
                        "HYPERTENSION", "OTHER_DISEASE", "CARDIOVASCULAR",
                        "OBESITY", "RENAL_CHRONIC", "TOBACCO"]

FEATURE_LABELS = {
    "AGE": "Age (years)",
    "SEX": "Sex (0=Female, 1=Male)",
    "HOSPITALIZED": "Hospitalized",
    "PNEUMONIA": "Pneumonia",
    "PREGNANT": "Pregnant",
    "DIABETES": "Diabetes",
    "COPD": "COPD",
    "ASTHMA": "Asthma",
    "IMMUNOSUPPRESSION": "Immunosuppression",
    "HYPERTENSION": "Hypertension",
    "OTHER_DISEASE": "Other Disease",
    "CARDIOVASCULAR": "Cardiovascular",
    "OBESITY": "Obesity",
    "RENAL_CHRONIC": "Chronic Kidney Disease",
    "TOBACCO": "Tobacco Use",
    "COVID_POSITIVE": "COVID Test Positive",
}

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Model Controls")
    active_model_name = st.selectbox(
        "Active model",
        list(models.keys()),
        index=list(models.keys()).index(best_model_name) if best_model_name in models else 0,
    )
    st.caption(f"Best model by AUC-ROC: **{best_model_name}**")

    st.divider()
    st.markdown("### Best Model Metrics")

    best_row = results_df.loc[best_model_name]
    st.metric("AUC-ROC", f"{best_row['AUC-ROC']:.4f}")
    st.metric("F1 Score", f"{best_row['F1 Score']:.4f}")
    st.metric("Accuracy", f"{best_row['Accuracy']:.4f}")
    st.metric("Precision", f"{best_row['Precision']:.4f}")
    st.metric("Recall", f"{best_row['Recall']:.4f}")

    st.divider()
    st.markdown("### Dataset Summary")
    st.metric("Total Patients", f"{len(df):,}")
    death_rate = df["DEATH"].mean() * 100
    st.metric("Mortality Rate", f"{death_rate:.2f}%")
    st.metric("Features", f"{len(feature_cols)}")

active_model = models[active_model_name]

# ─────────────────────────────────────────────
# Title
# ─────────────────────────────────────────────
st.markdown("# COVID-19 Mortality Prediction Cockpit")
st.caption("End-to-end ML workflow: from exploratory analysis through predictive modeling, model explainability, and interactive prediction.")

# ─────────────────────────────────────────────
# EXACTLY 4 TABS per rubric
# ─────────────────────────────────────────────
tab_exec, tab_eda, tab_perf, tab_explain_predict = st.tabs([
    "Executive Summary",
    "Descriptive Analytics",
    "Model Performance",
    "Explainability & Interactive Prediction",
])

# ═════════════════════════════════════════════
# TAB 1: EXECUTIVE SUMMARY (4 points)
# ═════════════════════════════════════════════
with tab_exec:
    st.markdown("## Executive Summary")

    st.markdown(
        "This project tackles one of the most consequential prediction problems in modern healthcare: "
        "**identifying which COVID-19 patients are most likely to die**. The dataset, curated from the "
        "Mexican government's epidemiological surveillance system, contains **1,021,977 anonymized patient "
        "records**, each described by 16 features spanning demographics (age, sex, pregnancy status), "
        "clinical indicators (hospitalization, pneumonia diagnosis, COVID test result), and ten pre-existing "
        "comorbidities including diabetes, hypertension, obesity, COPD, asthma, cardiovascular disease, "
        "chronic kidney disease, immunosuppression, tobacco use, and other diseases. The binary target "
        "variable, **DEATH**, records whether each patient survived or died. The prediction task is binary "
        "classification: given a patient's profile at intake, estimate the probability of mortality."
    )

    st.markdown(
        "**Why this matters:** In a clinical setting, the ability to accurately stratify patient risk is "
        "not merely a technical exercise — it is a tool for **resource allocation and triage**. During the "
        "COVID-19 pandemic, hospitals worldwide faced overwhelming patient volumes and had to make difficult "
        "decisions about which patients to prioritize for ICU beds, ventilators, and experimental treatments. "
        "A reliable mortality prediction model enables clinicians to identify the most vulnerable patients "
        "early, direct scarce resources where they will save the most lives, and provide families with "
        "honest prognostic information. Furthermore, understanding *which* comorbidities drive mortality "
        "risk — and by how much — helps public health officials design targeted vaccination campaigns and "
        "preventive interventions. This is precisely the kind of high-stakes, interpretable prediction "
        "problem where machine learning can make a tangible difference in human outcomes."
    )

    st.markdown(
        "**Approach and key findings:** We trained and compared six models of increasing complexity: "
        "Logistic Regression (baseline), Decision Tree (CART), Random Forest, XGBoost, LightGBM, and a "
        "Multi-Layer Perceptron (MLP) neural network. To address the severe class imbalance (only 7.3% "
        "of patients died), we constructed a balanced training subset of 10,000 patients (5,000 per class) "
        "and evaluated all models using **Stratified 5-Fold Cross-Validation** with **GridSearchCV** for "
        "hyperparameter tuning. The primary evaluation metric is **AUC-ROC**, which measures the model's "
        "ability to distinguish between survivors and non-survivors across all classification thresholds — "
        "a metric that is robust to class imbalance, as discussed in our course's COVID-19 CART tutorial."
    )

    st.markdown(
        f"The best-performing model is **{best_model_name}** with an AUC-ROC of "
        f"**{results_df.loc[best_model_name, 'AUC-ROC']:.4f}**, an F1 score of "
        f"**{results_df.loc[best_model_name, 'F1 Score']:.4f}**, and accuracy of "
        f"**{results_df.loc[best_model_name, 'Accuracy']:.4f}**. Notably, all six models achieved AUC-ROC "
        "scores above 0.94, indicating that the clinical and demographic features in this dataset carry "
        "strong predictive signal. SHAP analysis reveals that **hospitalization status**, **age**, and "
        "**pneumonia diagnosis** are the three most influential predictors of mortality — a finding that "
        "aligns with established clinical literature and provides actionable insight for triage protocols. "
        "The interactive prediction tool in this dashboard allows clinicians or analysts to input any "
        "patient profile and receive a real-time risk assessment with a full SHAP explanation of the "
        "prediction drivers."
    )

    # Key metrics cards
    st.divider()
    st.markdown("### Key Results at a Glance")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Best Model", best_model_name)
    c2.metric("AUC-ROC", f"{results_df.loc[best_model_name, 'AUC-ROC']:.4f}")
    c3.metric("F1 Score", f"{results_df.loc[best_model_name, 'F1 Score']:.4f}")
    c4.metric("Models Trained", "6")
    c5.metric("Dataset Size", f"{len(df):,}")

    st.markdown("### Data and Features")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Demographics:**
- `AGE` — Patient age in years
- `SEX` — Biological sex (0 = Female, 1 = Male)
- `PREGNANT` — Pregnancy status

**Clinical Status:**
- `HOSPITALIZED` — Whether the patient was hospitalized
- `PNEUMONIA` — Pneumonia diagnosis at intake
- `COVID_POSITIVE` — COVID-19 test result
""")
    with col2:
        st.markdown("""
**Pre-existing Conditions (10 comorbidities):**
- `DIABETES`, `HYPERTENSION`, `OBESITY`
- `COPD`, `ASTHMA`, `CARDIOVASCULAR`
- `RENAL_CHRONIC`, `IMMUNOSUPPRESSION`
- `TOBACCO`, `OTHER_DISEASE`

**Target Variable:**
- `DEATH` — Binary (0 = Survived, 1 = Died)
""")


# ═════════════════════════════════════════════
# TAB 2: DESCRIPTIVE ANALYTICS (4 points)
# ═════════════════════════════════════════════
with tab_eda:
    st.markdown("## Descriptive Analytics")
    st.caption("Interactive exploration of the COVID-19 patient dataset — each visualization includes a brief interpretation.")

    # Dataset summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Columns", f"{df.shape[1]}")
    col3.metric("Mortality Rate", f"{df['DEATH'].mean()*100:.2f}%")
    col4.metric("Median Age", f"{df['AGE'].median():.0f} years")

    # --- Visualization 1: Target Distribution ---
    st.markdown("### 1. Target Variable Distribution")
    fig_target = px.histogram(
        df, x="DEATH", color="DEATH",
        color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
        labels={"DEATH": "Outcome", "count": "Count"},
        title="Outcome Distribution: Survived vs. Died",
        category_orders={"DEATH": [0, 1]},
    )
    fig_target.update_layout(
        xaxis=dict(tickvals=[0, 1], ticktext=["Survived (0)", "Died (1)"]),
        showlegend=False, height=400,
    )
    st.plotly_chart(fig_target, use_container_width=True)
    st.markdown(
        "The dataset is **heavily imbalanced**: approximately 92.7% of patients survived while only 7.3% died. "
        "This imbalance means that a naive classifier predicting 'survived' for every patient would achieve ~93% accuracy, "
        "which is why we use **AUC-ROC** rather than raw accuracy as our primary evaluation metric. To train meaningful "
        "models, we created a balanced subset of 10,000 patients (5,000 per class) for training."
    )

    # --- Visualization 2: Age Distribution by Outcome ---
    st.markdown("### 2. Age Distribution by Outcome")
    fig_age = px.histogram(
        df, x="AGE", color="DEATH", barmode="overlay",
        color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
        labels={"AGE": "Age (years)", "DEATH": "Outcome"},
        title="Age Distribution: Survived vs. Died",
        opacity=0.7, marginal="box",
    )
    fig_age.update_layout(height=500)
    st.plotly_chart(fig_age, use_container_width=True)
    st.markdown(
        "Age is a powerful discriminator between survivors and non-survivors. The distribution of deceased patients "
        "(red) is shifted significantly to the right compared to survivors (green), with a median age roughly 20 years "
        "higher. The box plots at the top confirm this: the interquartile range for deceased patients sits almost "
        "entirely above the median age of survivors. This aligns with the well-documented clinical finding that "
        "advanced age is the single strongest risk factor for COVID-19 mortality."
    )

    # --- Visualization 3: Violin Plot — Age by Sex and Outcome ---
    st.markdown("### 3. Age Distribution by Sex and Outcome (Violin Plot)")
    fig_violin = px.violin(
        df, x="SEX", y="AGE", color="DEATH", box=True,
        color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
        labels={"SEX": "Sex", "AGE": "Age (years)", "DEATH": "Outcome"},
        title="Age Distribution by Sex and Outcome",
        category_orders={"SEX": [0, 1]},
    )
    fig_violin.update_layout(
        xaxis=dict(tickvals=[0, 1], ticktext=["Female", "Male"]),
        height=500,
    )
    st.plotly_chart(fig_violin, use_container_width=True)
    st.markdown(
        "This violin plot reveals an interaction between sex and age in predicting mortality. For both sexes, "
        "deceased patients are substantially older, but the effect is slightly more pronounced in males, whose "
        "mortality distribution peaks at a somewhat younger age than females. Male patients also show a broader "
        "spread of ages among the deceased, suggesting that men face elevated mortality risk across a wider age range."
    )

    # --- Visualization 4: Mortality Rate by Comorbidity ---
    st.markdown("### 4. Mortality Rate by Comorbidity")
    mort_rates = []
    for feat in COMORBIDITY_FEATURES:
        rate_with = df[df[feat] == 1]["DEATH"].mean() * 100
        rate_without = df[df[feat] == 0]["DEATH"].mean() * 100
        mort_rates.append({
            "Comorbidity": FEATURE_LABELS.get(feat, feat),
            "With Condition": rate_with,
            "Without Condition": rate_without,
        })
    mort_df = pd.DataFrame(mort_rates)
    mort_df = mort_df.sort_values("With Condition", ascending=True)

    fig_mort = go.Figure()
    fig_mort.add_trace(go.Bar(
        y=mort_df["Comorbidity"], x=mort_df["With Condition"],
        name="With Condition", orientation="h", marker_color="#e74c3c",
    ))
    fig_mort.add_trace(go.Bar(
        y=mort_df["Comorbidity"], x=mort_df["Without Condition"],
        name="Without Condition", orientation="h", marker_color="#2ecc71",
    ))
    fig_mort.update_layout(
        title="Mortality Rate (%) by Comorbidity Presence",
        xaxis_title="Mortality Rate (%)", barmode="group", height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_mort, use_container_width=True)
    st.markdown(
        "Every comorbidity increases mortality risk, but the magnitude varies dramatically. Chronic kidney disease "
        "and COPD show the largest gap between 'with' and 'without' mortality rates, while obesity and asthma show "
        "comparatively smaller increases. This suggests that renal and pulmonary comorbidities are particularly "
        "dangerous in the context of COVID-19, likely because the virus attacks the respiratory system and places "
        "additional strain on already-compromised kidneys through inflammatory cascades."
    )

    # --- Visualization 5: Hospitalization × Pneumonia ---
    st.markdown("### 5. Hospitalization and Pneumonia vs. Mortality")
    cross = df.groupby(["HOSPITALIZED", "PNEUMONIA"])["DEATH"].mean().reset_index()
    cross["DEATH"] = cross["DEATH"] * 100
    cross["HOSPITALIZED"] = cross["HOSPITALIZED"].map({0: "Not Hospitalized", 1: "Hospitalized"})
    cross["PNEUMONIA"] = cross["PNEUMONIA"].map({0: "No Pneumonia", 1: "Pneumonia"})
    fig_cross = px.bar(
        cross, x="HOSPITALIZED", y="DEATH", color="PNEUMONIA",
        barmode="group", color_discrete_sequence=["#3498db", "#e74c3c"],
        labels={"DEATH": "Mortality Rate (%)", "HOSPITALIZED": ""},
        title="Mortality Rate by Hospitalization and Pneumonia Status",
    )
    fig_cross.update_layout(height=450)
    st.plotly_chart(fig_cross, use_container_width=True)
    st.markdown(
        "The combination of hospitalization and pneumonia is the strongest clinical indicator of mortality in this "
        "dataset. Hospitalized patients with pneumonia have a mortality rate exceeding 30%, compared to less than 2% "
        "for non-hospitalized patients without pneumonia. This dramatic interaction effect confirms that these two "
        "features capture the severity of the patient's condition at intake and will likely dominate model predictions."
    )

    # --- Visualization 6: Correlation Heatmap ---
    st.markdown("### 6. Full Feature Correlation Heatmap")
    corr_cols = feature_cols + ["DEATH"]
    corr_matrix = df[corr_cols].corr()
    fig_corr = px.imshow(
        corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r",
        title="Feature Correlation Matrix (All Features Including Target)",
        aspect="auto",
    )
    fig_corr.update_layout(height=700)
    st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown(
        "The correlation heatmap reveals several important patterns. First, **HOSPITALIZED** and **PNEUMONIA** have "
        "the strongest positive correlations with DEATH (0.33 and 0.31 respectively), confirming their clinical "
        "significance. Second, **AGE** shows moderate positive correlation with DEATH (0.22) and also correlates "
        "with several comorbidities (diabetes, hypertension), suggesting that age acts as both a direct risk factor "
        "and a proxy for accumulated health burden. Third, the comorbidities themselves show mild inter-correlations "
        "(e.g., diabetes–hypertension at ~0.20), which is expected since metabolic conditions tend to cluster. "
        "These correlations inform our modeling strategy: tree-based models will naturally handle these interactions, "
        "while the logistic regression baseline may struggle with non-linear relationships."
    )

    # --- Visualization 7: Correlation with Mortality ---
    st.markdown("### 7. Feature Correlation with Mortality")
    death_corr = corr_matrix["DEATH"].drop("DEATH").sort_values(ascending=True)
    fig_dc = px.bar(
        x=death_corr.values, y=death_corr.index, orientation="h",
        labels={"x": "Pearson Correlation with DEATH", "y": "Feature"},
        title="Feature Correlation with Mortality (Ranked)",
        color=death_corr.values, color_continuous_scale="RdBu_r",
        range_color=[-0.5, 0.5],
    )
    fig_dc.update_layout(height=550, showlegend=False)
    st.plotly_chart(fig_dc, use_container_width=True)
    st.markdown(
        "This ranked bar chart provides a clear summary of each feature's linear association with mortality. "
        "Hospitalization, pneumonia, and age are the top three positively correlated features, while pregnancy "
        "and asthma show near-zero or slightly negative correlations — likely because younger, otherwise-healthy "
        "populations are overrepresented in those groups. This visualization serves as a useful sanity check: "
        "the features that correlate most strongly with mortality are exactly the ones we expect from clinical knowledge."
    )


# ═════════════════════════════════════════════
# TAB 3: MODEL PERFORMANCE (4 points)
# ═════════════════════════════════════════════
with tab_perf:
    st.markdown("## Model Performance Comparison")
    st.success(f"Best model by AUC-ROC: **{best_model_name}** (AUC = {results_df.loc[best_model_name, 'AUC-ROC']:.4f})")

    # --- Performance Summary Table ---
    st.markdown("### Performance Summary")
    st.markdown(
        "The table below compares all six models across five evaluation metrics. "
        "Green highlighting indicates the best value in each column."
    )
    styled_results = results_df.style.highlight_max(axis=0, color="#d4edda").format("{:.4f}")
    st.dataframe(styled_results, use_container_width=True)

    # --- Side-by-side bar charts ---
    col_bar1, col_bar2 = st.columns(2)

    with col_bar1:
        fig_f1 = px.bar(
            results_df.reset_index(), x="Model", y=["F1 Score", "Accuracy"],
            barmode="group", title="F1 Score & Accuracy by Model",
            color_discrete_sequence=["#3498db", "#2ecc71"],
        )
        fig_f1.update_layout(height=400, yaxis_range=[0.85, 0.95])
        st.plotly_chart(fig_f1, use_container_width=True)

    with col_bar2:
        fig_auc = px.bar(
            results_df.reset_index(), x="Model", y="AUC-ROC",
            title="AUC-ROC by Model",
            color="AUC-ROC", color_continuous_scale="Greens",
        )
        fig_auc.update_layout(height=400, yaxis_range=[0.93, 0.96], showlegend=False)
        st.plotly_chart(fig_auc, use_container_width=True)

    # --- ROC Curves ---
    st.markdown("### ROC Curves — All Models")
    fig_roc = go.Figure()
    colors = px.colors.qualitative.Set2
    for i, (model_name_roc, data) in enumerate(roc_data.items()):
        fig_roc.add_trace(go.Scatter(
            x=data["fpr"], y=data["tpr"],
            name=f"{model_name_roc} (AUC={data['auc']:.3f})",
            line=dict(color=colors[i % len(colors)], width=2),
        ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], name="Random Baseline",
        line=dict(color="gray", dash="dash", width=1),
    ))
    fig_roc.update_layout(
        title="ROC Curves — All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=550, legend=dict(x=0.55, y=0.05),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # --- Precision vs Recall scatter ---
    st.markdown("### Precision vs. Recall Tradeoff")
    fig_pr = px.scatter(
        results_df.reset_index(), x="Recall", y="Precision",
        text="Model", size="AUC-ROC", color="F1 Score",
        color_continuous_scale="RdYlGn",
        title="Precision vs. Recall (size = AUC-ROC, color = F1)",
    )
    fig_pr.update_traces(textposition="top center")
    fig_pr.update_layout(height=450)
    st.plotly_chart(fig_pr, use_container_width=True)

    # --- Model Comparison Commentary ---
    st.markdown("### Model Comparison Commentary")
    st.markdown(
        f"**{best_model_name}** achieves the highest AUC-ROC ({results_df.loc[best_model_name, 'AUC-ROC']:.4f}), "
        "narrowly outperforming LightGBM and Random Forest. All six models perform remarkably well (AUC > 0.94), "
        "which reflects the strong predictive signal in the clinical features. The Logistic Regression baseline "
        "is surprisingly competitive, suggesting that the relationship between features and mortality is largely "
        "monotonic — but the tree-based models capture non-linear interactions (e.g., age × comorbidity) that "
        "push their performance slightly higher. The MLP neural network performs comparably to the tree-based "
        "models but sacrifices interpretability. In a clinical deployment, we would recommend XGBoost or LightGBM "
        "paired with SHAP explanations, as they offer the best combination of predictive power and explainability."
    )

    # --- Training history (MLP) ---
    mlp_history_path = os.path.join(BASE, "plots", "9_mlp_training_history.png")
    if os.path.exists(mlp_history_path):
        st.markdown("### MLP Training History")
        st.image(mlp_history_path, use_container_width=True)
        st.markdown(
            "The MLP training curves show that the model converges within approximately 30 epochs. "
            "The validation loss closely tracks the training loss, indicating minimal overfitting — "
            "a result of the dropout regularization and early stopping applied during training."
        )

    # --- MLP Tuning Results ---
    mlp_tuning_path = os.path.join(BASE, "plots", "10_mlp_tuning.png")
    if os.path.exists(mlp_tuning_path):
        st.markdown("### MLP Hyperparameter Tuning Results (Bonus)")
        st.image(mlp_tuning_path, use_container_width=True)
        st.markdown(
            "We performed a grid search over hidden layer sizes, learning rates, and activation functions "
            "for the MLP. The plot above shows the validation performance across different configurations. "
            "The best MLP configuration uses two hidden layers of 128 units each with ReLU activation."
        )

    # --- Decision Tree Visualization ---
    tree_path = os.path.join(BASE, "plots", "8_decision_tree.png")
    if os.path.exists(tree_path):
        st.markdown("### Best Decision Tree (CART) Visualization")
        st.image(tree_path, use_container_width=True)
        st.markdown(
            "The pruned decision tree reveals the dominant decision logic: the first split is on AGE, "
            "followed by PNEUMONIA and HOSPITALIZED status. This transparent structure is exactly what "
            "makes CART valuable for interpretability — a clinician can trace any prediction path from "
            "root to leaf."
        )

    # --- Hyperparameters ---
    with st.expander("Tuned Hyperparameters (GridSearchCV)", expanded=False):
        for model_name_hp, params in best_params.items():
            st.markdown(f"**{model_name_hp}:**")
            st.json(params)


# ═════════════════════════════════════════════
# TAB 4: EXPLAINABILITY & INTERACTIVE PREDICTION (8 points)
# ═════════════════════════════════════════════
with tab_explain_predict:
    st.markdown("## Explainability & Interactive Prediction")
    st.caption(
        "This tab combines SHAP-based model explainability with interactive patient prediction. "
        "SHAP (SHapley Additive exPlanations) uses game-theoretic Shapley values to explain each "
        "feature's contribution to every prediction."
    )

    # --- SHAP Section ---
    st.markdown("---")
    st.markdown("### SHAP Global Explainability")

    shap_col1, shap_col2 = st.columns(2)

    shap_summary_path = os.path.join(BASE, "plots", "13_shap_summary.png")
    shap_bar_path = os.path.join(BASE, "plots", "14_shap_bar.png")
    shap_waterfall_died_path = os.path.join(BASE, "plots", "15_shap_waterfall.png")
    shap_waterfall_surv_path = os.path.join(BASE, "plots", "16_shap_waterfall_survived.png")

    with shap_col1:
        st.markdown("#### SHAP Summary Plot (Beeswarm)")
        if os.path.exists(shap_summary_path):
            st.image(shap_summary_path, use_container_width=True)
        st.markdown(
            "Each dot represents one patient. The horizontal position shows the SHAP value "
            "(impact on prediction), and the color shows the feature value (red = high, blue = low). "
            "**Hospitalization**, **age**, and **pneumonia** are the three most important features. "
            "Older patients (red dots for AGE) are pushed strongly toward higher mortality risk."
        )

    with shap_col2:
        st.markdown("#### SHAP Global Feature Importance (Bar)")
        if os.path.exists(shap_bar_path):
            st.image(shap_bar_path, use_container_width=True)
        st.markdown(
            "The bar plot ranks features by mean absolute SHAP value. Unlike Gini-based importance, "
            "SHAP values are grounded in cooperative game theory and provide consistent, fair attribution. "
            "Hospitalization status dominates, followed by age and pneumonia — confirming that disease "
            "severity at intake is the strongest predictor of outcome."
        )

    st.markdown("#### SHAP Waterfall Plots (Individual Predictions)")
    st.markdown(
        "Waterfall plots explain **individual predictions** by showing how each feature pushes the "
        "prediction away from the baseline (average prediction). This is what makes SHAP invaluable "
        "for clinical decision-making — a doctor can see exactly *why* a specific patient was flagged."
    )
    wf_col1, wf_col2 = st.columns(2)
    with wf_col1:
        st.markdown("**Patient Who Died**")
        if os.path.exists(shap_waterfall_died_path):
            st.image(shap_waterfall_died_path, use_container_width=True)
    with wf_col2:
        st.markdown("**Patient Who Survived**")
        if os.path.exists(shap_waterfall_surv_path):
            st.image(shap_waterfall_surv_path, use_container_width=True)

    st.markdown(
        "**Interpretation:** In the left plot, the patient's advanced age and pneumonia diagnosis "
        "push the prediction strongly toward mortality. In the right plot, younger age and absence "
        "of pneumonia push the prediction toward survival. This directional information is exactly "
        "what SHAP provides that traditional feature importance cannot."
    )

    # --- SHAP Insights for Decision-Makers ---
    st.markdown("#### How These Insights Help Decision-Makers")
    st.markdown(
        "A hospital administrator could use these SHAP insights to design a rapid triage protocol: "
        "patients who are hospitalized, elderly, and diagnosed with pneumonia should be immediately "
        "flagged for ICU-level monitoring. The fact that comorbidities like diabetes and chronic kidney "
        "disease also appear as significant SHAP features means that patients with multiple pre-existing "
        "conditions warrant additional scrutiny even if they initially appear stable."
    )

    # --- Interactive Prediction Section ---
    st.markdown("---")
    st.markdown("### Interactive Patient Prediction")
    st.markdown(
        f"Use the controls below to input patient characteristics and get a real-time mortality "
        f"risk prediction. The active model is **{active_model_name}** — you can change it in the sidebar."
    )

    # Model selector for prediction (also in sidebar, but explicit here per rubric)
    pred_model_name = st.selectbox(
        "Select prediction model",
        list(models.keys()),
        index=list(models.keys()).index(active_model_name),
        key="pred_model_select",
    )
    pred_model = models[pred_model_name]

    # Input controls
    st.markdown("#### Enter Patient Characteristics")
    col_left, col_right = st.columns(2)

    with col_left:
        age = st.slider("Age (years)", 0, 120, 55, help="Patient age in years")
        sex = st.selectbox("Sex", [("Female", 0), ("Male", 1)], format_func=lambda x: x[0])
        hospitalized = st.selectbox("Hospitalized?", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        pneumonia = st.selectbox("Pneumonia?", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        covid_pos = st.selectbox("COVID Test Positive?", [("No", 0), ("Yes", 1)], index=1, format_func=lambda x: x[0])
        pregnant = st.selectbox("Pregnant?", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])

    with col_right:
        diabetes = st.selectbox("Diabetes?", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        hypertension = st.selectbox("Hypertension?", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        obesity = st.selectbox("Obesity?", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        copd = st.selectbox("COPD?", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        asthma = st.selectbox("Asthma?", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        immuno = st.selectbox("Immunosuppression?", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        cardio = st.selectbox("Cardiovascular Disease?", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        kidney = st.selectbox("Chronic Kidney Disease?", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        tobacco = st.selectbox("Tobacco Use?", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        other = st.selectbox("Other Disease?", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])

    input_dict = {
        "SEX": sex[1], "HOSPITALIZED": hospitalized[1], "PNEUMONIA": pneumonia[1],
        "AGE": age, "PREGNANT": pregnant[1], "DIABETES": diabetes[1],
        "COPD": copd[1], "ASTHMA": asthma[1], "IMMUNOSUPPRESSION": immuno[1],
        "HYPERTENSION": hypertension[1], "OTHER_DISEASE": other[1],
        "CARDIOVASCULAR": cardio[1], "OBESITY": obesity[1],
        "RENAL_CHRONIC": kidney[1], "TOBACCO": tobacco[1],
        "COVID_POSITIVE": covid_pos[1],
    }
    input_df = pd.DataFrame([input_dict])[feature_cols]

    if st.button("Run Prediction", type="primary", use_container_width=True):
        # Scale features
        input_scaled = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=feature_cols)

        proba = pred_model.predict_proba(input_scaled_df)
        preds = pred_model.predict(input_scaled_df)

        death_prob = proba[0][1]
        prediction = preds[0]

        # Risk classification
        if death_prob < 0.2:
            risk_level, risk_color, risk_emoji = "Low Risk", "#2ecc71", "🟢"
        elif death_prob < 0.5:
            risk_level, risk_color, risk_emoji = "Moderate Risk", "#f39c12", "🟡"
        elif death_prob < 0.8:
            risk_level, risk_color, risk_emoji = "High Risk", "#e67e22", "🟠"
        else:
            risk_level, risk_color, risk_emoji = "Critical Risk", "#e74c3c", "🔴"

        st.divider()
        col_pred, col_shap_wf = st.columns([1, 1])

        with col_pred:
            st.markdown("#### Prediction Output")
            st.markdown(
                f"""
                <div style="background-color: #f8f9fa; border-left: 5px solid {risk_color};
                padding: 20px; border-radius: 8px; margin-bottom: 15px;">
                <h2 style="margin:0; color: {risk_color};">{death_prob*100:.1f}% Mortality Risk</h2>
                <p style="margin: 8px 0 0 0; font-size: 1.1em;">
                Risk Band: <strong style="color: {risk_color};">{risk_emoji} {risk_level}</strong></p>
                <p style="margin: 4px 0 0 0;">Model: <code>{pred_model_name}</code></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Plain-language summary
            age_val = input_df["AGE"].values[0]
            n_comorb = int(sum(input_df[COMORBIDITY_FEATURES].values[0]))
            hosp = "hospitalized" if input_df["HOSPITALIZED"].values[0] == 1 else "not hospitalized"
            pneu = "with pneumonia" if input_df["PNEUMONIA"].values[0] == 1 else "without pneumonia"

            st.markdown(
                f"**Plain-language summary:** This {int(age_val)}-year-old patient is {hosp}, "
                f"{pneu}, and presents {n_comorb} comorbidities. The model estimates a "
                f"**{death_prob*100:.1f}%** probability of mortality, classifying them as "
                f"**{risk_level}**."
            )

            # All-model spread
            st.markdown("#### All-Model Spread")
            spread_data = []
            for mname, mobj in models.items():
                p = mobj.predict_proba(input_scaled_df)[0][1]
                spread_data.append({"Model": mname, "Mortality Probability": p})
            spread_df = pd.DataFrame(spread_data).sort_values("Mortality Probability", ascending=False)

            fig_spread = px.bar(
                spread_df, x="Mortality Probability", y="Model", orientation="h",
                color="Mortality Probability", color_continuous_scale="RdYlGn_r",
                range_color=[0, 1],
            )
            fig_spread.update_layout(height=280, showlegend=False, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_spread, use_container_width=True)

        with col_shap_wf:
            st.markdown("#### SHAP Waterfall — This Patient")
            # Generate SHAP waterfall for custom input
            try:
                import shap
                # Use tree explainer for tree-based models
                tree_model_names = ["XGBoost", "LightGBM", "Random Forest", "Decision Tree (CART)"]
                if pred_model_name in tree_model_names:
                    explainer = shap.TreeExplainer(pred_model)
                    shap_values = explainer.shap_values(input_scaled_df)
                    # For binary classification, shap_values may be a list
                    if isinstance(shap_values, list):
                        sv = shap_values[1]  # class 1 (death)
                        base_val = explainer.expected_value[1]
                    else:
                        sv = shap_values
                        base_val = explainer.expected_value
                        if isinstance(base_val, np.ndarray):
                            base_val = base_val[1] if len(base_val) > 1 else base_val[0]

                    explanation = shap.Explanation(
                        values=sv[0],
                        base_values=base_val,
                        data=input_scaled_df.values[0],
                        feature_names=[FEATURE_LABELS.get(c, c) for c in feature_cols],
                    )

                    fig_wf, ax_wf = plt.subplots(figsize=(8, 6))
                    shap.plots.waterfall(explanation, max_display=12, show=False)
                    plt.title(f"SHAP Waterfall — {pred_model_name}", fontsize=12)
                    plt.tight_layout()
                    st.pyplot(fig_wf)
                    plt.close(fig_wf)
                else:
                    # For non-tree models, use KernelExplainer on a small background
                    st.info("SHAP waterfall is available for tree-based models (CART, RF, XGBoost, LightGBM). "
                            "Select a tree-based model for the waterfall plot.")

            except Exception as e:
                st.warning(f"Could not generate SHAP waterfall: {str(e)}")

            # Feature importance bar
            st.markdown("#### Feature Importance")
            if hasattr(pred_model, "feature_importances_"):
                importances = pred_model.feature_importances_
            elif hasattr(pred_model, "coef_"):
                importances = np.abs(pred_model.coef_[0])
            else:
                importances = np.zeros(len(feature_cols))

            imp_df = pd.DataFrame({
                "Feature": [FEATURE_LABELS.get(c, c) for c in feature_cols],
                "Importance": importances,
            }).sort_values("Importance", ascending=True)

            fig_imp = px.bar(
                imp_df, x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale="Reds",
                title=f"Feature Importance ({pred_model_name})",
            )
            fig_imp.update_layout(height=450, showlegend=False)
            st.plotly_chart(fig_imp, use_container_width=True)

        # Download buttons
        st.divider()
        dl_col1, dl_col2 = st.columns(2)
        pred_result = {
            "model": pred_model_name,
            "mortality_probability": float(death_prob),
            "risk_level": risk_level,
            "prediction": int(prediction),
            "input_features": {k: float(v) for k, v in input_df.iloc[0].to_dict().items()},
        }
        with dl_col1:
            st.download_button(
                "Download Prediction (JSON)",
                json.dumps(pred_result, indent=2),
                "prediction.json", "application/json",
                use_container_width=True,
            )
        with dl_col2:
            csv_out = pd.DataFrame([pred_result])
            st.download_button(
                "Download Prediction (CSV)",
                csv_out.to_csv(index=False),
                "prediction.csv", "text/csv",
                use_container_width=True,
            )


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()
st.caption(
    "MSIS 522 — Analytics and Machine Learning | University of Washington Foster School of Business | "
    "Prof. Léonard Boussioux | Built with Streamlit"
)
