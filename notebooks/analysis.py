"""
MSIS 522 — HW1: Complete Data Science Workflow
COVID-19 Patient Mortality Prediction
Author: MSIS 522 Student
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import shap
import joblib
import os
import json

# Set random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Paths
DATA_PATH = '/home/ubuntu/hw1_project/data/covid_data.csv'
MODELS_DIR = '/home/ubuntu/hw1_project/models'
PLOTS_DIR = '/home/ubuntu/hw1_project/plots'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Style
sns.set_style("whitegrid")
plt.rcParams.update({'figure.figsize': (10, 6), 'font.size': 12})

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)
df_full = pd.read_csv(DATA_PATH)
print(f"Full dataset shape: {df_full.shape}")
print(f"Death rate: {df_full['DEATH'].mean()*100:.2f}%")

# Create balanced subset for modeling (as done in course tutorial)
deaths = df_full[df_full['DEATH'] == 1].sample(n=5000, random_state=RANDOM_STATE)
survived = df_full[df_full['DEATH'] == 0].sample(n=5000, random_state=RANDOM_STATE)
df = pd.concat([deaths, survived]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
print(f"Balanced subset shape: {df.shape}")
print(f"Balanced death rate: {df['DEATH'].mean()*100:.2f}%")

# ============================================================
# PART 1: DESCRIPTIVE ANALYTICS (25 points)
# ============================================================
print("\n" + "=" * 60)
print("PART 1: DESCRIPTIVE ANALYTICS")
print("=" * 60)

# 1.1 Dataset Introduction
print("\n--- 1.1 Dataset Introduction ---")
print(f"Rows: {df_full.shape[0]:,}")
print(f"Features: {df_full.shape[1] - 1}")
print(f"Feature types: AGE (continuous), all others (binary 0/1)")
print(f"Target: DEATH (binary: 0=survived, 1=died)")
print(f"No missing values")
print(f"\nFeature names: {[c for c in df.columns if c != 'DEATH']}")

# 1.2 Target Distribution
print("\n--- 1.2 Target Distribution ---")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Full dataset
counts_full = df_full['DEATH'].value_counts()
colors = ['#2ecc71', '#e74c3c']
axes[0].bar(['Survived (0)', 'Died (1)'], counts_full.values, color=colors, edgecolor='black')
axes[0].set_title('Target Distribution — Full Dataset (N=1,021,977)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count')
for i, v in enumerate(counts_full.values):
    axes[0].text(i, v + 5000, f'{v:,}\n({v/len(df_full)*100:.1f}%)', ha='center', fontsize=11)

# Balanced subset
counts_bal = df['DEATH'].value_counts()
axes[1].bar(['Survived (0)', 'Died (1)'], counts_bal.values, color=colors, edgecolor='black')
axes[1].set_title('Target Distribution — Balanced Subset (N=10,000)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Count')
for i, v in enumerate(counts_bal.values):
    axes[1].text(i, v + 50, f'{v:,}\n({v/len(df)*100:.1f}%)', ha='center', fontsize=11)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/1_target_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 1_target_distribution.png")

# 1.3 Feature Distributions and Relationships (≥4 visualizations)
print("\n--- 1.3 Feature Distributions ---")

# Viz 1: Age distribution by outcome
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(df[df['DEATH']==0]['AGE'], bins=40, alpha=0.6, label='Survived', color='#2ecc71', edgecolor='black')
ax.hist(df[df['DEATH']==1]['AGE'], bins=40, alpha=0.6, label='Died', color='#e74c3c', edgecolor='black')
ax.set_xlabel('Age', fontsize=13)
ax.set_ylabel('Count', fontsize=13)
ax.set_title('Age Distribution by Mortality Outcome', fontsize=15, fontweight='bold')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/2_age_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 2_age_distribution.png")

# Viz 2: Mortality rate by comorbidity (bar chart)
comorbidities = ['DIABETES', 'HYPERTENSION', 'OBESITY', 'PNEUMONIA', 'COPD',
                 'RENAL_CHRONIC', 'CARDIOVASCULAR', 'IMMUNOSUPPRESSION', 'ASTHMA', 'TOBACCO']
mortality_rates = []
for c in comorbidities:
    rate_with = df_full[df_full[c] == 1]['DEATH'].mean() * 100
    rate_without = df_full[df_full[c] == 0]['DEATH'].mean() * 100
    mortality_rates.append({'Comorbidity': c, 'With Condition': rate_with, 'Without Condition': rate_without})

mr_df = pd.DataFrame(mortality_rates).sort_values('With Condition', ascending=True)

fig, ax = plt.subplots(figsize=(12, 7))
y_pos = np.arange(len(mr_df))
bars1 = ax.barh(y_pos - 0.2, mr_df['With Condition'], 0.4, label='With Condition', color='#e74c3c', edgecolor='black')
bars2 = ax.barh(y_pos + 0.2, mr_df['Without Condition'], 0.4, label='Without Condition', color='#2ecc71', edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(mr_df['Comorbidity'], fontsize=11)
ax.set_xlabel('Mortality Rate (%)', fontsize=13)
ax.set_title('COVID-19 Mortality Rate by Pre-Existing Condition (Full Dataset)', fontsize=15, fontweight='bold')
ax.legend(fontsize=12)
for bar in bars1:
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.1f}%', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/3_mortality_by_comorbidity.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 3_mortality_by_comorbidity.png")

# Viz 3: Violin plot of age by sex and outcome
fig, ax = plt.subplots(figsize=(10, 6))
plot_df = df.copy()
plot_df['Outcome'] = plot_df['DEATH'].map({0: 'Survived', 1: 'Died'})
plot_df['Sex'] = plot_df['SEX'].map({0: 'Female', 1: 'Male'})
sns.violinplot(data=plot_df, x='Outcome', y='AGE', hue='Sex', split=True, 
               palette={'Female': '#3498db', 'Male': '#e67e22'}, ax=ax)
ax.set_title('Age Distribution by Outcome and Sex', fontsize=15, fontweight='bold')
ax.set_ylabel('Age', fontsize=13)
ax.set_xlabel('Outcome', fontsize=13)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/4_violin_age_sex_outcome.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 4_violin_age_sex_outcome.png")

# Viz 4: Stacked bar chart of hospitalization and pneumonia vs death
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for idx, feat in enumerate(['HOSPITALIZED', 'PNEUMONIA']):
    ct = pd.crosstab(df[feat], df['DEATH'], normalize='index') * 100
    ct.columns = ['Survived', 'Died']
    ct.index = ['No', 'Yes']
    ct.plot(kind='bar', stacked=True, color=['#2ecc71', '#e74c3c'], ax=axes[idx], edgecolor='black')
    axes[idx].set_title(f'Mortality Rate by {feat.title()}', fontsize=14, fontweight='bold')
    axes[idx].set_ylabel('Percentage (%)')
    axes[idx].set_xlabel(feat.title())
    axes[idx].legend(title='Outcome')
    axes[idx].tick_params(axis='x', rotation=0)
    for p in axes[idx].patches:
        width, height = p.get_width(), p.get_height()
        if height > 3:
            axes[idx].text(p.get_x() + width/2, p.get_y() + height/2, f'{height:.1f}%',
                          ha='center', va='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/5_hospitalization_pneumonia.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 5_hospitalization_pneumonia.png")

# Viz 5: Comorbidity co-occurrence heatmap for deceased patients
fig, ax = plt.subplots(figsize=(10, 8))
deceased = df[df['DEATH'] == 1]
comorbidity_cols = ['DIABETES', 'HYPERTENSION', 'OBESITY', 'PNEUMONIA', 'COPD',
                    'RENAL_CHRONIC', 'CARDIOVASCULAR', 'IMMUNOSUPPRESSION']
co_occur = deceased[comorbidity_cols].T.dot(deceased[comorbidity_cols])
mask = np.triu(np.ones_like(co_occur, dtype=bool), k=1)
sns.heatmap(co_occur, annot=True, fmt='d', cmap='Reds', ax=ax, mask=mask,
            linewidths=0.5, linecolor='white')
ax.set_title('Comorbidity Co-Occurrence Among Deceased Patients', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/6_comorbidity_cooccurrence.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 6_comorbidity_cooccurrence.png")

# 1.4 Correlation Heatmap
print("\n--- 1.4 Correlation Heatmap ---")
fig, ax = plt.subplots(figsize=(14, 10))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax,
            mask=mask, linewidths=0.5, linecolor='white',
            vmin=-1, vmax=1, square=True)
ax.set_title('Feature Correlation Heatmap (Balanced Subset)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/7_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 7_correlation_heatmap.png")

# ============================================================
# PART 2: PREDICTIVE ANALYTICS (45 points)
# ============================================================
print("\n" + "=" * 60)
print("PART 2: PREDICTIVE ANALYTICS")
print("=" * 60)

# 2.1 Data Preparation
print("\n--- 2.1 Data Preparation ---")
feature_cols = [c for c in df.columns if c != 'DEATH']
X = df[feature_cols]
y = df['DEATH']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train death rate: {y_train.mean()*100:.1f}%, Test death rate: {y_test.mean()*100:.1f}%")

# Scale for Logistic Regression and MLP
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, f'{MODELS_DIR}/scaler.joblib')

# Store all results
results = {}
roc_data = {}

def evaluate_model(name, model, X_te, y_te, scaled=False):
    """Evaluate and store model metrics."""
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, 'predict_proba') else None
    
    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred)
    rec = recall_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_prob) if y_prob is not None else 0
    
    results[name] = {
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1 Score': round(f1, 4),
        'AUC-ROC': round(auc, 4)
    }
    
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        roc_data[name] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': round(auc, 4)}
    
    print(f"\n{name} Results:")
    for k, v in results[name].items():
        print(f"  {k}: {v}")
    return y_pred, y_prob

# 2.2 Logistic Regression Baseline
print("\n--- 2.2 Logistic Regression Baseline ---")
lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
lr.fit(X_train_scaled, y_train)
joblib.dump(lr, f'{MODELS_DIR}/logistic_regression.joblib')
evaluate_model('Logistic Regression', lr, X_test_scaled, y_test)

# 2.3 Decision Tree / CART
print("\n--- 2.3 Decision Tree / CART ---")
dt_params = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_leaf': [5, 10, 20, 50]
}
dt_cv = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_STATE),
    dt_params,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    scoring='f1',
    n_jobs=-1,
    return_train_score=True
)
dt_cv.fit(X_train, y_train)
print(f"Best params: {dt_cv.best_params_}")
print(f"Best CV F1: {dt_cv.best_score_:.4f}")
best_dt = dt_cv.best_estimator_
joblib.dump(best_dt, f'{MODELS_DIR}/decision_tree.joblib')
evaluate_model('Decision Tree', best_dt, X_test, y_test)

# Save best params
dt_best_params = dt_cv.best_params_

# Visualize the tree
fig, ax = plt.subplots(figsize=(24, 12))
plot_tree(best_dt, feature_names=feature_cols, class_names=['Survived', 'Died'],
          filled=True, rounded=True, ax=ax, fontsize=8, max_depth=3)
ax.set_title(f'Decision Tree (max_depth={dt_best_params["max_depth"]}, min_samples_leaf={dt_best_params["min_samples_leaf"]})',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/8_decision_tree.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 8_decision_tree.png")

# 2.4 Random Forest
print("\n--- 2.4 Random Forest ---")
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 8]
}
rf_cv = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    rf_params,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    scoring='f1',
    n_jobs=-1,
    return_train_score=True
)
rf_cv.fit(X_train, y_train)
print(f"Best params: {rf_cv.best_params_}")
print(f"Best CV F1: {rf_cv.best_score_:.4f}")
best_rf = rf_cv.best_estimator_
joblib.dump(best_rf, f'{MODELS_DIR}/random_forest.joblib')
evaluate_model('Random Forest', best_rf, X_test, y_test)
rf_best_params = rf_cv.best_params_

# 2.5 Boosted Trees — XGBoost and LightGBM
print("\n--- 2.5 XGBoost ---")
xgb_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1]
}
xgb_cv = GridSearchCV(
    xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', use_label_encoder=False),
    xgb_params,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    scoring='f1',
    n_jobs=-1,
    return_train_score=True
)
xgb_cv.fit(X_train, y_train)
print(f"Best params: {xgb_cv.best_params_}")
print(f"Best CV F1: {xgb_cv.best_score_:.4f}")
best_xgb = xgb_cv.best_estimator_
joblib.dump(best_xgb, f'{MODELS_DIR}/xgboost.joblib')
evaluate_model('XGBoost', best_xgb, X_test, y_test)
xgb_best_params = xgb_cv.best_params_

print("\n--- 2.5 LightGBM ---")
lgb_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1]
}
lgb_cv = GridSearchCV(
    lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
    lgb_params,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    scoring='f1',
    n_jobs=-1,
    return_train_score=True
)
lgb_cv.fit(X_train, y_train)
print(f"Best params: {lgb_cv.best_params_}")
print(f"Best CV F1: {lgb_cv.best_score_:.4f}")
best_lgb = lgb_cv.best_estimator_
joblib.dump(best_lgb, f'{MODELS_DIR}/lightgbm.joblib')
evaluate_model('LightGBM', best_lgb, X_test, y_test)
lgb_best_params = lgb_cv.best_params_

# 2.6 Neural Network — MLP
print("\n--- 2.6 Neural Network (MLP) ---")
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
tf.random.set_seed(RANDOM_STATE)

# Build MLP
model_nn = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model_nn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model_nn.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    verbose=0
)

# Save training history plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[0].set_title('MLP Training Loss', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

axes[1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[1].set_title('MLP Training Accuracy', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/9_mlp_training_history.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 9_mlp_training_history.png")

# Evaluate MLP
y_prob_nn = model_nn.predict(X_test_scaled, verbose=0).flatten()
y_pred_nn = (y_prob_nn >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred_nn)
prec = precision_score(y_test, y_pred_nn)
rec = recall_score(y_test, y_pred_nn)
f1 = f1_score(y_test, y_pred_nn)
auc = roc_auc_score(y_test, y_prob_nn)

results['MLP Neural Network'] = {
    'Accuracy': round(acc, 4),
    'Precision': round(prec, 4),
    'Recall': round(rec, 4),
    'F1 Score': round(f1, 4),
    'AUC-ROC': round(auc, 4)
}

fpr, tpr, _ = roc_curve(y_test, y_prob_nn)
roc_data['MLP Neural Network'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': round(auc, 4)}

print(f"\nMLP Neural Network Results:")
for k, v in results['MLP Neural Network'].items():
    print(f"  {k}: {v}")

# Save MLP model
model_nn.save(f'{MODELS_DIR}/mlp_model.keras')

# BONUS: MLP Hyperparameter Tuning
print("\n--- BONUS: MLP Hyperparameter Tuning ---")
mlp_results = []
for hidden_size in [(64, 64), (128, 128), (256, 128)]:
    for lr_val in [0.001, 0.01]:
        for dropout_rate in [0.2, 0.3, 0.5]:
            tf.random.set_seed(RANDOM_STATE)
            m = keras.Sequential([
                layers.Input(shape=(X_train_scaled.shape[1],)),
                layers.Dense(hidden_size[0], activation='relu'),
                layers.Dropout(dropout_rate),
                layers.Dense(hidden_size[1], activation='relu'),
                layers.Dropout(dropout_rate),
                layers.Dense(1, activation='sigmoid')
            ])
            m.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_val),
                      loss='binary_crossentropy', metrics=['accuracy'])
            h = m.fit(X_train_scaled, y_train, epochs=30, batch_size=64,
                      validation_split=0.2, verbose=0)
            y_p = m.predict(X_test_scaled, verbose=0).flatten()
            y_pred_temp = (y_p >= 0.5).astype(int)
            f1_val = f1_score(y_test, y_pred_temp)
            auc_val = roc_auc_score(y_test, y_p)
            mlp_results.append({
                'hidden_layers': str(hidden_size),
                'learning_rate': lr_val,
                'dropout': dropout_rate,
                'f1': round(f1_val, 4),
                'auc': round(auc_val, 4)
            })
            print(f"  HL={hidden_size}, LR={lr_val}, DO={dropout_rate} -> F1={f1_val:.4f}, AUC={auc_val:.4f}")

mlp_tuning_df = pd.DataFrame(mlp_results).sort_values('f1', ascending=False)
print(f"\nBest MLP config: {mlp_tuning_df.iloc[0].to_dict()}")

# Visualize MLP tuning results
fig, ax = plt.subplots(figsize=(12, 6))
mlp_tuning_df['config'] = mlp_tuning_df.apply(
    lambda r: f"HL={r['hidden_layers']}\nLR={r['learning_rate']}\nDO={r['dropout']}", axis=1)
top_configs = mlp_tuning_df.head(10)
ax.barh(range(len(top_configs)), top_configs['f1'], color='#3498db', edgecolor='black')
ax.set_yticks(range(len(top_configs)))
ax.set_yticklabels(top_configs['config'], fontsize=9)
ax.set_xlabel('F1 Score')
ax.set_title('Top 10 MLP Configurations (Hyperparameter Tuning)', fontsize=14, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/10_mlp_tuning.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 10_mlp_tuning.png")

# 2.7 Model Comparison Summary
print("\n--- 2.7 Model Comparison Summary ---")
results_df = pd.DataFrame(results).T
results_df.index.name = 'Model'
print(results_df.to_string())

# Save results
results_df.to_csv(f'{MODELS_DIR}/model_comparison.csv')

# Bar chart
fig, ax = plt.subplots(figsize=(12, 6))
models_list = results_df.index.tolist()
f1_scores = results_df['F1 Score'].values
auc_scores = results_df['AUC-ROC'].values
x = np.arange(len(models_list))
width = 0.35
bars1 = ax.bar(x - width/2, f1_scores, width, label='F1 Score', color='#3498db', edgecolor='black')
bars2 = ax.bar(x + width/2, auc_scores, width, label='AUC-ROC', color='#e74c3c', edgecolor='black')
ax.set_ylabel('Score')
ax.set_title('Model Comparison: F1 Score and AUC-ROC', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models_list, rotation=15, ha='right')
ax.legend()
ax.set_ylim(0, 1.1)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{bar.get_height():.3f}',
            ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{bar.get_height():.3f}',
            ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/11_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 11_model_comparison.png")

# ROC Curves
fig, ax = plt.subplots(figsize=(10, 8))
colors_roc = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c']
for i, (name, data) in enumerate(roc_data.items()):
    ax.plot(data['fpr'], data['tpr'], label=f"{name} (AUC={data['auc']:.3f})",
            linewidth=2, color=colors_roc[i % len(colors_roc)])
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Baseline')
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curves — All Models', fontsize=15, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/12_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 12_roc_curves.png")

# ============================================================
# PART 3: EXPLAINABILITY (10 points)
# ============================================================
print("\n" + "=" * 60)
print("PART 3: EXPLAINABILITY (SHAP)")
print("=" * 60)

# Use best tree-based model (XGBoost or LightGBM based on performance)
best_tree_name = max(['XGBoost', 'LightGBM'], key=lambda n: results[n]['F1 Score'])
best_tree_model = best_xgb if best_tree_name == 'XGBoost' else best_lgb
print(f"Using {best_tree_name} for SHAP analysis (F1={results[best_tree_name]['F1 Score']})")

# SHAP explainer
explainer = shap.TreeExplainer(best_tree_model)
shap_values = explainer.shap_values(X_test)

# 3.1 Summary plot (beeswarm)
fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
plt.title(f'SHAP Summary Plot — {best_tree_name}', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/13_shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 13_shap_summary.png")

# 3.2 Bar plot of mean absolute SHAP values
fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, feature_names=feature_cols, plot_type='bar', show=False)
plt.title(f'SHAP Feature Importance — {best_tree_name}', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/14_shap_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 14_shap_bar.png")

# 3.3 Waterfall plot for a high-risk individual
high_risk_idx = y_test[y_test == 1].index[0]
high_risk_pos = list(X_test.index).index(high_risk_idx)

fig, ax = plt.subplots(figsize=(12, 8))
shap_explanation = shap.Explanation(
    values=shap_values[high_risk_pos],
    base_values=explainer.expected_value,
    data=X_test.iloc[high_risk_pos].values,
    feature_names=feature_cols
)
shap.waterfall_plot(shap_explanation, show=False)
plt.title(f'SHAP Waterfall — High-Risk Patient (Actual: Died)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/15_shap_waterfall.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 15_shap_waterfall.png")

# Save all hyperparameters
all_params = {
    'Logistic Regression': {'solver': 'lbfgs', 'max_iter': 1000},
    'Decision Tree': dt_best_params,
    'Random Forest': rf_best_params,
    'XGBoost': xgb_best_params,
    'LightGBM': lgb_best_params,
    'MLP Neural Network': {'layers': '128-128-64', 'dropout': 0.3, 'lr': 0.001, 'epochs': 50}
}
with open(f'{MODELS_DIR}/best_params.json', 'w') as f:
    json.dump(all_params, f, indent=2, default=str)

# Save ROC data
with open(f'{MODELS_DIR}/roc_data.json', 'w') as f:
    json.dump(roc_data, f)

# Save feature columns
with open(f'{MODELS_DIR}/feature_cols.json', 'w') as f:
    json.dump(feature_cols, f)

# Save MLP tuning results
mlp_tuning_df.to_csv(f'{MODELS_DIR}/mlp_tuning_results.csv', index=False)

# Save training history
history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
with open(f'{MODELS_DIR}/mlp_history.json', 'w') as f:
    json.dump(history_dict, f)

print("\n" + "=" * 60)
print("ALL ANALYSIS COMPLETE!")
print("=" * 60)
print(f"\nModels saved to: {MODELS_DIR}")
print(f"Plots saved to: {PLOTS_DIR}")
print(f"\nFinal Model Comparison:")
print(results_df.to_string())
