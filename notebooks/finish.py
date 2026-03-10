"""Fast finish: plots, SHAP, and all metadata. Uses sklearn MLP (no TF overhead)."""
import pandas as pd, numpy as np, json, os, joblib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
import warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
import shap

RANDOM_STATE = 42; np.random.seed(RANDOM_STATE)
MODELS_DIR = '/home/ubuntu/hw1_project/models'
PLOTS_DIR = '/home/ubuntu/hw1_project/plots'

# --- Reload data ---
df_full = pd.read_csv('/home/ubuntu/hw1_project/data/covid_data.csv')
deaths = df_full[df_full['DEATH']==1].sample(n=5000, random_state=RANDOM_STATE)
survived = df_full[df_full['DEATH']==0].sample(n=5000, random_state=RANDOM_STATE)
df = pd.concat([deaths, survived]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
feature_cols = [c for c in df.columns if c != 'DEATH']
X = df[feature_cols]; y = df['DEATH']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
scaler = joblib.load(f'{MODELS_DIR}/scaler.joblib')
X_train_scaled = scaler.transform(X_train); X_test_scaled = scaler.transform(X_test)

# --- Load saved models ---
lr = joblib.load(f'{MODELS_DIR}/logistic_regression.joblib')
best_dt = joblib.load(f'{MODELS_DIR}/decision_tree.joblib')
best_rf = joblib.load(f'{MODELS_DIR}/random_forest.joblib')
best_xgb = joblib.load(f'{MODELS_DIR}/xgboost.joblib')
best_lgb = joblib.load(f'{MODELS_DIR}/lightgbm.joblib')

results = {}; roc_data = {}
def ev(name, mdl, Xt, yt):
    yp = mdl.predict(Xt); ypr = mdl.predict_proba(Xt)[:,1]
    r = {'Accuracy':round(accuracy_score(yt,yp),4), 'Precision':round(precision_score(yt,yp),4),
         'Recall':round(recall_score(yt,yp),4), 'F1 Score':round(f1_score(yt,yp),4),
         'AUC-ROC':round(roc_auc_score(yt,ypr),4)}
    results[name] = r
    fpr,tpr,_ = roc_curve(yt,ypr)
    roc_data[name] = {'fpr':fpr.tolist(),'tpr':tpr.tolist(),'auc':r['AUC-ROC']}
    print(f"{name}: F1={r['F1 Score']}, AUC={r['AUC-ROC']}")

ev('Logistic Regression', lr, X_test_scaled, y_test)
ev('Decision Tree', best_dt, X_test, y_test)
ev('Random Forest', best_rf, X_test, y_test)
ev('XGBoost', best_xgb, X_test, y_test)
ev('LightGBM', best_lgb, X_test, y_test)

# --- sklearn MLP ---
print("\n--- MLP (sklearn) ---")
mlp = MLPClassifier(hidden_layer_sizes=(128,128,64), activation='relu', solver='adam',
                    alpha=0.001, batch_size=64, learning_rate_init=0.001,
                    max_iter=200, random_state=RANDOM_STATE, early_stopping=True,
                    validation_fraction=0.2)
mlp.fit(X_train_scaled, y_train)
joblib.dump(mlp, f'{MODELS_DIR}/mlp_sklearn.joblib')
ev('MLP Neural Network', mlp, X_test_scaled, y_test)

# MLP loss curve
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(mlp.loss_curve_, label='Train Loss', linewidth=2)
ax.plot(mlp.validation_scores_, label='Val Accuracy', linewidth=2)
ax.set_xlabel('Iteration'); ax.set_ylabel('Score')
ax.set_title('MLP Training Curves', fontsize=14, fontweight='bold'); ax.legend()
plt.tight_layout(); plt.savefig(f'{PLOTS_DIR}/9_mlp_training_history.png', dpi=150, bbox_inches='tight'); plt.close()
print("Saved: 9_mlp_training_history.png")

# BONUS: MLP tuning
print("\n--- BONUS: MLP Tuning ---")
mlp_results = []
for hl in [(64,64),(128,128),(256,128,64)]:
    for lr_val in [0.001, 0.01]:
        for alpha in [0.0001, 0.001]:
            m = MLPClassifier(hidden_layer_sizes=hl, activation='relu', solver='adam',
                              alpha=alpha, learning_rate_init=lr_val, max_iter=150,
                              random_state=RANDOM_STATE, early_stopping=True, validation_fraction=0.2)
            m.fit(X_train_scaled, y_train)
            yp = m.predict(X_test_scaled); ypr = m.predict_proba(X_test_scaled)[:,1]
            f1v = f1_score(y_test, yp); aucv = roc_auc_score(y_test, ypr)
            mlp_results.append({'hidden_layers':str(hl),'lr':lr_val,'alpha':alpha,'f1':round(f1v,4),'auc':round(aucv,4)})
            print(f"  HL={hl}, LR={lr_val}, alpha={alpha} -> F1={f1v:.4f}")

mlp_df = pd.DataFrame(mlp_results).sort_values('f1', ascending=False)
mlp_df.to_csv(f'{MODELS_DIR}/mlp_tuning_results.csv', index=False)
fig, ax = plt.subplots(figsize=(12,6))
labels = mlp_df.apply(lambda r: f"HL={r['hidden_layers']}, LR={r['lr']}, a={r['alpha']}", axis=1)
ax.barh(range(len(labels)), mlp_df['f1'], color='#3498db', edgecolor='black')
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('F1 Score'); ax.set_title('MLP Hyperparameter Tuning', fontsize=14, fontweight='bold')
ax.invert_yaxis(); plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/10_mlp_tuning.png', dpi=150, bbox_inches='tight'); plt.close()

# --- Model Comparison Plots ---
print("\n--- Model Comparison ---")
results_df = pd.DataFrame(results).T; results_df.index.name = 'Model'
results_df.to_csv(f'{MODELS_DIR}/model_comparison.csv')
print(results_df.to_string())

fig, ax = plt.subplots(figsize=(12,6))
x = np.arange(len(results_df)); w = 0.35
b1 = ax.bar(x-w/2, results_df['F1 Score'], w, label='F1 Score', color='#3498db', edgecolor='black')
b2 = ax.bar(x+w/2, results_df['AUC-ROC'], w, label='AUC-ROC', color='#e74c3c', edgecolor='black')
ax.set_xticks(x); ax.set_xticklabels(results_df.index, rotation=15, ha='right')
ax.set_ylabel('Score'); ax.set_title('Model Comparison: F1 Score and AUC-ROC', fontsize=15, fontweight='bold')
ax.legend(); ax.set_ylim(0,1.1)
for bar in list(b1)+list(b2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{bar.get_height():.3f}', ha='center', fontsize=9)
plt.tight_layout(); plt.savefig(f'{PLOTS_DIR}/11_model_comparison.png', dpi=150, bbox_inches='tight'); plt.close()

fig, ax = plt.subplots(figsize=(10,8))
cols = ['#3498db','#2ecc71','#e74c3c','#9b59b6','#f39c12','#1abc9c']
for i,(name,d) in enumerate(roc_data.items()):
    ax.plot(d['fpr'],d['tpr'], label=f"{name} (AUC={d['auc']:.3f})", linewidth=2, color=cols[i])
ax.plot([0,1],[0,1],'k--',linewidth=1,label='Random')
ax.set_xlabel('FPR',fontsize=13); ax.set_ylabel('TPR',fontsize=13)
ax.set_title('ROC Curves — All Models', fontsize=15, fontweight='bold'); ax.legend(fontsize=11)
plt.tight_layout(); plt.savefig(f'{PLOTS_DIR}/12_roc_curves.png', dpi=150, bbox_inches='tight'); plt.close()

# --- SHAP ---
print("\n--- SHAP Analysis ---")
best_name = max(['XGBoost','LightGBM'], key=lambda n: results[n]['AUC-ROC'])
best_model = best_xgb if best_name=='XGBoost' else best_lgb
print(f"Using {best_name} for SHAP")

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(12,8))
shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
plt.title(f'SHAP Summary (Beeswarm) — {best_name}', fontsize=15, fontweight='bold')
plt.tight_layout(); plt.savefig(f'{PLOTS_DIR}/13_shap_summary.png', dpi=150, bbox_inches='tight'); plt.close()

plt.figure(figsize=(12,8))
shap.summary_plot(shap_values, X_test, feature_names=feature_cols, plot_type='bar', show=False)
plt.title(f'SHAP Feature Importance — {best_name}', fontsize=15, fontweight='bold')
plt.tight_layout(); plt.savefig(f'{PLOTS_DIR}/14_shap_bar.png', dpi=150, bbox_inches='tight'); plt.close()

# Waterfall for a deceased patient
hi = y_test[y_test==1].index[0]
pos = list(X_test.index).index(hi)
exp = shap.Explanation(values=shap_values[pos], base_values=explainer.expected_value,
                       data=X_test.iloc[pos].values, feature_names=feature_cols)
plt.figure(figsize=(12,8))
shap.waterfall_plot(exp, show=False)
plt.title('SHAP Waterfall — High-Risk Patient', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig(f'{PLOTS_DIR}/15_shap_waterfall.png', dpi=150, bbox_inches='tight'); plt.close()

# Waterfall for a survived patient
lo = y_test[y_test==0].index[0]
pos2 = list(X_test.index).index(lo)
exp2 = shap.Explanation(values=shap_values[pos2], base_values=explainer.expected_value,
                        data=X_test.iloc[pos2].values, feature_names=feature_cols)
plt.figure(figsize=(12,8))
shap.waterfall_plot(exp2, show=False)
plt.title('SHAP Waterfall — Low-Risk Patient (Survived)', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig(f'{PLOTS_DIR}/16_shap_waterfall_survived.png', dpi=150, bbox_inches='tight'); plt.close()

# --- Save metadata ---
dt_p = {k: best_dt.get_params()[k] for k in ['max_depth','min_samples_leaf']}
rf_p = {k: best_rf.get_params()[k] for k in ['max_depth','n_estimators']}
xgb_p = {k: (int(v) if isinstance(v,(np.integer,)) else v) for k,v in best_xgb.get_params().items() if k in ['max_depth','n_estimators','learning_rate']}
lgb_p = {k: (int(v) if isinstance(v,(np.integer,)) else v) for k,v in best_lgb.get_params().items() if k in ['max_depth','n_estimators','learning_rate']}

all_params = {'Logistic Regression':{'solver':'lbfgs','max_iter':1000},
              'Decision Tree':dt_p, 'Random Forest':rf_p, 'XGBoost':xgb_p, 'LightGBM':lgb_p,
              'MLP Neural Network':{'layers':'128-128-64','alpha':0.001,'lr':0.001}}
with open(f'{MODELS_DIR}/best_params.json','w') as f: json.dump(all_params,f,indent=2)
with open(f'{MODELS_DIR}/roc_data.json','w') as f: json.dump(roc_data,f)
with open(f'{MODELS_DIR}/feature_cols.json','w') as f: json.dump(feature_cols,f)

print("\n=== ALL DONE ===")
print(results_df.to_string())
