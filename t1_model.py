# ============================================================
# TIER 1 AVC SUSCEPTIBILITY MODEL COMPARISON
# ============================================================

# MODELS:
# 1. Logistic Regression
# 2. Random Forest
# 3. XGBoost
# 4. LightGBM
# 5. Neural Network (MLP)
# 6. Support Vector Machine (SVM)

# ============================================================
# 1. IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import joblib


import shap

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures

# MODELS
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv("tier1_with_kde.csv")

# ============================================================
# 3. TARGET VARIABLE
# ============================================================

# collision column:
# 1 = AVC
# 0 = absence point

target = "target"

# ============================================================
# 4. SELECT FEATURES
# ============================================================

features = [
    # ENVIRONMENTAL
    "NDVI_Mean",
    "monthly_drought_index (0-100)",
    "Monthly_Temperature_Average",
    "KDE_hotspot",

    # CYCLICAL TIME (Already pre-encoded in your data)
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "hour_sin",
    "hour_cos"
]

X = df[features]
y = df[target]

# ============================================================
# 5. TRAIN TEST SPLIT
# ============================================================

# Stratify=y keeps class ratios identical across training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ============================================================
# 6. SCALE DATA
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 7. SAMPLE WEIGHTS FOR MLP
# ============================================================

# MLPClassifier doesn't support class_weight, so we compute
# equivalent sample weights to pass directly into fit().
# This replicates the 'balanced' behaviour of other models.
sample_weights_train = compute_sample_weight(class_weight="balanced", y=y_train)

# ============================================================
# 8. DEFINE MODELS
# ============================================================

# Calculate imbalance ratio for XGBoost's scale_pos_weight
negative_cases = np.sum(y_train == 0)
positive_cases = np.sum(y_train == 1)
imbalance_ratio = negative_cases / positive_cases

models = {

    # 1. LOGISTIC REGRESSION
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    # 2. RANDOM FOREST
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    # 3. XGBOOST
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=imbalance_ratio,
        eval_metric="logloss",
        random_state=42
    ),

    # 4. LIGHTGBM
    "LightGBM": LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
        verbosity=-1
    ),

    # 5. NEURAL NETWORK (MLP)
    # Note: MLPClassifier has no class_weight param.
    # Class imbalance is handled via sample_weights passed to fit() below.
    "Neural Network": MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=64,
        learning_rate='adaptive',
        max_iter=500,
        random_state=42
    ),

    # 6. SUPPORT VECTOR MACHINE
    "SVM": SVC(
        kernel='rbf',
        probability=True,
        class_weight='balanced',
        random_state=42
    )
}

# ============================================================
# 9. TRAIN + EVALUATE (HOLD-OUT SET)
# ============================================================

results = []
trained_models = {}          # Store fitted models for PR curve plotting
test_probabilities = {}      # Store predicted probabilities for PR curves

for name, model in models.items():

    print("\n================================================")
    print(f"TRAINING: {name}")
    print("================================================")

    uses_scaled = name in ["Logistic Regression", "Neural Network", "SVM"]

    X_tr = X_train_scaled if uses_scaled else X_train
    X_te = X_test_scaled  if uses_scaled else X_test

    # FIT — pass sample weights for MLP to handle class imbalance
    if name == "Neural Network":
        model.fit(X_tr, y_train, sample_weight=sample_weights_train)
    else:
        model.fit(X_tr, y_train)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    # STORE FOR PR CURVES
    trained_models[name] = model
    test_probabilities[name] = y_prob

    # METRICS
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_prob)
    pr_auc    = average_precision_score(y_test, y_prob)
    cm        = confusion_matrix(y_test, y_pred)

    results.append({
        "Model":     name,
        "Accuracy":  accuracy,
        "Precision": precision,
        "Recall":    recall,
        "F1 Score":  f1,
        "ROC-AUC":   roc_auc,
        "PR-AUC":    pr_auc
    })

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

# ============================================================
# 10. 19-FOLD CROSS-VALIDATION (top 3 models by ROC-AUC)
# ============================================================

# Sort results to find the top 3 before running CV
results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
top_3_names = results_df["Model"].iloc[:3].tolist()

print("\n================================================")
print("5-FOLD CROSS-VALIDATION (top 3 models)")
print("================================================")

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_results = []

for name in top_3_names:
    model = models[name]
    uses_scaled = name in ["Logistic Regression", "Neural Network", "SVM"]
    X_cv = X_train_scaled if uses_scaled else np.array(X_train)

    scoring = {
        "roc_auc":          "roc_auc",
        "average_precision": "average_precision",
        "f1":               "f1",
        "recall":           "recall",
        "precision":        "precision"
    }

    # Note: sample_weight in CV for MLP is not straightforward with cross_validate;
    # the hold-out evaluation above already captures the imbalance effect.
    scores = cross_validate(model, X_cv, y_train, cv=cv, scoring=scoring, n_jobs=-1)

    cv_results.append({
        "Model":          name,
        "CV ROC-AUC":     f"{scores['test_roc_auc'].mean():.4f} ± {scores['test_roc_auc'].std():.4f}",
        "CV PR-AUC":      f"{scores['test_average_precision'].mean():.4f} ± {scores['test_average_precision'].std():.4f}",
        "CV F1":          f"{scores['test_f1'].mean():.4f} ± {scores['test_f1'].std():.4f}",
        "CV Recall":      f"{scores['test_recall'].mean():.4f} ± {scores['test_recall'].std():.4f}",
        "CV Precision":   f"{scores['test_precision'].mean():.4f} ± {scores['test_precision'].std():.4f}",
    })

    print(f"\n{name}")
    print(f"  ROC-AUC  : {scores['test_roc_auc'].mean():.4f} ± {scores['test_roc_auc'].std():.4f}")
    print(f"  PR-AUC   : {scores['test_average_precision'].mean():.4f} ± {scores['test_average_precision'].std():.4f}")
    print(f"  F1       : {scores['test_f1'].mean():.4f} ± {scores['test_f1'].std():.4f}")
    print(f"  Recall   : {scores['test_recall'].mean():.4f} ± {scores['test_recall'].std():.4f}")
    print(f"  Precision: {scores['test_precision'].mean():.4f} ± {scores['test_precision'].std():.4f}")

cv_df = pd.DataFrame(cv_results)

# ============================================================
# 11. PRECISION-RECALL CURVES
# ============================================================

print("\nGenerating precision-recall curves...")

fig, ax = plt.subplots(figsize=(8, 6))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for (name, y_prob), color in zip(test_probabilities.items(), colors):
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    pr_auc_val = average_precision_score(y_test, y_prob)
    ax.plot(rec, prec, label=f"{name} (PR-AUC={pr_auc_val:.3f})", color=color, linewidth=1.8)

# Baseline: a no-skill classifier at the positive class prevalence
baseline = positive_cases / (positive_cases + negative_cases)
ax.axhline(y=baseline, color='gray', linestyle='--', linewidth=1,
           label=f"No-skill baseline ({baseline:.3f})")

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves — Tier 1 Models")
ax.legend(loc="upper right", fontsize=8)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("tier1_pr_curves.png", dpi=150)
plt.close()
print("Saved: tier1_pr_curves.png")

# ============================================================
# 12. FINAL RESULTS TABLES
# ============================================================

print("\n================================================")
print("FINAL MODEL COMPARISON (hold-out set)")
print("================================================")
print(results_df.to_string(index=False))

print("\n================================================")
print("CROSS-VALIDATION SUMMARY (top 3 models)")
print("================================================")
print(cv_df.to_string(index=False))

# ============================================================
# 13. SAVE RESULTS
# ============================================================

results_df.to_csv("tier1_model_comparison_results.csv", index=False)
cv_df.to_csv("tier1_cv_results.csv", index=False)
print("\nResults saved: tier1_model_comparison_results.csv, tier1_cv_results.csv")

print(f"Average monthly drought index: {df['monthly_drought_index (0-100)'].mean():.2f}")
print(f"Average monthly temperature: {df['Monthly_Temperature_Average'].mean():.2f}")
print(f"Average KDE hotspot: {df['KDE_hotspot'].mean():.2f}")

joblib.dump(models["LightGBM"], "tier1_lgbm_model.pkl")

# ============================================================
# SHAP ANALYSIS — LightGBM (best tier 1 model)
# ============================================================

print("\nCalculating SHAP values (this may take a minute)...")

explainer   = shap.TreeExplainer(trained_models["LightGBM"])
shap_values = explainer.shap_values(X_test)

# Handle 3D array (n_samples, n_features, n_classes) — take class 1
if isinstance(shap_values, list):
    shap_vals = shap_values[1]
elif np.array(shap_values).ndim == 3:
    shap_vals = np.array(shap_values)[:, :, 1]
else:
    shap_vals = np.array(shap_values)

print(f"shap_vals shape: {shap_vals.shape}")  # should be (n_samples, 10)

# ---- PLOT 1: Summary (beeswarm) ----
plt.figure()
shap.summary_plot(shap_vals, X_test, feature_names=features, show=False)
plt.tight_layout()
plt.savefig("tier1_shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: tier1_shap_summary.png")

# ---- PLOT 2: Bar chart (mean absolute SHAP) ----
plt.figure()
shap.summary_plot(shap_vals, X_test, feature_names=features,
                  plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("tier1_shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: tier1_shap_bar.png")

# ---- PLOT 3: Dependence plots for key environmental features ----
env_features = ["NDVI_Mean", "monthly_drought_index (0-100)",
                "Monthly_Temperature_Average", "KDE_hotspot"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

X_test_arr = np.array(X_test)

for i, feat in enumerate(env_features):
    feat_idx       = features.index(feat)
    feat_vals      = X_test_arr[:, feat_idx]
    shap_vals_feat = shap_vals[:, feat_idx]

    axes[i].scatter(feat_vals, shap_vals_feat,
                    alpha=0.4, s=10, color="#1D9E75")
    axes[i].axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    axes[i].set_xlabel(feat)
    axes[i].set_ylabel("SHAP value (impact on risk)")
    axes[i].set_title(f"Effect of {feat} on Collision Risk")

plt.suptitle("Feature Effects on Collision Risk — Tier 1 (SHAP)", fontsize=13)
plt.tight_layout()
plt.savefig("tier1_shap_dependence.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: tier1_shap_dependence.png")

# ---- PLOT 4: Dependence plots for time features ----
time_features = ["hour_sin", "hour_cos", "month_sin", "month_cos"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, feat in enumerate(time_features):
    feat_idx       = features.index(feat)
    feat_vals      = X_test_arr[:, feat_idx]
    shap_vals_feat = shap_vals[:, feat_idx]

    axes[i].scatter(feat_vals, shap_vals_feat,
                    alpha=0.4, s=10, color="#534AB7")
    axes[i].axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    axes[i].set_xlabel(feat)
    axes[i].set_ylabel("SHAP value (impact on risk)")
    axes[i].set_title(f"Effect of {feat} on Collision Risk")

plt.suptitle("Time Feature Effects on Collision Risk — Tier 1 (SHAP)", fontsize=13)
plt.tight_layout()
plt.savefig("tier1_shap_time.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: tier1_shap_time.png")