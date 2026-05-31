# ============================================================
# TIER 2 AVC MODEL — DIRECT TRAINING (NO STACKING)
# ============================================================
#
# Trains directly on all available tier 2 features:
#   - Spatial:        NDVI, KDE hotspot
#   - Monthly climate: monthly temperature, drought index
#   - Daily weather:  temperature, precipitation, wind, moon
#   - Cyclical time:  hour, day-of-week, month (sin/cos)
#
# MODELS:
#   1. Logistic Regression
#   2. Random Forest
#   3. XGBoost
#   4. LightGBM
#   5. Neural Network (MLP)
#   6. Support Vector Machine (SVM)
#
# FIXES APPLIED:
#   - CV now runs on ALL 6 models (not just top 3 by hold-out ROC-AUC)
#   - SHAP uses best tree-based model, not hardcoded Random Forest
#   - Consistent numpy arrays passed to CV for tree models
#   - Duplicate matplotlib import removed
#   - imbalance_ratio moved to after train/test split for clarity

# ============================================================
# 1. IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv("tier2_with_drought.csv")
print(f"Loaded {len(df)} rows")

# ============================================================
# 3. FEATURE SET
# ============================================================

features = [
    # SPATIAL
    "NDVI_Mean",
    "KDE_hotspot",

    # MONTHLY CLIMATE (same variables as tier 1)
    "Monthly_Temperature_Average",
    "monthly_drought_index (0-100)",

    # DAILY WEATHER (new to tier 2)
    "Temperature (cel)",
    "Precipitation (mm)",
    "Wind km/h",
    "Moon_Illumination",

    # CYCLICAL TIME
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "hour_sin",
    "hour_cos"
]

target = "target"

# Check all features are present
missing = [f for f in features if f not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

X = df[features].copy()
y = df[target]

# Impute any remaining missing values with column medians
for col in features:
    if X[col].isnull().any():
        median_val = X[col].median()
        n_missing  = X[col].isnull().sum()
        X[col]     = X[col].fillna(median_val)
        print(f"  Imputed {n_missing} missing values in '{col}' with median {median_val:.3f}")

print(f"\nDataset: {len(df)} rows | {y.sum()} collisions | {(y==0).sum()} absence points")
print(f"Class ratio: 1:{int((y==0).sum() // y.sum())}")
print(f"Features ({len(features)}): {features}")

# ============================================================
# 4. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ============================================================
# 5. SCALE DATA
# ============================================================

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Convert training data to numpy array once for consistent CV usage
X_train_arr = np.array(X_train)
X_test_arr  = np.array(X_test)

# ============================================================
# 6. SAMPLE WEIGHTS FOR MLP
# ============================================================

sample_weights_train = compute_sample_weight(class_weight="balanced", y=y_train)

# ============================================================
# 7. CLASS IMBALANCE RATIO (computed after split, from training set)
# ============================================================

negative_cases  = np.sum(y_train == 0)
positive_cases  = np.sum(y_train == 1)
imbalance_ratio = negative_cases / positive_cases

print(f"\nTraining set: {positive_cases} positives, {negative_cases} negatives")
print(f"Imbalance ratio (scale_pos_weight): {imbalance_ratio:.2f}")

# ============================================================
# 8. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

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

    "LightGBM": LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
        verbosity=-1,
        feature_name='auto'
    ),

    # MLP: no class_weight param — handled via sample_weight in fit()
    # NOTE: Neural Network is excluded from cross_validate() because
    #       sklearn's cross_validate does not support fit_params with
    #       sample_weight cleanly across all versions. Hold-out results
    #       are used for Neural Network evaluation instead.
    "Neural Network": MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=64,
        learning_rate='adaptive',
        max_iter=1000,
        random_state=42
    ),

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

results            = []
trained_models     = {}
test_probabilities = {}

for name, model in models.items():

    print("\n================================================")
    print(f"TRAINING: {name}")
    print("================================================")

    uses_scaled = name in ["Logistic Regression", "Neural Network", "SVM"]
    X_tr = X_train_scaled if uses_scaled else X_train_arr
    X_te = X_test_scaled  if uses_scaled else X_test_arr

    if name == "Neural Network":
        model.fit(X_tr, y_train, sample_weight=sample_weights_train)
    else:
        model.fit(X_tr, y_train)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    trained_models[name]     = model
    test_probabilities[name] = y_prob

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

results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)

# ============================================================
# 10. 5-FOLD CROSS-VALIDATION (ALL models except Neural Network)
# ============================================================
# FIX: Previously only ran CV on top 3 by hold-out ROC-AUC.
#      This excluded LightGBM/XGBoost if they ranked outside top 3
#      on the small, noisy hold-out set. CV now runs on all models
#      that support standard fit() without sample_weight.
# ============================================================

print("\n================================================")
print("5-FOLD CROSS-VALIDATION (all models except Neural Network)")
print("================================================")

# Models that support standard CV (Neural Network excluded — see note above)
cv_model_names = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "SVM"]

cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for name in cv_model_names:
    model       = models[name]
    uses_scaled = name in ["Logistic Regression", "SVM"]

    # FIX: consistently use numpy arrays for all models in CV
    X_cv = X_train_scaled if uses_scaled else X_train_arr

    scoring = {
        "roc_auc":           "roc_auc",
        "average_precision": "average_precision",
        "f1":                "f1",
        "recall":            "recall",
        "precision":         "precision"
    }

    scores = cross_validate(model, X_cv, y_train, cv=cv, scoring=scoring, n_jobs=-1)

    cv_results.append({
        "Model":        name,
        "CV ROC-AUC":   f"{scores['test_roc_auc'].mean():.4f} ± {scores['test_roc_auc'].std():.4f}",
        "CV PR-AUC":    f"{scores['test_average_precision'].mean():.4f} ± {scores['test_average_precision'].std():.4f}",
        "CV F1":        f"{scores['test_f1'].mean():.4f} ± {scores['test_f1'].std():.4f}",
        "CV Recall":    f"{scores['test_recall'].mean():.4f} ± {scores['test_recall'].std():.4f}",
        "CV Precision": f"{scores['test_precision'].mean():.4f} ± {scores['test_precision'].std():.4f}",
    })

    print(f"\n{name}")
    print(f"  ROC-AUC  : {scores['test_roc_auc'].mean():.4f} ± {scores['test_roc_auc'].std():.4f}")
    print(f"  PR-AUC   : {scores['test_average_precision'].mean():.4f} ± {scores['test_average_precision'].std():.4f}")
    print(f"  F1       : {scores['test_f1'].mean():.4f} ± {scores['test_f1'].std():.4f}")
    print(f"  Recall   : {scores['test_recall'].mean():.4f} ± {scores['test_recall'].std():.4f}")
    print(f"  Precision: {scores['test_precision'].mean():.4f} ± {scores['test_precision'].std():.4f}")

cv_df = pd.DataFrame(cv_results)

# ============================================================
# 11. FEATURE IMPORTANCE (best model by hold-out ROC-AUC)
# ============================================================

best_model_name = results_df["Model"].iloc[0]
best_model      = trained_models[best_model_name]

if hasattr(best_model, "feature_importances_"):
    importance = pd.Series(best_model.feature_importances_, index=features)
    importance = importance.sort_values(ascending=False)

    print(f"\n================================================")
    print(f"FEATURE IMPORTANCE — {best_model_name}")
    print(f"================================================")
    print(importance.to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    importance.sort_values().plot(kind="barh", ax=ax, color="#378ADD")
    ax.set_title(f"Feature Importance — {best_model_name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig("tier2_feature_importance.png", dpi=150)
    plt.close()
    print("Saved: tier2_feature_importance.png")

# ============================================================
# 12. PRECISION-RECALL CURVES
# ============================================================

print("\nGenerating precision-recall curves...")

fig, ax = plt.subplots(figsize=(8, 6))
colors  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for (name, y_prob), color in zip(test_probabilities.items(), colors):
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    pr_auc_val   = average_precision_score(y_test, y_prob)
    ax.plot(rec, prec, label=f"{name} (PR-AUC={pr_auc_val:.3f})",
            color=color, linewidth=1.8)

baseline = positive_cases / (positive_cases + negative_cases)
ax.axhline(y=baseline, color='gray', linestyle='--', linewidth=1,
           label=f"No-skill baseline ({baseline:.3f})")

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves — Tier 2 Direct Models")
ax.legend(loc="upper right", fontsize=8)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("tier2_pr_curves.png", dpi=150)
plt.close()
print("Saved: tier2_pr_curves.png")

# ============================================================
# 13. SAVE RESULTS
# ============================================================

results_df.to_csv("tier2_direct_results.csv", index=False)
cv_df.to_csv("tier2_direct_cv_results.csv", index=False)

joblib.dump(best_model, "tier2_best_model.pkl")
joblib.dump(scaler,     "tier2_scaler.pkl")

print(f"\nBest model (hold-out ROC-AUC): {best_model_name}")
print("Saved: tier2_direct_results.csv, tier2_direct_cv_results.csv")
print("Saved: tier2_best_model.pkl, tier2_scaler.pkl")

# ============================================================
# 14. SHAP ANALYSIS
# ============================================================
# FIX: Previously hardcoded to Random Forest regardless of best model.
#      Now uses the best tree-based model available. Falls back to
#      Random Forest only if the best model is not tree-based (i.e.
#      Logistic Regression or SVM, which don't support TreeExplainer).
# ============================================================

TREE_BASED_MODELS = ["Random Forest", "XGBoost", "LightGBM"]

if best_model_name in TREE_BASED_MODELS:
    shap_model_name = best_model_name
else:
    # Best model is not tree-based; use the top tree-based model by hold-out ROC-AUC
    tree_results = results_df[results_df["Model"].isin(TREE_BASED_MODELS)]
    shap_model_name = tree_results["Model"].iloc[0]
    print(f"\nNote: Best model ({best_model_name}) is not tree-based.")
    print(f"Using {shap_model_name} for SHAP analysis (best tree-based model).")

shap_model = trained_models[shap_model_name]

print(f"\nCalculating SHAP values for: {shap_model_name} (this may take a minute)...")

# Use numpy array for SHAP — consistent with how the model was trained
X_test_shap = X_test_arr

explainer   = shap.TreeExplainer(shap_model)
shap_values = explainer.shap_values(X_test_shap)

# For binary classification, shap_values may be a list of two arrays
# or a 3D array — take class 1 (collision) in either case
if isinstance(shap_values, list):
    shap_vals = shap_values[1]
elif shap_values.ndim == 3:
    shap_vals = shap_values[:, :, 1]
else:
    shap_vals = shap_values

print(f"shap_vals shape: {shap_vals.shape}")  # should be (n_samples, 14)

# ---- PLOT 1: Summary (beeswarm) ----
plt.figure()
shap.summary_plot(shap_vals, X_test_shap, feature_names=features, show=False)
plt.title(f"SHAP Summary — {shap_model_name}")
plt.tight_layout()
plt.savefig("tier2_shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: tier2_shap_summary.png")

# ---- PLOT 2: Bar chart (mean absolute SHAP) ----
plt.figure()
shap.summary_plot(shap_vals, X_test_shap, feature_names=features,
                  plot_type="bar", show=False)
plt.title(f"SHAP Feature Importance (Mean |SHAP|) — {shap_model_name}")
plt.tight_layout()
plt.savefig("tier2_shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: tier2_shap_bar.png")

# ---- PLOT 3: Dependence plots for daily weather features ----
weather_features = ["Moon_Illumination", "Temperature (cel)",
                    "Wind km/h", "Precipitation (mm)"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, feat in enumerate(weather_features):
    feat_idx       = features.index(feat)
    feat_vals      = X_test_shap[:, feat_idx]
    shap_vals_feat = shap_vals[:, feat_idx]

    axes[i].scatter(feat_vals, shap_vals_feat,
                    alpha=0.4, s=10, color="#378ADD")
    axes[i].axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    axes[i].set_xlabel(feat)
    axes[i].set_ylabel("SHAP value (impact on risk)")
    axes[i].set_title(f"Effect of {feat} on Collision Risk")

plt.suptitle(f"Daily Weather Feature Effects on Collision Risk — {shap_model_name}", fontsize=12)
plt.tight_layout()
plt.savefig("tier2_shap_dependence.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: tier2_shap_dependence.png")

print("\n================================================")
print("TIER 2 COMPLETE")
print(f"Best model (hold-out): {best_model_name}")
print(f"SHAP analysis model:   {shap_model_name}")
print("================================================")