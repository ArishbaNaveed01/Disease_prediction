"""
Disease Prediction from Medical Data
--------------------------------------------
Objective : Predict the possibility of disease based on structured
            patient/medical data 

Dataset   : Breast Cancer Wisconsin (Diagnostic) Dataset
            - 569 patient records
            - 30 numeric features computed from digitized images of a
              breast mass (radius, texture, perimeter, area, smoothness,
              concavity, symmetry, etc.)
            - Target: malignant (disease present) vs benign (no disease)
            This dataset is one of the classic UCI-style medical
            datasets referenced in the task brief and ships directly
            with scikit-learn, so the script is fully reproducible
            without any external downloads.

Algorithms: Logistic Regression, Support Vector Machine (SVM),
            Random Forest, XGBoost   <-- all four requested in the task

Metrics   : Accuracy, Precision, Recall, F1-Score, ROC-AUC,
            Confusion Matrix, Cross-Validation

"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from xgboost import XGBClassifier

import os
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
sns.set_style("whitegrid")


# 1. LOAD DATA

print("=" * 70)
print("Loading dataset")
print("=" * 70)

data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
# target: 0 = malignant (disease), 1 = benign (no disease) in sklearn's
# raw encoding -> we flip it so 1 = disease (malignant), which is more
# intuitive for a "disease prediction" task.
df["diagnosis"] = df["target"].apply(lambda x: 1 if x == 0 else 0)
df.drop(columns=["target"], inplace=True)

print(f"Dataset shape: {df.shape}")
print(f"Features: {list(data.feature_names)[:5]} ... (+{len(data.feature_names)-5} more)")
print(f"Class balance (1=disease/malignant, 0=no disease/benign):")
print(df["diagnosis"].value_counts())
df.to_csv(f"{OUT_DIR}/dataset.csv", index=False)


# 2. EXPLORATORY DATA ANALYSIS 

print("\n" + "=" * 70)
print("Exploratory Data Analysis")
print("=" * 70)

plt.figure(figsize=(5, 4))
sns.countplot(x="diagnosis", data=df, palette=["#4C9F70", "#D1495B"])
plt.title("Class Distribution (0 = No Disease, 1 = Disease)")
plt.xlabel("Diagnosis")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_class_distribution.png", dpi=150)
plt.close()

plt.figure(figsize=(12, 10))
corr = df.corr(numeric_only=True)
top_corr_features = corr["diagnosis"].abs().sort_values(ascending=False).index[1:11]
sns.heatmap(df[top_corr_features.tolist() + ["diagnosis"]].corr(), annot=True,
            fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap - Top 10 Features vs Diagnosis")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_correlation_heatmap.png", dpi=150)
plt.close()

print("Saved EDA plots: 01_class_distribution.png, 02_correlation_heatmap.png")


# 3. FEATURE ENGINEERING / PREPROCESSING

print("\n" + "=" * 70)
print("Preprocessing (train/test split + scaling)")
print("=" * 70)

X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train set: {X_train.shape[0]} patients | Test set: {X_test.shape[0]} patients")


# 4. MODEL TRAINING - Logistic Regression, SVM, Random Forest, XGBoost

print("\n" + "=" * 70)
print(" Training models")
print("=" * 70)

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "SVM (RBF Kernel)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(
        n_estimators=300, use_label_encoder=False,
        eval_metric="logloss", random_state=RANDOM_STATE
    ),
}

results = []
roc_data = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for name, model in models.items():
    # Tree-based models don't strictly need scaling, but using scaled
    # data for all keeps the pipeline consistent and doesn't hurt them.
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="accuracy")

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc,
        "CV Accuracy (mean)": cv_scores.mean(),
        "CV Accuracy (std)": cv_scores.std(),
    })

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name] = (fpr, tpr, auc)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(f"{OUT_DIR}/cm_{safe_name}.png", dpi=150)
    plt.close()

    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
results_df.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)

# 5. MODEL COMPARISON PLOTS

print("\n" + "=" * 70)
print("Model comparison")
print("=" * 70)
print(results_df.to_string(index=False))

# Bar chart of metrics
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
plot_df = results_df.set_index("Model")[metrics_to_plot]
plot_df.plot(kind="bar", figsize=(10, 6), colormap="viridis")
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0.8, 1.02)
plt.legend(loc="lower right")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_model_comparison_bar.png", dpi=150)
plt.close()

# ROC curves - all models on one plot
plt.figure(figsize=(6, 5))
for name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - All Models")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_roc_curves.png", dpi=150)
plt.close()


# 6. FEATURE IMPORTANCE (Random Forest & XGBoost)

print("\n" + "=" * 70)
print("Feature importance")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, model_name in zip(axes, ["Random Forest", "XGBoost"]):
    model = models[model_name]
    importances = pd.Series(model.feature_importances_, index=X.columns)
    top10 = importances.sort_values(ascending=False).head(10)
    sns.barplot(x=top10.values, y=top10.index, ax=ax, palette="mako")
    ax.set_title(f"Top 10 Important Features - {model_name}")
    ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_feature_importance.png", dpi=150)
plt.close()

best_model_name = results_df.iloc[0]["Model"]
print(f"\nBest performing model by ROC-AUC: {best_model_name}")

# 7. SAVE SUMMARY REPORT

with open(f"{OUT_DIR}/results_summary.txt", "w") as f:
    f.write("CodeAlpha ML Internship - Task 4: Disease Prediction from Medical Data\n")
    f.write("=" * 72 + "\n\n")
    f.write(f"Dataset: Breast Cancer Wisconsin (Diagnostic), {df.shape[0]} patients, "
            f"{X.shape[1]} features\n\n")
    f.write("Model Comparison (sorted by ROC-AUC):\n\n")
    f.write(results_df.to_string(index=False))
    f.write(f"\n\nBest Model: {best_model_name}\n")

