"""
model.py  –  Train & save the Loan Prediction model
Run once before starting the Flask app:  python model.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib          # preferred over pickle for sklearn models
import os

# ── 1. Load data ─────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "loan_data_set.xls.csv")
data = pd.read_csv(DATA_PATH)

# The CSV stores all columns in one quoted field – split them out
data = data.iloc[:, 0].str.split(",", expand=True)
data.columns = [
    "Loan_ID", "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "ApplicantIncome", "CoapplicantIncome",
    "LoanAmount", "Loan_Amount_Term", "Credit_History",
    "Property_Area", "Loan_Status"
]

# ── 2. Clean ──────────────────────────────────────────────────────────────────
data.replace("", np.nan, inplace=True)
data.drop("Loan_ID", axis=1, inplace=True)

num_cols = [
    "ApplicantIncome", "CoapplicantIncome",
    "LoanAmount", "Loan_Amount_Term", "Credit_History"
]
data[num_cols] = data[num_cols].apply(pd.to_numeric, errors="coerce")
data["Dependents"] = data["Dependents"].replace("3+", 3)
data["Dependents"] = pd.to_numeric(data["Dependents"], errors="coerce")

data[num_cols + ["Dependents"]] = data[num_cols + ["Dependents"]].fillna(
    data[num_cols + ["Dependents"]].median()
)

cat_cols = ["Gender", "Married", "Education", "Self_Employed", "Property_Area", "Loan_Status"]
for col in cat_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

data.replace({
    "Gender":        {"Male": 1, "Female": 0},
    "Married":       {"Yes": 1, "No": 0},
    "Education":     {"Graduate": 1, "Not Graduate": 0},
    "Self_Employed": {"Yes": 1, "No": 0},
    "Loan_Status":   {"Y": 1, "N": 0},
    "Property_Area": {"Urban": 2, "Semiurban": 1, "Rural": 0},
}, inplace=True)

# ── 3. Split ──────────────────────────────────────────────────────────────────
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 4. Train ──────────────────────────────────────────────────────────────────
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
y_pred = clf.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")

print("=" * 50)
print(f"  Test Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
print(f"  CV Accuracy    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print("=" * 50)
print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))

# ── 6. Confusion matrix plot ──────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Rejected", "Approved"],
            yticklabels=["Rejected", "Approved"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
plt.tight_layout()
fig.savefig("confusion_matrix.png")
print("Confusion matrix saved → confusion_matrix.png")

# ── 7. SHAP feature importance ────────────────────────────────────────────────
explainer  = shap.TreeExplainer(clf)
shap_vals  = explainer.shap_values(X_test)
vals       = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

fig2, ax2 = plt.subplots(figsize=(8, 5))
mean_abs   = np.abs(vals).mean(axis=0)
feat_names = X.columns.tolist()
sorted_idx = np.argsort(mean_abs)
ax2.barh([feat_names[i] for i in sorted_idx],
         [mean_abs[i]   for i in sorted_idx],
         color="#6366f1")
ax2.set_title("Mean |SHAP| – Feature Importance")
ax2.set_xlabel("Mean absolute SHAP value")
plt.tight_layout()
fig2.savefig("shap_importance.png")
print("SHAP importance chart saved → shap_importance.png")

# ── 8. Save model with joblib ─────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "loan_model.pkl")
joblib.dump(clf, MODEL_PATH)
print(f"\n✅ Model saved  →  {MODEL_PATH}")
