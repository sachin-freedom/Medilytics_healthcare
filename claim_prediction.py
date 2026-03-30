import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

df = pd.read_csv("pre_processed_data.csv")

features = [
    "Department",
    "Procedure_Code",
    "Insurance_Type",
    "Claim_Amount",
    "Documentation_Delay_Days",
    "Length_of_Stay",
    "Previous_Denial_Count"
]

target = "Denial_Flag"

X = df[features].copy()
y = df[target]

# Identify categorical columns for one-hot encoding
categorical_features = [
    "Department",
    "Procedure_Code",
    "Insurance_Type"
]

# Apply one-hot encoding to categorical features
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

X = X.fillna(X.median(numeric_only=True))

numeric_features = [
    "Claim_Amount",
    "Documentation_Delay_Days",
    "Length_of_Stay",
    "Previous_Denial_Count"
]


scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

print("\nModel Evaluation")
print("Accuracy:", round(accuracy,4))
print("Precision:", round(precision,4))
print("Recall:", round(recall,4))
print("F1 Score:", round(f1,4))
print("ROC-AUC:", round(roc_auc,4))

# Ensure X has the same columns as X_train for prediction
X_processed = pd.get_dummies(df[features], columns=categorical_features, drop_first=True)
X_processed = X_processed.fillna(X_processed.median(numeric_only=True))
X_processed[numeric_features] = scaler.transform(X_processed[numeric_features])

df["Denial_Probability"] = model.predict_proba(X_processed)[:,1]

def risk_level(p):
    if p < 0.3:
        return "Low"
    elif p < 0.6:
        return "Medium"
    else:
        return "High"

df["Risk_Level"] = df["Denial_Probability"].apply(risk_level)

predictions = df[["Claim_ID","Denial_Probability","Risk_Level"]]
predictions.to_csv("denial_model_predictions.csv", index=False)

metrics = pd.DataFrame({
    "Accuracy":[accuracy],
    "Precision":[precision],
    "Recall":[recall],
    "F1_Score":[f1],
    "ROC_AUC":[roc_auc]
})

metrics.to_csv("denial_model_metrics.csv", index=False)

# Update feature importance to include one-hot encoded features
importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": model.coef_[0]
})

importance.to_csv("denial_feature_importance.csv", index=False)

os.makedirs("models", exist_ok=True)

joblib.dump(model,"models/denial_model.pkl")
joblib.dump(scaler,"models/scaler.pkl")

print("\nDenial prediction model completed successfully")