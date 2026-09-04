## import the libraries
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


##1. Load dataset

DATA_PATH = "data/supply_chain_data.csv"
df = pd.read_csv(DATA_PATH)

TARGET = "Late_Delivery_Risk"

CATEGORICAL_FEATURES = [
    "Shipping_Mode",
    "Order_Priority",
    "Customer_Segment",
    "Market",
    "Product_Category",
    "Order_Weekday",
]
NUMERIC_FEATURES = [
    "Order_Quantity",
    "Discount",
    "Price",
    "Days_for_Shipment_Scheduled",
    "Order_Month",
]

X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
y = df[TARGET]


# 2. Train - test split 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 3. Preprocessing: onehot encode categoricals, scale numerics

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("num", StandardScaler(), NUMERIC_FEATURES),
    ]
)

# handle class imbalance: ratio of negative to positive class
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
)


# 4. Full pipeline (preprocessing + model) — this is what gets deployed

pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

pipeline.fit(X_train, y_train)


# 5. Evaluate

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy :", round(accuracy, 4))
print("F1 score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))
print("\nClassification report:\n", classification_report(y_test, y_pred))


# 6. Feature importance (for the dashboard's Insights tab)

encoded_cols = pipeline.named_steps["preprocessor"].get_feature_names_out()
importances = pipeline.named_steps["model"].feature_importances_
feature_importance = (
    pd.Series(importances, index=encoded_cols)
    .sort_values(ascending=False)
    .head(15)
    .to_dict()
)


# 7. Save artifacts

joblib.dump(pipeline, "artifacts/model_pipeline.pkl")

metadata = {
    "categorical_features": CATEGORICAL_FEATURES,
    "numeric_features": NUMERIC_FEATURES,
    "feature_options": {
        col: sorted(df[col].unique().tolist()) for col in CATEGORICAL_FEATURES
    },
    "numeric_ranges": {
        col: (float(df[col].min()), float(df[col].max())) for col in NUMERIC_FEATURES
    },
    "target": TARGET,
    #here  everything below is only used to power the Streamlit "Insights" tab 

    "metrics": {
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "classification_report": report,
    },
    "confusion_matrix": cm.tolist(),
    "feature_importance": feature_importance,
}
joblib.dump(metadata, "artifacts/metadata.pkl")

print("\nSaved artifacts/model_pipeline.pkl and artifacts/metadata.pkl")
