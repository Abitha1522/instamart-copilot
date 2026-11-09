
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.utils import resample


# -------------------------------------------------------
# 1. PATHS AND SETTINGS
# -------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "clean"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CUTOFF_DATE = pd.Timestamp("2025-09-01")
LABEL_END_DATE = pd.Timestamp("2025-10-01")


# -------------------------------------------------------
# 2. LOAD DATA
# -------------------------------------------------------

orders = pd.read_csv(DATA_DIR / "orders.csv")
customers = pd.read_csv(DATA_DIR / "customers.csv")
products = pd.read_csv(DATA_DIR / "products.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"])
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

orders["status"] = orders["status"].astype(str).str.lower().str.strip()

print("=" * 55)
print("B2 - CUSTOMER LAPSE PREDICTION")
print("=" * 55)

print("Orders:", orders.shape)
print("Customers:", customers.shape)


# -------------------------------------------------------
# 3. CREATE CUSTOMER COHORT
# -------------------------------------------------------

completed = orders[orders["status"] == "completed"]

# Customers must have at least one completed order
# before 1 September 2025.

pre_cutoff_completed = completed[
    completed["order_ts"] < CUTOFF_DATE
]

cohort_ids = pre_cutoff_completed["customer_id"].unique()

cohort = customers[
    customers["customer_id"].isin(cohort_ids)
].copy()

print("\nCohort size:", len(cohort))


# -------------------------------------------------------
# 4. CREATE FEATURES USING PRE-CUTOFF DATA ONLY
# -------------------------------------------------------

pre_orders = orders[
    (orders["order_ts"] < CUTOFF_DATE)
    & (orders["customer_id"].isin(cohort_ids))
].copy()

pre_completed = pre_orders[
    pre_orders["status"] == "completed"
].copy()

# Add product category to orders.
pre_orders = pre_orders.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

pre_completed = pre_completed.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

# RFM features from completed orders.
rfm = pre_completed.groupby("customer_id").agg(
    last_order_date=("order_ts", "max"),
    frequency=("order_id", "count"),
    monetary=("order_amount", "sum")
)

rfm["recency"] = (
    CUTOFF_DATE - rfm["last_order_date"]
).dt.days

rfm = rfm.drop(columns=["last_order_date"])

# Average discount and cancellation rate.
order_features = pre_orders.groupby("customer_id").agg(
    avg_discount=("discount_pct", "mean"),
    total_orders=("order_id", "count")
)

cancelled = pre_orders[
    pre_orders["status"].isin(["cancelled", "canceled"])
].groupby("customer_id").size()

order_features["cancelled_orders"] = cancelled
order_features["cancelled_orders"] = (
    order_features["cancelled_orders"].fillna(0)
)

order_features["cancellation_rate"] = (
    order_features["cancelled_orders"]
    / order_features["total_orders"]
)

order_features = order_features[
    ["avg_discount", "cancellation_rate"]
]

# Average delivery time from available pre-cutoff values.
delivery = pre_completed.groupby("customer_id").agg(
    avg_delivery=("delivery_minutes", "mean")
)

# Favourite product category.
favourite_category = (
    pre_orders.groupby("customer_id")["category"]
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
    .rename("favorite_category")
)

# Customer profile features.
profile = cohort.set_index("customer_id")[
    ["city", "age", "gender", "signup_date"]
].copy()

profile["tenure_days"] = (
    CUTOFF_DATE - profile["signup_date"]
).dt.days

profile = profile.drop(columns=["signup_date"])

# Combine all features.
features = (
    rfm
    .join(order_features)
    .join(delivery)
    .join(favourite_category)
    .join(profile)
)

# Ensure every cohort customer is represented.
features = features.reset_index()

# Fill missing feature values.
numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount",
    "cancellation_rate",
    "avg_delivery",
    "age",
    "tenure_days"
]

categorical_features = [
    "favorite_category",
    "city",
    "gender"
]

for column in numeric_features:
    features[column] = pd.to_numeric(
        features[column], errors="coerce"
    )
    features[column] = features[column].fillna(
        features[column].median()
    )

for column in categorical_features:
    features[column] = (
        features[column]
        .fillna("Unknown")
        .astype(str)
    )

print("\nFeature dataset shape:", features.shape)


# -------------------------------------------------------
# 5. CREATE LAPSE LABEL
# -------------------------------------------------------

# September completed orders are used only for the label.
# They are NOT used as model features.

september_completed = completed[
    (completed["order_ts"] >= CUTOFF_DATE)
    & (completed["order_ts"] < LABEL_END_DATE)
]

active_ids = set(september_completed["customer_id"])

features["lapsed"] = (
    ~features["customer_id"].isin(active_ids)
).astype(int)

print("\nLAPSE LABEL SUMMARY")
print("-" * 45)
print(features["lapsed"].value_counts().sort_index())

print(
    "Lapsed percentage:",
    round(features["lapsed"].mean() * 100, 2),
    "%"
)

features.to_csv(
    OUTPUT_DIR / "b2_customer_features.csv",
    index=False
)


# -------------------------------------------------------
# 6. PREPARE TRAINING AND TEST DATA
# -------------------------------------------------------

X = features[
    numeric_features + categorical_features
]

y = features["lapsed"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\nTRAIN/TEST SPLIT")
print("-" * 45)
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# -------------------------------------------------------
# 7. PREPROCESSING
# -------------------------------------------------------

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        drop="first",
        handle_unknown="ignore"
    ))
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)
])


# -------------------------------------------------------
# 8. OVERSAMPLE TRAINING DATA ONLY
# -------------------------------------------------------

train_data = X_train.copy()
train_data["lapsed"] = y_train.values

majority = train_data[
    train_data["lapsed"] == 0
]

minority = train_data[
    train_data["lapsed"] == 1
]

minority_oversampled = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=42
)

balanced_train = pd.concat(
    [majority, minority_oversampled],
    ignore_index=True
)

X_train_balanced = balanced_train.drop(columns=["lapsed"])
y_train_balanced = balanced_train["lapsed"]

print("\nOVERSAMPLING SUMMARY")
print("-" * 45)
print(y_train_balanced.value_counts().sort_index())


# -------------------------------------------------------
# 9. HELPER FUNCTION FOR MODEL EVALUATION
# -------------------------------------------------------

def evaluate_model(name, model, X_eval, y_eval):
    predictions = model.predict(X_eval)

    accuracy = accuracy_score(y_eval, predictions)
    precision = precision_score(
        y_eval, predictions, zero_division=0
    )
    recall = recall_score(
        y_eval, predictions, zero_division=0
    )
    f1 = f1_score(
        y_eval, predictions, zero_division=0
    )

    print(f"\n{name}")
    print("-" * 45)
    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1 Score : {f1:.3f}")

    return f1


# -------------------------------------------------------
# 10. LOGISTIC REGRESSION - UNBALANCED
# -------------------------------------------------------

print("\nTraining Logistic Regression (unbalanced)...")

lr_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

lr_grid = GridSearchCV(
    lr_pipeline,
    param_grid={
        "model__C": [0.01, 0.1, 1, 10, 100]
    },
    cv=5,
    scoring="f1",
    n_jobs=-1
)

lr_grid.fit(X_train, y_train)

print("Best parameters:", lr_grid.best_params_)
print("Best CV F1:", round(lr_grid.best_score_, 3))

lr_unbalanced_f1 = evaluate_model(
    "LOGISTIC REGRESSION - UNBALANCED",
    lr_grid.best_estimator_,
    X_test,
    y_test
)


# -------------------------------------------------------
# 11. LOGISTIC REGRESSION - BALANCED
# -------------------------------------------------------

print("\nTraining Logistic Regression (balanced)...")

lr_balanced_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

lr_balanced_grid = GridSearchCV(
    lr_balanced_pipeline,
    param_grid={
        "model__C": [0.01, 0.1, 1, 10, 100]
    },
    cv=5,
    scoring="f1",
    n_jobs=-1
)

lr_balanced_grid.fit(
    X_train_balanced,
    y_train_balanced
)

print("Best parameters:", lr_balanced_grid.best_params_)
print("Best CV F1:", round(lr_balanced_grid.best_score_, 3))

lr_balanced_f1 = evaluate_model(
    "LOGISTIC REGRESSION - BALANCED",
    lr_balanced_grid.best_estimator_,
    X_test,
    y_test
)


# -------------------------------------------------------
# 12. RANDOM FOREST - BALANCED
# -------------------------------------------------------

print("\nTraining Random Forest...")

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

rf_grid = GridSearchCV(
    rf_pipeline,
    param_grid={
        "model__max_depth": [3, 5, 8],
        "model__min_samples_leaf": [5, 10, 20]
    },
    cv=5,
    scoring="f1",
    n_jobs=-1
)

rf_grid.fit(
    X_train_balanced,
    y_train_balanced
)

print("Best parameters:", rf_grid.best_params_)
print("Best CV F1:", round(rf_grid.best_score_, 3))

rf_f1 = evaluate_model(
    "RANDOM FOREST - BALANCED",
    rf_grid.best_estimator_,
    X_test,
    y_test
)


# -------------------------------------------------------
# 13. SELECT THE BEST MODEL
# -------------------------------------------------------

model_results = {
    "Logistic Regression - Unbalanced": (
        lr_unbalanced_f1, lr_grid.best_estimator_
    ),
    "Logistic Regression - Balanced": (
        lr_balanced_f1, lr_balanced_grid.best_estimator_
    ),
    "Random Forest - Balanced": (
        rf_f1, rf_grid.best_estimator_
    )
}

best_model_name = max(
    model_results,
    key=lambda name: model_results[name][0]
)

best_f1, best_model = model_results[best_model_name]

print("\nBEST MODEL")
print("-" * 45)
print("Model:", best_model_name)
print("Test F1:", round(best_f1, 3))

# Save the selected model.
joblib.dump(
    best_model,
    OUTPUT_DIR / "lapse_model.joblib"
)


# -------------------------------------------------------
# 14. CREATE RETENTION LIST
# -------------------------------------------------------

lapse_probabilities = best_model.predict_proba(X)[:, 1]
lapse_predictions = best_model.predict(X)

retention_list = pd.DataFrame({
    "customer_id": features["customer_id"],
    "lapse_probability": lapse_probabilities,
    "predicted_lapsed": lapse_predictions
})

retention_list = retention_list[
    retention_list["predicted_lapsed"] == 1
].sort_values(
    "lapse_probability",
    ascending=False
)

retention_list.to_csv(
    OUTPUT_DIR / "retention_list.csv",
    index=False
)

print("\nRetention list customers:", len(retention_list))


# -------------------------------------------------------
# 15. VALIDATION CHECKS
# -------------------------------------------------------

print("\n" + "=" * 55)
print("VALIDATION CHECKS")
print("=" * 55)

checks = []

def check(condition, message):
    if condition:
        print("PASS -", message)
        checks.append(True)
    else:
        print("FAIL -", message)
        checks.append(False)

check(
    len(features) == 1156,
    "Cohort contains 1156 customers"
)

check(
    int(features["lapsed"].sum()) == 443,
    "Lapsed customer count is 443"
)

check(
    len(X_train) == 924 and len(X_test) == 232,
    "Train/test split is 924/232"
)

check(
    y_train_balanced.value_counts().nunique() == 1,
    "Training classes are balanced after oversampling"
)

check(
    not features[numeric_features + categorical_features].isnull().any().any(),
    "Feature dataset contains no missing values"
)

check(
    best_f1 >= 0.55,
    "Best model test F1 is at least 0.55"
)

check(
    (OUTPUT_DIR / "customer_segments.csv").exists(),
    "Customer segments file was saved"
)
check(
    (OUTPUT_DIR / "b2_customer_features.csv").exists(),
    "B2 customer features file was saved"
)

check(
    (OUTPUT_DIR / "lapse_model.joblib").exists(),
    "Lapse model was saved"
)

check(
    (OUTPUT_DIR / "retention_list.csv").exists(),
    "Retention list was saved"
)

print("\nChecks passed:", sum(checks), "/", len(checks))

if all(checks):
    print("\nALL B2 CHECKPOINTS PASSED")
else:
    print("\nSOME B2 CHECKPOINTS FAILED - REVIEW THE OUTPUT")
