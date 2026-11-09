import sqlite3
from pathlib import Path
import pandas as pd

# Project folders
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
CLEAN_DIR = PROJECT_DIR / "data" / "clean"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# Load raw datasets
customers = pd.read_csv(RAW_DIR / "customers.csv")
products = pd.read_csv(RAW_DIR / "products.csv")
orders = pd.read_csv(RAW_DIR / "orders.csv")

# -------------------------------
# 1. CLEAN CUSTOMER DATA
# -------------------------------

# Standardize city names
customers["city"] = customers["city"].astype("string").str.strip().str.title()

# Normalize gender
customers["gender"] = (
    customers["gender"]
    .astype("string")
    .str.strip()
    .str.lower()
)

gender_map = {
    "m": "male",
    "male": "male",
    "f": "female",
    "female": "female"
}

customers["gender"] = customers["gender"].map(gender_map).fillna("unknown")

# Convert age to numeric
customers["age"] = pd.to_numeric(customers["age"], errors="coerce")

# Treat ages outside 10-100 as missing
customers.loc[
    ~customers["age"].between(10, 100),
    "age"
] = pd.NA

# Fill missing ages using city median
customers["age"] = customers.groupby("city")["age"].transform(
    lambda x: x.fillna(x.median())
)

# Fill any remaining missing ages with overall median
customers["age"] = customers["age"].fillna(customers["age"].median())

# Convert signup date
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)

# -------------------------------
# 2. CLEAN PRODUCT DATA
# -------------------------------

products["price_inr"] = pd.to_numeric(
    products["price_inr"], errors="coerce"
)

# -------------------------------
# 3. CLEAN ORDER DATA
# -------------------------------

# Convert order timestamp
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)

# Remove exact duplicate rows
orders = orders.drop_duplicates()

# Keep latest record for each order ID
orders = orders.sort_values("order_ts")
orders = orders.drop_duplicates(
    subset="order_id", keep="last"
)

# Convert numeric columns
orders["quantity"] = pd.to_numeric(
    orders["quantity"], errors="coerce"
)

orders["discount_pct"] = pd.to_numeric(
    orders["discount_pct"], errors="coerce"
)

orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

# Correct negative delivery times
orders.loc[
    orders["delivery_minutes"] < 0,
    "delivery_minutes"
] = pd.NA

# Calculate missing order amounts using product price
orders = orders.merge(
    products[["product_id", "price_inr"]],
    on="product_id",
    how="left"
)

calculated_amount = (
    orders["price_inr"]
    * orders["quantity"]
    * (1 - orders["discount_pct"] / 100)
)

orders["order_amount"] = orders["order_amount"].fillna(
    calculated_amount
)

orders = orders.drop(columns=["price_inr"])

# Fill missing delivery time for completed orders
completed_mask = orders["status"].str.lower().eq("completed")

category_data = orders.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

category_medians = category_data.loc[
    completed_mask
].groupby("category")["delivery_minutes"].median()

category_lookup = category_data["category"].map(category_medians)

missing_completed = (
    completed_mask & orders["delivery_minutes"].isna()
)

orders.loc[missing_completed, "delivery_minutes"] = (
    category_lookup[missing_completed]
)

# -------------------------------
# 4. SAVE CLEANED DATA
# -------------------------------

customers.to_csv(CLEAN_DIR / "customers.csv", index=False)
products.to_csv(CLEAN_DIR / "products.csv", index=False)
orders.to_csv(CLEAN_DIR / "orders.csv", index=False)

print("DATA CLEANING COMPLETED")
print("Customers:", customers.shape)
print("Products:", products.shape)
print("Orders:", orders.shape)

print("\nOrder status counts:")
print(orders["status"].value_counts(dropna=False))

print("\nMissing completed order amounts:",
      orders.loc[completed_mask, "order_amount"].isna().sum())

print("\nMissing completed delivery times:",
      orders.loc[completed_mask, "delivery_minutes"].isna().sum())

print("\nMissing cancelled delivery times:",
      orders.loc[
          orders["status"].str.lower().eq("cancelled"),
          "delivery_minutes"
      ].isna().sum())
# -------------------------------
# 5. SAVE DATA INTO SQLITE
# -------------------------------

DB_PATH = PROJECT_DIR / "data" / "instamart.db"

with sqlite3.connect(DB_PATH) as conn:
    customers.to_sql("customers", conn, if_exists="replace", index=False)
    products.to_sql("products", conn, if_exists="replace", index=False)
    orders.to_sql("orders", conn, if_exists="replace", index=False)

print("\nSQLite database created:", DB_PATH)