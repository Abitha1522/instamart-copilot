
from pathlib import Path
import pandas as pd

# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

ORDERS_PATH = ROOT / "data" / "clean" / "orders.csv"
CUSTOMERS_PATH = ROOT / "data" / "clean" / "customers.csv"
PRODUCTS_PATH = ROOT / "data" / "clean" / "products.csv"


# ==================================================
# 2. LOAD CLEANED DATASETS
# ==================================================

orders = pd.read_csv(ORDERS_PATH)
customers = pd.read_csv(CUSTOMERS_PATH)
products = pd.read_csv(PRODUCTS_PATH)

print("=" * 55)
print("A5 - PANDAS ANALYSIS")
print("=" * 55)

print("\nDataset shapes:")
print("Orders:", orders.shape)
print("Customers:", customers.shape)
print("Products:", products.shape)


# ==================================================
# 3. FILTER COMPLETED ORDERS
# ==================================================

completed = orders[
    orders["status"] == "completed"
].copy()


# ==================================================
# 4. MERGE CUSTOMER AND PRODUCT INFORMATION
# ==================================================

# Add customer city
completed = completed.merge(
    customers[["customer_id", "city"]],
    on="customer_id",
    how="left"
)

# Add product category
completed = completed.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

print("\nMerged columns:")
print(completed.columns.tolist())

print(
    "Missing categories:",
    completed["category"].isna().sum()
)


# ==================================================
# 5. REVENUE BY CITY AND CATEGORY
# ==================================================

pivot = pd.pivot_table(
    completed,
    index="city",
    columns="category",
    values="order_amount",
    aggfunc="sum",
    fill_value=0,
    margins=True,
    margins_name="All"
)

print("\n1. REVENUE BY CITY AND CATEGORY")
print(pivot.round(2))

grand_total = pivot.loc["All", "All"]

print("\nGrand Total Revenue:", round(grand_total, 2))


# ==================================================
# 6. CITY-WISE SUMMARY
# ==================================================

city_summary = completed.groupby("city").agg(
    total_revenue=("order_amount", "sum"),
    average_order_value=("order_amount", "mean"),
    order_count=("order_id", "count")
).reset_index()

print("\n2. CITY-WISE SUMMARY")
print(city_summary.round(2).to_string(index=False))


# ==================================================
# 7. ORDER AMOUNT DISTRIBUTION
# ==================================================

amount = completed["order_amount"]

mean_value = amount.mean()
median_value = amount.median()
skew_value = amount.skew()

q1 = amount.quantile(0.25)
q3 = amount.quantile(0.75)

iqr = q3 - q1

lower_fence = q1 - (1.5 * iqr)
upper_fence = q3 + (1.5 * iqr)

outliers = amount[
    (amount < lower_fence) |
    (amount > upper_fence)
]

print("\n3. ORDER AMOUNT DISTRIBUTION")
print("Mean:", round(mean_value, 2))
print("Median:", round(median_value, 2))
print("Skewness:", round(skew_value, 2))
print("Q1:", round(q1, 2))
print("Q3:", round(q3, 2))
print("IQR:", round(iqr, 2))
print("Lower Fence:", round(lower_fence, 2))
print("Upper Fence:", round(upper_fence, 2))
print("Number of Outliers:", len(outliers))


# ==================================================
# 8. CORRELATION ANALYSIS
# ==================================================

numeric_columns = [
    "order_amount",
    "discount_pct",
    "quantity",
    "delivery_minutes"
]

correlation = completed[numeric_columns].corr()

print("\n4. CORRELATION MATRIX")
print(correlation.round(3))

# Find strongest absolute correlation pair
correlation_pairs = correlation.abs().where(
    ~correlation.abs().eq(1)
).stack()

strongest_pair = correlation_pairs.idxmax()

actual_correlation = correlation.loc[
    strongest_pair[0],
    strongest_pair[1]
]

print("\nStrongest correlation pair:")
print(strongest_pair)
print("Correlation:", round(actual_correlation, 3))


# ==================================================
# 9. VALIDATION CHECKPOINTS
# ==================================================

print("\n5. VALIDATION CHECKS")
print("-" * 55)

checks = []


def check(description, condition):
    passed = bool(condition)
    checks.append(passed)

    if passed:
        print("PASS -", description)
    else:
        print("FAIL -", description)


# Total revenue validation
check(
    "Pandas revenue matches SQL revenue",
    abs(grand_total - 2998668.80) < 0.01
)


# Bengaluru validation
bengaluru = city_summary[
    city_summary["city"] == "Bengaluru"
].iloc[0]

check(
    "Bengaluru revenue matches reference",
    abs(bengaluru["total_revenue"] - 523643.40) < 0.01
)

check(
    "Bengaluru order count matches reference",
    bengaluru["order_count"] == 1894
)


# Mumbai validation
mumbai = city_summary[
    city_summary["city"] == "Mumbai"
].iloc[0]

check(
    "Mumbai revenue matches reference",
    abs(mumbai["total_revenue"] - 479058.50) < 0.01
)

check(
    "Mumbai order count matches reference",
    mumbai["order_count"] == 1613
)


# Outlier validation
check(
    "Upper outlier fence matches reference",
    abs(upper_fence - 645.30) < 0.01
)

check(
    "Outlier count matches reference",
    len(outliers) == 897
)


# ==================================================
# 10. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 55)

print(
    f"Checks passed: {sum(checks)}/{len(checks)}"
)

if all(checks):
    print("ALL PANDAS CHECKPOINTS PASSED")
else:
    print("SOME PANDAS CHECKPOINTS FAILED")

print("=" * 55)
