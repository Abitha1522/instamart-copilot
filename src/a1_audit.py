
from pathlib import Path
import pandas as pd

# Find the project folder
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"

# Load the three raw datasets
customers = pd.read_csv(RAW_DIR / "customers.csv")
products = pd.read_csv(RAW_DIR / "products.csv")
orders = pd.read_csv(RAW_DIR / "orders.csv")


# Function to audit each dataset
def audit_table(name, df):
    print("\n" + "=" * 60)
    print(f"AUDIT: {name.upper()}")
    print("=" * 60)

    # Number of rows and columns
    print("\nShape (rows, columns):")
    print(df.shape)

    # Data types
    print("\nColumn data types:")
    print(df.dtypes)

    # Missing values and percentages
    missing = df.isna().sum()
    percentage = (missing / len(df) * 100).round(2)

    missing_report = pd.DataFrame({
        "Missing Count": missing,
        "Missing Percentage": percentage
    })

    print("\nMissing values:")
    print(missing_report)


# Audit all three datasets
audit_table("Customers", customers)
audit_table("Products", products)
audit_table("Orders", orders)


# Additional raw data audit figures
print("\n" + "=" * 60)
print("IMPORTANT AUDIT FIGURES")
print("=" * 60)

# Orders audit
numeric_amount = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

numeric_delivery = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

non_numeric_amount = (
    orders["order_amount"].notna().sum()
    - numeric_amount.notna().sum()
)

print("\nORDERS")
print("Total rows:", len(orders))
print("Exact duplicate rows:", orders.duplicated().sum())
print("Pending rows:", (orders["status"] == "pending").sum())
print("Distinct order IDs:", orders["order_id"].nunique())
print("Non-numeric order amounts:", non_numeric_amount)
print("Negative delivery minutes:", (numeric_delivery < 0).sum())
print("Blank delivery minutes:", orders["delivery_minutes"].isna().sum())

# Customers audit
numeric_age = pd.to_numeric(customers["age"], errors="coerce")

print("\nCUSTOMERS")
print("Blank ages:", customers["age"].isna().sum())
print(
    "Ages outside 10-100:",
    ((numeric_age < 10) | (numeric_age > 100)).sum()
)
print("Raw city spellings:", customers["city"].nunique())
print(
    "Raw non-blank gender spellings:",
    customers["gender"].dropna().nunique()
)
print("Blank gender values:", customers["gender"].isna().sum())

print("\nAUDIT COMPLETED")
