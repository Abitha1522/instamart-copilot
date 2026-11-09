
from pathlib import Path
import sqlite3
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
CLEAN_DIR = PROJECT_DIR / "data" / "clean"
DB_PATH = PROJECT_DIR / "data" / "instamart.db"

customers = pd.read_csv(CLEAN_DIR / "customers.csv")
products = pd.read_csv(CLEAN_DIR / "products.csv")
orders = pd.read_csv(CLEAN_DIR / "orders.csv")

checks = []

def check(description, condition):
    passed = bool(condition)
    checks.append((description, passed))
    print(("PASS" if passed else "FAIL") + " - " + description)

# Customer validation
check("Customers has 1200 rows", len(customers) == 1200)
check("Customers has 6 cities", customers["city"].nunique() == 6)
check("Customers has no missing ages", customers["age"].isna().sum() == 0)
check(
    "Gender counts are correct",
    customers["gender"].value_counts().to_dict()
    == {"female": 610, "male": 560, "unknown": 30}
)
check(
    "Customer ages are between 18 and 60",
    customers["age"].between(18, 60).all()
)

# Product validation
check("Products has 40 rows", len(products) == 40)
check(
    "Product prices have no missing values",
    products["price_inr"].isna().sum() == 0
)

# Order validation
check("Orders has 12309 rows", len(orders) == 12309)
check("No pending orders remain", not orders["status"].eq("pending").any())
check(
    "Completed orders count is 11110",
    orders["status"].eq("completed").sum() == 11110
)
check(
    "Cancelled orders count is 1199",
    orders["status"].eq("cancelled").sum() == 1199
)
check(
    "Order amounts have no missing values",
    orders["order_amount"].isna().sum() == 0
)

completed = orders[orders["status"] == "completed"]
cancelled = orders[orders["status"] == "cancelled"]

check(
    "Completed orders have no missing delivery times",
    completed["delivery_minutes"].isna().sum() == 0
)
check(
    "Cancelled orders have 1199 missing delivery times",
    cancelled["delivery_minutes"].isna().sum() == 1199
)

revenue = completed["order_amount"].sum()
check(
    "Completed revenue is INR 2998668.80",
    abs(revenue - 2998668.80) < 0.01
)

# SQLite database validation
with sqlite3.connect(DB_PATH) as conn:
    tables = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }

    check(
        "SQLite has all three tables",
        {"customers", "products", "orders"}.issubset(tables)
    )

    check(
        "SQLite customers has 1200 rows",
        conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 1200
    )
    check(
        "SQLite products has 40 rows",
        conn.execute("SELECT COUNT(*) FROM products").fetchone()[0] == 40
    )
    check(
        "SQLite orders has 12309 rows",
        conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 12309
    )

# Final report
failed = [description for description, passed in checks if not passed]

print(f"\nChecks passed: {len(checks) - len(failed)}/{len(checks)}")

if failed:
    print("FAILED CHECKS:")
    for item in failed:
        print("-", item)
    raise SystemExit(1)

print("ALL CHECKPOINTS PASSED")
