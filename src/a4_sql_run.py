
from pathlib import Path
import sqlite3
import re
import pandas as pd

# Find the project root directory
ROOT = Path(__file__).resolve().parent.parent

# File paths
DB_PATH = ROOT / "data" / "instamart.db"
SQL_PATH = ROOT / "sql" / "queries.sql"


def main():

    # Check whether required files exist
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    if not SQL_PATH.exists():
        raise FileNotFoundError(f"SQL file not found: {SQL_PATH}")

    # Read SQL queries
    sql_text = SQL_PATH.read_text(encoding="utf-8")

    # Extract query labels and SQL statements
    labels = re.findall(r"(?m)^-- (Q\d+):.*$", sql_text)
    blocks = re.split(r"(?m)^-- Q\d+:.*$", sql_text)

    queries = {
        label: query.strip()
        for label, query in zip(labels, blocks[1:])
        if query.strip()
    }

    print("INSTAMART SQL ANALYSIS")
    print("=" * 50)

    # Connect to SQLite database
    with sqlite3.connect(DB_PATH) as connection:

        results = {}

        for label, query in queries.items():

            print(f"\n{label}")
            print("-" * 40)

            df = pd.read_sql_query(query, connection)
            results[label] = df

            print("Rows:", len(df))
            print(df.head(5).to_string(index=False))

        # Validate query results
        print("\nSQL VALIDATION")
        print("=" * 50)

        checks = []

        def check(description, condition):
            status = bool(condition)
            checks.append(status)
            print(
                f"{'PASS' if status else 'FAIL'} - {description}"
            )

        q1 = results["Q1"]
        q2 = results["Q2"]
        q3 = results["Q3"]
        q4 = results["Q4"]
        q5 = results["Q5"]
        q6 = results["Q6"]
        q7 = results["Q7"]

        # Q1
        check("Q1 returns 30 customers", len(q1) == 30)

        # Q2
        check("Q2 returns 30 city-category rows", len(q2) == 30)

        q2_total = q2["total_revenue"].sum()
        check(
            "Q2 total revenue matches INR 2998668.80",
            abs(q2_total - 2998668.80) < 0.01
        )

        max_row = q2.loc[q2["total_revenue"].idxmax()]
        check(
            "Q2 highest revenue is Bengaluru/Electronics",
            max_row["city"] == "Bengaluru"
            and max_row["category"] == "Electronics"
            and abs(max_row["total_revenue"] - 233909.45) < 0.01
        )

        # Q3
        check("Q3 returns 11 products", len(q3) == 11)

        # Q4
        check("Q4 returns 25 customers", len(q4) == 25)

        # Q5
        check("Q5 returns 550 customers", len(q5) == 550)

        average = q5["average_spend"].iloc[0]
        check(
            "Q5 average spend matches INR 2552.06",
            abs(average - 2552.06) < 0.02
        )

        # Q6
        mismatch_count = (
            q6["rank_num"] != q6["row_num"]
        ).sum()

        check(
            "Q6 has 8 rows where RANK differs from ROW_NUMBER",
            mismatch_count == 8
        )

        print("\nTop customer in each city:")
        print(
            q6[q6["row_num"] == 1][
                ["city", "customer_id", "total_spend"]
            ].to_string(index=False)
        )

        # Q7
        check("Q7 returns 183 days", len(q7) == 183)

        final_total = q7["cumulative_revenue"].iloc[-1]

        check(
            "Q7 final cumulative revenue matches INR 2998668.80",
            abs(final_total - 2998668.80) < 0.01
        )

        # Final summary
        print("\n" + "=" * 50)
        print(f"Checks passed: {sum(checks)}/{len(checks)}")

        if all(checks):
            print("ALL SQL CHECKPOINTS PASSED")
        else:
            print("SOME SQL CHECKPOINTS FAILED")


if __name__ == "__main__":
    main()
