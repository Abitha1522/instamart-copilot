
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

ORDERS_PATH = ROOT / "data" / "clean" / "orders.csv"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AS_OF_DATE = pd.Timestamp("2025-10-01")


# ==================================================
# 2. LOAD DATA
# ==================================================

orders = pd.read_csv(ORDERS_PATH)

print("=" * 55)
print("B1 - RFM ANALYSIS AND CUSTOMER SEGMENTATION")
print("=" * 55)

print("\nOrders dataset shape:", orders.shape)


# ==================================================
# 3. FILTER COMPLETED ORDERS
# ==================================================

orders["order_ts"] = pd.to_datetime(
    orders["order_ts"],
    errors="coerce"
)

completed = orders[
    orders["status"] == "completed"
].copy()

print("Completed orders:", len(completed))


# ==================================================
# 4. CALCULATE RFM VALUES
# ==================================================

rfm = completed.groupby("customer_id").agg(
    last_order_date=("order_ts", "max"),
    frequency=("order_id", "count"),
    monetary=("order_amount", "sum")
).reset_index()

# Recency = days since the customer's last order
rfm["recency"] = (
    AS_OF_DATE - rfm["last_order_date"]
).dt.days

# Keep customers with valid RFM values
rfm = rfm.dropna(
    subset=["recency", "frequency", "monetary"]
).copy()

rfm["recency"] = rfm["recency"].astype(int)

print("\nRFM dataset shape:", rfm.shape)


# ==================================================
# 5. RFM SUMMARY
# ==================================================

print("\nRFM SUMMARY")
print("-" * 55)

print(
    rfm[["recency", "frequency", "monetary"]]
    .describe()
    .round(2)
)


# Save RFM dataset
rfm.to_csv(
    OUTPUT_DIR / "b1_rfm.csv",
    index=False
)


# ==================================================
# 6. FEATURE SCALING
# ==================================================

features = ["recency", "frequency", "monetary"]

scaler = StandardScaler()

scaled_features = scaler.fit_transform(
    rfm[features]
)

print("\nFeature scaling completed.")


# ==================================================
# 7. FIND THE BEST NUMBER OF CLUSTERS
# ==================================================

print("\nK-MEANS CLUSTER EVALUATION")
print("-" * 55)

metrics = []

for k in range(2, 9):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(scaled_features)

    inertia = model.inertia_

    silhouette = silhouette_score(
        scaled_features,
        labels
    )

    metrics.append({
        "k": k,
        "inertia": inertia,
        "silhouette": silhouette
    })

    print(
        f"K={k} | "
        f"Inertia={inertia:.1f} | "
        f"Silhouette={silhouette:.3f}"
    )

metrics_df = pd.DataFrame(metrics)

metrics_df.to_csv(
    OUTPUT_DIR / "b1_kmeans_metrics.csv",
    index=False
)


# ==================================================
# 8. FINAL K-MEANS MODEL (K=4)
# ==================================================

final_model = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

rfm["cluster"] = final_model.fit_predict(
    scaled_features
)

print("\nFinal K-Means model trained with K=4.")


# ==================================================
# 9. CUSTOMER SEGMENT SUMMARY
# ==================================================

segment_summary = rfm.groupby("cluster").agg(
    customer_count=("customer_id", "count"),
    average_recency=("recency", "mean"),
    average_frequency=("frequency", "mean"),
    average_monetary=("monetary", "mean")
).reset_index()

print("\nCUSTOMER SEGMENT SUMMARY")
print("-" * 55)

print(segment_summary.round(2).to_string(index=False))

segment_summary.to_csv(
    OUTPUT_DIR / "b1_segment_summary.csv",
    index=False
)

rfm.to_csv(
    OUTPUT_DIR / "b1_rfm_clustered.csv",
    index=False
)


# ==================================================
# 10. PCA FOR DIMENSIONALITY REDUCTION
# ==================================================

pca = PCA(n_components=2)

pca_values = pca.fit_transform(
    scaled_features
)

pca_df = pd.DataFrame({
    "customer_id": rfm["customer_id"],
    "PC1": pca_values[:, 0],
    "PC2": pca_values[:, 1],
    "cluster": rfm["cluster"]
})

pca_df.to_csv(
    OUTPUT_DIR / "b1_pca.csv",
    index=False
)

print("\nPCA EXPLAINED VARIANCE")
print(
    "PC1:",
    round(pca.explained_variance_ratio_[0], 3)
)

print(
    "PC2:",
    round(pca.explained_variance_ratio_[1], 3)
)

print(
    "Total:",
    round(pca.explained_variance_ratio_.sum(), 3)
)


# ==================================================
# 11. VALIDATION CHECKPOINTS
# ==================================================

print("\nVALIDATION CHECKS")
print("-" * 55)

checks = []


def check(description, condition):
    passed = bool(condition)
    checks.append(passed)

    print(
        f"{'PASS' if passed else 'FAIL'} - {description}"
    )


check(
    "RFM contains 1175 customers",
    len(rfm) == 1175
)

check(
    "Median recency matches 19 days",
    rfm["recency"].median() == 19
)

check(
    "Mean frequency matches 9.46",
    abs(rfm["frequency"].mean() - 9.46) < 0.02
)

check(
    "Mean monetary value matches INR 2552.06",
    abs(rfm["monetary"].mean() - 2552.06) < 0.02
)

check(
    "Four customer clusters were created",
    rfm["cluster"].nunique() == 4
)

check(
    "PCA output contains two components",
    pca_df[["PC1", "PC2"]].shape[1] == 2
)

print("\n" + "=" * 55)

print(
    f"Checks passed: {sum(checks)}/{len(checks)}"
)

if all(checks):
    print("ALL B1 CHECKPOINTS PASSED")
else:
    print("SOME B1 CHECKPOINTS FAILED")

print("=" * 55)
