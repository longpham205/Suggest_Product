# src/clustering/behavior/behavior_train.py
import os
import sys
import joblib
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, root_dir)

from src.clustering.behavior.behavior_vectorizer import BehaviorVectorizer
from src.config.settings import (
    BEHAVIOR_FEATURES_PATH,
    BEHAVIOR_CLUSTER_MODEL_PATH,
    BEHAVIOR_CLUSTER_REPORT_PATH,
    BEHAVIOR_CLUSTER_TABLE_PATH,
    BEHAVIOR_CLUSTER_SCORE_PATH, 
    ORDERS_PATH,
    ORDER_PRIOR_PATH,
    PRODUCTS_PATH,
    RESULTS_DIR,
    N_BEHAVIOR_CLUSTERS,
    RANDOM_STATE
)

# =========================
# Helper: diễn giải cluster
# =========================
def interpret_cluster(row):
    if row["total_orders"] < 7 and row["reorder_ratio"] < 0.3:
        return "Ít đơn, ít mua", "Người dùng không thường xuyên"

    if row["total_orders"] > 35 and row["reorder_ratio"] > 0.7:
        return "Nhiều đơn, mua đều", "Người dùng trung thành"

    if row["total_orders"] > 15 and row["reorder_ratio"] > 0.4:
        return "Hoạt động ổn định", "Nhóm người dùng chủ lực"

    if row["total_orders"] >= 7 and row["reorder_ratio"] < 0.4:
        return "Mức trung bình", "Người dùng phổ thông"

    return "Khá tích cực", "Người dùng tiềm năng"


def train_behavior_clustering(save_assignments: bool = True):
    """
    Train behavior-based user clustering
    + Build behavior_cluster → department_score (for behavior_adjuster)
    """

    # =========================
    # 1. Load behavior features
    # =========================
    df = pd.read_csv(BEHAVIOR_FEATURES_PATH)

    if "user_id" not in df.columns:
        raise ValueError("Missing user_id in behavior features")

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if len(df) < N_BEHAVIOR_CLUSTERS:
        raise ValueError(
            f"Users ({len(df)}) < clusters ({N_BEHAVIOR_CLUSTERS})"
        )

    feature_cols = [c for c in df.columns if c != "user_id"]

    # =========================
    # 2. Vectorize
    # =========================
    vectorizer = BehaviorVectorizer()
    X = vectorizer.fit_transform(df)

    # =========================
    # 3. Train KMeans
    # =========================
    model = KMeans(
        n_clusters=N_BEHAVIOR_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=10
    )

    cluster_labels = model.fit_predict(X)

    silhouette = silhouette_score(X, cluster_labels)
    inertia = model.inertia_

    # =========================
    # 4. Save model
    # =========================
    os.makedirs(BEHAVIOR_CLUSTER_MODEL_PATH.parent, exist_ok=True)
    joblib.dump(model, BEHAVIOR_CLUSTER_MODEL_PATH)
    vectorizer.save()

    # =========================
    # 5. Assign cluster
    # =========================
    df_clustered = df.copy()
    df_clustered["cluster"] = cluster_labels

    os.makedirs(RESULTS_DIR / "behavior", exist_ok=True)

    if save_assignments:
        df_clustered[["user_id", "cluster"]].to_csv(
            RESULTS_DIR / "behavior" / "behavior_cluster_assignments.csv",
            index=False
        )

    # =========================
    # 6. Cluster summary
    # =========================
    cluster_summary = (
        df_clustered
        .groupby("cluster")[feature_cols]
        .mean()
        .round(2)
        .reset_index()
    )

    cluster_counts = (
        pd.Series(cluster_labels)
        .value_counts()
        .sort_index()
        .rename("num_users")
        .reset_index()
        .rename(columns={"index": "cluster"})
    )

    cluster_summary = cluster_summary.merge(cluster_counts, on="cluster")

    interp = cluster_summary.apply(interpret_cluster, axis=1)
    cluster_summary["Đặc trưng chính"] = [i[0] for i in interp]
    cluster_summary["Mô tả"] = [i[1] for i in interp]

    print("\nBẢNG ĐẶC TRƯNG VÀ DIỄN GIẢI CÁC CLUSTER\n")
    print(cluster_summary.to_string(index=False))
    print(f"\nSilhouette: {silhouette:.4f} | Inertia: {inertia:.2f}")

    # ======================================================
    # 7. BUILD behavior_cluster → department_score (🔥 CHUẨN)
    # ======================================================
    print("\n[INFO] Building behavior_cluster → department scores")

    orders = pd.read_csv(ORDERS_PATH)
    prior = pd.read_csv(ORDER_PRIOR_PATH)
    products = pd.read_csv(PRODUCTS_PATH)

    df_orders = (
        prior
        .merge(orders[["order_id", "user_id"]], on="order_id")
        .merge(products[["product_id", "department_id"]], on="product_id")
        .merge(df_clustered[["user_id", "cluster"]], on="user_id")
    )

    cluster_dept_scores = defaultdict(dict)

    grouped = (
        df_orders
        .groupby(["cluster", "department_id"])
        .size()
        .reset_index(name="cnt")
    )

    for cluster_id, g in grouped.groupby("cluster"):
        total = g["cnt"].sum()
        for _, r in g.iterrows():
            cluster_dept_scores[int(cluster_id)][
                int(r["department_id"])
            ] = float(r["cnt"] / total)

    os.makedirs(BEHAVIOR_CLUSTER_SCORE_PATH.parent, exist_ok=True)
    joblib.dump(dict(cluster_dept_scores), BEHAVIOR_CLUSTER_SCORE_PATH)

    print(
        f"[OK] Saved behavior_cluster → department scores:\n"
        f" {BEHAVIOR_CLUSTER_SCORE_PATH}"
    )

    # =========================
    # 8. Save reports
    # =========================
    cluster_summary.to_csv(BEHAVIOR_CLUSTER_TABLE_PATH, index=False)

    with open(BEHAVIOR_CLUSTER_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(cluster_summary.to_string(index=False))
        f.write("\n\n")
        f.write(f"Silhouette: {silhouette:.4f}\n")
        f.write(f"Inertia: {inertia:.2f}\n")

    print(f"\n[DONE] Results saved to {RESULTS_DIR}")

    return model, silhouette, cluster_summary


if __name__ == "__main__":
    train_behavior_clustering()
