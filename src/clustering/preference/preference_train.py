# src/clustering/preference/preference_train.py

import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import defaultdict


# =========================
# Path setup
# =========================
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, root_dir)

from src.clustering.preference.preference_vectorizer import PreferenceVectorizer
from src.config.settings import (
    PREFERENCE_FEATURES_PATH,
    PREFERENCE_CLUSTER_MODEL_PATH,
    PREFERENCE_CLUSTER_REPORT_PATH,
    PREFERENCE_CLUSTER_TABLE_PATH,
    PREFERENCE_CLUSTER_SCORE_PATH,
    RESULTS_DIR,
    N_PREFERENCE_CLUSTERS,
    RANDOM_STATE
)

RESULTS_DIR = os.path.join(root_dir, RESULTS_DIR)
os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================
# Helper: diễn giải cluster
# =========================
def interpret_preference_cluster(row, top_k=3):
    ignore_cols = {"cluster", "num_users"}
    feature_cols = [c for c in row.index if c not in ignore_cols]

    top_categories = (
        row[feature_cols]
        .sort_values(ascending=False)
        .head(top_k)
        .index.tolist()
    )

    return ", ".join(top_categories)


def train_preference_clustering(save_assignments: bool = True):
    """
    Preference-based user clustering
    - Pivot preference data (user × department)
    - L1 normalize
    - KMeans clustering
    - Summary + interpretation
    """

    # =========================
    # 1. Load preference data (LONG format)
    # =========================
    df_raw = pd.read_csv(PREFERENCE_FEATURES_PATH)

    required_cols = {"user_id", "department", "preference_score"}
    if not required_cols.issubset(df_raw.columns):
        raise ValueError(f"Preference data must contain {required_cols}")

    df_raw = (
        df_raw
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    # =========================
    # 2. Pivot → USER × DEPARTMENT (WIDE format)
    # =========================
    df = (
        df_raw
        .pivot_table(
            index="user_id",
            columns="department",
            values="preference_score",
            aggfunc="mean",
            fill_value=0.0
        )
        .reset_index()
    )

    if len(df) < N_PREFERENCE_CLUSTERS:
        raise ValueError(
            f"Number of users ({len(df)}) must be >= number of clusters ({N_PREFERENCE_CLUSTERS})"
        )

    feature_cols = [c for c in df.columns if c != "user_id"]

    # =========================
    # 3. Vectorize (L1 normalize)
    # =========================
    vectorizer = PreferenceVectorizer(norm="l1")
    X = vectorizer.fit_transform(df)

    # =========================
    # 4. Train KMeans
    # =========================
    model = KMeans(
        n_clusters=N_PREFERENCE_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=10
    )

    cluster_labels = model.fit_predict(X)

    # =========================
    # 5. Metrics
    # =========================
    silhouette = silhouette_score(X, cluster_labels)
    inertia = model.inertia_

    # =========================
    # 6. Save MODEL (checkpoints chỉ lưu model)
    # =========================
    os.makedirs(os.path.dirname(PREFERENCE_CLUSTER_MODEL_PATH), exist_ok=True)
    joblib.dump(model, PREFERENCE_CLUSTER_MODEL_PATH)
    vectorizer.save()

    # =========================
    # 7. Save assignments (RESULTS)
    # =========================
    df_assign = pd.DataFrame({
        "user_id": df["user_id"],
        "cluster": cluster_labels
    })

    if save_assignments:
        df_assign.to_csv(
            os.path.join(RESULTS_DIR, "preference/preference_cluster_assignments.csv"),
            index=False
        )

    # =========================
    # 8. CLUSTER SUMMARY
    # =========================
    df_clustered = df.copy()
    df_clustered["cluster"] = cluster_labels

    cluster_summary = (
        df_clustered
        .groupby("cluster")[feature_cols]
        .mean()
        .round(4)
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

    cluster_summary["Top preference categories"] = cluster_summary.apply(
        interpret_preference_cluster,
        axis=1
    )

    cluster_summary.to_csv(
        PREFERENCE_CLUSTER_TABLE_PATH,
        index=False
    )
    
        # =========================
    # 9. BUILD CLUSTER → DEPARTMENT SCORE MAP (FOR INFERENCE)
    # =========================
    # Input: df_raw (user_id, department, preference_score)
    #        df_assign (user_id, cluster)

    df_with_cluster = df_raw.merge(
        df_assign,
        on="user_id",
        how="inner"
    )

    cluster_department_scores = defaultdict(dict)

    for cluster_id, group in df_with_cluster.groupby("cluster"):
        dept_scores = (
            group
            .groupby("department")["preference_score"]
            .mean()
            .sort_values(ascending=False)
            .to_dict()
        )
        cluster_department_scores[int(cluster_id)] = {
            str(dept): float(score)
            for dept, score in dept_scores.items()
        }

    # Save to pkl (CHECKPOINT)
    from src.config.settings import (
        PREFERENCE_CLUSTER_SCORE_PATH
    )

    os.makedirs(
        os.path.dirname(PREFERENCE_CLUSTER_SCORE_PATH),
        exist_ok=True
    )

    joblib.dump(
        dict(cluster_department_scores),
        PREFERENCE_CLUSTER_SCORE_PATH
    )

    print(
        f"\n Preference cluster → department scores saved to:\n"
        f" {PREFERENCE_CLUSTER_SCORE_PATH}"
    )


    # =========================
    # 10. PRINT
    # =========================
    print("\nBẢNG DIỄN GIẢI PREFERENCE CLUSTERS\n")
    print(cluster_summary.to_string(index=False))

    print("\nClustering metrics:")
    print(f"- Number of clusters : {N_PREFERENCE_CLUSTERS}")
    print(f"- Silhouette score   : {silhouette:.4f}")
    print(f"- Inertia            : {inertia:.2f}")

    # =========================
    # 11. SAVE TXT REPORT
    # =========================
    report_path = PREFERENCE_CLUSTER_REPORT_PATH

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("BẢNG DIỄN GIẢI PREFERENCE CLUSTERS\n\n")
        f.write(cluster_summary.to_string(index=False))
        f.write("\n\nClustering metrics:\n")
        f.write(f"Number of clusters : {N_PREFERENCE_CLUSTERS}\n")
        f.write(f"Silhouette score   : {silhouette:.4f}\n")
        f.write(f"Inertia            : {inertia:.2f}\n")

    print(f"\n Preference cluster report saved to: {report_path}")

    return model, silhouette, cluster_summary


if __name__ == "__main__":
    train_preference_clustering()
