# src/clustering/lifecycle/assign_lifecycle.py

import os
import sys
import pandas as pd

# =========================
# Path setup
# =========================
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, root_dir)

from src.config.settings import (
    LIFECYCLE_FEATURES_PATH,
    RESULTS_DIR
)

RESULTS_DIR = os.path.join(root_dir, RESULTS_DIR)
os.makedirs(RESULTS_DIR, exist_ok=True)


class LifecycleAssigner:
    """
    Gán lifecycle stage cho user dựa trên ngưỡng tự động từ dữ liệu
    """

    REQUIRED_COLS = {"user_id", "total_orders", "active_days", "active_span"}

    def __init__(self, thresholds: dict):
        self.thresholds = thresholds

    @staticmethod
    def compute_thresholds(df: pd.DataFrame) -> dict:
        """
        Tính ngưỡng lifecycle dựa trên phân vị dữ liệu
        """
        return {
            "orders": df["total_orders"].quantile([0.4, 0.7, 0.9]).to_dict(),
            "days": df["active_days"].quantile([0.5, 0.8]).to_dict(),
            "span": df["active_span"].quantile([0.7]).to_dict()
        }

    def assign_stage(self, row):
        """
        Lifecycle assignment (data-driven)
        """
        t = self.thresholds

        if (
            row["total_orders"] >= t["orders"][0.9]
            and row["active_span"] >= t["span"][0.7]
        ):
            return "vip"

        if (
            row["total_orders"] >= t["orders"][0.7]
            and row["active_days"] >= t["days"][0.8]
        ):
            return "loyal"

        if row["total_orders"] >= t["orders"][0.4]:
            return "regular"

        return "new"

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"Missing lifecycle features: {missing}")

        df = df.copy()
        df["lifecycle_stage"] = df.apply(self.assign_stage, axis=1)
        return df


def assign_lifecycle(save_results: bool = True):
    """
    Run lifecycle assignment
    + Auto thresholding
    + Summary
    + Save results
    """

    # =========================
    # 1. Load data
    # =========================
    df = pd.read_csv(LIFECYCLE_FEATURES_PATH)

    # =========================
    # 2. Compute thresholds
    # =========================
    thresholds = LifecycleAssigner.compute_thresholds(df)

    print("\nLIFECYCLE THRESHOLDS (AUTO)\n")
    for k, v in thresholds.items():
        print(f"{k}: {v}")

    # =========================
    # 3. Assign lifecycle
    # =========================
    assigner = LifecycleAssigner(thresholds)
    df = assigner.run(df)

    # =========================
    # 4. Summary
    # =========================
    lifecycle_summary = (
        df["lifecycle_stage"]
        .value_counts()
        .rename_axis("lifecycle_stage")
        .reset_index(name="num_users")
    )

    # =========================
    # 5. Print
    # =========================
    print("\nBẢNG PHÂN BỐ LIFECYCLE STAGE\n")
    print(lifecycle_summary.to_string(index=False))

    # =========================
    # 6. Save results
    # =========================
    if save_results:
        lifecycle_dir = os.path.join(RESULTS_DIR, "lifecycle")
        os.makedirs(lifecycle_dir, exist_ok=True)

        # CSV: assignments
        df[["user_id", "lifecycle_stage"]].to_csv(
            os.path.join(lifecycle_dir, "lifecycle_assignments.csv"),
            index=False
        )

        # CSV: summary
        lifecycle_summary.to_csv(
            os.path.join(lifecycle_dir, "lifecycle_summary.csv"),
            index=False
        )

        # TXT report
        report_path = os.path.join(lifecycle_dir, "lifecycle_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("LIFECYCLE THRESHOLDS (AUTO)\n\n")
            for k, v in thresholds.items():
                f.write(f"{k}: {v}\n")
            f.write("\nBẢNG PHÂN BỐ LIFECYCLE STAGE\n\n")
            f.write(lifecycle_summary.to_string(index=False))

        print(f"\nLifecycle results saved to: {lifecycle_dir}")

    return df, lifecycle_summary


if __name__ == "__main__":
    assign_lifecycle()
