"""Integrity checks for generated generalization-audit artifacts."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import RESULT_DIR


def main() -> None:
    audit_dir = RESULT_DIR / "generalization_audit"
    required = [
        "dataset_profile_summary.csv",
        "fault_combination_counts.csv",
        "block_summary.csv",
        "leakage_audit.md",
        "validation_comparison.csv",
        "validation_summary.csv",
        "confusion_matrices.csv",
        "condition_holdout_plan.csv",
        "GENERALIZATION_AUDIT_REPORT.md",
        "execution_log.txt",
    ]
    missing = [name for name in required if not (audit_dir / name).exists()]
    assert not missing, f"Missing audit artifacts: {missing}"

    comparison = pd.read_csv(audit_dir / "validation_comparison.csv")
    confusion = pd.read_csv(audit_dir / "confusion_matrices.csv")
    combinations = pd.read_csv(audit_dir / "fault_combination_counts.csv")
    blocks = pd.read_csv(audit_dir / "block_summary.csv")
    holdout = pd.read_csv(audit_dir / "condition_holdout_plan.csv")

    assert len(comparison) == 3 * 2 * 2 * 5
    assert set(comparison["scheme"]) == {"random_cycle", "block_aware", "condition_holdout"}
    assert set(comparison["observation_seconds"]) == {5, 60}
    assert set(comparison["model"]) == {"logistic_regression", "random_forest"}
    assert comparison.groupby(["scheme", "observation_seconds", "model"])["fold"].nunique().eq(5).all()

    metric_columns = ["accuracy", "macro_f1", "normal_recall", "weak_recall", "severe_recall"]
    assert comparison[metric_columns].apply(lambda column: column.between(0, 1).all()).all()
    assert comparison.query("scheme == 'block_aware'")["shared_blocks"].eq(0).all()
    assert comparison.query("scheme == 'condition_holdout'")["shared_conditions"].eq(0).all()

    confusion_sums = (
        confusion.groupby(["scheme", "observation_seconds", "model", "fold"])["count"]
        .sum()
        .rename("confusion_count")
        .reset_index()
    )
    checked = comparison.merge(confusion_sums, on=["scheme", "observation_seconds", "model", "fold"], validate="one_to_one")
    assert checked["evaluation_cycles"].eq(checked["confusion_count"]).all()

    assert len(combinations) == 144
    assert combinations["cycle_count"].sum() == 2205
    assert len(blocks) == 194
    assert blocks["block_size"].sum() == 2205
    assert holdout["condition_id"].nunique() == 144
    assert holdout.groupby("condition_id")["fold"].nunique().eq(1).all()
    assert holdout["cycles"].sum() == 2205

    confusion_images = list(audit_dir.glob("confusion_*.png"))
    assert len(confusion_images) == 12
    print(
        "PASS: 60 model-fold results, 144 condition groups, 194 inferred blocks, "
        "group leakage checks, confusion totals, and 12 pooled confusion figures verified."
    )


if __name__ == "__main__":
    main()
