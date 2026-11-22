"""Create and audit cycle-level train/validation/test partitions."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import FEATURE_DIR, RANDOM_SEED, RESULT_DIR, ensure_directories


def infer_contiguous_blocks(metadata: pd.DataFrame) -> np.ndarray:
    """Infer runs where any experimental condition changes.

    UCI does not publish an explicit experiment/session ID. This conservative proxy
    groups adjacent cycles with identical five-column profiles. It is used for a
    robustness audit, not claimed to be a true acquisition-session identifier.
    """
    conditions = metadata[["cooler_condition", "valve_condition", "pump_leakage", "accumulator_pressure", "stable_flag"]]
    changed = conditions.ne(conditions.shift()).any(axis=1)
    return changed.cumsum().to_numpy(dtype=int)


def create_split(force: bool = False) -> pd.DataFrame:
    ensure_directories()
    output_path = DATA_DIR = FEATURE_DIR.parent / "split_assignments.csv"
    if output_path.exists() and not force:
        split = pd.read_csv(output_path)
        audit_split(split)
        print(f"Using existing split: {output_path}")
        return split

    metadata = pd.read_csv(FEATURE_DIR / "cycle_metadata.csv")
    ids = metadata["cycle_id"].to_numpy()
    y = metadata["pump_leakage"].to_numpy()
    train_ids, remainder_ids, y_train, y_remainder = train_test_split(
        ids, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
    )
    val_ids, test_ids = train_test_split(
        remainder_ids, test_size=0.50, random_state=RANDOM_SEED + 1, stratify=y_remainder
    )
    assignment = {int(cycle): "train" for cycle in train_ids}
    assignment.update({int(cycle): "validation" for cycle in val_ids})
    assignment.update({int(cycle): "test" for cycle in test_ids})
    split = metadata[["cycle_id", "pump_leakage"]].copy()
    split["split"] = split["cycle_id"].map(assignment)
    split["inferred_block_id"] = infer_contiguous_blocks(metadata)
    split.to_csv(output_path, index=False)
    audit_split(split)
    return split


def audit_split(split: pd.DataFrame) -> None:
    if split["cycle_id"].duplicated().any():
        raise AssertionError("A cycle ID appears more than once")
    if split["split"].isna().any() or set(split["split"]) != {"train", "validation", "test"}:
        raise AssertionError("Invalid split assignments")
    counts = split.groupby(["split", "pump_leakage"]).size().unstack(fill_value=0)
    if (counts == 0).any().any():
        raise AssertionError("At least one split is missing a pump class")
    expected_ids = set(range(1, 2206))
    if set(split["cycle_id"]) != expected_ids:
        raise AssertionError("Split does not contain exactly cycle IDs 1..2205")
    overlap = {}
    sets = {name: set(group["cycle_id"]) for name, group in split.groupby("split")}
    for left in sets:
        for right in sets:
            if left < right:
                overlap[f"{left}_{right}"] = len(sets[left] & sets[right])
    if any(overlap.values()):
        raise AssertionError(f"Split overlap detected: {overlap}")
    report = {
        "seed": RANDOM_SEED,
        "method": "cycle-level stratified random split (70/15/15)",
        "counts": split.groupby("split").size().to_dict(),
        "class_counts": {k: {str(kk): int(vv) for kk, vv in v.items()} for k, v in counts.to_dict("index").items()},
        "overlap_counts": overlap,
        "inferred_blocks": int(split["inferred_block_id"].nunique()) if "inferred_block_id" in split else None,
        "block_note": "Explicit experiment IDs are unavailable. Contiguous identical-profile runs are retained for group-CV robustness auditing.",
    }
    (RESULT_DIR / "split_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    create_split(force=args.force)


if __name__ == "__main__":
    main()
