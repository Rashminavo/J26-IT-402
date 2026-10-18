"""Build a window-level wearable dataset aligned to PhysioNet task ratings.

This script assigns each fully-contained 15-second feature window the numeric
self-reported rating for its task. It deliberately does NOT convert ratings to
Low/Moderate/High: those exploratory cutoffs must be learned from the training
participants inside each cross-validation split.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

RAW_ROOT = Path("data/raw/physionet")
FEATURE_PATH = Path("data/processed/f_stress_15s_window_features.csv")
TASK_LEVEL_PATH = Path("data/processed/f_stress_task_level_dataset.csv")
OUTPUT_PATH = Path("data/processed/f_stress_15s_labeled_windows.csv")
TASKS = [
    "Baseline",
    "TMCT",
    "First Rest",
    "Real Opinion",
    "Opposite Opinion",
    "Second Rest",
    "Subtract",
]
META_COLUMNS = {
    "participant_id", "session_id", "window_start_seconds", "window_end_seconds"
}


def _read_timestamps(path: Path) -> list[datetime]:
    values = [
        line.strip().strip('"')
        for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        if line.strip()
    ]
    return [datetime.fromisoformat(value) for value in values]


def main() -> None:
    for path in (FEATURE_PATH, TASK_LEVEL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    rating_files = list(RAW_ROOT.rglob("Stress_Level_v2.csv"))
    stress_roots = [p for p in RAW_ROOT.rglob("STRESS") if p.is_dir()]
    if len(rating_files) != 1 or len(stress_roots) != 1:
        raise RuntimeError(
            "Expected exactly one Stress_Level_v2.csv and one STRESS directory; "
            f"found {len(rating_files)} ratings files and {len(stress_roots)} STRESS directories."
        )

    features = pd.read_csv(FEATURE_PATH)
    task_level = pd.read_csv(TASK_LEVEL_PATH)
    ratings = pd.read_csv(rating_files[0])
    ratings.columns = [str(c).strip() for c in ratings.columns]
    id_column = ratings.columns[0]
    ratings[id_column] = ratings[id_column].astype(str).str.strip()
    ratings = ratings.set_index(id_column)
    if ratings.index.duplicated().any():
        raise ValueError("Duplicate participant IDs in Stress_Level_v2.csv")
    missing_tasks = set(TASKS) - set(ratings.columns)
    if missing_tasks:
        raise ValueError(f"Rating CSV is missing tasks: {sorted(missing_tasks)}")

    missing_feature_meta = META_COLUMNS - set(features.columns)
    if missing_feature_meta:
        raise ValueError(f"Feature CSV missing columns: {sorted(missing_feature_meta)}")

    features["participant_id"] = features["participant_id"].astype(str)
    features["session_id"] = features["session_id"].astype(str)
    feature_columns = [c for c in features.columns if c not in META_COLUMNS]
    if not feature_columns:
        raise ValueError("No physiological features found in input CSV")

    session_dirs = sorted(
        {
            p.parent
            for p in stress_roots[0].rglob("BVP.csv")
            if p.parent.name.startswith("f")
        },
        key=lambda p: p.name,
    )
    output_rows: list[dict] = []

    for folder in session_dirs:
        session_id = folder.name
        participant_id = session_id.split("_")[0]
        if participant_id not in ratings.index:
            raise ValueError(f"No rating row for {participant_id} (session {session_id})")

        session_windows = features[
            (features["participant_id"] == participant_id)
            & (features["session_id"] == session_id)
        ].copy()
        if session_windows.empty:
            raise ValueError(f"No feature windows found for session {session_id}")
        rating_row = ratings.loc[participant_id]

        # The dataset notes specify that f14_a contains the separate baseline.
        if session_id == "f14_a":
            baseline = float(rating_row["Baseline"])
            for _, window in session_windows.iterrows():
                output_rows.append({
                    **window.to_dict(),
                    "task": "Baseline",
                    "self_reported_stress_rating": baseline,
                })
            continue

        tag_path = folder / "tags.csv"
        tags = _read_timestamps(tag_path)
        if len(tags) < 9:
            raise ValueError(
                f"{session_id}: expected at least 9 tag timestamps; found {len(tags)}"
            )
        with (folder / "BVP.csv").open(encoding="utf-8-sig") as handle:
            recording_start = datetime.fromisoformat(handle.readline().strip())

        def seconds(index: int) -> float:
            return (tags[index] - recording_start).total_seconds()

        intervals: dict[str, tuple[float, float]] = {}
        # f14 baseline is represented by f14_a, not the f14_b recording.
        if session_id != "f14_b":
            intervals["Baseline"] = (0.0, seconds(0))
        intervals.update({
            "TMCT": (seconds(1), seconds(2)),
            "First Rest": (seconds(2), seconds(3)),
            "Real Opinion": (seconds(3), seconds(4)),
            "Opposite Opinion": (seconds(5), seconds(6)),
            "Second Rest": (seconds(6), seconds(7)),
            "Subtract": (seconds(7), seconds(8)),
        })

        for task, (start, end) in intervals.items():
            if task not in TASKS or end <= start:
                raise ValueError(f"Invalid task interval: {session_id}/{task}: {start}..{end}")
            rating_value = rating_row[task]
            if pd.isna(rating_value):
                raise ValueError(f"Missing rating: {participant_id}/{task}")
            # Full window must be inside task interval; boundary-crossing windows are skipped.
            mask = (
                (session_windows["window_start_seconds"] >= start - 1e-8)
                & (session_windows["window_end_seconds"] <= end + 1e-8)
            )
            selected = session_windows.loc[mask]
            if selected.empty:
                raise ValueError(f"No fully-contained windows for {session_id}/{task}")
            for _, window in selected.iterrows():
                output_rows.append({
                    **window.to_dict(),
                    "task": task,
                    "self_reported_stress_rating": float(rating_value),
                })

    result = pd.DataFrame(output_rows)
    expected_columns = [
        "participant_id", "session_id", "window_start_seconds", "window_end_seconds",
        "task", "self_reported_stress_rating", *feature_columns,
    ]
    result = result[expected_columns].sort_values(
        ["participant_id", "session_id", "window_start_seconds"]
    ).reset_index(drop=True)

    key_columns = ["participant_id", "session_id", "window_start_seconds"]
    if result.duplicated(key_columns).any():
        duplicate_rows = result.loc[result.duplicated(key_columns, keep=False), key_columns]
        raise ValueError("Some feature windows received multiple task labels:\n" + duplicate_rows.to_string(index=False))
    if result["self_reported_stress_rating"].isna().any():
        raise ValueError("Missing numeric ratings after alignment")
    numeric = result[feature_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Feature data contains NaN or infinite values")

    # Match the previously audited task-level dataset's window counts per task record.
    key = ["participant_id", "session_id", "task"]
    observed_counts = result.groupby(key).size().rename("observed_window_count").reset_index()
    expected_counts = task_level[key + ["window_count"]].copy()
    expected_counts["participant_id"] = expected_counts["participant_id"].astype(str)
    expected_counts["session_id"] = expected_counts["session_id"].astype(str)
    count_check = expected_counts.merge(observed_counts, on=key, how="outer", indicator=True)
    if not (count_check["_merge"] == "both").all():
        raise ValueError("Participant/task records do not match task-level dataset:\n" + count_check[count_check["_merge"] != "both"].to_string(index=False))
    mismatched = count_check[count_check["window_count"] != count_check["observed_window_count"]]
    if not mismatched.empty:
        raise ValueError("Window counts differ from audited task-level dataset:\n" + mismatched.to_string(index=False))

    participant_task_counts = result.groupby("participant_id")["task"].nunique()
    if len(participant_task_counts) != 18 or not (participant_task_counts == 7).all():
        raise ValueError("Expected 18 participants with 7 task ratings each")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print("=== WINDOW-LEVEL RATING ALIGNMENT DATASET CREATED ===")
    print("Output:", OUTPUT_PATH)
    print("Rows/windows:", len(result))
    print("Participants:", result["participant_id"].nunique())
    print("Sessions:", result["session_id"].nunique())
    print("Task records:", result[key].drop_duplicates().shape[0])
    print("Physiological features:", len(feature_columns))
    print("Duplicate windows:", int(result.duplicated(key_columns).sum()))
    print("Missing ratings:", int(result["self_reported_stress_rating"].isna().sum()))
    print("Missing feature values:", int(result[feature_columns].isna().sum().sum()))
    print("Non-finite feature values:", int((~np.isfinite(result[feature_columns].to_numpy(dtype=float))).sum()))
    print("\n=== WINDOW COUNTS BY TASK ===")
    print(result.groupby("task").agg(
        windows=("task", "size"),
        participants=("participant_id", "nunique"),
        min_rating=("self_reported_stress_rating", "min"),
        max_rating=("self_reported_stress_rating", "max"),
    ).reindex(TASKS).to_string())
    print("\nNOTE: Labels are numeric self-reported task ratings, not Low/Moderate/High classes.")
    print("Class cutoffs must be calculated within each training split during model evaluation.")
    print("No model was trained; raw/source files were not modified.")


if __name__ == "__main__":
    main()
