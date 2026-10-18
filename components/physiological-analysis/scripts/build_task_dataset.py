from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


RAW_ROOT = Path("data/raw/physionet")
FEATURE_PATH = Path("data/processed/f_stress_15s_window_features.csv")
OUTPUT_PATH = Path("data/processed/f_stress_task_level_dataset.csv")

TASK_NAMES = [
    "Baseline",
    "TMCT",
    "First Rest",
    "Real Opinion",
    "Opposite Opinion",
    "Second Rest",
    "Subtract",
]

METADATA_COLUMNS = {
    "participant_id",
    "session_id",
    "window_start_seconds",
    "window_end_seconds",
}


def main():
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"Feature dataset not found: {FEATURE_PATH}")

    ratings_matches = list(RAW_ROOT.rglob("Stress_Level_v2.csv"))
    stress_matches = [
        path for path in RAW_ROOT.rglob("STRESS") if path.is_dir()
    ]

    if len(ratings_matches) != 1:
        raise RuntimeError(
            f"Expected one Stress_Level_v2.csv, found {len(ratings_matches)}"
        )
    if len(stress_matches) != 1:
        raise RuntimeError(
            f"Expected one STRESS folder, found {len(stress_matches)}"
        )

    features = pd.read_csv(FEATURE_PATH)
    ratings = pd.read_csv(ratings_matches[0])

    required_feature_columns = METADATA_COLUMNS
    missing = required_feature_columns - set(features.columns)
    if missing:
        raise ValueError(f"Missing required feature columns: {sorted(missing)}")

    ratings.columns = [str(column).strip() for column in ratings.columns]
    id_column = ratings.columns[0]
    ratings[id_column] = ratings[id_column].astype(str).str.strip()
    ratings = ratings.set_index(id_column)

    if ratings.index.duplicated().any():
        raise ValueError("Duplicate participant IDs in ratings file.")

    missing_tasks = set(TASK_NAMES) - set(ratings.columns)
    if missing_tasks:
        raise ValueError(f"Missing rating tasks: {sorted(missing_tasks)}")

    features["participant_id"] = features["participant_id"].astype(str)
    features["session_id"] = features["session_id"].astype(str)

    feature_columns = [
        column for column in features.columns
        if column not in METADATA_COLUMNS
    ]

    if not feature_columns:
        raise ValueError("No model feature columns found.")

    session_dirs = sorted({
        path.parent
        for path in stress_matches[0].rglob("BVP.csv")
        if path.parent.name.startswith("f")
    }, key=lambda path: path.name)

    if not session_dirs:
        raise FileNotFoundError("No f* participant session folders found.")

    output_rows = []
    total_windows_used = 0

    for session_dir in session_dirs:
        session_id = session_dir.name
        participant_id = session_id.split("_")[0]

        if participant_id not in ratings.index:
            raise ValueError(
                f"No rating row for participant {participant_id}"
            )

        session_windows = features[
            (features["participant_id"] == participant_id)
            & (features["session_id"] == session_id)
        ].copy()

        if session_windows.empty:
            raise ValueError(
                f"No feature windows found for session {session_id}"
            )

        rating_row = ratings.loc[participant_id]

        # f14_a is documented as the separate baseline segment.
        if session_id == "f14_a":
            feature_windows = session_windows
            start_seconds = float(
                feature_windows["window_start_seconds"].min()
            )
            end_seconds = float(
                feature_windows["window_end_seconds"].max()
            )

            if pd.isna(rating_row["Baseline"]):
                raise ValueError("Missing f14 baseline rating.")

            tasks_for_session = [(
                "Baseline",
                float(rating_row["Baseline"]),
                feature_windows,
                start_seconds,
                end_seconds,
            )]

        else:
            tags_path = session_dir / "tags.csv"
            tag_strings = [
                line.strip().strip('"')
                for line in tags_path.read_text(
                    encoding="utf-8-sig",
                    errors="replace",
                ).splitlines()
                if line.strip()
            ]

            if len(tag_strings) < 9:
                raise ValueError(
                    f"{session_id}: expected at least 9 tag timestamps; "
                    f"found {len(tag_strings)}"
                )

            tags = [
                datetime.fromisoformat(value)
                for value in tag_strings
            ]

            with (session_dir / "BVP.csv").open(
                encoding="utf-8-sig"
            ) as handle:
                recording_start = datetime.fromisoformat(
                    handle.readline().strip()
                )

            def seconds_from_start(index):
                return (tags[index] - recording_start).total_seconds()

            intervals = {}

            # Baseline for f14 is recorded in f14_a, not f14_b.
            if session_id != "f14_b":
                intervals["Baseline"] = (0.0, seconds_from_start(0))

            intervals.update({
                "TMCT": (
                    seconds_from_start(1),
                    seconds_from_start(2),
                ),
                "First Rest": (
                    seconds_from_start(2),
                    seconds_from_start(3),
                ),
                "Real Opinion": (
                    seconds_from_start(3),
                    seconds_from_start(4),
                ),
                "Opposite Opinion": (
                    seconds_from_start(5),
                    seconds_from_start(6),
                ),
                "Second Rest": (
                    seconds_from_start(6),
                    seconds_from_start(7),
                ),
                "Subtract": (
                    seconds_from_start(7),
                    seconds_from_start(8),
                ),
            })

            tasks_for_session = []

            for task, (start_seconds, end_seconds) in intervals.items():
                if task not in TASK_NAMES:
                    continue

                rating = rating_row[task]
                if pd.isna(rating):
                    raise ValueError(
                        f"Missing rating: {participant_id}, {task}"
                    )

                if end_seconds <= start_seconds:
                    raise ValueError(
                        f"Invalid time interval: {session_id}, {task}"
                    )

                # Include only complete windows inside the task interval.
                mask = (
                    (session_windows["window_start_seconds"]
                     >= start_seconds - 1e-8)
                    & (session_windows["window_end_seconds"]
                       <= end_seconds + 1e-8)
                )
                selected = session_windows.loc[mask].copy()

                if selected.empty:
                    raise ValueError(
                        f"No complete feature windows for {session_id}, "
                        f"{task} ({start_seconds:.2f}s to "
                        f"{end_seconds:.2f}s)"
                    )

                tasks_for_session.append((
                    task,
                    float(rating),
                    selected,
                    float(start_seconds),
                    float(end_seconds),
                ))

        for task, rating, selected, start_seconds, end_seconds in tasks_for_session:
            numeric_features = selected[feature_columns]

            if numeric_features.isna().any().any():
                raise ValueError(
                    f"Missing feature values: {session_id}, {task}"
                )

            values = numeric_features.to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(
                    f"Non-finite feature values: {session_id}, {task}"
                )

            # One sample per participant-task: average its contained windows.
            feature_summary = numeric_features.mean().to_dict()

            row = {
                "participant_id": participant_id,
                "session_id": session_id,
                "task": task,
                "self_reported_stress_rating": rating,
                "window_count": len(selected),
                "task_start_seconds": start_seconds,
                "task_end_seconds": end_seconds,
                "task_duration_seconds": end_seconds - start_seconds,
                **feature_summary,
            }

            output_rows.append(row)
            total_windows_used += len(selected)

    dataset = pd.DataFrame(output_rows)

    if dataset.empty:
        raise RuntimeError("No task-level samples were generated.")

    if dataset.duplicated(["participant_id", "task"]).any():
        duplicated = dataset.loc[
            dataset.duplicated(["participant_id", "task"], keep=False),
            ["participant_id", "task", "session_id"],
        ]
        raise ValueError(
            "Duplicate participant-task records found:\n"
            + duplicated.to_string(index=False)
        )

    participant_task_counts = dataset.groupby("participant_id")["task"].nunique()
    if len(participant_task_counts) != 18:
        raise ValueError(
            f"Expected 18 participants, found {len(participant_task_counts)}"
        )
    if not (participant_task_counts == len(TASK_NAMES)).all():
        raise ValueError(
            "Each participant should have one record for each of 7 tasks:\n"
            + participant_task_counts.to_string()
        )

    if set(dataset["task"]) != set(TASK_NAMES):
        raise ValueError("The generated dataset does not contain all tasks.")

    model_features = dataset[feature_columns].to_numpy(dtype=float)
    if not np.isfinite(model_features).all():
        raise ValueError("Generated task-level features contain invalid values.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)

    print("\n=== TASK-LEVEL DATASET CREATED ===")
    print("Output:", OUTPUT_PATH)
    print("Task-level samples:", len(dataset))
    print("Participants:", dataset["participant_id"].nunique())
    print("Tasks:", dataset["task"].nunique())
    print("Physiological feature columns:", len(feature_columns))
    print("Candidate windows aggregated:", total_windows_used)
    print(
        "Rating range:",
        dataset["self_reported_stress_rating"].min(),
        "to",
        dataset["self_reported_stress_rating"].max(),
    )

    print("\n=== TASK SAMPLE COUNTS ===")
    print(
        dataset.groupby("task").agg(
            samples=("participant_id", "size"),
            windows_aggregated=("window_count", "sum"),
            minimum_rating=("self_reported_stress_rating", "min"),
            maximum_rating=("self_reported_stress_rating", "max"),
        ).to_string()
    )

    print("\n=== VALIDATION ===")
    print(
        "Duplicate participant-task pairs:",
        int(dataset.duplicated(["participant_id", "task"]).sum()),
    )
    print("Missing feature values:", int(dataset[feature_columns].isna().sum().sum()))
    print(
        "Non-finite feature values:",
        int((~np.isfinite(dataset[feature_columns].to_numpy(dtype=float))).sum()),
    )
    print("\nNo model trained. Ratings are still continuous self-reported scores.")


if __name__ == "__main__":
    main()
