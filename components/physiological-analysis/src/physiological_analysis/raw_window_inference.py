"""End-to-end raw wearable CSV -> 15-second features -> exploratory prediction.

Research prototype only. Predictions are relative categories derived from
self-reported task ratings, not clinically validated stress severity classes.
"""
from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Protocol

from .data_loader import load_heart_rate_csv, load_signal_csv
from .features import extract_heart_rate_features, extract_signal_features
from .window_stress_classifier import WindowLevelStressClassifier
from .windowing import segment_signal_into_windows, validate_window_sample_count

WINDOW_SIZE_SECONDS = 15.0
WINDOW_STRIDE_SECONDS = 7.5
SIGNAL_NAMES = ("eda", "bvp", "temp", "acc")
REQUIRED_FILES = ("HR.csv", "EDA.csv", "BVP.csv", "TEMP.csv", "ACC.csv")


class WindowClassifier(Protocol):
    feature_columns: tuple[str, ...]

    def predict(self, features: Mapping[str, Any]) -> dict[str, Any]: ...


def _read_heart_rate_metadata(path: Path) -> tuple[datetime, float]:
    """Read HR start time and sampling rate without changing the loader API."""
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 3 or not rows[0] or not rows[1]:
        raise ValueError(f"Invalid HR CSV metadata: {path}")
    try:
        start = datetime.fromisoformat(rows[0][0].strip())
        rate = float(rows[1][0].strip())
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Could not parse HR timestamp/rate: {path}") from exc
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError(f"Invalid HR sampling rate: {rate}")
    return start, rate


def _make_window_map(
    readings: list[Any], rate: float, offset: float, timeline_duration: float
) -> dict[float, dict[str, object]]:
    windows = segment_signal_into_windows(
        readings=readings,
        sampling_rate=rate,
        signal_start_offset_seconds=offset,
        window_size_seconds=WINDOW_SIZE_SECONDS,
        stride_seconds=WINDOW_STRIDE_SECONDS,
        timeline_duration_seconds=timeline_duration,
    )
    return {round(float(item["start_seconds"]), 8): item for item in windows}


def predict_session_windows(
    session_dir: str | Path,
    model_path: str | Path | None = None,
    classifier: WindowClassifier | None = None,
) -> list[dict[str, Any]]:
    """Generate complete windows and predict each with the saved window-level model.

    ``session_dir`` must contain HR.csv, EDA.csv, BVP.csv, TEMP.csv and ACC.csv
    in the same Empatica-style format used by the research dataset.
    BVP's timestamp is the shared timeline origin, matching the parity audit.
    Incomplete windows are excluded if any required signal lacks the expected
    number of samples.

    ``classifier`` is injectable for unit tests; normal use loads the persisted
    WindowLevelStressClassifier from ``model_path`` or its configured default.
    """
    directory = Path(session_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Wearable session folder not found: {directory}")

    missing_files = [name for name in REQUIRED_FILES if not (directory / name).is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"Session folder is missing required sensor files: {', '.join(missing_files)}"
        )

    if classifier is None:
        classifier = WindowLevelStressClassifier.load(model_path)
    elif model_path is not None:
        raise ValueError("Pass either classifier or model_path, not both.")

    feature_columns = tuple(classifier.feature_columns)
    if not feature_columns:
        raise ValueError("Classifier has no feature schema.")

    signals = {
        name: load_signal_csv(directory / f"{name.upper()}.csv")
        for name in SIGNAL_NAMES
    }
    hr_path = directory / "HR.csv"
    hr_start, hr_rate = _read_heart_rate_metadata(hr_path)
    hr_readings = load_heart_rate_csv(hr_path)

    # Training feature generation used BVP time as the session-relative origin.
    reference_start = datetime.fromisoformat(str(signals["bvp"]["timestamp"]))
    signal_starts = {"hr": hr_start}
    signal_starts.update({
        name: datetime.fromisoformat(str(signal["timestamp"]))
        for name, signal in signals.items()
    })
    offsets = {
        name: (timestamp - reference_start).total_seconds()
        for name, timestamp in signal_starts.items()
    }
    for name, offset in offsets.items():
        if offset < -1e-8:
            raise ValueError(
                f"{name.upper()} starts before the BVP reference timestamp; "
                "shared-timeline alignment requires review."
            )

    # ACC axes must be synchronously sampled for vector-magnitude features.
    acc_rates = [float(value) for value in signals["acc"]["sampling_rates"]]
    if len(acc_rates) != 3 or not all(
        math.isclose(rate, acc_rates[0], rel_tol=0.0, abs_tol=1e-9)
        for rate in acc_rates
    ):
        raise ValueError("ACC must contain three axes with matching sampling rates.")

    stream_ends = {
        "hr": offsets["hr"] + len(hr_readings) / hr_rate,
    }
    for name, signal in signals.items():
        rate = float(signal["sampling_rates"][0])
        stream_ends[name] = offsets[name] + int(signal["sample_count"]) / rate
    timeline_duration = max(stream_ends.values())

    windows_by_signal: dict[str, dict[float, dict[str, object]]] = {
        "hr": _make_window_map(hr_readings, hr_rate, offsets["hr"], timeline_duration)
    }
    for name, signal in signals.items():
        windows_by_signal[name] = _make_window_map(
            signal["readings"],
            float(signal["sampling_rates"][0]),
            offsets[name],
            timeline_duration,
        )

    required_streams = ("hr", "eda", "bvp", "temp", "acc")
    common_starts = set.intersection(*[
        set(windows_by_signal[name]) for name in required_streams
    ])

    rows: list[dict[str, Any]] = []
    rejected_incomplete = 0
    for start in sorted(common_starts):
        selected = {
            name: windows_by_signal[name][start]
            for name in required_streams
        }
        complete = True
        for name, item in selected.items():
            rate = hr_rate if name == "hr" else float(signals[name]["sampling_rates"][0])
            if not validate_window_sample_count(item, rate, WINDOW_SIZE_SECONDS):
                complete = False
                break
        if not complete:
            rejected_incomplete += 1
            continue

        features: dict[str, float] = extract_heart_rate_features(
            selected["hr"]["readings"]  # type: ignore[arg-type]
        )
        for name in SIGNAL_NAMES:
            original = signals[name]
            window_signal = {
                "timestamp": original["timestamp"],
                "sampling_rates": original["sampling_rates"],
                "readings": selected[name]["readings"],
                "sample_count": selected[name]["sample_count"],
                "channel_count": original["channel_count"],
            }
            features.update(extract_signal_features(name, window_signal))

        missing_features = [name for name in feature_columns if name not in features]
        extra_features = [name for name in features if name not in feature_columns]
        if missing_features or extra_features:
            raise ValueError(
                "Extracted feature schema does not match the saved model. "
                f"Missing: {missing_features}; unexpected: {extra_features}"
            )

        prediction = classifier.predict(features)
        row: dict[str, Any] = {
            "participant_id": directory.name.split("_")[0],
            "session_id": directory.name,
            "window_start_seconds": float(start),
            "window_end_seconds": float(start + WINDOW_SIZE_SECONDS),
            **{name: float(features[name]) for name in feature_columns},
            "predicted_class": prediction["predicted_class"],
            "class_id": prediction["class_id"],
            "model_name": prediction["model_name"],
            "research_use_only": prediction["research_use_only"],
            "clinical_validation": prediction["clinical_validation"],
            "warning": prediction["warning"],
        }
        rows.append(row)

    if not rows:
        raise ValueError(
            "No complete 15-second windows could be predicted from this session."
        )

    # Small internal summary used by the CLI, without modifying source data.
    predict_session_windows.last_summary = {  # type: ignore[attr-defined]
        "session_dir": str(directory),
        "timeline_duration_seconds": timeline_duration,
        "candidate_common_windows": len(common_starts),
        "rejected_incomplete_windows": rejected_incomplete,
        "predicted_windows": len(rows),
    }
    return rows
