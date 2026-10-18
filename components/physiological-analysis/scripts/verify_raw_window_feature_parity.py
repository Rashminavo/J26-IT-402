from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Make the repository's src package importable when this script is run from
# the physiological-analysis component directory.
sys.path.insert(0, str(Path("src").resolve()))

from physiological_analysis.data_loader import (  # noqa: E402
    load_heart_rate_csv,
    load_signal_csv,
)
from physiological_analysis.features import (  # noqa: E402
    extract_heart_rate_features,
    extract_signal_features,
)
from physiological_analysis.windowing import (  # noqa: E402
    segment_signal_into_windows,
    validate_window_sample_count,
)

DATA_ROOT = Path("data/raw/physionet")
FEATURES_PATH = Path("data/processed/f_stress_15s_window_features.csv")
WINDOW_SIZE_SECONDS = 15.0
STRIDE_SECONDS = 7.5
SESSION_ID = "f01"


def read_hr_metadata(path: Path) -> tuple[datetime, float]:
    """Read HR timestamp and sampling rate while keeping the existing loader API."""
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 3 or not rows[0] or not rows[1]:
        raise ValueError(f"Invalid HR CSV metadata: {path}")
    try:
        start = datetime.fromisoformat(rows[0][0].strip())
        rate = float(rows[1][0].strip())
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Could not parse HR timestamp/rate: {path}") from exc
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError(f"Invalid HR sampling rate: {rate}")
    return start, rate


def find_f01_folder() -> Path:
    """Find f01 specifically under the STRESS protocol, not AEROBIC/ANAEROBIC."""
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Raw data root does not exist: {DATA_ROOT}")
    candidates = [
        path for path in DATA_ROOT.rglob(SESSION_ID)
        if (
            path.is_dir()
            and path.name == SESSION_ID
            and path.parent.name.upper() == "STRESS"
            and (path / "HR.csv").is_file()
            and (path / "BVP.csv").is_file()
        )
    ]
    if len(candidates) != 1:
        found = [str(path) for path in candidates]
        raise RuntimeError(
            f"Expected exactly one STRESS/{SESSION_ID} session folder, "
            f"found {len(candidates)}: {found}"
        )
    return candidates[0]


def window_map(readings, rate: float, offset: float, duration: float):
    windows = segment_signal_into_windows(
        readings=readings,
        sampling_rate=rate,
        signal_start_offset_seconds=offset,
        window_size_seconds=WINDOW_SIZE_SECONDS,
        stride_seconds=STRIDE_SECONDS,
        timeline_duration_seconds=duration,
    )
    return {round(float(item["start_seconds"]), 8): item for item in windows}


def main() -> int:
    if not FEATURES_PATH.is_file():
        raise FileNotFoundError(f"Window feature dataset not found: {FEATURES_PATH}")

    session_dir = find_f01_folder()
    print("=== RAW-TO-WINDOW FEATURE PARITY AUDIT ===")
    print("Raw session:", session_dir)
    print("Reference feature CSV:", FEATURES_PATH)
    print(f"Window size: {WINDOW_SIZE_SECONDS:g}s | stride: {STRIDE_SECONDS:g}s")

    loaded_signals = {
        name: load_signal_csv(session_dir / f"{name.upper()}.csv")
        for name in ("eda", "bvp", "temp", "acc")
    }
    hr_path = session_dir / "HR.csv"
    hr_start, hr_rate = read_hr_metadata(hr_path)
    hr_readings = load_heart_rate_csv(hr_path)

    # BVP is the training dataset's session-relative timeline reference.
    reference_start = datetime.fromisoformat(str(loaded_signals["bvp"]["timestamp"]))
    starts = {"hr": hr_start}
    starts.update({
        name: datetime.fromisoformat(str(signal["timestamp"]))
        for name, signal in loaded_signals.items()
    })
    offsets = {
        name: (timestamp - reference_start).total_seconds()
        for name, timestamp in starts.items()
    }

    print("\n=== SIGNAL TIMING / RATE ===")
    for name in ("hr", "eda", "bvp", "temp", "acc"):
        if name == "hr":
            rates = [hr_rate]
            counts = len(hr_readings)
        else:
            rates = loaded_signals[name]["sampling_rates"]
            counts = loaded_signals[name]["sample_count"]
        print(f"{name.upper():4s} offset={offsets[name]:8.3f}s rates={rates} samples={counts}")
        if offsets[name] < -1e-8:
            raise ValueError(
                f"{name.upper()} starts before BVP reference; timeline handling needs review"
            )

    # Use each stream's actual coverage on the common BVP-relative timeline.
    stream_ends = {"hr": offsets["hr"] + len(hr_readings) / hr_rate}
    for name, signal in loaded_signals.items():
        rates = signal["sampling_rates"]
        reading_rate = float(rates[0])
        stream_ends[name] = offsets[name] + int(signal["sample_count"]) / reading_rate
    timeline_duration = max(stream_ends.values())

    windows_by_signal = {
        "hr": window_map(hr_readings, hr_rate, offsets["hr"], timeline_duration)
    }
    for name, signal in loaded_signals.items():
        windows_by_signal[name] = window_map(
            signal["readings"],
            float(signal["sampling_rates"][0]),
            offsets[name],
            timeline_duration,
        )

    common_starts = set.intersection(*[
        set(windows_by_signal[name])
        for name in ("hr", "eda", "bvp", "temp", "acc")
    ])
    complete_starts: list[float] = []
    generated_rows: list[dict[str, object]] = []
    incomplete_reasons: dict[str, int] = {}

    for start in sorted(common_starts):
        selected = {name: windows_by_signal[name][start]
                    for name in ("hr", "eda", "bvp", "temp", "acc")}
        valid = True
        for name, item in selected.items():
            rate = hr_rate if name == "hr" else float(loaded_signals[name]["sampling_rates"][0])
            if not validate_window_sample_count(item, rate, WINDOW_SIZE_SECONDS):
                incomplete_reasons[name] = incomplete_reasons.get(name, 0) + 1
                valid = False
        if not valid:
            continue

        feature_row: dict[str, object] = {
            "participant_id": SESSION_ID,
            "session_id": SESSION_ID,
            "window_start_seconds": start,
            "window_end_seconds": start + WINDOW_SIZE_SECONDS,
        }
        feature_row.update(extract_heart_rate_features(selected["hr"]["readings"]))
        for name in ("eda", "bvp", "temp", "acc"):
            original = loaded_signals[name]
            window_signal = {
                "timestamp": original["timestamp"],
                "sampling_rates": original["sampling_rates"],
                "readings": selected[name]["readings"],
                "sample_count": selected[name]["sample_count"],
                "channel_count": original["channel_count"],
            }
            feature_row.update(extract_signal_features(name, window_signal))
        generated_rows.append(feature_row)
        complete_starts.append(start)

    generated = pd.DataFrame(generated_rows).sort_values("window_start_seconds").reset_index(drop=True)
    reference = pd.read_csv(FEATURES_PATH)
    reference = reference[
        (reference["participant_id"].astype(str) == SESSION_ID)
        & (reference["session_id"].astype(str) == SESSION_ID)
    ].sort_values("window_start_seconds").reset_index(drop=True)

    metadata_cols = {
        "participant_id", "session_id", "window_start_seconds", "window_end_seconds"
    }
    generated_feature_cols = [c for c in generated.columns if c not in metadata_cols]
    reference_feature_cols = [c for c in reference.columns if c not in metadata_cols]
    missing_generated = sorted(set(reference_feature_cols) - set(generated_feature_cols))
    extra_generated = sorted(set(generated_feature_cols) - set(reference_feature_cols))

    print("\n=== WINDOW COUNTS ===")
    print("Reference f01 windows:", len(reference))
    print("Raw-generated complete windows:", len(generated))
    print("Common candidate starts before completeness check:", len(common_starts))
    print("Timeline duration:", round(timeline_duration, 3), "seconds")
    if incomplete_reasons:
        print("Incomplete windows rejected by signal:", incomplete_reasons)

    print("\n=== FEATURE SCHEMA ===")
    print("Reference feature columns:", len(reference_feature_cols))
    print("Generated feature columns:", len(generated_feature_cols))
    print("Missing generated feature columns:", missing_generated)
    print("Unexpected generated feature columns:", extra_generated)

    if len(reference) == len(generated) and "window_start_seconds" in reference and "window_start_seconds" in generated:
        starts_match = np.allclose(
            reference["window_start_seconds"].to_numpy(dtype=float),
            generated["window_start_seconds"].to_numpy(dtype=float),
            rtol=0,
            atol=1e-8,
        )
    else:
        starts_match = False
    print("Window start timestamps match:", starts_match)

    common_features = sorted(set(reference_feature_cols) & set(generated_feature_cols))
    if len(reference) == len(generated) and common_features:
        diffs = {}
        for column in common_features:
            ref_values = reference[column].to_numpy(dtype=float)
            gen_values = generated[column].to_numpy(dtype=float)
            diffs[column] = float(np.max(np.abs(ref_values - gen_values)))
        diff_series = pd.Series(diffs).sort_values(ascending=False)
        close_by_column = all(
            np.allclose(
                reference[column].to_numpy(dtype=float),
                generated[column].to_numpy(dtype=float),
                rtol=1e-6,
                atol=1e-6,
            )
            for column in common_features
        )
        print("All common feature values match within 1e-6:", close_by_column)
        print("Largest absolute differences (top 10):")
        print(diff_series.head(10).to_string())
    else:
        close_by_column = False
        print("Feature value comparison skipped because row counts/features differ.")

    passed = (
        starts_match
        and not missing_generated
        and not extra_generated
        and close_by_column
        and len(reference) > 0
    )
    print("\n=== RESULT ===")
    print("PASS: raw feature extraction matches saved training features." if passed else
          "REVIEW NEEDED: raw feature extraction does not yet match saved training features.")
    print("No files were written or modified.")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
