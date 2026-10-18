"""CLI: predict exploratory classes from raw Empatica-style sensor CSVs."""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physiological_analysis.raw_window_inference import predict_session_windows  # noqa: E402


def find_default_session() -> Path:
    raw_root = ROOT / "data" / "raw" / "physionet"
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw PhysioNet data root not found: {raw_root}")
    matches = [
        path for path in raw_root.rglob("f01")
        if path.is_dir()
        and path.parent.name.upper() == "STRESS"
        and all((path / name).is_file() for name in ("HR.csv", "EDA.csv", "BVP.csv", "TEMP.csv", "ACC.csv"))
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one STRESS/f01 session with required signals; found {len(matches)}: {matches}"
        )
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate 15-second wearable features and exploratory SVC predictions."
    )
    parser.add_argument("--session-dir", type=Path, default=None,
                        help="Folder containing HR.csv, EDA.csv, BVP.csv, TEMP.csv and ACC.csv.")
    parser.add_argument("--model", type=Path, default=None,
                        help="Optional path to the saved window-level SVC .joblib bundle.")
    parser.add_argument("--output", type=Path, default=None,
                        help="CSV output path; defaults to data/processed/model_comparison/raw_session_predictions.csv.")
    args = parser.parse_args()

    session_dir = args.session_dir or find_default_session()
    if not session_dir.is_absolute():
        session_dir = (Path.cwd() / session_dir).resolve()
    output_path = args.output or (
        ROOT / "data" / "processed" / "model_comparison" / "raw_session_predictions.csv"
    )
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()

    rows = predict_session_windows(session_dir=session_dir, model_path=args.model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["predicted_class"] for row in rows)
    summary = getattr(predict_session_windows, "last_summary", {})
    print("=== RAW WEARABLE -> FEATURES -> SVC PREDICTIONS ===")
    print("Session:", session_dir)
    print("Model:", rows[0]["model_name"])
    print("Window size: 15 seconds | stride: 7.5 seconds")
    print("Candidate common windows:", summary.get("candidate_common_windows"))
    print("Incomplete windows rejected:", summary.get("rejected_incomplete_windows"))
    print("Windows predicted:", len(rows))
    print("Prediction counts:", dict(counts))
    print("Output CSV:", output_path)
    print("Research prototype only; categories are exploratory self-report bands.")
    print("Clinical validation:", rows[0]["clinical_validation"])
    print("No raw/source files were modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
