"""Batch smoke-test raw STRESS sessions with a saved exploratory window SVC.

This script checks data/pipeline compatibility across sessions. It does not
estimate independent model performance: the saved model was trained on the
same participant set, and outputs remain exploratory self-report bands.
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physiological_analysis.raw_window_inference import predict_session_windows  # noqa: E402
from physiological_analysis.window_stress_classifier import WindowLevelStressClassifier  # noqa: E402

RAW_ROOT = ROOT / "data" / "raw" / "physionet"
MODEL_PATH = (
    ROOT / "data" / "processed" / "model_comparison"
    / "provisional_window_level_svc.joblib"
)
OUTPUT_DIR = ROOT / "data" / "processed" / "model_comparison"
REQUIRED_FILES = ("HR.csv", "EDA.csv", "BVP.csv", "TEMP.csv", "ACC.csv")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if not RAW_ROOT.is_dir():
        raise FileNotFoundError(f"PhysioNet raw data root not found: {RAW_ROOT}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Saved SVC model not found: {MODEL_PATH}")

    stress_roots = sorted(
        (path for path in RAW_ROOT.rglob("STRESS") if path.is_dir()),
        key=lambda path: str(path),
    )
    session_dirs = sorted(
        {
            child.resolve()
            for stress_root in stress_roots
            for child in stress_root.iterdir()
            if child.is_dir() and child.name.lower().startswith("f")
        },
        key=lambda path: path.name,
    )
    if not session_dirs:
        raise RuntimeError(f"No f* STRESS session folders found under {RAW_ROOT}")

    classifier = WindowLevelStressClassifier.load(MODEL_PATH)
    prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    print("=== BATCH RAW STRESS SESSION PIPELINE CHECK ===")
    print("Raw PhysioNet root:", RAW_ROOT)
    print("Saved model:", MODEL_PATH)
    print("Sessions discovered:", len(session_dirs))
    print("Purpose: robustness / pipeline compatibility smoke test only.")
    print("Model performance is not independently evaluated by this batch run.\n")

    for index, session_dir in enumerate(session_dirs, start=1):
        missing_files = [
            filename for filename in REQUIRED_FILES
            if not (session_dir / filename).is_file()
        ]
        print(f"[{index}/{len(session_dirs)}] {session_dir.name} ...", end=" ", flush=True)

        if missing_files:
            message = "Missing required files: " + ", ".join(missing_files)
            summary_rows.append({
                "session_id": session_dir.name,
                "session_path": str(session_dir),
                "status": "REJECTED_MISSING_FILES",
                "candidate_windows": 0,
                "incomplete_windows_rejected": 0,
                "predicted_windows": 0,
                "low_predictions": 0,
                "moderate_predictions": 0,
                "high_predictions": 0,
                "details": message,
            })
            print("REJECTED:", message)
            continue

        try:
            rows = predict_session_windows(session_dir=session_dir, classifier=classifier)
            info = getattr(predict_session_windows, "last_summary", {})
            counts = Counter(str(row["predicted_class"]) for row in rows)
            prediction_rows.extend(rows)
            summary_rows.append({
                "session_id": session_dir.name,
                "session_path": str(session_dir),
                "status": "PASS",
                "candidate_windows": info.get("candidate_common_windows", ""),
                "incomplete_windows_rejected": info.get("rejected_incomplete_windows", ""),
                "predicted_windows": len(rows),
                "low_predictions": counts.get("Low", 0),
                "moderate_predictions": counts.get("Moderate", 0),
                "high_predictions": counts.get("High", 0),
                "details": "",
            })
            print(f"PASS: {len(rows)} windows; predictions={dict(counts)}")
        except Exception as exc:  # retain failures so the whole audit completes
            summary_rows.append({
                "session_id": session_dir.name,
                "session_path": str(session_dir),
                "status": "ERROR",
                "candidate_windows": "",
                "incomplete_windows_rejected": "",
                "predicted_windows": 0,
                "low_predictions": 0,
                "moderate_predictions": 0,
                "high_predictions": 0,
                "details": f"{type(exc).__name__}: {exc}",
            })
            print(f"ERROR: {type(exc).__name__}: {exc}")

    summary_path = OUTPUT_DIR / "raw_stress_batch_session_summary.csv"
    predictions_path = OUTPUT_DIR / "raw_stress_batch_predictions.csv"
    write_csv(summary_path, summary_rows)
    if prediction_rows:
        write_csv(predictions_path, prediction_rows)

    summary_counts = Counter(row["status"] for row in summary_rows)
    print("\n=== BATCH SUMMARY ===")
    print("Sessions examined:", len(summary_rows))
    print("Passed:", summary_counts.get("PASS", 0))
    print("Rejected for missing files:", summary_counts.get("REJECTED_MISSING_FILES", 0))
    print("Other errors:", summary_counts.get("ERROR", 0))
    print("Total predictions generated:", len(prediction_rows))
    print("Session summary:", summary_path)
    if prediction_rows:
        print("Combined predictions:", predictions_path)
    print("\nResearch prototype only; no clinical validity is implied.")
    print("Raw/source dataset files were not modified.")

    return 0 if summary_counts.get("ERROR", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
