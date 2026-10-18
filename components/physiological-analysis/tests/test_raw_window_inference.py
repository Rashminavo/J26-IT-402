from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physiological_analysis.raw_window_inference import predict_session_windows  # noqa: E402

FEATURES = (
    "heart_rate_mean", "heart_rate_min", "heart_rate_max", "heart_rate_std",
    "eda_eda_mean", "eda_eda_std", "eda_eda_min", "eda_eda_max",
    "bvp_bvp_mean", "bvp_bvp_std", "bvp_bvp_min", "bvp_bvp_max",
    "temp_temp_mean", "temp_temp_std", "temp_temp_min", "temp_temp_max",
    "acc_x_mean", "acc_x_std", "acc_x_min", "acc_x_max",
    "acc_y_mean", "acc_y_std", "acc_y_min", "acc_y_max",
    "acc_z_mean", "acc_z_std", "acc_z_min", "acc_z_max",
    "acc_magnitude_mean", "acc_magnitude_std", "acc_magnitude_min", "acc_magnitude_max",
)


class FakeClassifier:
    feature_columns = FEATURES

    def predict(self, features):
        assert tuple(features.keys()) == FEATURES or set(features) == set(FEATURES)
        return {
            "predicted_class": "Low",
            "class_id": 0,
            "model_name": "fake-test-classifier",
            "research_use_only": True,
            "clinical_validation": False,
            "warning": "test only",
        }


def _write_csv(path: Path, rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def make_session(folder: Path) -> Path:
    folder.mkdir()
    start = "2013-06-12T16:18:58"
    # 30 seconds of readings, 15-second windows with 7.5-second stride -> 3 windows.
    _write_csv(folder / "HR.csv", [[start], [1], *[[60 + (i % 5)] for i in range(30)]])
    for filename, rate, count in [("EDA.csv", 4, 120), ("BVP.csv", 64, 1920), ("TEMP.csv", 4, 120)]:
        values = [[0.1 + ((i % 9) * 0.01)] for i in range(count)]
        _write_csv(folder / filename, [[start], [rate], *values])
    acc_values = [[float(i % 5), float((i + 1) % 7), float((i + 2) % 9)] for i in range(960)]
    _write_csv(folder / "ACC.csv", [[start], [32, 32, 32], *acc_values])
    return folder


def test_raw_session_produces_complete_window_predictions(tmp_path):
    session = make_session(tmp_path / "f01")
    rows = predict_session_windows(session, classifier=FakeClassifier())
    assert len(rows) == 3
    assert [row["window_start_seconds"] for row in rows] == [0.0, 7.5, 15.0]
    assert all(row["predicted_class"] == "Low" for row in rows)
    assert all(len([name for name in FEATURES if name in row]) == 32 for row in rows)


def test_raw_session_rejects_missing_required_sensor(tmp_path):
    session = make_session(tmp_path / "f01")
    (session / "TEMP.csv").unlink()
    with pytest.raises(FileNotFoundError, match="TEMP.csv"):
        predict_session_windows(session, classifier=FakeClassifier())


def test_raw_session_rejects_feature_schema_mismatch(tmp_path):
    session = make_session(tmp_path / "f01")

    class WrongSchemaClassifier(FakeClassifier):
        feature_columns = ("heart_rate_mean",)

    with pytest.raises(ValueError, match="schema does not match"):
        predict_session_windows(session, classifier=WrongSchemaClassifier())
