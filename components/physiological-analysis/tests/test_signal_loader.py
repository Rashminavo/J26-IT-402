
import pytest

from physiological_analysis.data_loader import (
    load_signal_csv,
    load_tags_csv,
)


def test_load_single_channel_signal(tmp_path):
    file_path = tmp_path / "EDA.csv"
    file_path.write_text(
        "2023-01-01 10:00:00\n"
        "4.0\n"
        "0.1\n"
        "0.2\n",
        encoding="utf-8",
    )

    result = load_signal_csv(file_path)

    assert result["timestamp"] == "2023-01-01 10:00:00"
    assert result["sampling_rates"] == [4.0]
    assert result["readings"] == [[0.1], [0.2]]
    assert result["sample_count"] == 2
    assert result["channel_count"] == 1


def test_load_three_axis_signal(tmp_path):
    file_path = tmp_path / "ACC.csv"
    file_path.write_text(
        "2023-01-01,2023-01-01,2023-01-01\n"
        "32.0,32.0,32.0\n"
        "-8.0,-17.0,54.0\n"
        "-14.0,-12.0,52.0\n",
        encoding="utf-8",
    )

    result = load_signal_csv(file_path)

    assert result["sampling_rates"] == [32.0, 32.0, 32.0]
    assert result["sample_count"] == 2
    assert result["channel_count"] == 3
    assert result["readings"][0] == [-8.0, -17.0, 54.0]


def test_load_signal_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_signal_csv(tmp_path / "missing.csv")


def test_load_signal_rejects_invalid_value(tmp_path):
    file_path = tmp_path / "EDA.csv"
    file_path.write_text(
        "2023-01-01\n4.0\ninvalid\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_signal_csv(file_path)


def test_load_tags_csv(tmp_path):
    file_path = tmp_path / "tags.csv"
    file_path.write_text(
        "2023-01-01 10:01:00\n"
        "2023-01-01 10:02:00\n",
        encoding="utf-8",
    )

    assert load_tags_csv(file_path) == [
        "2023-01-01 10:01:00",
        "2023-01-01 10:02:00",
    ]


def test_load_tags_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_tags_csv(tmp_path / "missing.csv")
