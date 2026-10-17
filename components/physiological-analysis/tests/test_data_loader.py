
import pytest

from physiological_analysis.data_loader import (
    load_heart_rate_csv,
    load_ibi_csv,
)


def test_load_heart_rate_csv(tmp_path):
    csv_file = tmp_path / "HR.csv"

    csv_file.write_text(
        "2023-01-01 10:00:00\n"
        "1\n"
        "60.0\n"
        "70.0\n"
        "80.0\n",
        encoding="utf-8",
    )

    readings = load_heart_rate_csv(csv_file)

    assert readings == [60.0, 70.0, 80.0]


def test_load_heart_rate_csv_missing_file(tmp_path):
    missing_file = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_heart_rate_csv(missing_file)


def test_load_heart_rate_csv_rejects_invalid_value(
    tmp_path,
):
    csv_file = tmp_path / "HR.csv"

    csv_file.write_text(
        "2023-01-01 10:00:00\n"
        "1\n"
        "60.0\n"
        "invalid\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_heart_rate_csv(csv_file)


def test_load_heart_rate_csv_rejects_non_finite_value(
    tmp_path,
):
    csv_file = tmp_path / "HR.csv"

    csv_file.write_text(
        "2023-01-01 10:00:00\n"
        "1\n"
        "nan\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_heart_rate_csv(csv_file)


def test_load_heart_rate_csv_rejects_empty_readings(
    tmp_path,
):
    csv_file = tmp_path / "HR.csv"

    csv_file.write_text(
        "2023-01-01 10:00:00\n"
        "1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_heart_rate_csv(csv_file)


def test_load_ibi_csv(tmp_path):
    csv_file = tmp_path / "IBI.csv"

    csv_file.write_text(
        "2023-01-01 10:00:00,2023-01-01 10:00:00\n"
        "12.8125,0.921875\n"
        "13.671875,0.859375\n",
        encoding="utf-8",
    )

    result = load_ibi_csv(csv_file)

    assert result["start_timestamp"] == "2023-01-01 10:00:00"
    assert result["readings"] == [
        [12.8125, 0.921875],
        [13.671875, 0.859375],
    ]
    assert result["sample_count"] == 2


def test_load_ibi_csv_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_ibi_csv(tmp_path / "missing.csv")


def test_load_ibi_csv_rejects_invalid_value(tmp_path):
    csv_file = tmp_path / "IBI.csv"

    csv_file.write_text(
        "2023-01-01 10:00:00,2023-01-01 10:00:00\n"
        "invalid,0.9\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_ibi_csv(csv_file)


def test_load_ibi_csv_rejects_non_finite_value(tmp_path):
    csv_file = tmp_path / "IBI.csv"

    csv_file.write_text(
        "2023-01-01 10:00:00,2023-01-01 10:00:00\n"
        "12.8,nan\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_ibi_csv(csv_file)


@pytest.mark.parametrize(
    "reading",
    [
        "-1,0.9",
        "12.8,0",
        "12.8,-0.9",
    ],
)
def test_load_ibi_csv_rejects_invalid_intervals(
    tmp_path, reading
):
    csv_file = tmp_path / "IBI.csv"

    csv_file.write_text(
        "2023-01-01 10:00:00,2023-01-01 10:00:00\n"
        f"{reading}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_ibi_csv(csv_file)


def test_load_ibi_csv_rejects_wrong_column_count(tmp_path):
    csv_file = tmp_path / "IBI.csv"

    csv_file.write_text(
        "2023-01-01 10:00:00,2023-01-01 10:00:00\n"
        "12.8,0.9,extra\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_ibi_csv(csv_file)


def test_load_ibi_csv_rejects_empty_readings(tmp_path):
    csv_file = tmp_path / "IBI.csv"

    csv_file.write_text(
        "2023-01-01 10:00:00,2023-01-01 10:00:00\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_ibi_csv(csv_file)