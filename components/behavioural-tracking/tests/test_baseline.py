import pytest

from behavioural_tracking.baseline import (
    calculate_baseline,
    calculate_reference_deviation,
)


def test_calculate_baseline():
    result = calculate_baseline(
        [200, 210, 190, 205]
    )

    assert result["mean"] == 201.25
    assert result["sample_count"] == 4.0
    assert result["std"] > 0


def test_reference_deviation():
    baseline = calculate_baseline(
        [200, 210, 190, 205]
    )

    result = calculate_reference_deviation(
        current_value=260,
        baseline=baseline,
    )

    assert result["z_score"] > 0
    assert 0 < result["deviation_score"] < 1


def test_empty_baseline_is_rejected():
    with pytest.raises(ValueError):
        calculate_baseline([])