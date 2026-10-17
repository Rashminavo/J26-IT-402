from physiological_analysis.features import (
    count_available_features,
    extract_features,
)


def test_extract_physiological_features():
    result = extract_features(
        {
            "heart_rate": 78.0,
            "body_temperature": 36.8,
        }
    )

    assert result["heart_rate"] == 78.0
    assert result["body_temperature"] == 36.8


def test_count_available_features():
    result = count_available_features(
        {
            "heart_rate": 78.0,
            "body_temperature": 36.8,
        }
    )

    assert result == 2
from pytest import raises

from physiological_analysis.features import (
    extract_heart_rate_features,
)


def test_extract_heart_rate_features():
    result = extract_heart_rate_features(
        [60.0, 70.0, 80.0]
    )

    assert result["heart_rate_mean"] == 70.0
    assert result["heart_rate_min"] == 60.0
    assert result["heart_rate_max"] == 80.0
    assert round(result["heart_rate_std"], 2) == 8.16


def test_extract_heart_rate_features_rejects_empty_list():
    with raises(ValueError):
        extract_heart_rate_features([])


def test_extract_heart_rate_features_rejects_invalid_values():
    with raises(ValueError):
        extract_heart_rate_features([60.0, float("nan")])



from physiological_analysis.features import extract_signal_features


def test_extract_eda_signal_features():
    result = extract_signal_features(
        "eda",
        {
            "sampling_rates": [4.0],
            "readings": [[0.1], [0.2], [0.3]],
        },
    )

    assert result["eda_eda_mean"] == 0.2
    assert result["eda_eda_min"] == 0.1
    assert result["eda_eda_max"] == 0.3
    assert round(result["eda_eda_std"], 6) == round(
        0.0816496580927726, 6
    )


def test_extract_acc_signal_features():
    result = extract_signal_features(
        "acc",
        {
            "sampling_rates": [32.0, 32.0, 32.0],
            "readings": [
                [3.0, 4.0, 0.0],
                [0.0, 0.0, 5.0],
            ],
        },
    )

    assert result["acc_x_mean"] == 1.5
    assert result["acc_y_mean"] == 2.0
    assert result["acc_z_mean"] == 2.5
    assert result["acc_magnitude_mean"] == 5.0
    assert result["acc_magnitude_min"] == 5.0
    assert result["acc_magnitude_max"] == 5.0


def test_extract_signal_features_rejects_unsupported_signal():
    import pytest

    with pytest.raises(ValueError):
        extract_signal_features("unknown", {
            "sampling_rates": [1.0],
            "readings": [[1.0]],
        })


def test_extract_signal_features_rejects_non_finite_values():
    import pytest

    with pytest.raises(ValueError):
        extract_signal_features("eda", {
            "sampling_rates": [4.0],
            "readings": [[float("nan")]],
        })

def test_extract_ibi_features():
    from physiological_analysis.features import extract_ibi_features

    result = extract_ibi_features(
        [
            [1.0, 0.8],
            [2.0, 1.0],
            [3.0, 0.9],
        ]
    )

    assert round(result["ibi_mean_seconds"], 6) == round(0.9, 6)
    assert round(result["ibi_std_seconds"], 6) == round(0.0816496581, 6)
    assert round(result["ibi_rmssd_seconds"], 6) == round(
        (0.025 ** 0.5), 6
    )


def test_extract_ibi_features_rejects_non_increasing_elapsed_times():
    import pytest
    from physiological_analysis.features import extract_ibi_features

    with pytest.raises(ValueError):
        extract_ibi_features(
            [[2.0, 0.8], [2.0, 0.9]]
        )
