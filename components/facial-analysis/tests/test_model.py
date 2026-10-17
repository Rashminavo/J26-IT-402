import numpy as np
import pytest

from facial_analysis.model import (
    MOBILENET_V3_RGB_MEAN,
    MOBILENET_V3_RGB_STD,
    preprocess_mobilenet_v3_sequence,
)


def test_preprocessing_converts_thwc_uint8_rgb_to_tchw_float32():
    sequence = np.zeros((16, 224, 224, 3), dtype=np.uint8)

    result = preprocess_mobilenet_v3_sequence(sequence)

    assert result.shape == (16, 3, 224, 224)
    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    assert np.isfinite(result).all()


def test_preprocessing_applies_expected_imagenet_normalization():
    sequence = np.zeros((1, 224, 224, 3), dtype=np.uint8)

    result = preprocess_mobilenet_v3_sequence(sequence)

    expected = -np.asarray(MOBILENET_V3_RGB_MEAN, dtype=np.float32) / np.asarray(
        MOBILENET_V3_RGB_STD, dtype=np.float32
    )
    np.testing.assert_allclose(result[0, :, 0, 0], expected, rtol=1e-6, atol=1e-6)


def test_preprocessing_rescales_white_pixel_before_normalization():
    sequence = np.full((1, 224, 224, 3), 255, dtype=np.uint8)

    result = preprocess_mobilenet_v3_sequence(sequence)

    expected = (
        np.ones(3, dtype=np.float32)
        - np.asarray(MOBILENET_V3_RGB_MEAN, dtype=np.float32)
    ) / np.asarray(MOBILENET_V3_RGB_STD, dtype=np.float32)
    np.testing.assert_allclose(result[0, :, 0, 0], expected, rtol=1e-6, atol=1e-6)


def test_preprocessing_preserves_channel_order_and_does_not_mutate_input():
    sequence = np.zeros((1, 224, 224, 3), dtype=np.uint8)
    sequence[..., 0] = 255
    original = sequence.copy()

    result = preprocess_mobilenet_v3_sequence(sequence)

    assert result[0, 0, 0, 0] > result[0, 1, 0, 0]
    assert result[0, 0, 0, 0] > result[0, 2, 0, 0]
    np.testing.assert_array_equal(sequence, original)


def test_preprocessing_rejects_non_array_input():
    with pytest.raises(TypeError, match="NumPy array"):
        preprocess_mobilenet_v3_sequence([[[[0, 0, 0]]]])


def test_preprocessing_rejects_wrong_rank():
    with pytest.raises(ValueError, match="Expected sequence with shape"):
        preprocess_mobilenet_v3_sequence(np.zeros((224, 224, 3), dtype=np.uint8))


def test_preprocessing_rejects_empty_time_dimension():
    with pytest.raises(ValueError, match="at least one frame"):
        preprocess_mobilenet_v3_sequence(np.zeros((0, 224, 224, 3), dtype=np.uint8))


def test_preprocessing_rejects_wrong_spatial_size():
    with pytest.raises(ValueError, match="spatial size"):
        preprocess_mobilenet_v3_sequence(np.zeros((16, 200, 224, 3), dtype=np.uint8))


def test_preprocessing_rejects_wrong_channel_count():
    with pytest.raises(ValueError, match="3 RGB channels"):
        preprocess_mobilenet_v3_sequence(np.zeros((16, 224, 224, 1), dtype=np.uint8))


def test_preprocessing_rejects_non_uint8_pixels():
    with pytest.raises(TypeError, match="uint8 RGB pixels"):
        preprocess_mobilenet_v3_sequence(np.zeros((16, 224, 224, 3), dtype=np.float32))
