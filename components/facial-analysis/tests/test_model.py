import numpy as np
import pytest
import torch
from torchvision.models import MobileNet_V3_Large_Weights

from facial_analysis.model import (
    MOBILENET_V3_FEATURE_DIM,
    MOBILENET_V3_RGB_MEAN,
    MOBILENET_V3_RGB_STD,
    MOBILENET_V3_WEIGHTS,
    MobileNetV3LargeFeatureExtractor,
    preprocess_mobilenet_v3_sequence,
)


def make_sequence(frames=2):
    return np.zeros((frames, 224, 224, 3), dtype=np.uint8)


def test_preprocessing_shape_dtype_and_contiguity():
    result = preprocess_mobilenet_v3_sequence(make_sequence())

    assert result.shape == (2, 3, 224, 224)
    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    assert np.isfinite(result).all()


def test_black_pixels_use_image_net_normalization():
    result = preprocess_mobilenet_v3_sequence(
        np.zeros((1, 224, 224, 3), dtype=np.uint8)
    )

    expected = -np.asarray(MOBILENET_V3_RGB_MEAN) / np.asarray(
        MOBILENET_V3_RGB_STD
    )
    np.testing.assert_allclose(result[0, :, 112, 112], expected, atol=1e-5)


def test_white_pixels_use_image_net_normalization():
    result = preprocess_mobilenet_v3_sequence(
        np.full((1, 224, 224, 3), 255, dtype=np.uint8)
    )

    expected = (
        1.0 - np.asarray(MOBILENET_V3_RGB_MEAN)
    ) / np.asarray(MOBILENET_V3_RGB_STD)
    np.testing.assert_allclose(result[0, :, 112, 112], expected, atol=1e-5)


def test_rgb_channel_order_and_input_immutability():
    sequence = np.empty((1, 224, 224, 3), dtype=np.uint8)
    sequence[:] = (255, 0, 127)
    original = sequence.copy()

    result = preprocess_mobilenet_v3_sequence(sequence)

    values = np.asarray((255, 0, 127), dtype=np.float32) / 255.0
    expected = (
        values - np.asarray(MOBILENET_V3_RGB_MEAN)
    ) / np.asarray(MOBILENET_V3_RGB_STD)

    np.testing.assert_allclose(result[0, :, 112, 112], expected, atol=1e-5)
    np.testing.assert_array_equal(sequence, original)


def test_preprocessing_rejects_non_numpy_input():
    with pytest.raises(TypeError, match="NumPy array"):
        preprocess_mobilenet_v3_sequence([1, 2, 3])


def test_preprocessing_rejects_wrong_rank():
    with pytest.raises(ValueError, match="shape"):
        preprocess_mobilenet_v3_sequence(np.zeros((224, 224, 3), dtype=np.uint8))


def test_preprocessing_rejects_empty_temporal_dimension():
    with pytest.raises(ValueError, match="at least one frame"):
        preprocess_mobilenet_v3_sequence(make_sequence(frames=0))


def test_preprocessing_rejects_wrong_spatial_size():
    sequence = np.zeros((1, 223, 224, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="224, 224"):
        preprocess_mobilenet_v3_sequence(sequence)


def test_preprocessing_rejects_wrong_channel_count():
    sequence = np.zeros((1, 224, 224, 1), dtype=np.uint8)

    with pytest.raises(ValueError, match="3 RGB channels"):
        preprocess_mobilenet_v3_sequence(sequence)


def test_preprocessing_rejects_non_uint8_input():
    sequence = np.zeros((1, 224, 224, 3), dtype=np.float32)

    with pytest.raises(TypeError, match="uint8"):
        preprocess_mobilenet_v3_sequence(sequence)


def test_pinned_weight_version_is_explicit():
    assert MOBILENET_V3_WEIGHTS is (
        MobileNet_V3_Large_Weights.IMAGENET1K_V2
    )


@pytest.fixture(scope="module")
def untrained_extractor():
    # Avoid network access in the automated test suite.
    return MobileNetV3LargeFeatureExtractor(weights=None, device="cpu")


def test_extractor_initializes_in_eval_mode(untrained_extractor):
    assert untrained_extractor.initialized is True
    assert untrained_extractor.model.training is False


def test_extractor_returns_one_feature_vector_per_frame(untrained_extractor):
    result = untrained_extractor.extract(make_sequence(frames=2))

    assert result.shape == (2, MOBILENET_V3_FEATURE_DIM)
    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    assert np.isfinite(result).all()


def test_extractor_rejects_invalid_input(untrained_extractor):
    sequence = np.zeros((1, 224, 224, 3), dtype=np.float32)

    with pytest.raises(TypeError, match="uint8"):
        untrained_extractor.extract(sequence)
