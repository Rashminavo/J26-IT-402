from abc import ABC, abstractmethod

import numpy as np


# ImageNet normalization constants used by the common TorchVision MobileNetV3
# pretrained-weight recipes. The selected weight enum and its complete inference
# transform remain to be fixed during model integration.
MOBILENET_V3_RGB_MEAN = (0.485, 0.456, 0.406)
MOBILENET_V3_RGB_STD = (0.229, 0.224, 0.225)
MOBILENET_V3_INPUT_SIZE = (224, 224)


def preprocess_mobilenet_v3_sequence(
    sequence: np.ndarray,
) -> np.ndarray:
    """Convert RGB uint8 THWC face ROIs to normalized float32 TCHW input.

    Contract:
    - Input shape: (time, 224, 224, 3), RGB, uint8, values in [0, 255].
    - Output shape: (time, 3, 224, 224), float32.
    - Converts pixels to [0, 1] then applies ImageNet mean/std normalization.
    - Does not resize or crop: the upstream face-ROI extractor already returns
      square 224x224 crops. Apply any chosen weight-specific spatial transform
      at most once in the model integration layer.

    This is a deterministic preprocessing helper, not MobileNet inference.
    It requires only NumPy so preprocessing contract tests can run before the
    PyTorch/TorchVision environment is installed.
    """
    if not isinstance(sequence, np.ndarray):
        raise TypeError("sequence must be a NumPy array")

    if sequence.ndim != 4:
        raise ValueError(
            "Expected sequence with shape (time, height, width, channels)"
        )

    if sequence.shape[0] <= 0:
        raise ValueError("sequence must contain at least one frame")

    if sequence.shape[1:3] != MOBILENET_V3_INPUT_SIZE:
        raise ValueError(
            "Expected face ROIs with spatial size (224, 224); "
            f"received {sequence.shape[1:3]}"
        )

    if sequence.shape[3] != 3:
        raise ValueError(
            "Expected 3 RGB channels in the last dimension"
        )

    if sequence.dtype != np.uint8:
        raise TypeError(
            f"Expected uint8 RGB pixels in [0, 255]; received {sequence.dtype}"
        )

    # Normalize in THWC layout, then transpose to TCHW. Copy to ensure a
    # contiguous output array and never mutate the caller's input.
    pixels = sequence.astype(np.float32) / np.float32(255.0)
    mean = np.asarray(MOBILENET_V3_RGB_MEAN, dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.asarray(MOBILENET_V3_RGB_STD, dtype=np.float32).reshape(1, 1, 1, 3)
    normalized = (pixels - mean) / std

    return np.ascontiguousarray(np.transpose(normalized, (0, 3, 1, 2)))


class FacialFeatureExtractor(ABC):
    """
    Common interface for facial feature extraction.
    """

    @abstractmethod
    def extract(
        self,
        sequence: np.ndarray,
    ) -> np.ndarray:
        """
        Convert a temporal face sequence into feature representation.
        """


class MobileNetV3LargeFeatureExtractor(
    FacialFeatureExtractor
):
    """
    MobileNetV3-Large research interface.

    Actual PyTorch/TorchVision implementation will be added after:
    dataset preparation, input pipeline validation and model environment
    setup are complete.
    """

    model_name = "MobileNetV3-Large"

    def __init__(self) -> None:
        self.initialized = False

    def extract(
        self,
        sequence: np.ndarray,
    ) -> np.ndarray:
        if sequence.ndim != 4:
            raise ValueError(
                "Expected sequence with shape "
                "(time, height, width, channels)"
            )

        raise NotImplementedError(
            "MobileNetV3-Large model integration is "
            "pending the model-development stage."
        )