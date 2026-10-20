from abc import ABC, abstractmethod

import numpy as np
import torch
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    mobilenet_v3_large,
)


# Pin the exact pretrained weight enum for reproducible preprocessing.
MOBILENET_V3_WEIGHTS = MobileNet_V3_Large_Weights.IMAGENET1K_V2
MOBILENET_V3_RGB_MEAN = (0.485, 0.456, 0.406)
MOBILENET_V3_RGB_STD = (0.229, 0.224, 0.225)
MOBILENET_V3_INPUT_SIZE = (224, 224)
MOBILENET_V3_FEATURE_DIM = 960

# Official resize, center-crop, scaling and normalization transform.
_MOBILENET_V3_TRANSFORM = MOBILENET_V3_WEIGHTS.transforms()


def _validate_sequence(sequence: np.ndarray) -> None:
    """Validate uint8 RGB face ROIs in THWC layout."""
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
        raise ValueError("Expected 3 RGB channels in the last dimension")

    if sequence.dtype != np.uint8:
        raise TypeError(
            f"Expected uint8 RGB pixels in [0, 255]; received {sequence.dtype}"
        )


def preprocess_mobilenet_v3_sequence(
    sequence: np.ndarray,
) -> np.ndarray:
    """Apply the pinned TorchVision transform to uint8 RGB THWC frames.

    Input:
        uint8 NumPy array with shape (time, 224, 224, 3), RGB [0, 255].

    Output:
        Contiguous float32 NumPy array with shape (time, 3, 224, 224).

    The official IMAGENET1K_V2 transform handles resize to 232, center crop
    to 224, conversion to [0, 1], and ImageNet mean/std normalization.
    Do not normalize this output again.
    """
    _validate_sequence(sequence)

    # Create a tensor view without mutating the caller's input.
    frames = torch.from_numpy(np.ascontiguousarray(sequence))
    frames = frames.permute(0, 3, 1, 2)

    transformed = _MOBILENET_V3_TRANSFORM(frames)

    return np.ascontiguousarray(
        transformed.cpu().numpy(),
        dtype=np.float32,
    )


class FacialFeatureExtractor(ABC):
    """Common interface for facial feature extraction."""

    @abstractmethod
    def extract(self, sequence: np.ndarray) -> np.ndarray:
        """Convert a temporal face sequence into feature representations."""


class MobileNetV3LargeFeatureExtractor(FacialFeatureExtractor):
    """Extract one 960-dimensional pretrained MobileNetV3 feature per frame.

    This class returns frame-level visual features. It does not predict
    emotions, classify micro-expressions, or aggregate temporal evidence.
    """

    model_name = "MobileNetV3-Large"
    feature_dim = MOBILENET_V3_FEATURE_DIM

    def __init__(
        self,
        weights: MobileNet_V3_Large_Weights | None = MOBILENET_V3_WEIGHTS,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available in this environment"
            )

        # weights=None is useful for deterministic-shape tests without
        # downloading weights. The input transform remains pinned to V2.
        self.weights = weights
        self.model = mobilenet_v3_large(weights=weights)
        self.model.to(self.device)
        self.model.eval()
        self.initialized = True

    def extract(self, sequence: np.ndarray) -> np.ndarray:
        """Return float32 features with shape (time, 960), preserving order."""
        preprocessed = preprocess_mobilenet_v3_sequence(sequence)
        inputs = torch.from_numpy(preprocessed).to(self.device)

        with torch.inference_mode():
            feature_maps = self.model.features(inputs)
            pooled = self.model.avgpool(feature_maps)
            features = torch.flatten(pooled, start_dim=1)

        result = features.cpu().numpy()

        if result.ndim != 2 or result.shape[1] != self.feature_dim:
            raise RuntimeError(
                "Unexpected MobileNetV3 feature shape: "
                f"{result.shape}"
            )

        if not np.isfinite(result).all():
            raise RuntimeError("MobileNetV3 produced non-finite features")

        return np.ascontiguousarray(result, dtype=np.float32)
