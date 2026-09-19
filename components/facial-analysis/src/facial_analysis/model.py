from abc import ABC, abstractmethod

import numpy as np


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