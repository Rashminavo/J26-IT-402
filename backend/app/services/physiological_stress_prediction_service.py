"""Service adapter for the exploratory window-level physiological classifier.

This service intentionally returns an exploratory class prediction rather than
creating a numeric component score in the shared score store.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
COMPONENT_SOURCE = (
    PROJECT_ROOT / "components" / "physiological-analysis" / "src"
)

# The repository uses a src-layout component that is not installed as a wheel.
# Make the component import explicit for this monorepo backend process.
component_source_text = str(COMPONENT_SOURCE)
if component_source_text not in sys.path:
    sys.path.insert(0, component_source_text)

from physiological_analysis.window_stress_classifier import (  # noqa: E402
    WindowLevelStressClassifier,
)


@lru_cache(maxsize=1)
def _load_classifier() -> WindowLevelStressClassifier:
    """Load the model once per backend process.

    For local development, the classifier's default artifact path is used.
    For deployments, set J26_PHYSIOLOGICAL_MODEL_PATH to the separately
    provisioned .joblib artifact path.
    """
    model_path = os.environ.get("J26_PHYSIOLOGICAL_MODEL_PATH")
    return WindowLevelStressClassifier.load(model_path or None)


def predict_window_stress(features: Mapping[str, Any]) -> dict[str, Any]:
    """Predict an exploratory self-report band from one 15-second window."""
    return _load_classifier().predict(features)
