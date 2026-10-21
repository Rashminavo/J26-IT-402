
from pathlib import Path
from threading import Lock
from typing import Any

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[4]
    / "backend"
    / "artifacts"
    / "language_baseline"
    / "mentalbert_weighted"
    / "model"
)

MAX_LENGTH = 256

_tokenizer: Any = None
_model: Any = None
_device: torch.device | None = None
_model_lock = Lock()


def load_mentalbert(
    model_path: str | Path | None = None,
) -> tuple[Any, Any, torch.device]:
    """Load the trained MentalBERT model once and reuse it."""
    global _tokenizer, _model, _device

    with _model_lock:
        if _model is not None and _tokenizer is not None and _device is not None:
            return _tokenizer, _model, _device

        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

        if not path.is_dir():
            raise FileNotFoundError(
                f"MentalBERT model directory not found: {path}"
            )

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        tokenizer = AutoTokenizer.from_pretrained(str(path))
        model = AutoModelForSequenceClassification.from_pretrained(
            str(path)
        )
        model.to(device)
        model.eval()

        _tokenizer = tokenizer
        _model = model
        _device = device

        return _tokenizer, _model, _device


def analyze_with_mentalbert(text: str) -> dict[str, Any]:
    """
    Return MentalBERT class probabilities.

    These are model probabilities, not clinical diagnoses or
    calibrated mental-health risk percentages.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not text.strip():
        raise ValueError("text must not be empty")

    tokenizer, model, device = load_mentalbert()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0].cpu().tolist()

    predicted_class = int(max(range(len(probabilities)), key=probabilities.__getitem__))

    return {
        "model": "MentalBERT",
        "device": str(device),
        "predicted_class": predicted_class,
        "class_probabilities": {
            str(index): float(probability)
            for index, probability in enumerate(probabilities)
        },
        "max_length": MAX_LENGTH,
        "truncated": len(tokenizer.encode(text, add_special_tokens=True)) > MAX_LENGTH,
    }
