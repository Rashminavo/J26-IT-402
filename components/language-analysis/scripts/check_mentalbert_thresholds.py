
from pathlib import Path

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from language_analysis.dataset import load_labeled_split


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MAX_LENGTH = 256

# Project root / saved model path
PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_DIR = (
    PROJECT_ROOT
    / "backend"
    / "artifacts"
    / "language_baseline"
    / "mentalbert_weighted"
    / "model"
)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print("Loading development dataset...")

    zip_path = Path.home() / "Downloads" / "daicwoz.zip"

    if not zip_path.exists():
        raise FileNotFoundError(
            f"DAIC-WOZ ZIP not found: {zip_path}"
        )

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Saved model not found: {MODEL_DIR}"
        )

    dev_rows = load_labeled_split(
        zip_path, "dev"
    )["participants"]

    labels = np.array(
        [int(row["phq8_binary"]) for row in dev_rows]
    )

    print("Development participants:", len(dev_rows))
    print("Actual labels:", {
        0: int(np.sum(labels == 0)),
        1: int(np.sum(labels == 1)),
    })

    # Choose GPU if available
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("GPU not available; using CPU.")

    print("\nLoading saved MentalBERT model...")

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR)
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR)
    )

    model.to(device)
    model.eval()

    # --------------------------------------------------
    # Get class 1 probabilities
    # --------------------------------------------------

    probabilities = []

    print("\nCalculating probabilities...")

    with torch.no_grad():
        for index, row in enumerate(dev_rows, start=1):
            transcript = str(row["transcript"])

            encoded = tokenizer(
                transcript,
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

            logits = model(**encoded).logits

            probs = torch.softmax(logits, dim=-1)

            # Probability assigned to class 1
            class_1_probability = probs[0, 1].item()

            probabilities.append(class_1_probability)

            print(
                f"Participant {index:02d}/{len(dev_rows)}: "
                f"class_1_probability="
                f"{class_1_probability:.4f}"
            )

    probabilities = np.array(probabilities)

    print("\nProbability summary")
    print("-------------------")
    print("Minimum:", round(float(probabilities.min()), 4))
    print("Maximum:", round(float(probabilities.max()), 4))
    print("Mean:", round(float(probabilities.mean()), 4))
    print("Standard deviation:", round(float(probabilities.std()), 4))

    # --------------------------------------------------
    # Compare thresholds
    # --------------------------------------------------

    thresholds = [
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
    ]

    all_results = []

    print("\nTHRESHOLD COMPARISON")
    print("=" * 95)

    print(
        f"{'Threshold':>9} | "
        f"{'Accuracy':>8} | "
        f"{'Balanced Acc':>12} | "
        f"{'Macro F1':>8} | "
        f"{'Class 1 Recall':>14} | "
        f"{'Class 1 Precision':>16}"
    )

    print("-" * 95)

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            labels, predictions
        )

        balanced_accuracy = balanced_accuracy_score(
            labels, predictions
        )

        macro_f1 = f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )

        class_1_recall = recall_score(
            labels,
            predictions,
            pos_label=1,
            zero_division=0,
        )

        class_1_precision = precision_score(
            labels,
            predictions,
            pos_label=1,
            zero_division=0,
        )

        matrix = confusion_matrix(
            labels,
            predictions,
            labels=[0, 1],
        ).tolist()

        result = {
            "threshold": threshold,
            "accuracy": float(accuracy),
            "balanced_accuracy": float(balanced_accuracy),
            "macro_f1": float(macro_f1),
            "class_1_recall": float(class_1_recall),
            "class_1_precision": float(class_1_precision),
            "confusion_matrix": matrix,
        }

        all_results.append(result)

        print(
            f"{threshold:9.2f} | "
            f"{accuracy:8.3f} | "
            f"{balanced_accuracy:12.3f} | "
            f"{macro_f1:8.3f} | "
            f"{class_1_recall:14.3f} | "
            f"{class_1_precision:16.3f}"
        )

        print("  Confusion matrix:", matrix)

    # --------------------------------------------------
    # Save experiment results
    # --------------------------------------------------

    output_dir = MODEL_DIR.parent

    results_path = (
        output_dir / "threshold_comparison.json"
    )

    output = {
        "model": str(MODEL_DIR),
        "dataset": "DAIC-WOZ development split",
        "participants": len(dev_rows),
        "max_length": MAX_LENGTH,
        "threshold_results": all_results,
        "warning": (
            "Thresholds are exploratory and evaluated on the "
            "development set. Do not treat these results as "
            "independent final-test performance."
        ),
    }

    results_path.write_text(
        __import__("json").dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nResults saved to:")
    print(results_path)

    print(
        "\nNote: This script only evaluates the saved model. "
        "It does not retrain MentalBERT."
    )


if __name__ == "__main__":
    main()
