
import argparse
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

from language_analysis.dataset import load_labeled_split


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a DAIC-WOZ text baseline."
    )
    parser.add_argument(
        "--zip",
        type=Path,
        default=Path.home() / "Downloads" / "daicwoz.zip",
        help="Path to the DAIC-WOZ ZIP file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts") / "language_baseline",
        help="Directory for the trained model and metrics.",
    )
    args = parser.parse_args()

    if not args.zip.is_file():
        raise FileNotFoundError(f"Dataset ZIP not found: {args.zip}")

    train_data = load_labeled_split(args.zip, "train")
    dev_data = load_labeled_split(args.zip, "dev")

    train_rows = train_data["participants"]
    dev_rows = dev_data["participants"]

    train_texts = [row["transcript"] for row in train_rows]
    train_labels = [row["phq8_binary"] for row in train_rows]
    dev_texts = [row["transcript"] for row in dev_rows]
    dev_labels = [row["phq8_binary"] for row in dev_rows]

    if len(set(train_labels)) < 2:
        raise ValueError("Training data must contain both label classes.")

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    max_features=10000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(train_texts, train_labels)
    predictions = model.predict(dev_texts)

    metrics = {
        "task": "PHQ8_Binary classification",
        "train_participants": len(train_rows),
        "dev_participants": len(dev_rows),
        "missing_train_transcripts": train_data["missing_transcript_count"],
        "missing_dev_transcripts": dev_data["missing_transcript_count"],
        "accuracy": accuracy_score(dev_labels, predictions),
        "f1_macro": f1_score(
            dev_labels, predictions, average="macro", zero_division=0
        ),
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": confusion_matrix(
            dev_labels, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            dev_labels,
            predictions,
            labels=[0, 1],
            target_names=["PHQ8_Binary_0", "PHQ8_Binary_1"],
            output_dict=True,
            zero_division=0,
        ),
        "warning": (
            "Research baseline only; not a clinically validated "
            "diagnostic or risk assessment tool."
        ),
    }

    args.output.mkdir(parents=True, exist_ok=True)

    import joblib

    joblib.dump(model, args.output / "model.joblib")

    with (args.output / "metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model and metrics to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
