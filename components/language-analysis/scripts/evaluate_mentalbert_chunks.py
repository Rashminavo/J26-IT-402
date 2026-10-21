
from pathlib import Path
import json
import sys

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path.cwd()

ZIP_PATH = Path.home() / "Downloads" / "daicwoz.zip"

MODEL_DIR = (
    ROOT
    / "backend"
    / "artifacts"
    / "language_baseline"
    / "mentalbert_weighted"
    / "model"
)

OUTPUT_DIR = (
    ROOT
    / "backend"
    / "artifacts"
    / "language_baseline"
    / "chunk_evaluation"
)

CHUNK_SIZE = 256
CHUNK_OVERLAP = 32
BATCH_SIZE = 8
THRESHOLD = 0.5

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATASET IMPORT
# ============================================================

SOURCE_DIR = ROOT / "components" / "language-analysis" / "src"
sys.path.insert(0, str(SOURCE_DIR))

from language_analysis.dataset import load_labeled_split


# ============================================================
# CHUNKING
# ============================================================

def make_chunks(text, tokenizer):
    """
    Split a transcript into overlapping token chunks.

    CHUNK_SIZE counts content tokens, excluding special tokens.
    Special tokens are added later by the tokenizer.
    """

    token_ids = tokenizer(
        str(text),
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    if not token_ids:
        return [[]]

    step = CHUNK_SIZE - CHUNK_OVERLAP

    chunks = [
        token_ids[start:start + CHUNK_SIZE]
        for start in range(0, len(token_ids), step)
    ]

    return chunks


# ============================================================
# PREDICT ONE PARTICIPANT
# ============================================================



def predict_participant(transcript, tokenizer, model, device):
    """
    Split a transcript into chunks, predict each chunk,
    and average the class-1 probabilities.
    """

    chunks = make_chunks(transcript, tokenizer)
    chunk_probabilities = []

    for start in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[start:start + BATCH_SIZE]

        # Convert token IDs back to text, then let the tokenizer
        # add the correct special tokens and padding.
        chunk_texts = [
            tokenizer.decode(
                chunk,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for chunk in batch_chunks
        ]

        encoded = tokenizer(
            chunk_texts,
            add_special_tokens=True,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.no_grad():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=1)[:, 1]

        chunk_probabilities.extend(
            probabilities.detach().cpu().numpy().tolist()
        )

    participant_probability = float(np.mean(chunk_probabilities))

    return participant_probability, len(chunks)



# ============================================================
# EVALUATION
# ============================================================

def evaluate_split(
    split_name,
    tokenizer,
    model,
    device,
):
    print(f"\nEvaluating {split_name.upper()} split...")

    dataset = load_labeled_split(
        ZIP_PATH,
        split_name,
    )

    rows = dataset["participants"]
    split_results = []

    for index, row in enumerate(rows, start=1):
        probability, number_of_chunks = predict_participant(
            row["transcript"],
            tokenizer,
            model,
            device,
        )

        label = int(row["phq8_binary"])
        prediction = int(probability >= THRESHOLD)

        participant_id = row.get(
            "participant_id",
            row.get("participant", index),
        )

        split_results.append({
            "split": split_name,
            "participant_id": str(participant_id),
            "label": label,
            "number_of_chunks": number_of_chunks,
            "probability_class_1": probability,
            "prediction": prediction,
        })

        if index % 10 == 0 or index == len(rows):
            print(
                f"  Processed {index}/{len(rows)} participants"
            )

    return split_results


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(results):
    y_true = np.array(
        [item["label"] for item in results]
    )

    y_pred = np.array(
        [item["prediction"] for item in results]
    )

    y_prob = np.array(
        [item["probability_class_1"] for item in results]
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "participants": len(results),
        "threshold": THRESHOLD,
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "class_1_precision": float(
            report["1"]["precision"]
        ),
        "class_1_recall": float(
            report["1"]["recall"]
        ),
        "class_1_f1": float(
            report["1"]["f1-score"]
        ),
        "confusion_matrix_labels_0_1": confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).tolist(),
        "roc_auc": (
            float(roc_auc_score(y_true, y_prob))
            if len(np.unique(y_true)) == 2
            else None
        ),
        "average_precision": (
            float(average_precision_score(y_true, y_prob))
            if len(np.unique(y_true)) == 2
            else None
        ),
    }

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"Dataset ZIP not found: {ZIP_PATH}"
        )

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Trained model directory not found: {MODEL_DIR}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Model directory:", MODEL_DIR)
    print("Chunk size:", CHUNK_SIZE)
    print("Chunk overlap:", CHUNK_OVERLAP)
    print("Batch size:", BATCH_SIZE)
    print("Prediction threshold:", THRESHOLD)

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR)
    )

    print("Loading trained model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR)
    )

    model.to(device)
    model.eval()

    # Check that content tokens plus special tokens fit the model.
    max_positions = getattr(
        model.config,
        "max_position_embeddings",
        512,
    )

    special_tokens = tokenizer.num_special_tokens_to_add(
        pair=False
    )

    if CHUNK_SIZE + special_tokens > max_positions:
        raise ValueError(
            f"CHUNK_SIZE ({CHUNK_SIZE}) plus special tokens "
            f"({special_tokens}) exceeds the model limit "
            f"({max_positions}). Reduce CHUNK_SIZE."
        )

    all_results = []

    for split_name in ["train", "dev"]:
        split_results = evaluate_split(
            split_name,
            tokenizer,
            model,
            device,
        )

        all_results.extend(split_results)

        metrics = calculate_metrics(split_results)

        print(f"\n{split_name.upper()} RESULTS")
        print(json.dumps(metrics, indent=2))

    # Save participant-level predictions.
    predictions_path = (
        OUTPUT_DIR / "participant_predictions.json"
    )

    predictions_path.write_text(
        json.dumps(all_results, indent=2),
        encoding="utf-8",
    )

    # Save metrics separately for each split.
    metrics_output = {}

    for split_name in ["train", "dev"]:
        split_results = [
            result
            for result in all_results
            if result["split"] == split_name
        ]

        metrics_output[split_name] = calculate_metrics(
            split_results
        )

    metrics_path = OUTPUT_DIR / "metrics.json"

    metrics_path.write_text(
        json.dumps(metrics_output, indent=2),
        encoding="utf-8",
    )

    print("\nEvaluation completed.")
    print("Predictions saved to:", predictions_path)
    print("Metrics saved to:", metrics_path)

    print(
        "\nNOTE: These results are exploratory. "
        "They are not independent test results or "
        "clinical validation."
    )


if __name__ == "__main__":
    main()
