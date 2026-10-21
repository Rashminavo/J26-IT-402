
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from language_analysis.dataset import load_labeled_split


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "mental/mental-bert-base-uncased"

SEED = 42
MAX_LENGTH = 256
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
EPOCHS = 3
LEARNING_RATE = 2e-5


# --------------------------------------------------
# Dataset
# --------------------------------------------------

class TranscriptDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        encoded = self.tokenizer(
            str(row["transcript"]),
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        item = {
            key: value.squeeze(0)
            for key, value in encoded.items()
        }

        item["labels"] = torch.tensor(
            int(row["phq8_binary"]),
            dtype=torch.long,
        )

        return item


# --------------------------------------------------
# Metrics
# --------------------------------------------------

def compute_metrics(prediction):
    logits = prediction.predictions

    if isinstance(logits, tuple):
        logits = logits[0]

    labels = prediction.label_ids
    predicted_labels = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(
            labels, predicted_labels
        ),
        "balanced_accuracy": balanced_accuracy_score(
            labels, predicted_labels
        ),
        "macro_f1": f1_score(
            labels,
            predicted_labels,
            average="macro",
            zero_division=0,
        ),
        "class_1_precision": precision_score(
            labels,
            predicted_labels,
            pos_label=1,
            zero_division=0,
        ),
        "class_1_recall": recall_score(
            labels,
            predicted_labels,
            pos_label=1,
            zero_division=0,
        ),
        "class_1_f1": f1_score(
            labels,
            predicted_labels,
            pos_label=1,
            zero_division=0,
        ),
    }


# --------------------------------------------------
# Trainer with class-weighted loss
# --------------------------------------------------

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        inputs = inputs.copy()
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits

        weights = self.class_weights.to(
            device=logits.device,
            dtype=logits.dtype,
        )

        loss_function = nn.CrossEntropyLoss(
            weight=weights
        )

        loss = loss_function(
            logits.reshape(-1, model.config.num_labels),
            labels.reshape(-1),
        )

        if return_outputs:
            return loss, outputs

        return loss


# --------------------------------------------------
# Main training process
# --------------------------------------------------

def main():
    set_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not available. Check your PyTorch installation."
        )

    print("GPU:", torch.cuda.get_device_name(0))

    zip_path = Path.home() / "Downloads" / "daicwoz.zip"

    train_rows = load_labeled_split(
        zip_path, "train"
    )["participants"]

    dev_rows = load_labeled_split(
        zip_path, "dev"
    )["participants"]

    train_labels = np.array([
        int(row["phq8_binary"])
        for row in train_rows
    ])

    dev_labels = np.array([
        int(row["phq8_binary"])
        for row in dev_rows
    ])

    print("\nTrain participants:", len(train_rows))
    print("Development participants:", len(dev_rows))

    train_counts = np.bincount(
        train_labels, minlength=2
    )

    dev_counts = np.bincount(
        dev_labels, minlength=2
    )

    print("Train label counts:", {
        0: int(train_counts[0]),
        1: int(train_counts[1]),
    })

    print("Development label counts:", {
        0: int(dev_counts[0]),
        1: int(dev_counts[1]),
    })

    if np.any(train_counts == 0):
        raise ValueError(
            "Both classes must be present in the training data."
        )

    # Inverse-frequency weights calculated ONLY from training labels.
    # The less frequent class receives a higher weight.
    class_weights = (
        len(train_labels)
        / (2.0 * train_counts.astype(np.float32))
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
    )

    print("\nClass weights:")
    print({
        0: round(float(class_weights[0]), 4),
        1: round(float(class_weights[1]), 4),
    })

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    model.config.problem_type = (
        "single_label_classification"
    )

    train_dataset = TranscriptDataset(
        train_rows, tokenizer
    )

    dev_dataset = TranscriptDataset(
        dev_rows, tokenizer
    )

    output_dir = (
        Path(__file__).resolve().parents[3]
        / "backend"
        / "artifacts"
        / "language_baseline"
        / "mentalbert_weighted"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),

        num_train_epochs=EPOCHS,

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        learning_rate=LEARNING_RATE,
        weight_decay=0.01,

        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,

        fp16=True,

        logging_steps=10,
        save_total_limit=1,

        seed=SEED,
        data_seed=SEED,

        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    print("\nStarting class-weighted MentalBERT fine-tuning...")
    trainer.train()

    print("\nEvaluating the best model on the development set...")
    evaluation = trainer.evaluate()

    prediction_output = trainer.predict(dev_dataset)

    logits = prediction_output.predictions

    if isinstance(logits, tuple):
        logits = logits[0]

    predicted_labels = np.argmax(
        logits, axis=-1
    )

    actual_labels = prediction_output.label_ids

    results = {
        "model": MODEL_NAME,
        "experiment": "class_weighted_loss",
        "task": "PHQ8_Binary classification",
        "train_participants": len(train_rows),
        "dev_participants": len(dev_rows),
        "train_label_counts": {
            "0": int(train_counts[0]),
            "1": int(train_counts[1]),
        },
        "dev_label_counts": {
            "0": int(dev_counts[0]),
            "1": int(dev_counts[1]),
        },
        "class_weights": {
            "0": float(class_weights[0]),
            "1": float(class_weights[1]),
        },
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": (
            GRADIENT_ACCUMULATION_STEPS
        ),
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "evaluation": evaluation,
        "confusion_matrix": confusion_matrix(
            actual_labels,
            predicted_labels,
            labels=[0, 1],
        ).tolist(),
        "classification_report": classification_report(
            actual_labels,
            predicted_labels,
            labels=[0, 1],
            target_names=[
                "PHQ8_Binary_0",
                "PHQ8_Binary_1",
            ],
            output_dict=True,
            zero_division=0,
        ),
        "limitation": (
            "Exploratory university research only. "
            "Not clinically validated."
        ),
    }

    model_dir = output_dir / "model"

    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    metrics_path = output_dir / "metrics.json"

    metrics_path.write_text(
        json.dumps(
            results,
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )

    print("\nFinal development results:")
    print(json.dumps(results, indent=2, default=float))

    print("\nSaved model to:", model_dir)
    print("Saved metrics to:", metrics_path)


if __name__ == "__main__":
    main()
