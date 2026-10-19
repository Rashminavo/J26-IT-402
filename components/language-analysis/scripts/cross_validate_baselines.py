
import json
from pathlib import Path
from sklearn.naive_bayes import MultinomialNB

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from language_analysis.dataset import load_labeled_split


def evaluate(name, model, texts, labels, participant_ids, cv):
    predictions = cross_val_predict(
        model, texts, labels, cv=cv, method="predict"
    )

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    participant_predictions = [
        {
            "participant_id": str(participant_id),
            "actual_label": int(actual),
            "predicted_label": int(predicted),
            "correct": bool(actual == predicted),
        }
        for participant_id, actual, predicted in zip(
            participant_ids, labels, predictions
        )
    ]

    return {
        "model": name,
        "predictions": participant_predictions,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "accuracy": round(accuracy_score(labels, predictions), 3),
        "balanced_accuracy": round(
            balanced_accuracy_score(labels, predictions), 3
        ),
        "macro_f1": round(
            f1_score(labels, predictions, average="macro", zero_division=0),
            3,
        ),
        "class_1_precision": round(
            precision_score(labels, predictions, pos_label=1, zero_division=0),
            3,
        ),
        "class_1_recall": round(
            recall_score(labels, predictions, pos_label=1, zero_division=0),
            3,
        ),
        "class_1_f1": round(
            f1_score(labels, predictions, pos_label=1, zero_division=0),
            3,
        ),
    }



def main():
    zip_path = Path.home() / "Downloads" / "daicwoz.zip"
    rows = load_labeled_split(zip_path, "train")["participants"]

    texts = [row["transcript"] for row in rows]
    labels = [row["phq8_binary"] for row in rows]
    participant_ids = [row["participant_id"] for row in rows]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_unigram": Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 1),
                min_df=2,
                max_features=5000,
                sublinear_tf=True,
            )),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
            )),

        ]),
        "logistic_unigram_bigram": Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=10000,
                sublinear_tf=True,
            )),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                C=0.1,
                max_iter=2000,
                random_state=42,
            )),
        ]),
        "linear_svc": Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=10000,
                sublinear_tf=True,
            )),
            ("classifier", LinearSVC(
                class_weight="balanced",
                C=0.1,
                random_state=42,
            )),
        ]),

        
        "naive_bayes_word": Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=10000,
                sublinear_tf=True,
            )),
            ("classifier", MultinomialNB(alpha=0.5)),
        ]),
        "logistic_character_ngrams": Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_features=20000,
                sublinear_tf=True,
            )),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                C=0.1,
                max_iter=2000,
                random_state=42,
            )),
        ]),

    }

    results = [
    evaluate(name, model, texts, labels, participant_ids, cv)
    for name, model in models.items()
]

    
    output_data = {
        "participants": len(rows),
        "label_counts": {
            "class_0": labels.count(0),
            "class_1": labels.count(1),
        },
        "evaluation": "5-fold stratified cross-validation",
        "results": results,
        "limitation": (
            "Exploratory research only; small dataset and no clinical validation."
        ),
    }

    output_path = (
        Path(__file__).resolve().parents[3]
        / "backend"
        / "artifacts"
        / "language_baseline"
        / "cross_validation_metrics.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(output_data, indent=2),
        encoding="utf-8",
    )

    print(f"Results saved to: {output_path}")
    print(json.dumps(output_data, indent=2))



if __name__ == "__main__":
    main()
