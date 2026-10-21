
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from language_analysis.dataset import load_labeled_split


def main() -> None:
    zip_path = Path.home() / "Downloads" / "daicwoz.zip"

    train = load_labeled_split(zip_path, "train")["participants"]
    dev = load_labeled_split(zip_path, "dev")["participants"]

    x_train = [row["transcript"] for row in train]
    y_train = [row["phq8_binary"] for row in train]
    x_dev = [row["transcript"] for row in dev]
    y_dev = [row["phq8_binary"] for row in dev]

    experiments = {
        "logistic_word_unigram": Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 1),
                max_features=5000,
                min_df=2,
                sublinear_tf=True,
            )),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
            )),
        ]),
        "logistic_word_unigram_bigram": Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=10000,
                min_df=2,
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
                max_features=10000,
                min_df=2,
                sublinear_tf=True,
            )),
            ("classifier", LinearSVC(
                class_weight="balanced",
                C=0.1,
                random_state=42,
            )),
        ]),
    }

    results = []

    for name, model in experiments.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_dev)
        report = classification_report(
            y_dev,
            predictions,
            labels=[0, 1],
            output_dict=True,
            zero_division=0,
        )

        result = {
            "model": name,
            "accuracy": round(accuracy_score(y_dev, predictions), 3),
            "macro_f1": round(
                f1_score(y_dev, predictions, average="macro",
                         zero_division=0),
                3,
            ),
            "class_1_precision": round(report["1"]["precision"], 3),
            "class_1_recall": round(report["1"]["recall"], 3),
            "class_1_f1": round(report["1"]["f1-score"], 3),
        }
        results.append(result)

    print(json.dumps(results, indent=2))
    print(
        "\nExploratory comparison only. The small dev set is not "
        "sufficient to establish clinical validity."
    )


if __name__ == "__main__":
    main()
