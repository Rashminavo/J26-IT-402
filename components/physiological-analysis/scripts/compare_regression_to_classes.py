"""Compare direct 3-class classification with regression predictions binned to classes.

Uses existing out-of-fold (OOF) predictions only; it does not retrain or save a model.
Each outer fold's label cutoffs are learned from that fold's training participants and
applied unchanged to that fold's held-out participants. These remain exploratory
relative rating bands, not clinically validated stress severity cutoffs.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

BASE = Path("data/processed/model_comparison")
REG_PATH = BASE / "out_of_fold_predictions.csv"
CLS_PATH = BASE / "classification_oof_predictions.csv"
CUT_PATH = BASE / "classification_label_cutoffs.csv"
OUT_PATH = BASE / "regression_to_classes_comparison.csv"
FOLD_OUT_PATH = BASE / "regression_to_classes_fold_metrics.csv"
LABELS = [0, 1, 2]
NAMES = {0: "Low", 1: "Moderate", 2: "High"}
KEYS = ["outer_fold", "participant_id", "session_id", "task"]


def label_score(value: float, lower: float, upper: float) -> int:
    if value <= lower:
        return 0
    if value <= upper:
        return 1
    return 2


def calc_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
    }


def attach_fold_cutoffs(frame: pd.DataFrame, cutoffs: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Attach fold cutoffs while avoiding duplicate-column suffixes.

    The classification OOF file already stores both cutoff columns per row.
    The regression OOF file may not, so this accepts either schema and verifies
    pre-existing values against the authoritative one-row-per-fold cutoff file.
    """
    lower_col = "lower_rating_cutoff"
    upper_col = "upper_rating_cutoff"
    has_lower = lower_col in frame.columns
    has_upper = upper_col in frame.columns
    if has_lower != has_upper:
        raise ValueError(
            f"{source_name} OOF file has only one of {lower_col}/{upper_col}."
        )

    if not has_lower:
        return frame.merge(cutoffs, on="outer_fold", how="left", validate="many_to_one")

    expected = cutoffs.rename(columns={
        lower_col: "_expected_lower_rating_cutoff",
        upper_col: "_expected_upper_rating_cutoff",
    })
    checked = frame.merge(expected, on="outer_fold", how="left", validate="many_to_one")
    if checked[["_expected_lower_rating_cutoff", "_expected_upper_rating_cutoff"]].isna().any().any():
        raise ValueError(f"Missing authoritative fold cutoffs for {source_name} OOF predictions.")
    if checked[[lower_col, upper_col]].isna().any().any():
        raise ValueError(f"Missing embedded fold cutoffs in {source_name} OOF predictions.")
    if not np.allclose(checked[lower_col], checked["_expected_lower_rating_cutoff"]):
        raise ValueError(f"Embedded lower cutoffs do not match the cutoff file for {source_name}.")
    if not np.allclose(checked[upper_col], checked["_expected_upper_rating_cutoff"]):
        raise ValueError(f"Embedded upper cutoffs do not match the cutoff file for {source_name}.")
    return checked.drop(columns=["_expected_lower_rating_cutoff", "_expected_upper_rating_cutoff"])


def main():
    for path in [REG_PATH, CLS_PATH, CUT_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required report not found: {path}")

    reg = pd.read_csv(REG_PATH)
    cls = pd.read_csv(CLS_PATH)
    cuts = pd.read_csv(CUT_PATH)

    required_reg = set(KEYS + ["model", "actual_rating", "predicted_rating"])
    required_cls = set(KEYS + ["model", "actual_rating", "actual_class_id", "predicted_class_id"])
    required_cuts = {"outer_fold", "lower_rating_cutoff", "upper_rating_cutoff"}
    if not required_reg.issubset(reg.columns):
        raise ValueError(f"Regression OOF file missing: {sorted(required_reg - set(reg.columns))}")
    if not required_cls.issubset(cls.columns):
        raise ValueError(f"Classification OOF file missing: {sorted(required_cls - set(cls.columns))}")
    if not required_cuts.issubset(cuts.columns):
        raise ValueError(f"Cutoff file missing: {sorted(required_cuts - set(cuts.columns))}")

    cuts = cuts[["outer_fold", "lower_rating_cutoff", "upper_rating_cutoff"]].copy()
    if cuts["outer_fold"].duplicated().any():
        raise ValueError("Expected one cutoff record per outer fold.")
    if reg.duplicated(["model"] + KEYS).any():
        raise ValueError("Duplicate model/fold/participant/task rows in regression OOF predictions.")
    if cls.duplicated(["model"] + KEYS).any():
        raise ValueError("Duplicate model/fold/participant/task rows in classification OOF predictions.")

    # Confirm the same participant/task was held out in the same fold for both experiments.
    reg_keys = reg[KEYS + ["actual_rating"]].drop_duplicates()
    cls_keys = cls[KEYS + ["actual_rating"]].drop_duplicates()
    check = reg_keys.merge(cls_keys, on=KEYS, how="outer", suffixes=("_reg", "_cls"), indicator=True)
    if not (check["_merge"] == "both").all():
        raise ValueError("Regression and classification OOF rows/fold assignments do not match.")
    if not np.allclose(check["actual_rating_reg"], check["actual_rating_cls"]):
        raise ValueError("Actual ratings differ between regression and classification OOF results.")

    cls = attach_fold_cutoffs(cls, cuts, "classification")
    if cls[["lower_rating_cutoff", "upper_rating_cutoff"]].isna().any().any():
        raise ValueError("Missing fold-specific rating cutoffs.")
    recomputed_actual = [
        label_score(rating, low, high)
        for rating, low, high in zip(
            cls["actual_rating"], cls["lower_rating_cutoff"], cls["upper_rating_cutoff"]
        )
    ]
    if not np.array_equal(np.asarray(recomputed_actual), cls["actual_class_id"].to_numpy(dtype=int)):
        raise ValueError("Stored actual class labels do not match fold-specific cutoffs.")

    # Direct classifier OOF predictions.
    comparison_rows = []
    fold_rows = []
    for model, group in cls.groupby("model", sort=False):
        y_true = group["actual_class_id"].to_numpy(dtype=int)
        y_pred = group["predicted_class_id"].to_numpy(dtype=int)
        full = calc_metrics(y_true, y_pred)
        per_fold = []
        for fold, fold_group in group.groupby("outer_fold"):
            values = calc_metrics(
                fold_group["actual_class_id"].to_numpy(dtype=int),
                fold_group["predicted_class_id"].to_numpy(dtype=int),
            )
            per_fold.append(values)
            fold_rows.append({
                "approach": "Direct classifier",
                "model": model,
                "outer_fold": int(fold),
                "samples": len(fold_group),
                **values,
            })
        comparison_rows.append({
            "approach": "Direct classifier",
            "model": model,
            "oof_samples": len(group),
            "oof_participants": group["participant_id"].nunique(),
            **full,
            "mean_fold_macro_f1": np.mean([item["macro_f1"] for item in per_fold]),
            "std_fold_macro_f1": np.std([item["macro_f1"] for item in per_fold]),
            "mean_fold_balanced_accuracy": np.mean([item["balanced_accuracy"] for item in per_fold]),
        })

    # Convert regression OOF predictions to the same per-fold relative classes.
    reg = attach_fold_cutoffs(reg, cuts, "regression")
    if reg[["lower_rating_cutoff", "upper_rating_cutoff"]].isna().any().any():
        raise ValueError("Missing fold-specific cutoffs for regression predictions.")
    reg["actual_class_id"] = [
        label_score(rating, low, high)
        for rating, low, high in zip(
            reg["actual_rating"], reg["lower_rating_cutoff"], reg["upper_rating_cutoff"]
        )
    ]
    reg["predicted_class_id"] = [
        label_score(prediction, low, high)
        for prediction, low, high in zip(
            reg["predicted_rating"], reg["lower_rating_cutoff"], reg["upper_rating_cutoff"]
        )
    ]

    for model, group in reg.groupby("model", sort=False):
        y_true = group["actual_class_id"].to_numpy(dtype=int)
        y_pred = group["predicted_class_id"].to_numpy(dtype=int)
        full = calc_metrics(y_true, y_pred)
        per_fold = []
        for fold, fold_group in group.groupby("outer_fold"):
            values = calc_metrics(
                fold_group["actual_class_id"].to_numpy(dtype=int),
                fold_group["predicted_class_id"].to_numpy(dtype=int),
            )
            per_fold.append(values)
            fold_rows.append({
                "approach": "Regression then bin",
                "model": model,
                "outer_fold": int(fold),
                "samples": len(fold_group),
                **values,
            })
        comparison_rows.append({
            "approach": "Regression then bin",
            "model": model,
            "oof_samples": len(group),
            "oof_participants": group["participant_id"].nunique(),
            **full,
            "mean_fold_macro_f1": np.mean([item["macro_f1"] for item in per_fold]),
            "std_fold_macro_f1": np.std([item["macro_f1"] for item in per_fold]),
            "mean_fold_balanced_accuracy": np.mean([item["balanced_accuracy"] for item in per_fold]),
        })

    summary = pd.DataFrame(comparison_rows).sort_values(
        ["mean_fold_macro_f1", "macro_f1"], ascending=False
    )
    fold_df = pd.DataFrame(fold_rows)
    summary.to_csv(OUT_PATH, index=False)
    fold_df.to_csv(FOLD_OUT_PATH, index=False)

    print("=== DIRECT CLASSIFICATION VS REGRESSION-THEN-BIN ===")
    print("Uses existing OOF predictions; no models were retrained.")
    print("All participants remain held out by the existing grouped folds.")
    print("Same fold-specific training-rating cutoffs were applied to both approaches.")
    print("Labels are exploratory relative self-report bands, not clinically validated severity labels.\n")
    cols = ["approach", "model", "oof_samples", "oof_participants", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "mean_fold_macro_f1", "std_fold_macro_f1"]
    print(summary[cols].round(3).to_string(index=False))
    print("\n=== FOLD-WISE MACRO-F1 ===")
    print(fold_df.pivot_table(index=["approach", "model"], columns="outer_fold", values="macro_f1").round(3).to_string())
    print(f"\nSaved summary: {OUT_PATH}")
    print(f"Saved fold metrics: {FOLD_OUT_PATH}")
    print("No final/deployment model was saved.")


if __name__ == "__main__":
    main()
