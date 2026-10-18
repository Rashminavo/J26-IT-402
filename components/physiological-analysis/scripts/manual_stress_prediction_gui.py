import csv
import math
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physiological_analysis.window_stress_classifier import (
    WindowLevelStressClassifier,
)

CSV_PATH = ROOT / "data" / "processed" / "f_stress_15s_window_features.csv"
MODEL_PATH = (
    ROOT / "data" / "processed" / "model_comparison"
    / "provisional_window_level_svc.joblib"
)

if not CSV_PATH.is_file():
    raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")

classifier = WindowLevelStressClassifier.load(model_path=MODEL_PATH)
feature_names = list(classifier.feature_columns)

with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)
    if reader.fieldnames is None:
        raise ValueError("The dataset has no column headers.")
    rows = list(reader)

if not rows:
    raise ValueError("The dataset contains no rows.")

for column in ("participant_id", "session_id",
               "window_start_seconds", "window_end_seconds"):
    if column not in rows[0]:
        raise ValueError(f"Required dataset column missing: {column}")

for name in feature_names:
    if name not in rows[0]:
        raise ValueError(f"Model feature missing from dataset: {name}")

participants = sorted({
    row["participant_id"] for row in rows
    if row.get("participant_id", "").strip()
})

if not participants:
    raise ValueError("No participant IDs found in the dataset.")

app = tk.Tk()
app.title("Physiological Stress Predictor — Research Prototype")
app.geometry("1180x830")
app.minsize(900, 650)

ttk.Label(
    app,
    text="Physiological Stress Predictor",
    font=("Segoe UI", 18, "bold"),
).pack(pady=(12, 4))

ttk.Label(
    app,
    text="Choose a real dataset sample or inspect its extracted features.",
    wraplength=1050,
).pack(pady=(0, 10))

selector = ttk.LabelFrame(app, text="Dataset Sample Selector", padding=10)
selector.pack(fill="x", padx=14, pady=4)

participant_var = tk.StringVar()
session_var = tk.StringVar()
window_var = tk.StringVar()
sample_info = tk.StringVar()
result = tk.StringVar(value="Choose a sample and click Predict Stress Level.")

ttk.Label(selector, text="Participant").grid(
    row=0, column=0, sticky="w", padx=5, pady=5
)
participant_combo = ttk.Combobox(
    selector, textvariable=participant_var,
    values=participants, state="readonly", width=15
)
participant_combo.grid(row=1, column=0, sticky="ew", padx=5)

ttk.Label(selector, text="Session").grid(
    row=0, column=1, sticky="w", padx=5, pady=5
)
session_combo = ttk.Combobox(
    selector, textvariable=session_var,
    state="readonly", width=18
)
session_combo.grid(row=1, column=1, sticky="ew", padx=5)

ttk.Label(selector, text="15-second window").grid(
    row=0, column=2, sticky="w", padx=5, pady=5
)
window_combo = ttk.Combobox(
    selector, textvariable=window_var,
    state="readonly", width=38
)
window_combo.grid(row=1, column=2, sticky="ew", padx=5)

selector.columnconfigure(0, weight=1)
selector.columnconfigure(1, weight=1)
selector.columnconfigure(2, weight=2)

ttk.Label(
    app,
    textvariable=sample_info,
    wraplength=1100,
).pack(fill="x", padx=18, pady=8)

ttk.Label(
    app,
    text=(
        "Edit feature values only when you intend to test a hypothetical input. "
        "Arbitrary values may not represent real physiological measurements."
    ),
    wraplength=1100,
).pack(padx=12, pady=(0, 8))

container = ttk.Frame(app)
container.pack(fill="both", expand=True, padx=12, pady=4)

canvas = tk.Canvas(container, highlightthickness=0)
scrollbar = ttk.Scrollbar(
    container, orient="vertical", command=canvas.yview
)
fields_frame = ttk.Frame(canvas)
window_id = canvas.create_window(
    (0, 0), window=fields_frame, anchor="nw"
)
canvas.configure(yscrollcommand=scrollbar.set)

fields_frame.bind(
    "<Configure>",
    lambda event: canvas.configure(scrollregion=canvas.bbox("all")),
)
canvas.bind(
    "<Configure>",
    lambda event: canvas.itemconfigure(window_id, width=event.width),
)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

entries = {}
for index, name in enumerate(feature_names):
    group = index // 16
    row_index = index % 16
    label_column = group * 2
    entry_column = label_column + 1

    ttk.Label(
        fields_frame, text=name, width=30
    ).grid(
        row=row_index, column=label_column,
        sticky="w", padx=(8, 4), pady=4
    )

    entry = ttk.Entry(fields_frame, width=20)
    entry.grid(
        row=row_index, column=entry_column,
        sticky="ew", padx=(0, 16), pady=4
    )
    entries[name] = entry

window_rows = {}

def load_selected_sample():
    label = window_var.get()
    selected = window_rows.get(label)

    if selected is None:
        return

    try:
        for name, entry in entries.items():
            raw = selected.get(name, "").strip()
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError(f"Non-finite feature value: {name}")
            entry.delete(0, tk.END)
            entry.insert(0, raw)

        info = (
            f"Participant: {selected['participant_id']} | "
            f"Session: {selected['session_id']} | "
            f"Window: {selected['window_start_seconds']}–"
            f"{selected['window_end_seconds']} seconds"
        )

        for optional_column in ("task", "activity"):
            optional_value = selected.get(optional_column, "").strip()
            if optional_value:
                info += f" | {optional_column}: {optional_value}"

        sample_info.set(info)
        result.set("Sample loaded. Click Predict Stress Level to run the model.")

    except (ValueError, TypeError) as exc:
        messagebox.showerror("Dataset input error", str(exc))

def refresh_windows(event=None):
    global window_rows

    participant = participant_var.get()
    session = session_var.get()

    matching_rows = [
        (index, row)
        for index, row in enumerate(rows)
        if row.get("participant_id") == participant
        and row.get("session_id") == session
    ]

    window_rows = {}
    labels = []

    for index, row in matching_rows:
        start = row["window_start_seconds"]
        end = row["window_end_seconds"]
        label = f"{start}–{end} sec | dataset row {index + 1}"

        task_name = row.get("task", "").strip()
        if task_name:
            label = f"{start}–{end} sec | {task_name} | row {index + 1}"

        window_rows[label] = row
        labels.append(label)

    window_combo["values"] = labels

    if labels:
        window_var.set(labels[0])
        load_selected_sample()
    else:
        window_var.set("")
        sample_info.set("No windows available for this session.")
        result.set("Choose another session.")

def refresh_sessions(event=None):
    participant = participant_var.get()
    sessions = sorted({
        row["session_id"] for row in rows
        if row.get("participant_id") == participant
        and row.get("session_id", "").strip()
    })

    session_combo["values"] = sessions
    session_var.set(sessions[0] if sessions else "")
    refresh_windows()

def predict():
    try:
        features = {}

        for name, entry in entries.items():
            raw = entry.get().strip()
            if not raw:
                raise ValueError(f"Please enter a value for {name}.")

            value = float(raw)
            if not math.isfinite(value):
                raise ValueError(f"Value must be finite for {name}.")

            features[name] = value

        prediction = classifier.predict(features)

        result.set(
            f"Predicted class: {prediction['predicted_class']}\n"
            f"Model: {prediction['model_name']} | "
            f"Window size: {prediction['window_size_seconds']} seconds\n"
            f"Research prototype: {prediction['research_use_only']} | "
            f"Clinical validation: {prediction['clinical_validation']}\n"
            f"{prediction['warning']}"
        )

    except (ValueError, TypeError, KeyError) as exc:
        messagebox.showerror("Invalid input", str(exc))
    except Exception as exc:
        messagebox.showerror(
            "Prediction error", f"{type(exc).__name__}: {exc}"
        )

def restore_sample():
    load_selected_sample()

participant_combo.bind("<<ComboboxSelected>>", refresh_sessions)
session_combo.bind("<<ComboboxSelected>>", refresh_windows)
window_combo.bind("<<ComboboxSelected>>", lambda event: load_selected_sample())

buttons = ttk.Frame(app)
buttons.pack(pady=8)

ttk.Button(
    buttons, text="Predict Stress Level", command=predict
).pack(side="left", padx=8)

ttk.Button(
    buttons, text="Restore Selected Sample", command=restore_sample
).pack(side="left", padx=8)

ttk.Label(
    app, textvariable=result,
    wraplength=1120, justify="left"
).pack(fill="x", padx=18, pady=(2, 14))

participant_var.set(participants[0])
refresh_sessions()
app.mainloop()
