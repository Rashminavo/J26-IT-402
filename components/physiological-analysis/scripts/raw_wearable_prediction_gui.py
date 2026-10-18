import csv
import sys
import threading
from collections import Counter
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physiological_analysis.raw_window_inference import predict_session_windows

REQUIRED_FILES = ("HR.csv", "EDA.csv", "BVP.csv", "TEMP.csv", "ACC.csv")
OUTPUT_DIR = ROOT / "data" / "processed" / "model_comparison"

prediction_rows = []

app = tk.Tk()
app.title("Raw Wearable Stress Predictor — Research Prototype")
app.geometry("1180x790")
app.minsize(900, 650)

folder_var = tk.StringVar()
status_var = tk.StringVar(value="Select a raw wearable session folder.")
summary_var = tk.StringVar(value="No predictions generated yet.")
detail_var = tk.StringVar(
    value="Select a predicted window in the table to inspect its features."
)

ttk.Label(
    app,
    text="Raw Wearable Stress Predictor",
    font=("Segoe UI", 18, "bold"),
).pack(pady=(12, 4))

ttk.Label(
    app,
    text=(
        "Raw sensor CSVs → validated 15-second windows → feature extraction "
        "→ saved SVC model predictions"
    ),
    wraplength=1100,
).pack(padx=10, pady=(0, 8))

ttk.Label(
    app,
    text=(
        "Supported input: one Empatica-style session folder containing "
        "HR.csv, EDA.csv, BVP.csv, TEMP.csv and ACC.csv."
    ),
    wraplength=1100,
).pack(padx=10, pady=(0, 8))

folder_frame = ttk.LabelFrame(app, text="1. Select Wearable Session", padding=10)
folder_frame.pack(fill="x", padx=12, pady=5)

folder_entry = ttk.Entry(folder_frame, textvariable=folder_var)
folder_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))

def browse_folder():
    selected = filedialog.askdirectory(title="Select raw wearable session folder")
    if not selected:
        return

    folder_var.set(selected)
    session_dir = Path(selected)
    missing = [
        name for name in REQUIRED_FILES
        if not (session_dir / name).is_file()
    ]

    if missing:
        status_var.set("Missing required files: " + ", ".join(missing))
    else:
        status_var.set("Required sensor files found. Ready to predict.")

ttk.Button(
    folder_frame, text="Browse Folder...", command=browse_folder
).pack(side="right")

action_frame = ttk.Frame(app)
action_frame.pack(fill="x", padx=12, pady=6)

results_frame = ttk.LabelFrame(app, text="2. Window Predictions", padding=8)
results_frame.pack(fill="both", expand=True, padx=12, pady=5)

columns = (
    "start", "end", "predicted", "hr", "eda", "bvp", "temp", "acc"
)
tree = ttk.Treeview(
    results_frame, columns=columns, show="headings", height=15
)

headers = {
    "start": ("Start (s)", 85),
    "end": ("End (s)", 85),
    "predicted": ("Predicted class", 125),
    "hr": ("HR mean", 105),
    "eda": ("EDA mean", 105),
    "bvp": ("BVP mean", 105),
    "temp": ("Temp mean", 105),
    "acc": ("ACC magnitude mean", 150),
}

for column, (heading, width) in headers.items():
    tree.heading(column, text=heading)
    tree.column(column, width=width, minwidth=65, anchor="center")

vertical_scroll = ttk.Scrollbar(
    results_frame, orient="vertical", command=tree.yview
)
horizontal_scroll = ttk.Scrollbar(
    results_frame, orient="horizontal", command=tree.xview
)
tree.configure(
    yscrollcommand=vertical_scroll.set,
    xscrollcommand=horizontal_scroll.set,
)

tree.grid(row=0, column=0, sticky="nsew")
vertical_scroll.grid(row=0, column=1, sticky="ns")
horizontal_scroll.grid(row=1, column=0, sticky="ew")
results_frame.rowconfigure(0, weight=1)
results_frame.columnconfigure(0, weight=1)

summary_label = ttk.Label(
    app, textvariable=summary_var, wraplength=1120, justify="left"
)
summary_label.pack(fill="x", padx=15, pady=5)

detail_frame = ttk.LabelFrame(
    app, text="Selected Window Details", padding=6
)
detail_frame.pack(fill="x", padx=12, pady=4)

detail_text = tk.Text(detail_frame, height=5, wrap="word", state="disabled")
detail_scroll = ttk.Scrollbar(
    detail_frame, orient="vertical", command=detail_text.yview
)
detail_text.configure(yscrollcommand=detail_scroll.set)
detail_text.pack(side="left", fill="both", expand=True)
detail_scroll.pack(side="right", fill="y")

def show_details(event=None):
    selected = tree.selection()
    if not selected:
        return

    index = int(selected[0].split("-", 1)[1])
    row = prediction_rows[index]

    lines = [
        f"Window: {row['window_start_seconds']}–"
        f"{row['window_end_seconds']} seconds",
        f"Prediction: {row['predicted_class']} | "
        f"Model: {row['model_name']}",
        "",
        "Extracted features:",
    ]

    feature_keys = [
        key for key in row
        if key not in {
            "participant_id", "session_id",
            "window_start_seconds", "window_end_seconds",
            "predicted_class", "class_id", "model_name",
            "research_use_only", "clinical_validation", "warning",
        }
    ]

    lines.extend(f"{key}: {row[key]}" for key in feature_keys)
    lines.extend([
        "",
        f"Research use only: {row['research_use_only']}",
        f"Clinical validation: {row['clinical_validation']}",
        row["warning"],
    ])

    detail_text.configure(state="normal")
    detail_text.delete("1.0", tk.END)
    detail_text.insert("1.0", "\n".join(lines))
    detail_text.configure(state="disabled")

tree.bind("<<TreeviewSelect>>", show_details)

def finish_error(message):
    predict_button.configure(state="normal")
    browse_button.configure(state="normal")
    export_button.configure(state="normal" if prediction_rows else "disabled")
    status_var.set("Prediction failed.")
    messagebox.showerror("Raw wearable prediction error", message)

def finish_predictions(rows, summary):
    global prediction_rows
    prediction_rows = rows

    for item in tree.get_children():
        tree.delete(item)

    def fmt(value):
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return str(value)

    for index, row in enumerate(rows):
        tree.insert(
            "",
            "end",
            iid=f"row-{index}",
            values=(
                fmt(row["window_start_seconds"]),
                fmt(row["window_end_seconds"]),
                row["predicted_class"],
                fmt(row["heart_rate_mean"]),
                fmt(row["eda_eda_mean"]),
                fmt(row["bvp_bvp_mean"]),
                fmt(row["temp_temp_mean"]),
                fmt(row["acc_magnitude_mean"]),
            ),
        )

    counts = Counter(row["predicted_class"] for row in rows)
    summary_var.set(
        f"Predictions generated: {len(rows)} | "
        f"Candidate common windows: {summary.get('candidate_common_windows', 'N/A')} | "
        f"Incomplete windows rejected: "
        f"{summary.get('rejected_incomplete_windows', 'N/A')}\n"
        f"Class counts: {dict(counts)}"
    )

    status_var.set(
        "Prediction complete. Select a row to inspect its features, "
        "or export all predicted windows to CSV."
    )

    predict_button.configure(state="normal")
    browse_button.configure(state="normal")
    export_button.configure(state="normal")

    if tree.get_children():
        first_item = tree.get_children()[0]
        tree.selection_set(first_item)
        tree.focus(first_item)
        show_details()

def prediction_worker(session_dir):
    try:
        rows = predict_session_windows(session_dir=session_dir)
        summary = dict(getattr(predict_session_windows, "last_summary", {}))
        app.after(0, finish_predictions, rows, summary)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        app.after(0, finish_error, message)

def start_prediction():
    session_text = folder_var.get().strip()

    if not session_text:
        messagebox.showwarning(
            "Select folder", "Please select a wearable session folder first."
        )
        return

    session_dir = Path(session_text).expanduser()

    if not session_dir.is_dir():
        messagebox.showerror(
            "Invalid folder", f"Folder not found:\n{session_dir}"
        )
        return

    missing = [
        name for name in REQUIRED_FILES
        if not (session_dir / name).is_file()
    ]
    if missing:
        messagebox.showerror(
            "Missing sensor files",
            "The selected folder is missing:\n" + "\n".join(missing),
        )
        return

    predict_button.configure(state="disabled")
    browse_button.configure(state="disabled")
    export_button.configure(state="disabled")
    status_var.set("Processing sensor files and generating predictions...")
    summary_var.set("Please wait while the model processes the session.")

    for item in tree.get_children():
        tree.delete(item)

    threading.Thread(
        target=prediction_worker,
        args=(session_dir,),
        daemon=True,
    ).start()

def export_csv():
    if not prediction_rows:
        messagebox.showwarning(
            "No results", "Generate predictions before exporting."
        )
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suggested_name = (
        f"{Path(folder_var.get()).name}_stress_predictions.csv"
        if folder_var.get().strip()
        else "raw_stress_predictions.csv"
    )

    output_path = filedialog.asksaveasfilename(
        title="Export all window predictions",
        initialdir=str(OUTPUT_DIR),
        initialfile=suggested_name,
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
    )
    if not output_path:
        return

    try:
        fieldnames = list(prediction_rows[0].keys())
        with Path(output_path).open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(prediction_rows)

        status_var.set(f"Exported {len(prediction_rows)} predictions to CSV.")
        messagebox.showinfo(
            "Export complete",
            f"Saved {len(prediction_rows)} predicted windows to:\n{output_path}",
        )
    except (OSError, csv.Error) as exc:
        messagebox.showerror("CSV export failed", str(exc))

predict_button = ttk.Button(
    action_frame, text="Predict Raw Session", command=start_prediction
)
predict_button.pack(side="left", padx=(0, 8))

browse_button = next(
    widget for widget in folder_frame.winfo_children()
    if isinstance(widget, ttk.Button)
)

export_button = ttk.Button(
    action_frame, text="Export Results CSV...", command=export_csv,
    state="disabled"
)
export_button.pack(side="left", padx=8)

ttk.Label(
    app, textvariable=status_var, wraplength=1120, justify="left"
).pack(fill="x", padx=15, pady=(5, 4))

ttk.Label(
    app,
    text=(
        "Research prototype only. Predicted classes are exploratory relative "
        "bands of self-reported task ratings, not clinically validated "
        "stress-severity assessments. This GUI expects the research dataset's "
        "Empatica-style CSV format."
    ),
    wraplength=1120,
    justify="left",
).pack(fill="x", padx=15, pady=(0, 12))

app.mainloop()
