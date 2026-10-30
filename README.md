# Physiological Analysis Component

Part of the **AI Mental Health Framework** (Project ID: J26-IT-402).

This component provides data validation, physiological feature extraction, signal windowing, and exploratory wearable-based stress classification. It is designed to process physiological measurements and to provide a structured result that can be consumed by the wider framework.

> **Research-use notice:** The current machine-learning outputs are exploratory categories derived from self-reported task ratings. They are not clinically validated stress-severity assessments and must not be used for diagnosis or treatment decisions.

## 1. Component scope

The component contains two related processing paths. They should not be confused:

### A. Reference physiological-analysis pipeline

The reference pipeline in `src/physiological_analysis/pipeline.py` follows these stages:

1. Validate the supplied physiological measurements.
2. Extract descriptive physiological features.
3. Pass the features through the standardized-feature boundary.
4. Return the analysis result using the component contract.

The current reference implementation marks normalization and Random Forest stress classification as pending. The contract defines the output structure, but it does not by itself calculate a completed physiological stress score.

### B. Provisional window-level SVC inference path

The separate window-level inference path uses a saved scikit-learn pipeline to predict one exploratory class for each 15-second window:

`Raw wearable CSVs → timestamp alignment → 15-second windows → 32 features → StandardScaler → RBF SVC → predicted class`

The three output classes are:

- `Low`
- `Moderate`
- `High`

These are relative bands derived from self-reported stress ratings in the development data. They are not clinical labels.

## 2. Sensor data and features

The current 32-feature window-level model uses the following sensor groups:

| Sensor | Features | Count |
|---|---|---:|
| Heart rate (HR) | Mean, minimum, maximum, standard deviation | 4 |
| Electrodermal activity (EDA) | Mean, minimum, maximum, standard deviation | 4 |
| Blood volume pulse (BVP) | Mean, minimum, maximum, standard deviation | 4 |
| Temperature (TEMP) | Mean, minimum, maximum, standard deviation | 4 |
| Accelerometer (ACC) | Mean, minimum, maximum, standard deviation for X, Y, Z and magnitude | 16 |
| **Total** | | **32** |

For raw-session inference, the expected Empatica-style files are:

- `HR.csv`
- `EDA.csv`
- `BVP.csv`
- `TEMP.csv`
- `ACC.csv`

The raw-window inference code aligns the sensor streams on a shared timeline using BVP as the reference time, creates 15-second windows with a 7.5-second stride, rejects incomplete windows, extracts the features, and predicts each valid window.

A 15-second window with a 7.5-second stride overlaps the previous window by 50%. Predictions are returned per window; the current raw-window function does **not** calculate one final session-level label.

## 3. Machine-learning model

The provisional window-level model is an SVC with an RBF kernel inside a scikit-learn pipeline:

`StandardScaler → SVC (RBF kernel)`

The recorded `accuracy619` model configuration is:

- Kernel: `rbf`
- `C`: `0.5`
- `gamma`: `scale`
- Input: one 15-second window represented by 32 named physiological features
- Output: one class ID mapped to `Low`, `Moderate`, or `High`

The model bundle is stored with its feature schema, class mapping, rating cut-offs, model metadata, and research-use warnings. The SVC is configured without probability calibration; its class prediction should not be presented as a calibrated probability or confidence percentage.

### Saved model artifacts

The separately saved candidate used for the recorded 61.90% cross-validation experiment is:

```text
data/processed/model_comparison/provisional_window_level_svc_accuracy619.joblib
```

Related files:

```text
data/processed/model_comparison/provisional_window_level_svc_accuracy619_metadata.json
data/processed/model_comparison/provisional_window_level_svc_accuracy619_tuning.csv
```

**Model-path note:** `WindowLevelStressClassifier` has a default artifact path pointing to `provisional_window_level_svc.joblib`. To use the `accuracy619` artifact in raw-session inference, pass its path explicitly with the `--model` argument. Do not assume the default path selects the `accuracy619` artifact.

## 4. Training data and evaluation

The recorded window-level SVC experiment used the processed labelled dataset:

```text
data/processed/f_stress_15s_labeled_windows.csv
```

Recorded dataset summary for this experiment:

- 6,928 labelled windows
- 18 participants
- 19 sessions
- 126 unique participant-task self-reported ratings
- 32 numeric physiological features per window

The processed labels are derived from participant-reported task ratings. In the evaluation workflow, rating cut-offs are learned from unique participant-task ratings in the training portion of each fold. Participant grouping is used so that windows from the same participant are not split across the training and held-out groups.

### Evaluation result (exploratory)

The recorded nested participant-grouped cross-validation result for the selected 32-feature SVC workflow was:

| Metric | Result |
|---|---:|
| Task-level accuracy | 61.90% |
| Balanced accuracy | 50.57% |
| Macro F1-score | 50.66% |
| Weighted F1-score | 58.10% |
| Supplementary window-level accuracy | 62.50% |

**Interpretation:** 61.90% is a cross-validation estimate from the evaluation workflow. It is not an independent test-set score for the final model refitted on all development data. The final full-data model fit does not create a new unbiased performance estimate. Moderate-class recall was limited in the recorded evaluation, so the overall accuracy should not be interpreted on its own.

### Model comparison

Candidate classifier and feature-ablation experiments are implemented in the scripts under `scripts/`. The main no-class-balance window-level evaluation script is:

```text
scripts/train_window_level_accuracy_no_class_balance_20261020.py
```

Its comparison report is:

```text
data/processed/window_level_accuracy_no_class_balance_20261020/window_level_classification_comparison.csv
```

The experiments compare candidate approaches using metrics including accuracy, balanced accuracy, macro F1-score, and class-wise performance. Different experiment scripts may use different feature sets or training settings; compare results only when their datasets, splits, labels, and evaluation levels are aligned.

## 5. Data sources

The research repository contains datasets downloaded from public research resources:

- **PhysioNet Wearable Device Dataset, version 1.0.1:** <https://physionet.org/content/wearable-device-dataset/1.0.1/>
- **WESAD (Wearable Stress and Affect Detection):** <https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detect>

The recorded labelled-window dataset used for the SVC experiment is derived from the PhysioNet stress-session data. WESAD is also present in the research data collection, but its presence in the repository does not mean that it was combined with the data used to train this specific SVC model. Any future cross-dataset use requires checking sensor compatibility, feature definitions, sampling rates, label meaning, and the dataset's usage terms.

Please follow the original dataset providers' citation and usage requirements when using or redistributing the source data. Do not commit restricted or large raw datasets to Git unless the project and dataset terms explicitly allow it.

## 6. Project structure

The principal source modules are:

```text
src/physiological_analysis/
├── __init__.py
├── contract.py                  # Shared output structure
├── data_loader.py               # Sensor CSV loading and parsing
├── features.py                  # Physiological feature extraction
├── normalization.py             # Standardized-feature boundary
├── pipeline.py                  # Reference analysis pipeline
├── raw_window_inference.py      # Raw sensor CSV to window predictions
├── stress_classifier.py         # Exploratory task-level classifier wrapper
├── validation.py                # Measurement validation
├── window_stress_classifier.py  # Saved window-level SVC wrapper
└── windowing.py                 # Time-window segmentation and checks
```

Other useful paths:

```text
scripts/                          # Training, evaluation and demo scripts
tests/                            # Component unit and integration tests
data/raw/physionet/               # Raw PhysioNet data (locally provided)
data/raw/wesad/                   # Raw WESAD data (locally provided)
data/processed/                   # Processed feature tables and model outputs
```

The actual `data/` contents depend on which datasets and generated artifacts are available in the local working copy.

## 7. Environment setup

The development environment used for this component was Python 3.12 on Windows with PowerShell.

From the `components/physiological-analysis` directory:

```powershell
# Create a virtual environment (skip if .venv already exists)
py -3.12 -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1

# Install dependencies declared by the component
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The model-training and inference workflow uses packages including NumPy, pandas, scikit-learn, and joblib. The test suite uses pytest. Make sure the component's dependency file includes the packages required by the scripts you intend to run.

## 8. Run tests

Run from the component root:

```powershell
python -m pytest -q
```

To use the explicitly configured virtual environment without activating it:

```powershell
& ".\.venv\Scripts\python.exe" -m pytest -q
```

## 9. Run the raw wearable-session inference script

The raw-session CLI expects a folder containing all five required CSV files (`HR.csv`, `EDA.csv`, `BVP.csv`, `TEMP.csv`, and `ACC.csv`). From the component root:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "src")

& ".\.venv\Scripts\python.exe" ".\scripts\predict_raw_wearable_session.py" `
  --session-dir "PATH_TO_SESSION_FOLDER" `
  --model ".\data\processed\model_comparison\provisional_window_level_svc_accuracy619.joblib"
```

Replace `PATH_TO_SESSION_FOLDER` with the actual directory containing the five sensor CSV files. The script writes per-window predictions to a CSV output file. It does not produce one aggregated session-level stress label.

## 10. Run the offline prediction GUI

If the processed feature dataset and the GUI script are present in the local checkout, run:

```powershell
& ".\.venv\Scripts\python.exe" ".\scripts\manual_stress_prediction_gui_accuracy619.py"
```

This GUI is an offline demonstration that selects a preprocessed feature row and predicts that selected window. It is not a live wearable stream and does not automatically aggregate predictions across all windows in a session.

## 11. Re-run the recorded model evaluation or fit

To re-run the nested participant-grouped evaluation:

```powershell
& ".\.venv\Scripts\python.exe" ".\scripts\train_window_level_accuracy_no_class_balance_20261020.py"
```

To run the final provisional SVC fitting script associated with the `accuracy619` artifact:

```powershell
& ".\.venv\Scripts\python.exe" ".\scripts\train_provisional_window_svc_accuracy619_20261020.py"
```

These scripts expect the processed labelled-window dataset to be available at the documented path. Re-running the final-fit script can overwrite its output artifact and metadata files. The final full-data fit is not an independent test and should not be used to claim a new unbiased accuracy score.

## 12. Limitations and future work

- The `Low`, `Moderate`, and `High` outputs are exploratory relative bands based on self-reported task ratings, not clinically validated diagnoses or severity levels.
- The reported 61.90% is a participant-grouped cross-validation estimate, not independent external validation of the final saved model.
- Moderate-class recall was limited in the recorded evaluation and needs further investigation.
- The current GUI predicts a selected preprocessed window; it is not a real-time wearable application.
- The raw-window inference path produces a prediction for each valid window. A validated session-level aggregation method has not been implemented in this path.
- The reference `pipeline.py` still marks normalization and Random Forest classification as pending; its output contract does not imply that a completed physiological stress score is already calculated.
- Applying the model to a different dataset requires compatible sensor measurements, feature extraction, sampling rates, units, and label interpretation. A prediction on another dataset alone does not establish external accuracy or generalisation.
- Further external validation, larger and more diverse participant datasets, and appropriate ethical review for any new human data collection are required before any clinical use is considered.

## 13. Responsible use

This component is intended for academic research and software prototyping only. It is not a medical device, does not provide a clinical diagnosis, and must not be used as the sole basis for health-related decisions.
