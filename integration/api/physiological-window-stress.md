# Exploratory physiological window-stress API

## Endpoint

`POST /api/v1/physiological/predict-window-stress`

This endpoint is separate from `/api/v1/physiological/analyze` and does not create a shared numeric component score. It returns exploratory relative bands trained from self-reported task ratings; these classes are **not clinically validated stress-severity categories**.

## Request

```json
{
  "observation_id": "OBS-WINDOW-001",
  "user_id": "USER-001",
  "session_id": "SESSION-001",
  "window_start_seconds": 15.0,
  "window_end_seconds": 30.0,
  "features": {
    "heart_rate_mean": 70.14,
    "heart_rate_min": 69.11,
    "heart_rate_max": 71.71,
    "heart_rate_std": 0.96,
    "eda_eda_mean": 0.28,
    "eda_eda_std": 0.003,
    "eda_eda_min": 0.279,
    "eda_eda_max": 0.292,
    "bvp_bvp_mean": 0.499,
    "bvp_bvp_std": 28.11,
    "bvp_bvp_min": -75.17,
    "bvp_bvp_max": 49.56,
    "temp_temp_mean": 30.04,
    "temp_temp_std": 0.021,
    "temp_temp_min": 30.01,
    "temp_temp_max": 30.07,
    "acc_x_mean": -32.86,
    "acc_x_std": 1.67,
    "acc_x_min": -38.0,
    "acc_x_max": -26.0,
    "acc_y_mean": 16.59,
    "acc_y_std": 0.62,
    "acc_y_min": 15.0,
    "acc_y_max": 18.0,
    "acc_z_mean": 53.63,
    "acc_z_std": 1.18,
    "acc_z_min": 50.0,
    "acc_z_max": 57.0,
    "acc_magnitude_mean": 65.08,
    "acc_magnitude_std": 0.56,
    "acc_magnitude_min": 63.12,
    "acc_magnitude_max": 67.63
  }
}
```

The `features` object must contain the 32 exact model feature names. Window duration must be 15 seconds. These numeric values are illustrative only; real features must be produced by the verified raw-window feature pipeline.

## Response

A successful request returns HTTP `200` with `data.prediction.predicted_class` (`Low`, `Moderate`, or `High`), the class ID, model metadata, window interval, and explicit research-only / non-clinical-validation markers. The endpoint does not write to the shared score store.

Common errors: `400` invalid JSON/body or required fields; `422` invalid window bounds or feature values; `503` model artifact unavailable or prediction service unavailable.

## Model artifact and environment

The `.joblib` model is under the physiologic component's `data/` directory, which is Git-ignored. A deployment must provision that file separately. Optionally set `J26_PHYSIOLOGICAL_MODEL_PATH` to the artifact's full path. Backend requirements pin the scikit-learn, joblib, and NumPy versions used to train/serialize this artifact.

## Limitations

This is an offline/research integration only. Live smartwatch/HealthKit/Health Connect transport is not implemented by this endpoint. The experimental rating bands are not clinically validated, and the model should not be presented as a diagnostic tool.
