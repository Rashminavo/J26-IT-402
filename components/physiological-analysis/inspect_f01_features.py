from pathlib import Path

from physiological_analysis.data_loader import (
    load_heart_rate_csv,
    load_signal_csv,
)
from physiological_analysis.features import (
    extract_heart_rate_features,
    extract_signal_features,
)

root = Path("data/raw/physionet")
matches = list(root.rglob("STRESS/f01"))

if not matches:
    raise FileNotFoundError("Could not find STRESS/f01 dataset folder")

participant_dir = matches[0]

print(f"Participant folder: {participant_dir}")

hr = load_heart_rate_csv(participant_dir / "HR.csv")
hr_features = extract_heart_rate_features(hr)

print("\n=== Heart Rate ===")
print(f"Readings: {len(hr)}")
for name, value in hr_features.items():
    print(f"{name}: {value:.6f}")

for sensor in ("EDA", "BVP", "TEMP", "ACC"):
    signal = load_signal_csv(participant_dir / f"{sensor}.csv")
    features = extract_signal_features(sensor, signal)

    print(f"\n=== {sensor} ===")
    print(f"Sampling rates: {signal['sampling_rates']}")
    print(f"Samples: {signal['sample_count']}")

    for name, value in features.items():
        print(f"{name}: {value:.6f}")
