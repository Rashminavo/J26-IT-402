from pathlib import Path

from physiological_analysis.data_loader import (
    load_ibi_csv,
    load_tags_csv,
)

root = Path("data/raw/physionet")
matches = list(root.rglob("STRESS/f01"))

if not matches:
    raise FileNotFoundError("STRESS/f01 folder not found")

participant_dir = matches[0]

ibi = load_ibi_csv(participant_dir / "IBI.csv")
tags = load_tags_csv(participant_dir / "tags.csv")

print("=== IBI SUMMARY ===")
print("Recording start:", ibi["start_timestamp"])
print("IBI samples:", ibi["sample_count"])
print("First 5 IBI readings:", ibi["readings"][:5])
print("Last 5 IBI readings:", ibi["readings"][-5:])

elapsed = [row[0] for row in ibi["readings"]]
print("First elapsed time:", elapsed[0])
print("Last elapsed time:", elapsed[-1])
print(
    "Elapsed times strictly increasing:",
    all(a < b for a, b in zip(elapsed, elapsed[1:]))
)

print("\n=== TAGS ===")
print("Tag count:", len(tags))
print("First 5 tags:", tags[:5])
print("Last 5 tags:", tags[-5:])
