
from pathlib import Path

from language_analysis.dataset import load_labeled_split


zip_path = Path.home() / "Downloads" / "daicwoz.zip"

for split in ("train", "dev"):
    result = load_labeled_split(zip_path, split)

    print(f"\n{split.upper()} SPLIT")
    print("Loaded participants:", result["loaded_count"])
    print("Missing transcript count:", result["missing_transcript_count"])
    print("Missing participant IDs:", result["missing_transcript_ids"])

    if result["participants"]:
        sample = result["participants"][0]
        print("First participant ID:", sample["participant_id"])
        print("PHQ-8 binary label:", sample["phq8_binary"])
        print("PHQ-8 score:", sample["phq8_score"])
        print("Transcript characters:", sample["character_count"])
