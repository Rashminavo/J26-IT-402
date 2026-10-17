import csv
import io
import zipfile
from pathlib import Path

zip_path = Path.home() / "Downloads" / "daicwoz.zip"

split_files = {
    "train": "daicwoz/train_split_Depression_AVEC2017.csv",
    "dev": "daicwoz/dev_split_Depression_AVEC2017.csv",
    "test": "daicwoz/test_split_Depression_AVEC2017.csv",
}

with zipfile.ZipFile(zip_path) as archive:
    archive_names = set(archive.namelist())
    participant_sets = {}

    for split_name, file_path in split_files.items():
        with archive.open(file_path) as file:
            text_file = io.TextIOWrapper(
                file,
                encoding="utf-8-sig",
                errors="replace",
            )
            rows = list(csv.DictReader(text_file))

        if not rows:
            print(f"{split_name}: no participant rows found")
            participant_sets[split_name] = set()
            continue

        id_column = next(
            (
                name
                for name in ("Participant_ID", "participant_ID")
                if name in rows[0]
            ),
            None,
        )

        if id_column is None:
            print(f"{split_name}: participant ID column not found")
            participant_sets[split_name] = set()
            continue

        ids = {
            str(row[id_column]).strip()
            for row in rows
            if row.get(id_column)
        }
        participant_sets[split_name] = ids

        found = {
            participant_id
            for participant_id in ids
            if f"daicwoz/{participant_id}_TRANSCRIPT.csv" in archive_names
        }

        print(
            f"{split_name}: participants={len(ids)}, "
            f"transcripts_found={len(found)}, "
            f"transcripts_missing={len(ids - found)}"
        )

    print(
        "Train/dev overlap:",
        len(participant_sets["train"] & participant_sets["dev"]),
    )
    print(
        "Train/test overlap:",
        len(participant_sets["train"] & participant_sets["test"]),
    )
    print(
        "Dev/test overlap:",
        len(participant_sets["dev"] & participant_sets["test"]),
    )