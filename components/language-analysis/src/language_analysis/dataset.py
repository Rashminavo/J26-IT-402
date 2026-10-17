
import csv
import io
import zipfile
from pathlib import Path
from typing import Any


def load_participant_transcript(
    zip_path: str | Path,
    participant_id: str | int,
    participant_only: bool = True,
) -> dict[str, Any]:
    """Load one participant's transcript directly from the DAIC-WOZ ZIP."""
    participant_id = str(participant_id)
    transcript_path = f"daicwoz/{participant_id}_TRANSCRIPT.csv"

    with zipfile.ZipFile(zip_path, "r") as archive:
        try:
            with archive.open(transcript_path) as file:
                text_file = io.TextIOWrapper(
                    file,
                    encoding="utf-8-sig",
                    errors="replace",
                )
                reader = csv.DictReader(text_file, delimiter="\t")
                rows = list(reader)
                fieldnames = reader.fieldnames
        except KeyError as exc:
            raise FileNotFoundError(
                f"Transcript not found for participant {participant_id}"
            ) from exc

    if not fieldnames or not {
        "start_time", "stop_time", "speaker", "value"
    }.issubset(fieldnames):
        raise ValueError(f"Unexpected transcript format: {transcript_path}")

    if participant_only:
        rows = [row for row in rows if row["speaker"] == "Participant"]

    transcript = " ".join(
        row["value"].strip()
        for row in rows
        if row["value"].strip()
    )

    return {
        "participant_id": participant_id,
        "transcript": transcript,
        "row_count": len(rows),
        "character_count": len(transcript),
    }


def load_labeled_split(
    zip_path: str | Path,
    split: str,
    participant_only: bool = True,
) -> dict[str, Any]:
    """Load labeled train/dev participants with available transcripts."""
    split_files = {
        "train": "daicwoz/train_split_Depression_AVEC2017.csv",
        "dev": "daicwoz/dev_split_Depression_AVEC2017.csv",
    }

    if split not in split_files:
        raise ValueError("split must be 'train' or 'dev'")

    loaded = []
    missing_transcripts = []

    with zipfile.ZipFile(zip_path, "r") as archive:
        with archive.open(split_files[split]) as file:
            text_file = io.TextIOWrapper(
                file,
                encoding="utf-8-sig",
                errors="replace",
            )
            rows = list(csv.DictReader(text_file))

        available_files = set(archive.namelist())

        for row in rows:
            participant_id = str(row.get("Participant_ID", "")).strip()

            if not participant_id:
                continue

            transcript_path = f"daicwoz/{participant_id}_TRANSCRIPT.csv"

            if transcript_path not in available_files:
                missing_transcripts.append(participant_id)
                continue

            loaded.append({
                "participant_id": participant_id,
                "phq8_binary": int(row["PHQ8_Binary"]),
                "phq8_score": int(row["PHQ8_Score"]),
            })

    for participant in loaded:
        transcript_data = load_participant_transcript(
            zip_path,
            participant["participant_id"],
            participant_only=participant_only,
        )
        participant["transcript"] = transcript_data["transcript"]
        participant["character_count"] = transcript_data["character_count"]

    return {
        "split": split,
        "participants": loaded,
        "loaded_count": len(loaded),
        "missing_transcript_ids": missing_transcripts,
        "missing_transcript_count": len(missing_transcripts),
    }
