from pathlib import Path

import pytest

from language_analysis.dataset import load_participant_transcript


ZIP_PATH = Path.home() / "Downloads" / "daicwoz.zip"


@pytest.mark.skipif(not ZIP_PATH.exists(), reason="DAIC-WOZ ZIP not available")
def test_load_participant_transcript_from_zip():
    result = load_participant_transcript(ZIP_PATH, "300")

    assert result["participant_id"] == "300"
    assert result["row_count"] > 0
    assert result["character_count"] > 0
    assert isinstance(result["transcript"], str)


@pytest.mark.skipif(not ZIP_PATH.exists(), reason="DAIC-WOZ ZIP not available")
def test_transcript_does_not_include_interviewer_rows():
    result = load_participant_transcript(
        ZIP_PATH,
        "300",
        participant_only=True,
    )

    assert result["row_count"] > 0
    assert result["character_count"] > 0


@pytest.mark.skipif(not ZIP_PATH.exists(), reason="DAIC-WOZ ZIP not available")
def test_missing_participant_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_participant_transcript(ZIP_PATH, "999999")


@pytest.mark.skipif(not ZIP_PATH.exists(), reason="DAIC-WOZ ZIP not available")
def test_load_train_split_with_labels():
    from language_analysis.dataset import load_labeled_split

    result = load_labeled_split(ZIP_PATH, "train")

    assert result["split"] == "train"
    assert result["loaded_count"] == 107
    assert result["missing_transcript_count"] == 0
    assert result["participants"]

    participant = result["participants"][0]
    assert "transcript" in participant
    assert isinstance(participant["phq8_binary"], int)
    assert isinstance(participant["phq8_score"], int)


@pytest.mark.skipif(not ZIP_PATH.exists(), reason="DAIC-WOZ ZIP not available")
def test_dev_split_reports_missing_transcript():
    from language_analysis.dataset import load_labeled_split

    result = load_labeled_split(ZIP_PATH, "dev")

    assert result["loaded_count"] == 34
    assert result["missing_transcript_count"] == 1
    assert result["missing_transcript_ids"] == ["458"]


def test_invalid_split_is_rejected(tmp_path):
    from language_analysis.dataset import load_labeled_split

    with pytest.raises(ValueError, match="split must be"):
        load_labeled_split(tmp_path / "unused.zip", "test")
