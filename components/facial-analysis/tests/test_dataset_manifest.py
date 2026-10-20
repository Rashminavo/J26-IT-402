import csv

import pytest

from facial_analysis.dataset_manifest import (
    MANIFEST_COLUMNS,
    DatasetManifestError,
    DatasetSample,
    load_dataset_manifest,
    validate_dataset_records,
)


def valid_row(**overrides):
    row = {
        "source_dataset": "CASME_II",
        "sample_id": "sample_001",
        "subject_id": "sub01",
        "video_path": "sub01/sample_001.avi",
        "native_label": "Disgust",
        "onset_frame": "10",
        "apex_frame": "15",
        "offset_frame": "20",
        "fps": "200",
        "action_units": "AU4+AU9",
    }
    row.update(overrides)
    return row


def test_validate_valid_record_preserves_native_label_and_ids():
    [sample] = validate_dataset_records([valid_row()])

    assert isinstance(sample, DatasetSample)
    assert sample.source_dataset == "CASME_II"
    assert sample.sample_id == "sample_001"
    assert sample.subject_id == "sub01"
    assert sample.native_label == "Disgust"
    assert (sample.onset_frame, sample.apex_frame, sample.offset_frame) == (10, 15, 20)
    assert sample.fps == 200.0
    assert sample.video_path == "sub01/sample_001.avi"


def test_blank_action_units_and_missing_temporal_annotations_are_allowed():
    row = valid_row(onset_frame="", apex_frame="", offset_frame="", fps="", action_units="")
    [sample] = validate_dataset_records([row])

    assert sample.onset_frame is None
    assert sample.apex_frame is None
    assert sample.offset_frame is None
    assert sample.fps is None
    assert sample.action_units is None


def test_duplicate_dataset_sample_id_is_rejected():
    with pytest.raises(DatasetManifestError, match="duplicate sample_id"):
        validate_dataset_records([valid_row(), valid_row(subject_id="sub02")])


def test_same_sample_id_from_different_datasets_is_allowed():
    rows = [valid_row(), valid_row(source_dataset="SAMM")]
    samples = validate_dataset_records(rows)
    assert len(samples) == 2


@pytest.mark.parametrize(
    "overrides",
    [
        {"sample_id": ""},
        {"subject_id": "  "},
        {"native_label": ""},
        {"source_dataset": ""},
        {"video_path": ""},
    ],
)
def test_required_text_fields_must_not_be_blank(overrides):
    with pytest.raises(DatasetManifestError, match="must not be empty"):
        validate_dataset_records([valid_row(**overrides)])


@pytest.mark.parametrize(
    "overrides",
    [
        {"onset_frame": "21", "apex_frame": "15", "offset_frame": "20"},
        {"onset_frame": "10", "apex_frame": "22", "offset_frame": "20"},
        {"onset_frame": "-1", "apex_frame": "15", "offset_frame": "20"},
        {"onset_frame": "1.5", "apex_frame": "15", "offset_frame": "20"},
    ],
)
def test_invalid_temporal_annotation_is_rejected(overrides):
    with pytest.raises(DatasetManifestError):
        validate_dataset_records([valid_row(**overrides)])


def test_partial_temporal_annotation_is_rejected():
    with pytest.raises(DatasetManifestError, match="must either all be supplied"):
        validate_dataset_records([valid_row(apex_frame="", offset_frame="")])


@pytest.mark.parametrize("fps", ["0", "-1", "nan", "inf", "not-a-number"])
def test_invalid_fps_is_rejected(fps):
    with pytest.raises(DatasetManifestError, match="fps"):
        validate_dataset_records([valid_row(fps=fps)])


@pytest.mark.parametrize("video_path", ["../outside.avi", "sub01/../../outside.avi", r"C:\\dataset\\video.avi", r"\\server\share\video.avi", "/absolute/video.avi"])
def test_unsafe_video_paths_are_rejected(video_path):
    with pytest.raises(DatasetManifestError, match="relative|traverse"):
        validate_dataset_records([valid_row(video_path=video_path)])


def test_video_file_existence_can_be_required(tmp_path):
    video = tmp_path / "sub01" / "sample_001.avi"
    video.parent.mkdir()
    video.write_bytes(b"test fixture")

    samples = validate_dataset_records(
        [valid_row()], data_root=tmp_path, require_video_files=True
    )
    assert samples[0].sample_id == "sample_001"

    with pytest.raises(DatasetManifestError, match="does not exist"):
        validate_dataset_records(
            [valid_row(video_path="sub01/missing.avi")],
            data_root=tmp_path,
            require_video_files=True,
        )


def test_require_video_files_needs_data_root():
    with pytest.raises(DatasetManifestError, match="data_root is required"):
        validate_dataset_records([valid_row()], require_video_files=True)


def test_load_csv_manifest_and_header_only_template(tmp_path):
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerow(valid_row())

    [sample] = load_dataset_manifest(manifest)
    assert sample.sample_id == "sample_001"

    template = tmp_path / "template.csv"
    with template.open("w", encoding="utf-8", newline="") as file:
        csv.writer(file).writerow(MANIFEST_COLUMNS)
    assert load_dataset_manifest(template) == []


def test_manifest_header_must_match_canonical_schema(tmp_path):
    manifest = tmp_path / "bad.csv"
    manifest.write_text("sample_id,subject_id\nabc,sub01\n", encoding="utf-8")

    with pytest.raises(DatasetManifestError, match="Invalid manifest header"):
        load_dataset_manifest(manifest)


def test_unknown_record_columns_are_rejected():
    row = valid_row(unmapped_source_column="value")
    with pytest.raises(DatasetManifestError, match="unexpected column"):
        validate_dataset_records([row])
