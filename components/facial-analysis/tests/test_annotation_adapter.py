import csv

import pytest

from facial_analysis.annotation_adapter import (
    AnnotationAdapterError,
    adapt_annotation_rows,
    records_from_table_rows,
    write_canonical_manifest,
)
from facial_analysis.dataset_manifest import load_dataset_manifest


def source_row(**overrides):
    row = {
        "Clip ID": "sample_001",
        "Participant": "sub01",
        "Emotion (original)": "Disgust",
        "Relative clip file": "sub01/sample_001.avi",
        "Onset frame": 10,
        "Apex frame": 15,
        "Offset frame": 20,
        "Sampling rate": 200,
        "AUs": "AU4+AU9",
        "Irrelevant source note": "ignored explicitly",
    }
    row.update(overrides)
    return row


MAPPING = {
    "sample_id": "Clip ID",
    "subject_id": "Participant",
    "native_label": "Emotion (original)",
    "video_path": "Relative clip file",
    "onset_frame": "Onset frame",
    "apex_frame": "Apex frame",
    "offset_frame": "Offset frame",
    "fps": "Sampling rate",
    "action_units": "AUs",
}


def test_adapter_uses_explicit_mapping_and_preserves_native_labels():
    [sample] = adapt_annotation_rows(
        [source_row()], source_dataset="SYNTHETIC_TEST", column_map=MAPPING
    )
    assert sample.source_dataset == "SYNTHETIC_TEST"
    assert sample.sample_id == "sample_001"
    assert sample.subject_id == "sub01"
    assert sample.native_label == "Disgust"
    assert (sample.onset_frame, sample.apex_frame, sample.offset_frame) == (10, 15, 20)
    assert sample.fps == 200.0
    assert sample.action_units == "AU4+AU9"


def test_adapter_does_not_need_actual_dataset_or_external_packages():
    [sample] = adapt_annotation_rows(
        [source_row(onset="", **{})],
        source_dataset="SYNTHETIC_TEST",
        column_map={
            "sample_id": "Clip ID",
            "subject_id": "Participant",
            "native_label": "Emotion (original)",
            "video_path": "Relative clip file",
        },
    )
    assert sample.onset_frame is None
    assert sample.native_label == "Disgust"


@pytest.mark.parametrize(
    "mapping, message",
    [
        ({"subject_id": "Participant", "native_label": "Emotion (original)", "video_path": "Relative clip file"}, "sample_id"),
        ({"sample_id": "Clip ID", "subject_id": "Participant", "native_label": "Emotion (original)"}, "video_path"),
        ({"sample_id": "Clip ID", "subject_id": "Participant", "native_label": "Emotion (original)", "video_path": "Relative clip file", "onset_frame": "Onset frame"}, "Map onset_frame, apex_frame"),
        ({"sample_id": "Clip ID", "subject_id": "Participant", "native_label": "Emotion (original)", "video_path": "Relative clip file", "not_a_field": "X"}, "Unsupported canonical field"),
        ({"sample_id": "ID", "subject_id": "ID", "native_label": "Emotion (original)", "video_path": "Relative clip file"}, "cannot map to multiple"),
    ],
)
def test_invalid_column_mapping_is_rejected(mapping, message):
    with pytest.raises(AnnotationAdapterError, match=message):
        adapt_annotation_rows([source_row()], source_dataset="SYNTHETIC_TEST", column_map=mapping)


def test_video_path_resolver_can_be_used_instead_of_a_path_column():
    mapping = {key: value for key, value in MAPPING.items() if key != "video_path"}
    [sample] = adapt_annotation_rows(
        [source_row()],
        source_dataset="SYNTHETIC_TEST",
        column_map=mapping,
        video_path_resolver=lambda row: f"{row['Participant']}/{row['Clip ID']}.avi",
    )
    assert sample.video_path == "sub01/sample_001.avi"


def test_resolver_and_video_path_column_are_mutually_exclusive():
    with pytest.raises(AnnotationAdapterError, match="not both"):
        adapt_annotation_rows(
            [source_row()],
            source_dataset="SYNTHETIC_TEST",
            column_map=MAPPING,
            video_path_resolver=lambda row: "sub01/sample_001.avi",
        )


def test_missing_mapped_source_value_is_reported_by_manifest_validator():
    with pytest.raises(AnnotationAdapterError, match="sample_id.*must not be empty"):
        adapt_annotation_rows(
            [source_row(**{"Clip ID": ""})],
            source_dataset="SYNTHETIC_TEST",
            column_map=MAPPING,
        )


def test_table_rows_use_exact_headers_and_skip_blank_rows():
    records = records_from_table_rows(
        [
            (" Clip ID ", "Emotion original", "Participant"),
            ("clip-1", "repression", "p01"),
            (None, None, None),
        ]
    )
    assert records == [{"Clip ID": "clip-1", "Emotion original": "repression", "Participant": "p01"}]


@pytest.mark.parametrize(
    "rows, message",
    [
        ([], "no header row"),
        ([(None, "")], "empty header"),
        ([("id", "id")], "duplicate column headers"),
        ([("id", "label"), ("only-one-cell",)], "expected 2"),
    ],
)
def test_invalid_tabular_headers_or_rows_are_rejected(rows, message):
    with pytest.raises(AnnotationAdapterError, match=message):
        records_from_table_rows(rows)


def test_write_and_load_canonical_manifest_round_trip(tmp_path):
    samples = adapt_annotation_rows(
        [source_row()], source_dataset="SYNTHETIC_TEST", column_map=MAPPING
    )
    path = write_canonical_manifest(samples, tmp_path / "manifest.csv")
    [loaded] = load_dataset_manifest(path)
    assert loaded == samples[0]


def test_written_csv_uses_canonical_header(tmp_path):
    samples = adapt_annotation_rows(
        [source_row()], source_dataset="SYNTHETIC_TEST", column_map=MAPPING
    )
    path = write_canonical_manifest(samples, tmp_path / "manifest.csv")
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
    assert header == [
        "source_dataset", "sample_id", "subject_id", "video_path", "native_label",
        "onset_frame", "apex_frame", "offset_frame", "fps", "action_units",
    ]
