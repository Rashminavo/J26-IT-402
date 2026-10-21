"""Dataset-source-neutral adapters for annotation tables.

This module deliberately does not assume any CASME II workbook column names and
has no pandas/openpyxl dependency. A caller reads a source table into row mappings,
then supplies an explicit mapping from canonical field names to source headers.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from .dataset_manifest import (
    MANIFEST_COLUMNS,
    TEMPORAL_COLUMNS,
    DatasetManifestError,
    DatasetSample,
    validate_dataset_records,
)

CANONICAL_SOURCE_FIELDS = tuple(
    column for column in MANIFEST_COLUMNS if column != "source_dataset"
)
REQUIRED_MAPPED_FIELDS = ("sample_id", "subject_id", "native_label")


class AnnotationAdapterError(ValueError):
    """Raised when a source-to-canonical annotation mapping is invalid."""


def records_from_table_rows(
    table_rows: Iterable[Sequence[object]],
) -> list[dict[str, object]]:
    """Convert a tabular iterator (for example, worksheet.iter_rows(values_only=True))
    into row mappings using its first row as the exact header row.

    Header spelling is preserved apart from trimming outer whitespace. Empty and
    duplicate header names are rejected. Fully blank rows are skipped; non-empty
    rows with a different number of cells are rejected rather than misaligned.
    """
    iterator = iter(table_rows)
    try:
        raw_header = next(iterator)
    except StopIteration as exc:
        raise AnnotationAdapterError("Annotation table is empty; no header row found.") from exc

    headers = ["" if value is None else str(value).strip() for value in raw_header]
    if not headers or all(not name for name in headers):
        raise AnnotationAdapterError("Annotation table has an empty header row.")
    if any(not name for name in headers):
        raise AnnotationAdapterError("Annotation table contains an empty column header.")
    if len(headers) != len(set(headers)):
        raise AnnotationAdapterError("Annotation table contains duplicate column headers.")

    records: list[dict[str, object]] = []
    for row_number, raw_values in enumerate(iterator, start=2):
        values = list(raw_values)
        if not any(value is not None and str(value).strip() for value in values):
            continue
        if len(values) != len(headers):
            raise AnnotationAdapterError(
                f"Annotation table row {row_number} has {len(values)} cells; "
                f"expected {len(headers)}."
            )
        records.append(dict(zip(headers, values)))
    return records


def _validate_mapping(
    column_map: Mapping[str, str],
    *,
    video_path_resolver: Callable[[Mapping[str, object]], str] | None,
) -> dict[str, str]:
    if not isinstance(column_map, Mapping):
        raise AnnotationAdapterError("column_map must be a mapping of canonical fields to source headers.")

    unknown_fields = sorted(set(column_map) - set(CANONICAL_SOURCE_FIELDS))
    if unknown_fields:
        raise AnnotationAdapterError(
            "Unsupported canonical field(s) in column_map: " + ", ".join(unknown_fields)
        )

    cleaned: dict[str, str] = {}
    for canonical_field, source_header in column_map.items():
        if not isinstance(source_header, str) or not source_header.strip():
            raise AnnotationAdapterError(
                f"Source header for '{canonical_field}' must be a non-empty string."
            )
        cleaned[canonical_field] = source_header.strip()

    for field in REQUIRED_MAPPED_FIELDS:
        if field not in cleaned:
            raise AnnotationAdapterError(
                f"column_map must explicitly map required field '{field}'."
            )

    if video_path_resolver is None and "video_path" not in cleaned:
        raise AnnotationAdapterError(
            "Map 'video_path' explicitly or provide video_path_resolver."
        )
    if video_path_resolver is not None and "video_path" in cleaned:
        raise AnnotationAdapterError(
            "Use either a 'video_path' column mapping or video_path_resolver, not both."
        )

    temporal_mapped = [field in cleaned for field in TEMPORAL_COLUMNS]
    if any(temporal_mapped) and not all(temporal_mapped):
        raise AnnotationAdapterError(
            "Map onset_frame, apex_frame, and offset_frame together, or omit all three."
        )

    duplicate_headers: dict[str, list[str]] = {}
    for canonical_field, source_header in cleaned.items():
        duplicate_headers.setdefault(source_header, []).append(canonical_field)
    ambiguous = [
        f"'{header}' -> {', '.join(fields)}"
        for header, fields in duplicate_headers.items()
        if len(fields) > 1
    ]
    if ambiguous:
        raise AnnotationAdapterError(
            "A source header cannot map to multiple canonical fields: " + "; ".join(ambiguous)
        )
    return cleaned


def adapt_annotation_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    source_dataset: str,
    column_map: Mapping[str, str],
    video_path_resolver: Callable[[Mapping[str, object]], str] | None = None,
    data_root: str | Path | None = None,
    require_video_files: bool = False,
) -> list[DatasetSample]:
    """Map source-specific row dictionaries into the canonical manifest schema.

    ``column_map`` keys are canonical field names; values are exact source header
    names from the annotation file. Native labels and IDs are not remapped. The
    resulting rows are passed through the existing canonical manifest validator.
    """
    dataset_name = str(source_dataset).strip()
    if not dataset_name:
        raise AnnotationAdapterError("source_dataset must not be empty.")

    cleaned_map = _validate_mapping(
        column_map,
        video_path_resolver=video_path_resolver,
    )

    canonical_rows: list[dict[str, object]] = []
    for row_number, source_row in enumerate(rows, start=2):
        if not isinstance(source_row, Mapping):
            raise AnnotationAdapterError(
                f"Annotation row {row_number} must be a mapping of headers to values."
            )

        canonical: dict[str, object] = {column: "" for column in MANIFEST_COLUMNS}
        canonical["source_dataset"] = dataset_name
        for canonical_field, source_header in cleaned_map.items():
            canonical[canonical_field] = source_row.get(source_header, "")

        if video_path_resolver is not None:
            try:
                canonical["video_path"] = video_path_resolver(source_row)
            except Exception as exc:
                raise AnnotationAdapterError(
                    f"Could not resolve video_path for annotation row {row_number}: {exc}"
                ) from exc

        canonical_rows.append(canonical)

    try:
        return validate_dataset_records(
            canonical_rows,
            data_root=data_root,
            require_video_files=require_video_files,
        )
    except DatasetManifestError as exc:
        raise AnnotationAdapterError(str(exc)) from exc


def write_canonical_manifest(
    samples: Iterable[DatasetSample],
    output_path: str | Path,
) -> Path:
    """Write validated samples to a canonical UTF-8 CSV manifest."""
    path = Path(output_path)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "source_dataset": sample.source_dataset,
                    "sample_id": sample.sample_id,
                    "subject_id": sample.subject_id,
                    "video_path": sample.video_path,
                    "native_label": sample.native_label,
                    "onset_frame": "" if sample.onset_frame is None else sample.onset_frame,
                    "apex_frame": "" if sample.apex_frame is None else sample.apex_frame,
                    "offset_frame": "" if sample.offset_frame is None else sample.offset_frame,
                    "fps": "" if sample.fps is None else sample.fps,
                    "action_units": "" if sample.action_units is None else sample.action_units,
                }
            )
    return path
