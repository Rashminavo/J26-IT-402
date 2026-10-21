"""Canonical metadata manifest handling for facial-analysis datasets.

This module validates metadata only. It does not decode videos, download datasets,
remap source labels, or produce model/risk scores.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Mapping, Sequence


MANIFEST_COLUMNS = (
    "source_dataset",
    "sample_id",
    "subject_id",
    "video_path",
    "native_label",
    "onset_frame",
    "apex_frame",
    "offset_frame",
    "fps",
    "action_units",
)

REQUIRED_TEXT_COLUMNS = (
    "source_dataset",
    "sample_id",
    "subject_id",
    "video_path",
    "native_label",
)
TEMPORAL_COLUMNS = ("onset_frame", "apex_frame", "offset_frame")


class DatasetManifestError(ValueError):
    """Raised when a canonical dataset manifest is invalid."""


@dataclass(frozen=True)
class DatasetSample:
    """One annotated facial-expression event from a source dataset.

    ``native_label`` and ``subject_id`` are preserved rather than mapped into a
    different label scheme. Frame annotations and FPS may be absent for datasets
    that do not publish them, but the three frame annotations must be supplied
    together when any one of them is supplied.
    """

    source_dataset: str
    sample_id: str
    subject_id: str
    video_path: str
    native_label: str
    onset_frame: int | None = None
    apex_frame: int | None = None
    offset_frame: int | None = None
    fps: float | None = None
    action_units: str | None = None


def _text_value(row: Mapping[str, object], column: str, row_number: int) -> str:
    value = row.get(column)
    text_value = "" if value is None else str(value).strip()
    if not text_value:
        raise DatasetManifestError(
            f"Row {row_number}: '{column}' must not be empty."
        )
    return text_value


def _normalise_relative_video_path(value: object, row_number: int) -> str:
    raw = "" if value is None else str(value).strip()
    if not raw:
        raise DatasetManifestError(
            f"Row {row_number}: 'video_path' must not be empty."
        )

    # Canonical manifests use forward slashes regardless of host operating system.
    normalised = raw.replace("\\", "/")
    posix_path = PurePosixPath(normalised)
    windows_path = PureWindowsPath(raw)

    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
    ):
        raise DatasetManifestError(
            f"Row {row_number}: 'video_path' must be relative to the dataset root."
        )

    if normalised in {"", "."} or ".." in posix_path.parts:
        raise DatasetManifestError(
            f"Row {row_number}: 'video_path' must not be empty or traverse parent directories."
        )

    return posix_path.as_posix()


def _optional_frame_values(
    row: Mapping[str, object], row_number: int
) -> tuple[int | None, int | None, int | None]:
    raw_values = [
        "" if row.get(column) is None else str(row.get(column)).strip()
        for column in TEMPORAL_COLUMNS
    ]
    supplied = [bool(value) for value in raw_values]

    if any(supplied) and not all(supplied):
        raise DatasetManifestError(
            f"Row {row_number}: onset_frame, apex_frame, and offset_frame "
            "must either all be supplied or all be blank."
        )

    if not any(supplied):
        return None, None, None

    parsed: list[int] = []
    for column, value in zip(TEMPORAL_COLUMNS, raw_values):
        try:
            frame_number = int(value)
        except (TypeError, ValueError) as exc:
            raise DatasetManifestError(
                f"Row {row_number}: '{column}' must be a non-negative integer."
            ) from exc
        if frame_number < 0:
            raise DatasetManifestError(
                f"Row {row_number}: '{column}' must be a non-negative integer."
            )
        parsed.append(frame_number)

    onset, apex, offset = parsed
    if not onset <= apex <= offset:
        raise DatasetManifestError(
            f"Row {row_number}: frame order must satisfy "
            "onset_frame <= apex_frame <= offset_frame."
        )

    return onset, apex, offset


def _optional_fps(row: Mapping[str, object], row_number: int) -> float | None:
    raw = "" if row.get("fps") is None else str(row.get("fps")).strip()
    if not raw:
        return None

    try:
        fps = float(raw)
    except (TypeError, ValueError) as exc:
        raise DatasetManifestError(
            f"Row {row_number}: 'fps' must be a finite positive number."
        ) from exc

    if not math.isfinite(fps) or fps <= 0:
        raise DatasetManifestError(
            f"Row {row_number}: 'fps' must be a finite positive number."
        )
    return fps


def _parse_sample(row: Mapping[str, object], row_number: int) -> DatasetSample:
    unknown_columns = set(row) - set(MANIFEST_COLUMNS)
    if unknown_columns:
        unknown = ", ".join(sorted(unknown_columns))
        raise DatasetManifestError(
            f"Row {row_number}: unexpected column(s): {unknown}. "
            "Map source-specific annotation columns into the canonical schema first."
        )

    text_values = {
        column: _text_value(row, column, row_number)
        for column in REQUIRED_TEXT_COLUMNS
    }
    video_path = _normalise_relative_video_path(row.get("video_path"), row_number)
    onset, apex, offset = _optional_frame_values(row, row_number)
    fps = _optional_fps(row, row_number)

    action_units_value = row.get("action_units")
    action_units_text = (
        "" if action_units_value is None else str(action_units_value).strip()
    )

    return DatasetSample(
        source_dataset=text_values["source_dataset"],
        sample_id=text_values["sample_id"],
        subject_id=text_values["subject_id"],
        video_path=video_path,
        native_label=text_values["native_label"],
        onset_frame=onset,
        apex_frame=apex,
        offset_frame=offset,
        fps=fps,
        action_units=action_units_text or None,
    )


def validate_dataset_records(
    rows: Sequence[Mapping[str, object]],
    *,
    data_root: str | Path | None = None,
    require_video_files: bool = False,
) -> list[DatasetSample]:
    """Validate canonical row dictionaries and return typed samples.

    Paths are checked for absolute paths and parent-directory traversal even when
    file existence is not required. When ``data_root`` is provided, resolved paths
    must stay inside that root (including through existing symlinks). Set
    ``require_video_files=True`` only when the authorised dataset files are locally
    available and should be checked on disk.
    """
    if require_video_files and data_root is None:
        raise DatasetManifestError(
            "data_root is required when require_video_files=True."
        )

    resolved_root = Path(data_root).expanduser().resolve() if data_root is not None else None
    samples: list[DatasetSample] = []
    seen_keys: dict[tuple[str, str], int] = {}
    errors: list[str] = []

    for row_number, row in enumerate(rows, start=2):
        try:
            sample = _parse_sample(row, row_number)
            key = (sample.source_dataset, sample.sample_id)
            if key in seen_keys:
                raise DatasetManifestError(
                    f"Row {row_number}: duplicate sample_id '{sample.sample_id}' "
                    f"for source_dataset '{sample.source_dataset}' "
                    f"(first seen on row {seen_keys[key]})."
                )

            if resolved_root is not None:
                relative_parts = PurePosixPath(sample.video_path).parts
                resolved_video = resolved_root.joinpath(*relative_parts).resolve()
                if not resolved_video.is_relative_to(resolved_root):
                    raise DatasetManifestError(
                        f"Row {row_number}: 'video_path' resolves outside data_root."
                    )
                if require_video_files and not resolved_video.is_file():
                    raise DatasetManifestError(
                        f"Row {row_number}: video file does not exist: "
                        f"{sample.video_path}."
                    )

            seen_keys[key] = row_number
            samples.append(sample)
        except DatasetManifestError as exc:
            errors.append(str(exc))

    if errors:
        raise DatasetManifestError("\n".join(errors))

    return samples


def load_dataset_manifest(
    manifest_path: str | Path,
    *,
    data_root: str | Path | None = None,
    require_video_files: bool = False,
) -> list[DatasetSample]:
    """Read and validate a UTF-8 CSV in the canonical manifest format.

    A header-only template is accepted and returns an empty list. Training code
    should explicitly reject an empty result before starting an experiment.
    """
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise DatasetManifestError("Manifest CSV is empty and has no header.")

        fieldnames = [name if name is not None else "" for name in reader.fieldnames]
        if len(fieldnames) != len(set(fieldnames)):
            raise DatasetManifestError("Manifest CSV contains duplicate column names.")

        missing_columns = [column for column in MANIFEST_COLUMNS if column not in fieldnames]
        unexpected_columns = [column for column in fieldnames if column not in MANIFEST_COLUMNS]
        if missing_columns or unexpected_columns:
            details: list[str] = []
            if missing_columns:
                details.append("missing column(s): " + ", ".join(missing_columns))
            if unexpected_columns:
                details.append("unexpected column(s): " + ", ".join(unexpected_columns))
            raise DatasetManifestError("Invalid manifest header; " + "; ".join(details) + ".")

        rows = [dict(row) for row in reader]

    return validate_dataset_records(
        rows,
        data_root=data_root,
        require_video_files=require_video_files,
    )
