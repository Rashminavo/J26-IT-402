# Source-neutral annotation adapter (C3)

This is a source-neutral adapter milestone for use before the CASME II dataset has been acquired. It deliberately **does not assume any specific CASME II workbook column names** and uses only the Python standard library in the adapter itself.

## What it does

- Converts tabular rows (e.g. `openpyxl` worksheet rows with `values_only=True`) to dictionaries using the exact supplied header names.
- Maps those headers to the existing canonical `DatasetSample` fields via an explicit `column_map`.
- Preserves source/subject/sample IDs and the native label; it does not remap emotion classes.
- Uses the current `validate_dataset_records` validator for required fields, relative paths, frame order, FPS, duplicate IDs and optional on-disk file checks.
- Writes validated samples into the existing canonical CSV manifest format.

It does **not** load `.xlsx` files directly, require `pandas` or `openpyxl`, download dataset files, or assume a particular workbook sheet/column layout. Once the authorised dataset is available, the calling script can read its desired worksheet with an optional spreadsheet reader, inspect the actual header row, and provide an explicit mapping.

## Usage outline when the annotation file becomes available

```python
from openpyxl import load_workbook  # optional dependency in the local environment only
from facial_analysis.annotation_adapter import (
    records_from_table_rows,
    adapt_annotation_rows,
    write_canonical_manifest,
)

workbook = load_workbook("<local-annotation-file.xlsx>", read_only=True, data_only=True)
worksheet = workbook["<confirmed-sheet-name>"]
rows = records_from_table_rows(worksheet.iter_rows(values_only=True))

# Fill these names only after inspecting the actual header row.
column_map = {
    "sample_id": "<actual sample-id header>",
    "subject_id": "<actual subject-id header>",
    "native_label": "<actual native-label header>",
    # Use either a video_path source column OR video_path_resolver below.
    "video_path": "<actual relative-video-path header>",
    # Map all of the following together if the source actually has them:
    # "onset_frame": "<actual onset header>",
    # "apex_frame": "<actual apex header>",
    # "offset_frame": "<actual offset header>",
}

samples = adapt_annotation_rows(
    rows,
    source_dataset="CASME_II",
    column_map=column_map,
    data_root="<local dataset root>",
    require_video_files=True,
)
write_canonical_manifest(samples, "<output manifest path.csv>")
workbook.close()
```

Do not leave placeholders in a real run. Do not commit raw videos or restricted annotations to Git. Dataset access/use terms must be checked before acquiring or processing the data.

## Verification

The companion `tests/test_annotation_adapter.py` uses invented synthetic rows only; it does not claim to represent the real CASME II annotation layout or provide real dataset samples.
