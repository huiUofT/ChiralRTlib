# IG chromatogram data processing

This directory contains the public, command-line version of the data-extraction
workflow used to build a manually curated chromatographic test set. It replaces
the original workstation/HPC-specific paths with explicit arguments and records
the inputs, parameters, software versions, and SHA-256 checksums for every run.

## Workflow

```text
chemical metadata + mzML files
  -> 01_prepare_racemate_input.py
  -> selected_racemates.csv
  -> 02_extract_eic_peaks.py
  -> automated_peak_report.csv + review plots
  -> manual review (positive/ and negative/ folders)
  -> 03_build_curated_table.py
  -> curated.csv + positive.csv + negative.csv
```

The manual-review step is intentionally explicit: moving a review plot into
`positive/` means two chromatographic peaks were accepted; moving it into
`negative/` means the compound was judged unseparated. Unreviewed plots are not
included. The curated-table command records every reviewed image by SHA-256.

## Installation

Python 3.11 was used for the original run. In a clean environment:

```bash
python -m pip install -r data_processing/requirements.txt
```

The requirements file fixes the portable upstream package releases used by the
workflow.

## Run

Commands below assume the repository root is the working directory.

```bash
python data_processing/01_prepare_racemate_input.py \
  --chemical-list /path/to/chemical_metadata.csv \
  --mzml-dir /path/to/mzml \
  --output work/selected_racemates.csv \
  --provenance work/01_prepare.provenance.json

python data_processing/02_extract_eic_peaks.py \
  --input-table work/selected_racemates.csv \
  --mzml-dir /path/to/mzml \
  --output work/automated_peak_report.csv \
  --plot-dir work/review/unreviewed \
  --workers 16 \
  --provenance work/02_extract.provenance.json

# Manually sort plots into work/review/positive and work/review/negative.

python data_processing/03_build_curated_table.py \
  --input-table work/selected_racemates.csv \
  --peak-report work/automated_peak_report.csv \
  --positive-dir work/review/positive \
  --negative-dir work/review/negative \
  --output-dir work/curated \
  --provenance work/03_curate.provenance.json
```

Each command refuses ambiguous mzML mappings, duplicate compound IDs, missing
required columns, and partial extraction results. CSV rows are sorted by stable
keys, so process completion order does not change file content. Provenance JSON
uses paths relative to the invocation directory when possible; timestamps and
host information are excluded so otherwise identical runs have stable metadata.

## Scientific parameters

The extraction defaults reproduce the original workflow:

- positive-mode target: `[M+H]+`, using proton mass `1.007276466621` Da;
- EIC window: `3.0 ppm` around the target m/z;
- minimum peak height: `100,000`;
- Savitzky-Golay window/polyorder: `3/2`;
- minimum peak distance: `20` scans;
- prominence: `2.0 x 100,000`;
- reported peaks: the two most intense detected peaks, ordered by retention
  time, with each position snapped to the local raw-intensity maximum within
  five scans;
- retention time is stored in minutes and rounded to three decimals.

All parameters can be overridden on `02_extract_eic_peaks.py`; overrides are
written into provenance. Changing them creates a different data-processing
protocol and should be reported.

## Traceability

Every processing step produces a provenance JSON containing its parameters,
software versions, and input/output checksums. Verify one of these records from
the same working directory used to run the pipeline:

```bash
python data_processing/verify_manifest.py \
  work/02_extract.provenance.json
```

This directory contains code and documentation only. Raw mzML files, chemical
tables, generated plots, manually reviewed images, derived tables, and
run-specific manifests are not included. Users supply their own data and retain
their generated provenance records alongside their results.
