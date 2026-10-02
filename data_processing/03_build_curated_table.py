#!/usr/bin/env python3
"""Build curated positive/negative tables from manually classified review plots."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from provenance import display_path, write_provenance

OUTPUT_COLUMNS = [
    "mol_id", "smiles", "column", "stage1_label", "manual_class", "Peak1",
    "Peak2", "delta_rt", "mean_rt", "MZ", "Raw_File", "Plate_Number", "Well",
    "Stereochem", "Formula", "MONOISOTOPIC_MASS", "original_qc_flag",
    "original_num_peaks", "algorithm_peak1", "algorithm_peak2", "review_image",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-table", type=Path, required=True)
    parser.add_argument("--peak-report", type=Path, required=True)
    parser.add_argument("--positive-dir", type=Path, required=True)
    parser.add_argument("--negative-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--column", default="IG")
    parser.add_argument("--expected-positive", type=int, default=None)
    parser.add_argument("--expected-negative", type=int, default=None)
    return parser.parse_args()


def read_csv(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {row[key]: row for row in rows}
    if len(result) != len(rows): raise ValueError(f"Duplicate {key} values in {path}")
    return result


def reviewed(directory: Path, known_ids: set[str]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(directory.glob("*.png")):
        matches = [mol_id for mol_id in known_ids if f"_{mol_id}_" in path.name]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one compound ID in review filename: {path.name}")
        mol_id = matches[0]
        if mol_id in found: raise ValueError(f"Duplicate reviewed image for {mol_id} in {directory}")
        found[mol_id] = path
    return found


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    args = parse_args()
    inputs = read_csv(args.input_table, "Compound_ID")
    reports = read_csv(args.peak_report, "Compound_ID")
    known = set(inputs)
    positive = reviewed(args.positive_dir, known)
    negative = reviewed(args.negative_dir, known)
    overlap = set(positive) & set(negative)
    if overlap: raise ValueError(f"IDs occur in both manual classes: {sorted(overlap)}")
    selected = set(positive) | set(negative)
    missing = selected - set(reports)
    if missing: raise ValueError(f"Reviewed IDs missing from peak report: {sorted(missing)}")
    if args.expected_positive is not None and len(positive) != args.expected_positive:
        raise ValueError(f"Expected {args.expected_positive} positives, found {len(positive)}")
    if args.expected_negative is not None and len(negative) != args.expected_negative:
        raise ValueError(f"Expected {args.expected_negative} negatives, found {len(negative)}")
    rows: list[dict[str, object]] = []
    for mol_id in sorted(selected):
        source, report = inputs[mol_id], reports[mol_id]
        is_positive = mol_id in positive
        first, second = report["Peak1"], report["Peak2"]
        if is_positive and (not first or not second):
            raise ValueError(f"Manual positive lacks two RT values: {mol_id}")
        if is_positive:
            peak1, peak2 = float(first), float(second)
            values: tuple[object, object, object, object] = (
                f"{peak1:.3f}", f"{peak2:.3f}", f"{peak2 - peak1:.3f}",
                f"{(peak1 + peak2) / 2:.4f}")
            label, manual_class, image = 1, "positive_separated", positive[mol_id]
        else:
            values = ("", "", "", "")
            label, manual_class, image = 0, "negative_unseparated", negative[mol_id]
        rows.append({
            "mol_id": mol_id, "smiles": source["SMILES"], "column": args.column,
            "stage1_label": label, "manual_class": manual_class,
            "Peak1": values[0], "Peak2": values[1], "delta_rt": values[2],
            "mean_rt": values[3], "MZ": source["MZ"], "Raw_File": source["Raw_File"],
            "Plate_Number": source["Plate_Number"], "Well": source["Well"],
            "Stereochem": source["Stereochem"], "Formula": source["Formula"],
            "MONOISOTOPIC_MASS": source["MONOISOTOPIC_MASS"],
            "original_qc_flag": report["QC_Flag"],
            "original_num_peaks": report["Num_Peaks"], "algorithm_peak1": first,
            "algorithm_peak2": second, "review_image": display_path(image),
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_path = args.output_dir / "curated.csv"
    positive_path = args.output_dir / "positive.csv"
    negative_path = args.output_dir / "negative.csv"
    write_csv(all_path, rows)
    write_csv(positive_path, [row for row in rows if row["stage1_label"] == 1])
    write_csv(negative_path, [row for row in rows if row["stage1_label"] == 0])
    review_files = [*positive.values(), *negative.values()]
    write_provenance(args.provenance, command=Path(__file__).name,
                     parameters={"column": args.column, "positive_rows": len(positive),
                                 "negative_rows": len(negative),
                                 "unreviewed_rows": len(inputs) - len(rows)},
                     inputs=[args.input_table, args.peak_report, *review_files],
                     outputs=[all_path, positive_path, negative_path], packages=[])
    print(f"Wrote {len(rows)} curated rows ({len(positive)} positive, {len(negative)} negative)")


if __name__ == "__main__":
    main()

