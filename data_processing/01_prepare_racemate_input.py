#!/usr/bin/env python3
"""Select high-confidence racemates and map each sample to one mzML file."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from provenance import write_provenance

TARGET_STEREOCHEM = "Racemic diastereomer with known relative stereochemistry"
PROTON_MASS = 1.007276466621
OUTPUT_COLUMNS = [
    "Compound_ID", "SMILES", "Stereochem", "Chemical_Name", "Formula",
    "MONOISOTOPIC_MASS", "Ion_Mode", "Adduct", "MZ", "Plate_ID",
    "Plate_Number", "Well", "Sample_Key", "Raw_File", "Raw_Path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chemical-list", type=Path, required=True)
    parser.add_argument("--mzml-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--stereochem", default=TARGET_STEREOCHEM)
    parser.add_argument("--proton-mass", type=float, default=PROTON_MASS)
    parser.add_argument("--expected-count", type=int, default=None,
                        help="Fail unless this many rows are selected.")
    return parser.parse_args()


def plate_number(value: str) -> int:
    match = re.fullmatch(r"P0*(\d+)", value.strip(), re.IGNORECASE)
    if not match:
        raise ValueError(f"Unsupported Plate_ID_simpled value: {value!r}")
    return int(match.group(1))


def index_mzml(directory: Path) -> dict[tuple[int, str], Path]:
    pattern = re.compile(r"_PL(\d+)([A-H]\d{2})_", re.IGNORECASE)
    index: dict[tuple[int, str], Path] = {}
    for path in sorted(directory.rglob("*.mzML")):
        match = pattern.search(path.name)
        if not match:
            continue
        key = (int(match.group(1)), match.group(2).upper())
        if key in index:
            raise ValueError(f"Duplicate mzML mapping for {key}: {index[key]} and {path}")
        index[key] = path.resolve()
    return index


def main() -> None:
    args = parse_args()
    raw_files = index_mzml(args.mzml_dir)
    rows: list[dict[str, object]] = []
    used_files: set[Path] = set()
    with args.chemical_list.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"SMILES", "SGC ID for component", "PLATE_ID", "Well",
                    "Plate_ID_simpled", "Stereochem.data", "Chemical name",
                    "formula", "MONOISOTOPIC_MASS"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Chemical list is missing columns: {sorted(missing)}")
        for source in reader:
            if source["Stereochem.data"].strip() != args.stereochem:
                continue
            plate = plate_number(source["Plate_ID_simpled"])
            well = source["Well"].strip().upper()
            raw_path = raw_files.get((plate, well))
            if raw_path is None:
                raise FileNotFoundError(f"No mzML found for plate {plate}, well {well}")
            mass = float(source["MONOISOTOPIC_MASS"])
            used_files.add(raw_path)
            rows.append({
                "Compound_ID": source["SGC ID for component"].strip(),
                "SMILES": source["SMILES"].strip(),
                "Stereochem": source["Stereochem.data"].strip(),
                "Chemical_Name": source["Chemical name"].strip(),
                "Formula": source["formula"].strip(),
                "MONOISOTOPIC_MASS": f"{mass:.7f}", "Ion_Mode": "positive",
                "Adduct": "[M+H]+", "MZ": f"{mass + args.proton_mass:.7f}",
                "Plate_ID": source["PLATE_ID"].strip(), "Plate_Number": plate,
                "Well": well, "Sample_Key": f"PL{plate}{well}",
                "Raw_File": raw_path.name,
                "Raw_Path": raw_path.relative_to(args.mzml_dir.resolve().parent).as_posix(),
            })
    rows.sort(key=lambda row: str(row["Compound_ID"]))
    ids = [str(row["Compound_ID"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate Compound_ID values in selected rows")
    if args.expected_count is not None and len(rows) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} rows, found {len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    write_provenance(args.provenance, command=Path(__file__).name,
                     parameters={"stereochem": args.stereochem,
                                 "proton_mass_da": args.proton_mass,
                                 "selected_rows": len(rows)},
                     inputs=[args.chemical_list, *used_files], outputs=[args.output],
                     packages=[])
    print(f"Wrote {len(rows)} compounds mapped to {len(used_files)} mzML files: {args.output}")


if __name__ == "__main__":
    main()
