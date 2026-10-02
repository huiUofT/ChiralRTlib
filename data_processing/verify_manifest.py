#!/usr/bin/env python3
"""Verify files recorded in a pipeline-generated provenance JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from provenance import sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd(),
                        help="Base directory for relative paths (default: current directory).")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    failures: list[str] = []
    records = [*manifest.get("inputs", []), *manifest.get("outputs", [])]
    for record in records:
        recorded_path = Path(record["path"])
        path = recorded_path if recorded_path.is_absolute() else args.root / recorded_path
        if not path.is_file():
            failures.append(f"missing file: {record['path']}")
            continue
        if path.stat().st_size != record["bytes"]:
            failures.append(f"size mismatch: {record['path']}")
            continue
        actual = sha256(path)
        if actual != record["sha256"]: failures.append(f"checksum mismatch: {record['path']}")
    if failures: raise SystemExit("Verification FAILED:\n- " + "\n- ".join(failures))
    print(f"Verification passed: {len(records)} files")


if __name__ == "__main__":
    main()
