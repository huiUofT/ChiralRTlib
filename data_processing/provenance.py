"""Deterministic provenance helpers for the public data-processing scripts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any, Iterable


def sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path, base: Path | None = None) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to((base or Path.cwd()).resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def file_record(path: Path, base: Path | None = None) -> dict[str, Any]:
    return {
        "path": display_path(path, base),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def software_versions(packages: Iterable[str]) -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def write_provenance(
    path: Path,
    *,
    command: str,
    parameters: dict[str, Any],
    inputs: Iterable[Path],
    outputs: Iterable[Path],
    packages: Iterable[str],
) -> None:
    base = Path.cwd()
    document = {
        "schema_version": 1,
        "command": command,
        "parameters": parameters,
        "software": software_versions(packages),
        "inputs": [file_record(item, base) for item in sorted(inputs)],
        "outputs": [file_record(item, base) for item in sorted(outputs)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")

