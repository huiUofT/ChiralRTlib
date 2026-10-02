#!/usr/bin/env python3
"""Extract targeted EICs from mzML files, detect peaks, and make review plots."""

from __future__ import annotations

import argparse
import csv
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyopenms import MSExperiment, MzMLFile
from scipy.signal import find_peaks, savgol_filter
from tqdm import tqdm

from provenance import write_provenance

REPORT_COLUMNS = ["Compound_ID", "MZ", "File", "Column", "Status", "Num_Peaks",
                  "Peak1", "Peak2", "QC_Flag", "QC_Reason", "Image_Path"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-table", type=Path, required=True)
    parser.add_argument("--mzml-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot-dir", type=Path, default=None)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--column", default="IG")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--ppm", type=float, default=3.0)
    parser.add_argument("--noise-threshold", type=float, default=1e5)
    parser.add_argument("--win-len", type=int, default=3)
    parser.add_argument("--polyorder", type=int, default=2)
    parser.add_argument("--distance", type=int, default=20)
    parser.add_argument("--prom-factor", type=float, default=2.0)
    parser.add_argument("--search-window", type=int, default=5)
    parser.add_argument("--max-rt", type=float, default=30.0, help="Plot limit in minutes.")
    return parser.parse_args()


def extract_eics(exp: MSExperiment, target_mzs: list[float], ppm: float) -> tuple[list[list[float]], list[list[float]]]:
    targets = np.asarray(target_mzs, dtype=np.float64)
    delta = targets * ppm / 1e6
    starts, ends = targets - delta, targets + delta
    times = [[] for _ in targets]
    intensities = [[] for _ in targets]
    for spectrum in exp:
        if spectrum.getMSLevel() != 1:
            continue
        rt_min = spectrum.getRT() / 60.0
        mzs, values = spectrum.get_peaks()
        scan_values = np.zeros(len(targets), dtype=np.float64)
        if len(mzs):
            left = np.searchsorted(mzs, starts)
            right = np.searchsorted(mzs, ends)
            for index in np.flatnonzero(right > left):
                scan_values[index] = np.max(values[left[index]:right[index]])
        for index, value in enumerate(scan_values):
            times[index].append(rt_min)
            intensities[index].append(float(value))
    return times, intensities


def analyze(times: list[float], intensities: list[float], params: dict[str, Any]) -> dict[str, Any]:
    x = np.asarray(times, dtype=np.float64)
    y = np.asarray(intensities, dtype=np.float64)
    if len(y) < 5 or not len(y) or float(np.max(y)) < params["noise_threshold"]:
        return {"status": "NO_SIGNAL", "NumPeaks": 0, "NumReportedPeaks": 0}
    win_len = max(3, int(params["win_len"]))
    if win_len % 2 == 0:
        win_len += 1
    polyorder = min(max(1, int(params["polyorder"])), win_len - 1)
    try:
        smooth = y if len(y) <= win_len else savgol_filter(y, win_len, polyorder)
        smooth = np.maximum(np.asarray(smooth, dtype=np.float64), 0.0)
    except Exception as error:
        return {"status": f"ERROR_SMOOTH: {error}", "NumPeaks": 0,
                "NumReportedPeaks": 0}
    peaks, _ = find_peaks(
        smooth, height=params["noise_threshold"],
        distance=max(1, int(params["distance"])),
        prominence=max(params["noise_threshold"] * params["prom_factor"], 1000.0),
    )
    if not len(peaks):
        return {"status": "FAIL_NO_PEAK", "NumPeaks": 0, "NumReportedPeaks": 0}
    selected = np.sort(peaks[np.argsort(smooth[peaks])[-2:]])
    found: list[float] = []
    radius = max(0, int(params["search_window"]))
    for peak in selected:
        start, end = max(0, peak - radius), min(len(y), peak + radius + 1)
        raw_index = start + int(np.argmax(y[start:end]))
        found.append(round(float(x[raw_index]), 3))
    return {"status": "SUCCESS", "Peak1": found[0],
            "Peak2": found[1] if len(found) > 1 else None,
            "NumPeaks": int(len(peaks)), "NumReportedPeaks": len(found)}


def qc(result: dict[str, Any]) -> tuple[str, str]:
    if result["status"] == "NO_SIGNAL":
        return "RED", "No Signal"
    if result["status"] != "SUCCESS":
        return "RED", f"Error: {result['status']}"
    if result["NumPeaks"] == 1:
        return "YELLOW", "Single Peak"
    return "GREEN", "Pass"


def save_plot(x: list[float], y: list[float], result: dict[str, Any], path: Path,
              params: dict[str, Any]) -> None:
    x_values, y_values = np.asarray(x), np.asarray(y)
    fig, axis = plt.subplots(figsize=(12, 5), dpi=150)
    try:
        axis.fill_between(x_values, y_values, color="#B0BEC5", alpha=0.3, label="Raw data")
        axis.plot(x_values, y_values, color="#78909C", linewidth=0.5, alpha=0.5)
        win_len = int(params["win_len"])
        if win_len % 2 == 0: win_len += 1
        polyorder = min(int(params["polyorder"]), win_len - 1)
        smooth = y_values if len(y_values) <= win_len else savgol_filter(y_values, win_len, polyorder)
        smooth = np.maximum(smooth, 0.0)
        axis.plot(x_values, smooth, color="#1565C0", linewidth=1.0, label="Smoothed")
        for label in ("Peak1", "Peak2"):
            value = result.get(label)
            if value is None: continue
            index = int(np.abs(x_values - float(value)).argmin())
            axis.plot(x_values[index], smooth[index], "v", color="#D50000", markersize=7)
            axis.annotate(f"{float(value):.3f} min", (x_values[index], smooth[index]),
                          xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8)
        axis.set(xlabel="Retention time (min)", ylabel="Intensity",
                 title=f"Status: {result['status']} | {path.stem}", xlim=(0, params["max_rt"]))
        axis.grid(linestyle=":", alpha=0.5); axis.legend(loc="upper right", fontsize=8)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(path)
    finally:
        plt.close(fig)


def process_file(task: tuple[str, list[dict[str, str]], str | None, dict[str, Any], str]) -> list[dict[str, Any]]:
    filename, rows, plot_root, params, column = task
    path = Path(filename)
    exp = MSExperiment(); MzMLFile().load(str(path), exp)
    targets = [float(row["MZ"]) for row in rows]
    times, values = extract_eics(exp, targets, params["ppm"])
    results: list[dict[str, Any]] = []
    for row, target, x, y in zip(rows, targets, times, values):
        result = analyze(x, y, params)
        flag, reason = qc(result)
        image_path = ""
        if plot_root and len(x) > 10:
            safe_id = re.sub(r'[^A-Za-z0-9_.-]', "_", row["Compound_ID"])
            image = Path(plot_root) / flag / f"{flag}_{safe_id}_{path.name}.png"
            save_plot(x, y, result, image, params)
            image_path = image.as_posix()
        results.append({"Compound_ID": row["Compound_ID"], "MZ": target,
                        "File": path.name, "Column": column, "Status": result["status"],
                        "Num_Peaks": result["NumPeaks"], "Peak1": result.get("Peak1", ""),
                        "Peak2": result.get("Peak2", ""), "QC_Flag": flag,
                        "QC_Reason": reason, "Image_Path": image_path})
    return results


def main() -> None:
    args = parse_args()
    if args.workers < 1: raise ValueError("--workers must be at least 1")
    with args.input_table.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Compound_ID", "MZ", "Raw_File"}
        missing = required.difference(reader.fieldnames or [])
        if missing: raise ValueError(f"Input table is missing columns: {sorted(missing)}")
        rows = list(reader)
    ids = [row["Compound_ID"] for row in rows]
    if not rows: raise ValueError("Input table contains no data rows")
    if len(ids) != len(set(ids)): raise ValueError("Duplicate Compound_ID values in input")
    available: dict[str, Path] = {}
    for path in sorted(args.mzml_dir.rglob("*.mzML")):
        if path.name in available: raise ValueError(f"Duplicate mzML basename: {path.name}")
        available[path.name] = path.resolve()
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row["Raw_File"] not in available: raise FileNotFoundError(row["Raw_File"])
        grouped.setdefault(row["Raw_File"], []).append(row)
    params = {name: getattr(args, name) for name in
              ("ppm", "noise_threshold", "win_len", "polyorder", "distance",
               "prom_factor", "search_window", "max_rt")}
    tasks = [(str(available[name]), sorted(group, key=lambda item: item["Compound_ID"]),
              str(args.plot_dir) if args.plot_dir else None, params, args.column)
             for name, group in sorted(grouped.items())]
    all_results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        futures = [pool.submit(process_file, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), unit="mzML"):
            all_results.extend(future.result())
    all_results.sort(key=lambda row: (str(row["Compound_ID"]), str(row["File"])))
    if len(all_results) != len(rows):
        raise RuntimeError(f"Expected {len(rows)} results, obtained {len(all_results)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_COLUMNS, lineterminator="\n")
        writer.writeheader(); writer.writerows(all_results)
    mzml_inputs = [available[name] for name in sorted(grouped)]
    outputs = [args.output, *(Path(row["Image_Path"]) for row in all_results
                              if row["Image_Path"])]
    write_provenance(args.provenance, command=Path(__file__).name,
                     parameters={**params, "column": args.column, "workers": args.workers,
                                 "result_rows": len(all_results)},
                     inputs=[args.input_table, *mzml_inputs], outputs=outputs,
                     packages=["numpy", "scipy", "matplotlib", "pyopenms", "tqdm"])
    print(f"Wrote {len(all_results)} results: {args.output}")


if __name__ == "__main__":
    main()
