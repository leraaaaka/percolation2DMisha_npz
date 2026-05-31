#!/usr/bin/env python3
"""
Analyze percolation scan_all.npz files produced by percolation2D.

Use cases
---------
1) One image / one folder containing repeated runs:
   python analyze_percolation_npz.py /path/to/sample_folder --sample-name Stack6

2) Many images / parent folder where each immediate subfolder is one image:
   python analyze_percolation_npz.py /path/to/parent_folder --batch

Optional metadata CSV for the final across-sample graph:
   sample,cardiomyocytes_pct,fibroblasts_pct
   Stack6,25,75

   python analyze_percolation_npz.py /path/to/parent_folder --batch --metadata metadata.csv

The script never reruns the original simulation. It reads existing .npz files only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required. Install it with: pip install matplotlib") from exc


REQUIRED_KEYS = ("x0", "x1", "y1", "loss")


def natural_key(path: Path) -> List[object]:
    """Sort file names naturally: run2 before run10."""
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(p) if p.isdigit() else p for p in parts]


def fmt(value: float | int | str | None, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def percentile(values: Sequence[float], q: float) -> float:
    """Compatibility wrapper for NumPy versions with different percentile APIs."""
    try:
        return float(np.percentile(values, q, method="linear"))
    except TypeError:  # older NumPy
        return float(np.percentile(values, q, interpolation="linear"))


@dataclass
class RunBest:
    sample: str
    run_number: int
    file_name: str
    n_functions: int
    sorted_by_loss_in_file: bool
    best_x0: float
    best_x1: float
    best_y1: float
    min_loss: float


@dataclass
class SampleSummary:
    sample: str
    n_runs: int
    median_x0: float
    min_x0: float
    max_x0: float
    q1_x0: float
    q3_x0: float
    mean_x0: float
    sd_x0: float
    stable_x0_by_median_min_loss: float
    stable_x0_median_min_loss: float
    global_best_x0: float
    global_best_x1: float
    global_best_y1: float
    global_min_loss: float
    global_best_file: str
    global_best_run: int


class ValidationError(RuntimeError):
    pass


def load_run(path: Path) -> Tuple[Dict[str, np.ndarray], List[str], bool]:
    """Load and validate one NPZ file. Returns arrays, warnings, sorted flag."""
    warnings: List[str] = []
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = [key for key in REQUIRED_KEYS if key not in data.files]
            if missing:
                raise ValidationError(f"missing required arrays: {missing}")
            arrays = {key: np.asarray(data[key], dtype=float).reshape(-1) for key in REQUIRED_KEYS}
    except Exception as exc:
        raise ValidationError(f"cannot read {path.name}: {exc}") from exc

    lengths = {key: len(arr) for key, arr in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValidationError(f"array lengths differ: {lengths}")
    if lengths["loss"] == 0:
        raise ValidationError("arrays are empty")

    for key, arr in arrays.items():
        if not np.all(np.isfinite(arr)):
            raise ValidationError(f"array {key} contains NaN or inf")

    if np.any((arrays["x0"] < 0) | (arrays["x0"] > 1)):
        warnings.append("x0 contains values outside [0, 1]")
    if np.any((arrays["x1"] < 0) | (arrays["x1"] > 1)):
        warnings.append("x1 contains values outside [0, 1]")
    if np.any((arrays["y1"] < 0) | (arrays["y1"] > 1)):
        warnings.append("y1 contains values outside [0, 1]")
    if np.any(arrays["loss"] < 0):
        warnings.append("loss contains negative values")

    sorted_in_file = bool(np.all(np.diff(arrays["loss"]) >= 0))
    if not sorted_in_file:
        warnings.append("rows were not sorted by loss; the script sorted a copy before analysis")
        order = np.argsort(arrays["loss"], kind="stable")
        arrays = {key: arr[order] for key, arr in arrays.items()}

    return arrays, warnings, sorted_in_file


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def analyze_sample(sample_dir: Path, sample_name: str, output_dir: Path) -> SampleSummary:
    npz_files = sorted(sample_dir.glob("*.npz"), key=natural_key)
    if not npz_files:
        raise ValidationError(f"no .npz files found in {sample_dir}")

    sample_output = output_dir / sample_name
    sample_output.mkdir(parents=True, exist_ok=True)

    run_rows: List[RunBest] = []
    warnings_rows: List[Dict[str, object]] = []
    loss_by_x0_per_run: List[Dict[str, object]] = []
    global_candidates: List[Tuple[float, int, str, float, float, float]] = []

    for run_number, path in enumerate(npz_files, start=1):
        try:
            arrays, warnings, sorted_in_file = load_run(path)
        except ValidationError as exc:
            warnings_rows.append({"sample": sample_name, "file_name": path.name, "warning": str(exc)})
            continue

        for warning in warnings:
            warnings_rows.append({"sample": sample_name, "file_name": path.name, "warning": warning})

        best_idx = int(np.argmin(arrays["loss"]))
        run_rows.append(
            RunBest(
                sample=sample_name,
                run_number=run_number,
                file_name=path.name,
                n_functions=len(arrays["loss"]),
                sorted_by_loss_in_file=sorted_in_file,
                best_x0=float(arrays["x0"][best_idx]),
                best_x1=float(arrays["x1"][best_idx]),
                best_y1=float(arrays["y1"][best_idx]),
                min_loss=float(arrays["loss"][best_idx]),
            )
        )

        global_candidates.append(
            (
                float(arrays["loss"][best_idx]),
                run_number,
                path.name,
                float(arrays["x0"][best_idx]),
                float(arrays["x1"][best_idx]),
                float(arrays["y1"][best_idx]),
            )
        )

        for x0 in sorted(np.unique(arrays["x0"])):
            mask = np.isclose(arrays["x0"], x0, rtol=0, atol=1e-12)
            x0_losses = arrays["loss"][mask]
            loss_by_x0_per_run.append(
                {
                    "sample": sample_name,
                    "run_number": run_number,
                    "file_name": path.name,
                    "x0": float(x0),
                    "n_functions_for_x0": int(mask.sum()),
                    "min_loss_for_x0": float(np.min(x0_losses)),
                }
            )

    if not run_rows:
        raise ValidationError(f"all NPZ files failed validation in {sample_dir}")

    best_x0_values = np.asarray([row.best_x0 for row in run_rows], dtype=float)
    ddof = 1 if len(best_x0_values) > 1 else 0

    # Aggregate the minimum loss obtained for each x0 in each independent run.
    x0_groups: Dict[float, List[float]] = {}
    for row in loss_by_x0_per_run:
        x0_groups.setdefault(float(row["x0"]), []).append(float(row["min_loss_for_x0"]))

    loss_by_x0_summary: List[Dict[str, object]] = []
    for x0 in sorted(x0_groups):
        losses = np.asarray(x0_groups[x0], dtype=float)
        loss_by_x0_summary.append(
            {
                "sample": sample_name,
                "x0": float(x0),
                "n_runs": len(losses),
                "median_min_loss": float(np.median(losses)),
                "q1_min_loss": percentile(losses, 25),
                "q3_min_loss": percentile(losses, 75),
                "min_min_loss": float(np.min(losses)),
                "max_min_loss": float(np.max(losses)),
                "mean_min_loss": float(np.mean(losses)),
                "sd_min_loss": float(np.std(losses, ddof=(1 if len(losses) > 1 else 0))),
            }
        )

    stable_row = min(loss_by_x0_summary, key=lambda row: float(row["median_min_loss"]))
    global_best = min(global_candidates, key=lambda item: item[0])

    summary = SampleSummary(
        sample=sample_name,
        n_runs=len(run_rows),
        median_x0=float(np.median(best_x0_values)),
        min_x0=float(np.min(best_x0_values)),
        max_x0=float(np.max(best_x0_values)),
        q1_x0=percentile(best_x0_values, 25),
        q3_x0=percentile(best_x0_values, 75),
        mean_x0=float(np.mean(best_x0_values)),
        sd_x0=float(np.std(best_x0_values, ddof=ddof)),
        stable_x0_by_median_min_loss=float(stable_row["x0"]),
        stable_x0_median_min_loss=float(stable_row["median_min_loss"]),
        global_best_x0=float(global_best[3]),
        global_best_x1=float(global_best[4]),
        global_best_y1=float(global_best[5]),
        global_min_loss=float(global_best[0]),
        global_best_file=str(global_best[2]),
        global_best_run=int(global_best[1]),
    )

    # Frequencies of best x0 values across independent runs.
    unique_x0, counts = np.unique(best_x0_values, return_counts=True)
    frequency_rows = [
        {
            "sample": sample_name,
            "x0": float(x0),
            "count": int(count),
            "frequency": float(count / len(best_x0_values)),
        }
        for x0, count in zip(unique_x0, counts)
    ]

    write_csv(sample_output / "run_level_results.csv", [asdict(row) for row in run_rows], list(asdict(run_rows[0]).keys()))
    write_csv(sample_output / "loss_by_x0_per_run.csv", loss_by_x0_per_run, list(loss_by_x0_per_run[0].keys()))
    write_csv(sample_output / "loss_by_x0_summary.csv", loss_by_x0_summary, list(loss_by_x0_summary[0].keys()))
    write_csv(sample_output / "best_x0_frequencies.csv", frequency_rows, list(frequency_rows[0].keys()))
    write_csv(sample_output / "sample_summary.csv", [asdict(summary)], list(asdict(summary).keys()))
    write_csv(sample_output / "warnings.csv", warnings_rows, ["sample", "file_name", "warning"])

    # Diagnostic figure: median minimum loss for each x0.
    xs = np.asarray([float(row["x0"]) for row in loss_by_x0_summary])
    ys = np.asarray([float(row["median_min_loss"]) for row in loss_by_x0_summary])
    q1 = np.asarray([float(row["q1_min_loss"]) for row in loss_by_x0_summary])
    q3 = np.asarray([float(row["q3_min_loss"]) for row in loss_by_x0_summary])
    yerr = np.vstack([ys - q1, q3 - ys])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=4)
    ax.scatter([summary.stable_x0_by_median_min_loss], [summary.stable_x0_median_min_loss], marker="s", s=75)
    ax.set_xlabel("Значение x0")
    ax.set_ylabel("Медиана минимальной ошибки моделирования L")
    ax.set_title(f"Диагностическая зависимость ошибки от x0: {sample_name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(sample_output / "diagnostic_loss_by_x0.png", dpi=300)
    fig.savefig(sample_output / "diagnostic_loss_by_x0.svg")
    plt.close(fig)

    # Simple text report for quick manual reading.
    report = {
        **asdict(summary),
        "best_x0_values_by_run": [float(value) for value in best_x0_values],
        "notes": [
            "median_x0 is the main estimate for the sample",
            "min_x0-max_x0 is a descriptive observed range, not a confidence interval",
            "stable_x0_by_median_min_loss is an independent diagnostic check",
        ],
    }
    (sample_output / "quick_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary


def read_metadata(path: Path) -> Dict[str, Dict[str, float]]:
    metadata: Dict[str, Dict[str, float]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"sample", "cardiomyocytes_pct", "fibroblasts_pct"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValidationError(f"metadata CSV must contain columns: {sorted(required)}")
        for row in reader:
            metadata[row["sample"]] = {
                "cardiomyocytes_pct": float(row["cardiomyocytes_pct"]),
                "fibroblasts_pct": float(row["fibroblasts_pct"]),
            }
    return metadata


def make_across_sample_plot(summaries: Sequence[SampleSummary], metadata_path: Path, output_dir: Path) -> None:
    metadata = read_metadata(metadata_path)
    rows = []
    for summary in summaries:
        if summary.sample not in metadata:
            raise ValidationError(f"sample {summary.sample!r} is missing from metadata CSV")
        meta = metadata[summary.sample]
        rows.append(
            {
                **asdict(summary),
                "cardiomyocytes_pct": meta["cardiomyocytes_pct"],
                "fibroblasts_pct": meta["fibroblasts_pct"],
            }
        )

    rows.sort(key=lambda row: float(row["cardiomyocytes_pct"]))
    write_csv(output_dir / "all_samples_summary_with_metadata.csv", rows, list(rows[0].keys()))

    xs = np.asarray([float(row["cardiomyocytes_pct"]) for row in rows])
    ys = np.asarray([float(row["median_x0"]) for row in rows])
    lower = ys - np.asarray([float(row["min_x0"]) for row in rows])
    upper = np.asarray([float(row["max_x0"]) for row in rows]) - ys

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(xs, ys, yerr=np.vstack([lower, upper]), marker="o", capsize=4)
    for row in rows:
        ax.annotate(str(row["sample"]), (float(row["cardiomyocytes_pct"]), float(row["median_x0"])), xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Доля кардиомиоцитов в сокультуре, %")
    ax.set_ylabel("Медианная оценка порога перколяции x0")
    ax.set_title("Оценка порога перколяции для исследуемых образцов")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "percolation_threshold_by_cardiomyocytes_pct.png", dpi=300)
    fig.savefig(output_dir / "percolation_threshold_by_cardiomyocytes_pct.svg")
    plt.close(fig)


def make_metadata_template(summaries: Sequence[SampleSummary], output_dir: Path) -> None:
    rows = [
        {"sample": summary.sample, "cardiomyocytes_pct": "", "fibroblasts_pct": ""}
        for summary in summaries
    ]
    write_csv(output_dir / "metadata_template.csv", rows, ["sample", "cardiomyocytes_pct", "fibroblasts_pct"])


def discover_samples(input_path: Path, batch: bool, sample_name: str | None) -> List[Tuple[str, Path]]:
    if not input_path.exists():
        raise ValidationError(f"input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() != ".npz":
            raise ValidationError("input file must be .npz")
        # Copying is unnecessary; analyze the containing folder but only this file via temp directory is overkill.
        # Instead require a folder for clarity.
        raise ValidationError("please place the .npz file in its own folder and pass the folder path")

    if batch:
        samples: List[Tuple[str, Path]] = []
        for child in sorted([p for p in input_path.iterdir() if p.is_dir()], key=natural_key):
            if any(child.glob("*.npz")):
                samples.append((child.name, child))
        if not samples:
            raise ValidationError("batch mode found no subfolders containing .npz files")
        return samples

    if not any(input_path.glob("*.npz")):
        raise ValidationError("the selected folder contains no .npz files")
    return [(sample_name or input_path.name, input_path)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze percolation2D scan_all.npz files without rerunning the simulation."
    )
    parser.add_argument("input", type=Path, help="sample folder, or parent folder in --batch mode")
    parser.add_argument("--sample-name", help="sample label for single-folder mode")
    parser.add_argument("--batch", action="store_true", help="treat each immediate subfolder as one sample")
    parser.add_argument("--metadata", type=Path, help="optional CSV with sample, cardiomyocytes_pct, fibroblasts_pct")
    parser.add_argument("--output", type=Path, help="output folder; default: <input>/analysis_output")
    args = parser.parse_args()

    try:
        input_path = args.input.resolve()
        output_dir = (args.output.resolve() if args.output else input_path / "analysis_output")
        output_dir.mkdir(parents=True, exist_ok=True)

        samples = discover_samples(input_path, args.batch, args.sample_name)
        summaries: List[SampleSummary] = []
        for sample_name, sample_dir in samples:
            print(f"Analyzing {sample_name}: {sample_dir}")
            summary = analyze_sample(sample_dir, sample_name, output_dir)
            summaries.append(summary)
            print(
                f"  runs={summary.n_runs}; median x0={summary.median_x0:.6f}; "
                f"range={summary.min_x0:.6f}-{summary.max_x0:.6f}; "
                f"stable x0={summary.stable_x0_by_median_min_loss:.6f}"
            )

        write_csv(output_dir / "all_samples_summary.csv", [asdict(s) for s in summaries], list(asdict(summaries[0]).keys()))
        make_metadata_template(summaries, output_dir)

        if args.metadata:
            make_across_sample_plot(summaries, args.metadata.resolve(), output_dir)
            print("Across-sample plot created using metadata CSV.")
        else:
            print("Metadata not supplied. Fill metadata_template.csv later to create the across-sample plot.")

        print(f"Done. Results saved to: {output_dir}")
        return 0
    except ValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
