from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from .config import ExperimentConfig
from .train import run
from .utils import write_json


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_reused(values: list[str]) -> dict[int, Path]:
    reused: dict[int, Path] = {}
    for value in values:
        try:
            seed_text, directory = value.split("=", maxsplit=1)
            seed = int(seed_text)
        except ValueError as error:
            raise ValueError("--reuse-result must use SEED=DIRECTORY") from error
        reused[seed] = Path(directory)
    return reused


def _result_row(seed: int, directory: Path) -> dict[str, int | float | str]:
    metrics_path = directory / "test_metrics.json"
    config_path = directory / "config.json"
    if not metrics_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"Incomplete result directory: {directory}")
    config = _read_json(config_path)
    if int(config["seed"]) != seed:
        raise ValueError(f"Expected seed {seed}, but {directory} contains seed {config['seed']}")
    metrics = _read_json(metrics_path)
    return {
        "seed": seed,
        "result_dir": str(directory),
        "best_epoch": int(metrics["best_epoch"]),
        "best_val_loss": float(metrics["best_val_loss"]),
        "test_accuracy": float(metrics["test"]["accuracy"]),
        "test_macro_f1": float(metrics["test"]["f1_macro"]),
        "elapsed_seconds": float(metrics["elapsed_seconds"]),
    }


def summarize_rows(rows: list[dict[str, int | float | str]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("At least one completed result is required")
    accuracy = [float(row["test_accuracy"]) for row in rows]
    macro_f1 = [float(row["test_macro_f1"]) for row in rows]
    return {
        "num_seeds": len(rows),
        "seeds": [int(row["seed"]) for row in rows],
        "test_accuracy_mean": fmean(accuracy),
        "test_accuracy_std": stdev(accuracy) if len(accuracy) > 1 else 0.0,
        "test_macro_f1_mean": fmean(macro_f1),
        "test_macro_f1_std": stdev(macro_f1) if len(macro_f1) > 1 else 0.0,
        "runs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and aggregate multiple MalNet GAT seeds")
    parser.add_argument("--config", default="configs/directed_concat_baseline.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 7, 21, 42, 84])
    parser.add_argument("--output-root", default="runs/directed_concat_multiseed")
    parser.add_argument("--device", default=None, help="Override config device")
    parser.add_argument(
        "--reuse-result",
        action="append",
        default=[],
        metavar="SEED=DIRECTORY",
        help="Reuse a completed run, for example 42=/kaggle/working/directed_concat_baseline",
    )
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")

    base = ExperimentConfig.from_yaml(args.config)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        reused = _parse_reused(args.reuse_result)
    except ValueError as error:
        parser.error(str(error))

    rows: list[dict[str, int | float | str]] = []
    for seed in args.seeds:
        result_dir = reused.get(seed, output_root / f"seed_{seed}")
        metrics_path = result_dir / "test_metrics.json"
        if metrics_path.is_file():
            print(f"seed={seed}: reusing {result_dir}")
        else:
            if seed in reused:
                parser.error(f"Reused result is incomplete: {result_dir}")
            config = replace(base, seed=seed, output_dir=str(result_dir))
            if args.device is not None:
                config.device = args.device
            print(f"seed={seed}: training -> {result_dir}")
            run(config)
        row = _result_row(seed, result_dir)
        rows.append(row)
        print(
            f"seed={seed}: accuracy={float(row['test_accuracy']):.4f} "
            f"macro_f1={float(row['test_macro_f1']):.4f}"
        )

    summary = summarize_rows(rows)
    write_json(output_root / "multiseed_summary.json", summary)
    with (output_root / "multiseed_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"complete: n={summary['num_seeds']} "
        f"accuracy={summary['test_accuracy_mean']:.4f}+/-{summary['test_accuracy_std']:.4f} "
        f"macro_f1={summary['test_macro_f1_mean']:.4f}+/-{summary['test_macro_f1_std']:.4f}"
    )


if __name__ == "__main__":
    main()
