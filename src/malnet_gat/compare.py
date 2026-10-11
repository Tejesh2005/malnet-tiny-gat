from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import ExperimentConfig
from .train import run
from .utils import write_json

SUPPORTED_MODELS = ("gcn", "graphsage", "gin", "gat")


def compare(
    base_config: ExperimentConfig,
    output_dir: str | Path,
    models: list[str],
    resume: bool = False,
) -> Path:
    unknown = sorted(set(models) - set(SUPPORTED_MODELS))
    if unknown:
        raise ValueError(f"Unsupported models: {', '.join(unknown)}")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for model_type in models:
        model_output = root / model_type
        metrics_path = model_output / "test_metrics.json"
        config = replace(
            base_config,
            model_type=model_type,
            output_dir=str(model_output),
        )
        if metrics_path.is_file():
            print(f"Skipping completed model={model_type}: {metrics_path}")
        else:
            print(f"Starting model={model_type}")
            run(config, resume=resume)
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        rows.append(
            {
                "model": model_type,
                "best_epoch": metrics["best_epoch"],
                "best_val_accuracy": metrics["best_val_accuracy"],
                "test_accuracy": metrics["test"]["accuracy"],
                "macro_f1": metrics["test"]["f1_macro"],
                "parameters": metrics["parameter_count"],
                "elapsed_seconds": metrics["elapsed_seconds"],
            }
        )

    csv_path = root / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(root / "comparison.json", {"models": rows})

    positions = np.arange(len(rows))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    accuracy_bars = ax.bar(
        positions - width / 2,
        [float(row["test_accuracy"]) for row in rows],
        width,
        label="Accuracy",
    )
    f1_bars = ax.bar(
        positions + width / 2,
        [float(row["macro_f1"]) for row in rows],
        width,
        label="Macro-F1",
    )
    ax.set(
        title="MalNet-Tiny GNN comparison",
        ylabel="Score",
        xticks=positions,
        xticklabels=[str(row["model"]).upper() for row in rows],
        ylim=(0, 1),
    )
    ax.bar_label(accuracy_bars, fmt="%.3f", padding=3)
    ax.bar_label(f1_bars, fmt="%.3f", padding=3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(root / "comparison.png", dpi=180)
    plt.close(fig)

    print("\nmodel       val_acc  test_acc  macro_f1  parameters")
    for row in rows:
        print(
            f"{row['model']:<11} {row['best_val_accuracy']:.4f}   "
            f"{row['test_accuracy']:.4f}    {row['macro_f1']:.4f}    "
            f"{row['parameters']}"
        )
    print(f"comparison={csv_path}")
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare GCN, GraphSAGE, GIN, and GAT under one protocol"
    )
    parser.add_argument(
        "--config", default="configs/paper_exact_graph_dropout.yaml"
    )
    parser.add_argument("--output-dir", default="runs/model_comparison")
    parser.add_argument("--models", default=",".join(SUPPORTED_MODELS))
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    if args.device is not None:
        config.device = args.device
    if args.epochs is not None:
        config.epochs = args.epochs
    models = [item.strip().lower() for item in args.models.split(",") if item.strip()]
    compare(config, args.output_dir, models, resume=args.resume)


if __name__ == "__main__":
    main()
