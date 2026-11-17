from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

from .compare import SUPPORTED_MODELS, compare
from .config import ExperimentConfig
from .robustness import PERTURBATION_TYPES, evaluate_robustness
from .train import run
from .utils import write_json


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _ensure_run(config: ExperimentConfig) -> Path:
    metrics = Path(config.output_dir) / "test_metrics.json"
    if not metrics.is_file():
        run(config)
    return Path(config.output_dir) / "best_model.pt"


def _metric_row(study: str, variant: str, result_dir: Path) -> dict[str, object]:
    payload = _read_json(result_dir / "test_metrics.json")
    test = payload["test"]
    assert isinstance(test, dict)
    return {
        "study": study,
        "variant": variant,
        "test_accuracy": test["accuracy"],
        "macro_f1": test["f1_macro"],
        "mean_perturbed_accuracy": "",
        "mean_attack_success_rate": "",
    }


def run_ablations(
    base: ExperimentConfig,
    output_dir: str | Path,
    studies: list[str],
    robustness_samples: int | None = None,
) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    if "architectures" in studies:
        architecture_root = root / "architectures"
        compare(base, architecture_root, list(SUPPORTED_MODELS))
        for model_type in SUPPORTED_MODELS:
            rows.append(_metric_row("architectures", model_type, architecture_root / model_type))

    if "features" in studies:
        for feature_profile in ("ldp", "directed_ldp", "enhanced"):
            data_root = (
                base.data_root
                if feature_profile == base.feature_profile
                else str(Path(base.data_root).with_name(f"malnet_tiny_{feature_profile}_ablation"))
            )
            result_dir = root / "features" / feature_profile
            config = replace(
                base,
                feature_profile=feature_profile,
                data_root=data_root,
                output_dir=str(result_dir),
                robust_training=False,
            )
            _ensure_run(config)
            rows.append(_metric_row("features", feature_profile, result_dir))

    if "robustness" in studies:
        for robust_training in (False, True):
            variant = "robust_training" if robust_training else "standard_training"
            result_dir = root / "robustness" / variant
            config = replace(
                base,
                output_dir=str(result_dir),
                robust_training=robust_training,
            )
            checkpoint = _ensure_run(config)
            robustness_dir = result_dir / "stress_test"
            report = evaluate_robustness(
                checkpoint,
                robustness_dir,
                fractions=[0.01, 0.05, 0.10],
                perturbation_types=list(PERTURBATION_TYPES),
                device_name=config.device,
                max_samples=robustness_samples,
            )
            row = _metric_row("robustness", variant, result_dir)
            perturbed_rows = [
                item for item in report["rows"] if item["perturbation"] != "clean"
            ]
            row["mean_perturbed_accuracy"] = sum(
                float(item["accuracy"]) for item in perturbed_rows
            ) / len(perturbed_rows)
            row["mean_attack_success_rate"] = sum(
                float(item["attack_success_rate"]) for item in perturbed_rows
            ) / len(perturbed_rows)
            rows.append(row)

    csv_path = root / "ablation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(root / "ablation.json", {"studies": studies, "rows": rows})
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run architecture, feature, and robustness ablations on MalNet-Tiny"
    )
    parser.add_argument("--config", default="configs/extended_comparison.yaml")
    parser.add_argument("--output-dir", default="runs/extended_ablations")
    parser.add_argument(
        "--studies",
        default="architectures,features,robustness",
        help="Comma-separated subset of architectures,features,robustness",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-robustness-samples", type=int, default=None)
    args = parser.parse_args()
    studies = [item.strip() for item in args.studies.split(",") if item.strip()]
    unknown = sorted(set(studies) - {"architectures", "features", "robustness"})
    if unknown:
        parser.error(f"Unsupported studies: {', '.join(unknown)}")
    config = ExperimentConfig.from_yaml(args.config)
    if args.device is not None:
        config.device = args.device
    if args.epochs is not None:
        config.epochs = args.epochs
    path = run_ablations(
        config,
        args.output_dir,
        studies,
        robustness_samples=args.max_robustness_samples,
    )
    print(f"ablation={path}")


if __name__ == "__main__":
    main()
