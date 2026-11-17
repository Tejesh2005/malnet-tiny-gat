from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

# Keep Matplotlib's cache inside the writable project tree on managed machines.
_MPL_CONFIG = Path(os.environ.get("MPLCONFIGDIR", Path.cwd() / ".cache" / "matplotlib"))
_MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_MPL_CONFIG)

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from .config import ExperimentConfig
from .data import infer_class_names, load_splits, make_pre_transform
from .engine import predict
from .metrics import classification_metrics
from .train import build_model
from .utils import resolve_device, seed_everything, write_json

PERTURBATION_TYPES = ("add_nodes", "add_edges", "rewire", "add_paths")


def _random_index(high: int, generator: torch.Generator) -> int:
    return int(torch.randint(high, (1,), generator=generator).item())


def _recompute_features(
    graph: Data,
    feature_profile: str,
    log_features: bool,
) -> Data:
    graph.x = None
    transform = make_pre_transform(
        feature_profile=feature_profile,
        log_features=log_features,
        remove_isolated_nodes=False,
    )
    return transform(graph)


def perturb_graph(
    graph: Data,
    perturbation: str,
    fraction: float,
    feature_profile: str,
    log_features: bool,
    generator: torch.Generator,
) -> tuple[Data, int]:
    """Apply a deterministic topology-only stress test and recompute node features.

    These graph mutations cannot be mapped back to executable APKs because MalNet-Tiny
    omits code and node identities. They are therefore robustness stress tests, not a
    claim of functionality-preserving code-level adversarial examples.
    """

    if perturbation not in PERTURBATION_TYPES:
        raise ValueError(f"Unsupported perturbation: {perturbation}")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    result = graph.clone()
    edge_index = result.edge_index.clone().cpu()
    num_nodes = int(result.num_nodes)
    num_edges = int(edge_index.size(1))
    modifications = 0

    if num_nodes == 0:
        return _recompute_features(result, feature_profile, log_features), 0

    if perturbation == "add_nodes":
        count = max(1, math.ceil(num_nodes * fraction))
        new_nodes = torch.arange(num_nodes, num_nodes + count, dtype=torch.long)
        parents = torch.randint(num_nodes, (count,), generator=generator)
        added = torch.stack([parents, new_nodes])
        edge_index = torch.cat([edge_index, added], dim=1)
        num_nodes += count
        modifications = count * 2

    elif perturbation == "add_edges":
        requested = max(1, math.ceil(max(num_edges, 1) * fraction))
        existing = set(map(tuple, edge_index.t().tolist()))
        added_edges: list[tuple[int, int]] = []
        attempts = 0
        while len(added_edges) < requested and attempts < requested * 30:
            source = _random_index(num_nodes, generator)
            target = _random_index(num_nodes, generator)
            candidate = (source, target)
            attempts += 1
            if source == target or candidate in existing:
                continue
            existing.add(candidate)
            added_edges.append(candidate)
        if added_edges:
            added = torch.tensor(added_edges, dtype=torch.long).t().contiguous()
            edge_index = torch.cat([edge_index, added], dim=1)
        modifications = len(added_edges)

    elif perturbation == "rewire":
        if num_edges:
            count = min(num_edges, max(1, math.ceil(num_edges * fraction)))
            selected = torch.randperm(num_edges, generator=generator)[:count]
            for edge_position in selected.tolist():
                source = int(edge_index[0, edge_position])
                target = _random_index(num_nodes, generator)
                if num_nodes > 1:
                    while target == source:
                        target = _random_index(num_nodes, generator)
                edge_index[1, edge_position] = target
            modifications = count

    else:  # add_paths
        path_count = max(1, math.ceil(max(num_edges, 1) * fraction / 3))
        additions: list[tuple[int, int]] = []
        for _ in range(path_count):
            source = _random_index(num_nodes, generator)
            target = _random_index(num_nodes, generator)
            first = num_nodes
            second = num_nodes + 1
            num_nodes += 2
            additions.extend([(source, first), (first, second), (second, target)])
        added = torch.tensor(additions, dtype=torch.long).t().contiguous()
        edge_index = torch.cat([edge_index, added], dim=1)
        modifications = path_count * 5

    result.edge_index = edge_index
    result.num_nodes = num_nodes
    result = _recompute_features(result, feature_profile, log_features)
    return result, modifications


def augment_batch(
    batch: Batch,
    probability: float,
    fraction: float,
    perturbation_types: list[str],
    feature_profile: str,
    log_features: bool,
    generator: torch.Generator,
) -> Batch:
    """Keep originals and append perturbed copies selected for robust training."""

    originals = batch.to_data_list()
    augmented = list(originals)
    for graph in originals:
        if float(torch.rand(1, generator=generator)) > probability:
            continue
        perturbation = perturbation_types[_random_index(len(perturbation_types), generator)]
        changed, _ = perturb_graph(
            graph,
            perturbation,
            fraction,
            feature_profile,
            log_features,
            generator,
        )
        augmented.append(changed)
    return Batch.from_data_list(augmented)


def evaluate_robustness(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    fractions: list[float],
    perturbation_types: list[str],
    device_name: str = "auto",
    max_samples: int | None = None,
) -> dict[str, object]:
    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ExperimentConfig(**checkpoint["config"])
    config.validate()
    seed_everything(config.seed)
    _, _, test_set = load_splits(
        config.data_root,
        config.feature_profile,
        config.log_features,
        config.remove_isolated_nodes,
        config.split_strategy,
        config.split_seed,
    )
    if max_samples is not None:
        test_graphs = [test_set[index] for index in range(min(max_samples, len(test_set)))]
    else:
        test_graphs = [test_set[index] for index in range(len(test_set))]
    class_names = infer_class_names(test_set)
    model = build_model(config, checkpoint["in_channels"], checkpoint["num_classes"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    loader_options = {
        "batch_size": config.batch_size,
        "shuffle": False,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    clean_result = predict(
        model, DataLoader(test_graphs, **loader_options), criterion, device
    )
    clean_metrics = classification_metrics(
        clean_result["labels"], clean_result["predictions"], class_names
    )
    rows: list[dict[str, object]] = [
        {
            "perturbation": "clean",
            "fraction": 0.0,
            "accuracy": clean_metrics["accuracy"],
            "macro_f1": clean_metrics["f1_macro"],
            "prediction_change_rate": 0.0,
            "attack_success_rate": 0.0,
            "average_modifications": 0.0,
        }
    ]
    labels = np.asarray(clean_result["labels"])
    clean_predictions = np.asarray(clean_result["predictions"])
    originally_correct = clean_predictions == labels

    for perturbation_index, perturbation in enumerate(perturbation_types):
        for fraction_index, fraction in enumerate(fractions):
            generator = torch.Generator().manual_seed(
                config.seed + 1000 * (perturbation_index + 1) + fraction_index
            )
            perturbed: list[Data] = []
            modification_counts: list[int] = []
            for graph in test_graphs:
                changed, count = perturb_graph(
                    graph,
                    perturbation,
                    fraction,
                    config.feature_profile,
                    config.log_features,
                    generator,
                )
                perturbed.append(changed)
                modification_counts.append(count)
            result = predict(model, DataLoader(perturbed, **loader_options), criterion, device)
            predictions = np.asarray(result["predictions"])
            metrics = classification_metrics(labels, predictions, class_names)
            changed_predictions = predictions != clean_predictions
            successful = originally_correct & (predictions != labels)
            attack_success = (
                float(successful.sum() / originally_correct.sum())
                if originally_correct.any()
                else 0.0
            )
            rows.append(
                {
                    "perturbation": perturbation,
                    "fraction": fraction,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["f1_macro"],
                    "prediction_change_rate": float(changed_predictions.mean()),
                    "attack_success_rate": attack_success,
                    "average_modifications": float(np.mean(modification_counts)),
                }
            )
            print(
                f"perturbation={perturbation} fraction={fraction:.3f} "
                f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['f1_macro']:.4f} "
                f"asr={attack_success:.4f}"
            )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "robustness.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "checkpoint": str(checkpoint_path),
        "scope": "graph-level structural stress test; not executable APK perturbation",
        "samples": len(test_graphs),
        "rows": rows,
    }
    write_json(output / "robustness.json", payload)

    fig, ax = plt.subplots(figsize=(9, 5))
    for perturbation in perturbation_types:
        selected = [row for row in rows if row["perturbation"] == perturbation]
        ax.plot(
            [float(row["fraction"]) * 100 for row in selected],
            [float(row["accuracy"]) for row in selected],
            marker="o",
            label=perturbation,
        )
    ax.axhline(float(clean_metrics["accuracy"]), color="black", linestyle="--", label="clean")
    ax.set(
        title="MalNet-Tiny graph-level robustness",
        xlabel="Perturbation budget (%)",
        ylabel="Accuracy",
        ylim=(0, 1),
    )
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "robustness.png", dpi=180)
    plt.close(fig)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate graph-level structural robustness on MalNet-Tiny"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/robustness")
    parser.add_argument("--fractions", default="0.01,0.05,0.10")
    parser.add_argument("--types", default=",".join(PERTURBATION_TYPES))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    fractions = [float(value) for value in args.fractions.split(",") if value.strip()]
    perturbation_types = [value.strip() for value in args.types.split(",") if value.strip()]
    unknown = sorted(set(perturbation_types) - set(PERTURBATION_TYPES))
    if unknown:
        parser.error(f"Unsupported perturbation types: {', '.join(unknown)}")
    evaluate_robustness(
        args.checkpoint,
        args.output_dir,
        fractions,
        perturbation_types,
        args.device,
        args.max_samples,
    )


if __name__ == "__main__":
    main()
