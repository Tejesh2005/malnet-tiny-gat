from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

# Keep Matplotlib's cache inside the writable project tree on managed machines.
_MPL_CONFIG = Path(os.environ.get("MPLCONFIGDIR", Path.cwd() / ".cache" / "matplotlib"))
_MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_MPL_CONFIG)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from torch import nn
from torch_geometric.loader import DataLoader

from .config import ExperimentConfig
from .data import load_splits
from .engine import predict
from .metrics import classification_metrics
from .train import build_model
from .utils import resolve_device, seed_everything, write_json


def evaluate(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    device_name: str = "auto",
    make_tsne: bool = True,
) -> dict[str, object]:
    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ExperimentConfig(**checkpoint["config"])
    seed_everything(config.seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _, _, test_set = load_splits(
        config.data_root,
        config.feature_profile,
        config.log_features,
        config.remove_isolated_nodes,
        config.split_strategy,
        config.split_seed,
    )
    loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_model(config, checkpoint["in_channels"], checkpoint["num_classes"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    result = predict(model, loader, criterion, device, include_embeddings=True)
    class_names = checkpoint["class_names"]
    metrics = classification_metrics(result["labels"], result["predictions"], class_names)
    payload = {"loss": float(result["loss"]), **metrics}
    write_json(output_dir / "metrics.json", payload)

    probabilities = result["probabilities"]
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["sample_index", "true_label", "predicted_label"] + [
            f"prob_{name}" for name in class_names
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (truth, prediction, probs) in enumerate(
            zip(result["labels"], result["predictions"], probabilities, strict=True)
        ):
            row = {
                "sample_index": index,
                "true_label": class_names[int(truth)],
                "predicted_label": class_names[int(prediction)],
            }
            row.update({f"prob_{name}": float(probs[i]) for i, name in enumerate(class_names)})
            writer.writerow(row)

    matrix = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set(xlabel="Predicted", ylabel="True", title="MalNet-Tiny test confusion matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    embeddings = result["embeddings"]
    np.savez_compressed(
        output_dir / "embeddings.npz", embeddings=embeddings, labels=result["labels"]
    )
    if make_tsne:
        projection = TSNE(
            n_components=2,
            perplexity=min(30, len(embeddings) - 1),
            init="pca",
            learning_rate="auto",
            random_state=config.seed,
        ).fit_transform(embeddings)
        fig, ax = plt.subplots(figsize=(8, 6))
        for label, name in enumerate(class_names):
            mask = result["labels"] == label
            ax.scatter(projection[mask, 0], projection[mask, 1], s=12, alpha=0.75, label=name)
        ax.set(title="GAT graph embeddings (t-SNE)", xticks=[], yticks=[])
        ax.legend(frameon=False, markerscale=1.5)
        fig.tight_layout()
        fig.savefig(output_dir / "tsne.png", dpi=180)
        plt.close(fig)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained MalNet-Tiny GAT")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-tsne", action="store_true")
    args = parser.parse_args()
    metrics = evaluate(args.checkpoint, args.output_dir, args.device, not args.no_tsne)
    print(f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
