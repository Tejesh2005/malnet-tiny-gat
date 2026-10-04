from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch import nn
from torch_geometric.loader import DataLoader

from .config import ExperimentConfig
from .data import infer_class_names, load_splits
from .engine import predict, train_epoch
from .metrics import classification_metrics
from .model import MalNetGAT
from .utils import resolve_device, seed_everything, write_json


def build_model(config: ExperimentConfig, in_channels: int, num_classes: int) -> MalNetGAT:
    return MalNetGAT(
        in_channels=in_channels,
        num_classes=num_classes,
        hidden_channels=config.hidden_channels,
        num_layers=config.num_layers,
        heads=config.heads,
        concat_heads=config.concat_heads,
        activation=config.activation,
        layer_norm=config.layer_norm,
        dropout=config.dropout,
        attention_dropout=config.attention_dropout,
        dropout_location=config.dropout_location,
        pooling=config.pooling,
        classifier_hidden=config.classifier_hidden,
    )


def run(config: ExperimentConfig) -> Path:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", config.to_dict())

    train_set, val_set, test_set = load_splits(
        config.data_root,
        config.feature_profile,
        config.log_features,
        config.remove_isolated_nodes,
        config.split_strategy,
        config.split_seed,
    )
    class_names = infer_class_names(train_set)
    generator = torch.Generator().manual_seed(config.seed)
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)

    in_channels = train_set[0].num_node_features
    num_classes = len(class_names)
    model = build_model(config, in_channels, num_classes).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        if config.lr_scheduler == "cosine"
        else None
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    best_path = output_dir / "best_model.pt"
    best_val_loss = float("inf")
    best_val_accuracy = float("-inf")
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    start = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, criterion, device)
        val_result = predict(model, val_loader, criterion, device)
        val_metrics = classification_metrics(
            val_result["labels"], val_result["predictions"], class_names
        )
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": float(val_result["loss"]),
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.4f} val_loss={val_result['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1_macro']:.4f}"
        )
        val_loss = float(val_result["loss"])
        val_accuracy = float(val_metrics["accuracy"])
        improved = (
            val_loss < best_val_loss
            if config.checkpoint_metric == "val_loss"
            else val_accuracy > best_val_accuracy
        )
        if improved:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config.to_dict(),
                    "class_names": class_names,
                    "in_channels": in_channels,
                    "num_classes": num_classes,
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_accuracy": best_val_accuracy,
                    "selection_metric": config.checkpoint_metric,
                },
                best_path,
            )
        else:
            stale_epochs += 1
        if scheduler is not None:
            scheduler.step()
        if config.patience > 0 and stale_epochs >= config.patience:
            print(f"Early stopping after {epoch} epochs")
            break

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_result = predict(model, test_loader, criterion, device)
    test_metrics = classification_metrics(
        test_result["labels"], test_result["predictions"], class_names
    )
    summary = {
        "best_epoch": checkpoint["epoch"],
        "best_val_loss": checkpoint["best_val_loss"],
        "best_val_accuracy": checkpoint["best_val_accuracy"],
        "selection_metric": checkpoint["selection_metric"],
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "device": str(device),
        "paper_reported_accuracy": 0.846,
        "test": {"loss": float(test_result["loss"]), **test_metrics},
    }
    write_json(output_dir / "test_metrics.json", summary)
    print(f"test_acc={test_metrics['accuracy']:.4f} test_f1={test_metrics['f1_macro']:.4f}")
    print(f"checkpoint={best_path}")
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MalNet-Tiny GAT baseline")
    parser.add_argument("--config", default="configs/paper_baseline.yaml")
    parser.add_argument("--device", default=None, help="Override config device, e.g. cpu or cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    if args.device is not None:
        config.device = args.device
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    run(config)


if __name__ == "__main__":
    main()
