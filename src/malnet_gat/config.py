from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    seed: int = 42
    data_root: str = "data/malnet_tiny_ldp"
    output_dir: str = "runs/paper_baseline"
    feature_profile: str = "ldp"
    log_features: bool = True
    remove_isolated_nodes: bool = False
    split_strategy: str = "official"
    split_seed: int = 42
    hidden_channels: int = 64
    num_layers: int = 3
    heads: int = 4
    concat_heads: bool = False
    activation: str = "elu"
    layer_norm: bool = True
    dropout: float = 0.3
    pooling: str = "mean_max"
    classifier_hidden: bool = True
    batch_size: int = 16
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    label_smoothing: float = 0.0
    patience: int = 20
    lr_scheduler: str = "cosine"
    checkpoint_metric: str = "val_loss"
    num_workers: int = 0
    device: str = "auto"

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {', '.join(unknown)}")
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        if self.feature_profile not in {"ldp", "directed_ldp", "constant"}:
            raise ValueError("feature_profile must be 'ldp', 'directed_ldp', or 'constant'")
        if self.pooling not in {"mean", "max", "mean_max"}:
            raise ValueError("pooling must be 'mean', 'max', or 'mean_max'")
        if self.split_strategy not in {"official", "paper_stratified"}:
            raise ValueError("split_strategy must be 'official' or 'paper_stratified'")
        if self.activation not in {"elu", "relu"}:
            raise ValueError("activation must be 'elu' or 'relu'")
        if self.lr_scheduler not in {"cosine", "none"}:
            raise ValueError("lr_scheduler must be 'cosine' or 'none'")
        if self.checkpoint_metric not in {"val_loss", "val_accuracy"}:
            raise ValueError("checkpoint_metric must be 'val_loss' or 'val_accuracy'")
        if self.num_layers < 1 or self.heads < 1 or self.hidden_channels < 1:
            raise ValueError("num_layers, heads, and hidden_channels must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.patience < 0:
            raise ValueError("patience must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
