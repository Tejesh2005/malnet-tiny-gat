from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import MalNetTiny
from torch_geometric.transforms import BaseTransform, LocalDegreeProfile

KNOWN_CLASS_NAMES = ["adware", "benign", "downloader", "trojan", "addisplay"]


class ConstantFeatures(BaseTransform):
    """Featureless-graph control: attach one scalar 1 to every node."""

    def forward(self, data: Data) -> Data:
        data.x = torch.ones((data.num_nodes, 1), dtype=torch.float32)
        return data


class Log1pFeatures(BaseTransform):
    """Compress heavy-tailed non-negative structural values."""

    def forward(self, data: Data) -> Data:
        if data.x is None:
            raise ValueError("Log1pFeatures requires node features")
        data.x = torch.log1p(data.x.float())
        return data


class StructuralProfile(BaseTransform):
    """Local degree profile used as a topology-only node descriptor.

    PyG's LocalDegreeProfile returns [degree, min, max, mean, std] of neighbor
    degree. No APK bytes, API names, node identities, or edge attributes enter
    the model.
    """

    def __init__(self, log_features: bool = True) -> None:
        self.ldp = LocalDegreeProfile()
        self.log_features = log_features

    def forward(self, data: Data) -> Data:
        data = self.ldp(data)
        data.x = torch.nan_to_num(data.x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if self.log_features:
            data.x = torch.log1p(data.x)
        return data


def make_pre_transform(feature_profile: str, log_features: bool = True) -> BaseTransform:
    if feature_profile == "ldp":
        return StructuralProfile(log_features=log_features)
    if feature_profile == "constant":
        return ConstantFeatures()
    raise ValueError(f"Unsupported feature profile: {feature_profile}")


def load_splits(
    root: str | Path,
    feature_profile: str = "ldp",
    log_features: bool = True,
) -> tuple[MalNetTiny, MalNetTiny, MalNetTiny]:
    transform = make_pre_transform(feature_profile, log_features)
    root = str(root)
    train = MalNetTiny(root=root, split="train", pre_transform=transform)
    val = MalNetTiny(root=root, split="val", pre_transform=transform)
    test = MalNetTiny(root=root, split="test", pre_transform=transform)
    return train, val, test


def infer_class_names(dataset: MalNetTiny) -> list[str]:
    """Recover label order from the release split files when available."""
    raw_split = Path(dataset.raw_dir) / "split_info_tiny" / "type"
    label_by_type: dict[str, int] = {}
    # PyG assigns ids in train -> val -> test file order.
    next_label = 0
    for split in ("train", "val", "test"):
        path = raw_split / f"{split}.txt"
        if not path.exists():
            continue
        for item in path.read_text(encoding="utf-8").splitlines():
            malware_type = item.replace("\\", "/").split("/", 1)[0]
            if malware_type and malware_type not in label_by_type:
                label_by_type[malware_type] = next_label
                next_label += 1
    if label_by_type:
        return [name for name, _ in sorted(label_by_type.items(), key=lambda pair: pair[1])]
    return KNOWN_CLASS_NAMES.copy()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stats(dataset: Iterable[Data], limit: int | None = None) -> dict[str, object]:
    nodes: list[int] = []
    edges: list[int] = []
    labels: Counter[int] = Counter()
    for index, graph in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        nodes.append(graph.num_nodes)
        edges.append(graph.num_edges)
        labels[int(graph.y)] += 1
    tensor_nodes = torch.tensor(nodes, dtype=torch.float32)
    tensor_edges = torch.tensor(edges, dtype=torch.float32)
    return {
        "graphs_scanned": len(nodes),
        "class_counts": dict(sorted(labels.items())),
        "nodes": {
            "min": min(nodes),
            "max": max(nodes),
            "mean": round(float(tensor_nodes.mean()), 3),
            "median": round(float(tensor_nodes.median()), 3),
        },
        "edges": {
            "min": min(edges),
            "max": max(edges),
            "mean": round(float(tensor_edges.mean()), 3),
            "median": round(float(tensor_edges.median()), 3),
        },
    }


def inspect_dataset(root: str | Path, limit: int | None = None) -> dict[str, object]:
    train, val, test = load_splits(root)
    class_names = infer_class_names(train)
    report: dict[str, object] = {
        "dataset": "MalNet-Tiny",
        "source": MalNetTiny.data_url,
        "split_source": MalNetTiny.split_url,
        "class_names_by_label": class_names,
        "num_node_features": train.num_node_features,
        "splits": {},
    }
    for name, dataset in (("train", train), ("val", val), ("test", test)):
        report["splits"][name] = {"size": len(dataset), **_stats(dataset, limit)}
    split_dir = Path(train.raw_dir) / "split_info_tiny" / "type"
    report["split_file_sha256"] = {
        path.name: _file_sha256(path) for path in sorted(split_dir.glob("*.txt"))
    }
    return report


def download_main() -> None:
    parser = argparse.ArgumentParser(description="Download and preprocess MalNet-Tiny")
    parser.add_argument("--root", default="data/malnet_tiny_ldp")
    parser.add_argument("--feature-profile", choices=["ldp", "constant"], default="ldp")
    parser.add_argument("--no-log-features", action="store_true")
    args = parser.parse_args()
    train, val, test = load_splits(
        args.root, args.feature_profile, log_features=not args.no_log_features
    )
    print(f"Ready: train={len(train)} val={len(val)} test={len(test)}")
    print(f"Node features: {train.num_node_features}; classes: {train.num_classes}")


def inspect_main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a processed MalNet-Tiny release")
    parser.add_argument("--root", default="data/malnet_tiny_ldp")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = inspect_dataset(args.root, args.limit)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    inspect_main()
