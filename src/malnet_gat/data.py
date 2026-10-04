from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, Dataset, Subset
from torch_geometric.data import Data
from torch_geometric.datasets import MalNetTiny
from torch_geometric.transforms import (
    BaseTransform,
    Compose,
    LocalDegreeProfile,
    RemoveIsolatedNodes,
)

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


class DirectedStructuralProfile(BaseTransform):
    """Direction-aware topology profile for function-call graphs.

    Features are in-degree, out-degree, total degree, followed by min/max/mean/std
    total-degree statistics for outgoing and incoming neighbors. The resulting 11
    values retain caller/callee asymmetry that the five-value LDP does not expose.
    """

    def __init__(self, log_features: bool = True) -> None:
        self.log_features = log_features

    @staticmethod
    def _neighbor_stats(index: torch.Tensor, values: torch.Tensor, size: int) -> torch.Tensor:
        counts = values.new_zeros(size)
        counts.scatter_add_(0, index, torch.ones_like(values))

        sums = values.new_zeros(size)
        sums.scatter_add_(0, index, values)
        squared_sums = values.new_zeros(size)
        squared_sums.scatter_add_(0, index, values.square())
        safe_counts = counts.clamp_min(1)
        means = sums / safe_counts
        variances = (squared_sums / safe_counts - means.square()).clamp_min(0)

        minimums = values.new_full((size,), float("inf"))
        maximums = values.new_full((size,), float("-inf"))
        minimums.scatter_reduce_(0, index, values, reduce="amin", include_self=True)
        maximums.scatter_reduce_(0, index, values, reduce="amax", include_self=True)
        has_neighbors = counts > 0
        minimums = torch.where(has_neighbors, minimums, torch.zeros_like(minimums))
        maximums = torch.where(has_neighbors, maximums, torch.zeros_like(maximums))
        return torch.stack([minimums, maximums, means, variances.sqrt()], dim=-1)

    def forward(self, data: Data) -> Data:
        source, target = data.edge_index
        size = data.num_nodes
        out_degree = torch.bincount(source, minlength=size).float()
        in_degree = torch.bincount(target, minlength=size).float()
        total_degree = in_degree + out_degree
        outgoing_stats = self._neighbor_stats(source, total_degree[target], size)
        incoming_stats = self._neighbor_stats(target, total_degree[source], size)
        data.x = torch.cat(
            [
                in_degree.unsqueeze(-1),
                out_degree.unsqueeze(-1),
                total_degree.unsqueeze(-1),
                outgoing_stats,
                incoming_stats,
            ],
            dim=-1,
        )
        if self.log_features:
            data.x = torch.log1p(data.x)
        return data


def make_pre_transform(
    feature_profile: str,
    log_features: bool = True,
    remove_isolated_nodes: bool = False,
) -> BaseTransform:
    transforms: list[BaseTransform] = []
    if remove_isolated_nodes:
        transforms.append(RemoveIsolatedNodes())
    if feature_profile == "ldp":
        transforms.append(StructuralProfile(log_features=log_features))
    elif feature_profile == "directed_ldp":
        transforms.append(DirectedStructuralProfile(log_features=log_features))
    elif feature_profile == "constant":
        transforms.append(ConstantFeatures())
    else:
        raise ValueError(f"Unsupported feature profile: {feature_profile}")
    return Compose(transforms) if len(transforms) > 1 else transforms[0]


def load_splits(
    root: str | Path,
    feature_profile: str = "ldp",
    log_features: bool = True,
    remove_isolated_nodes: bool = False,
    split_strategy: str = "official",
    split_seed: int = 42,
) -> tuple[Dataset, Dataset, Dataset]:
    transform = make_pre_transform(feature_profile, log_features, remove_isolated_nodes)
    root = str(root)
    train = MalNetTiny(root=root, split="train", pre_transform=transform)
    val = MalNetTiny(root=root, split="val", pre_transform=transform)
    test = MalNetTiny(root=root, split="test", pre_transform=transform)
    if split_strategy == "paper_stratified":
        combined = ConcatDataset([train, val, test])
        labels = [int(combined[index].y) for index in range(len(combined))]
        train_indices, val_indices, test_indices = stratified_split_indices(labels, split_seed)
        return (
            Subset(combined, train_indices),
            Subset(combined, val_indices),
            Subset(combined, test_indices),
        )
    if split_strategy != "official":
        raise ValueError(f"Unsupported split strategy: {split_strategy}")
    return train, val, test


def stratified_split_indices(
    labels: list[int], seed: int
) -> tuple[list[int], list[int], list[int]]:
    """Create the paper-stated deterministic 80/10/10 stratified split."""
    indices = list(range(len(labels)))
    train_indices, remainder = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )
    remainder_labels = [labels[index] for index in remainder]
    val_indices, test_indices = train_test_split(
        remainder,
        test_size=0.5,
        random_state=seed,
        stratify=remainder_labels,
    )
    return list(train_indices), list(val_indices), list(test_indices)


def _malnet_base(dataset: Dataset) -> MalNetTiny:
    if isinstance(dataset, MalNetTiny):
        return dataset
    if isinstance(dataset, Subset):
        return _malnet_base(dataset.dataset)
    if isinstance(dataset, ConcatDataset):
        return _malnet_base(dataset.datasets[0])
    raise TypeError(f"Cannot locate MalNetTiny base for {type(dataset).__name__}")


def infer_class_names(dataset: Dataset) -> list[str]:
    """Recover label order from the release split files when available."""
    raw_split = Path(_malnet_base(dataset).raw_dir) / "split_info_tiny" / "type"
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
    parser.add_argument(
        "--feature-profile", choices=["ldp", "directed_ldp", "constant"], default="ldp"
    )
    parser.add_argument("--no-log-features", action="store_true")
    parser.add_argument("--remove-isolated-nodes", action="store_true")
    parser.add_argument(
        "--split-strategy", choices=["official", "paper_stratified"], default="official"
    )
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    train, val, test = load_splits(
        args.root,
        args.feature_profile,
        log_features=not args.no_log_features,
        remove_isolated_nodes=args.remove_isolated_nodes,
        split_strategy=args.split_strategy,
        split_seed=args.split_seed,
    )
    print(f"Ready: train={len(train)} val={len(val)} test={len(test)}")
    print(f"Node features: {train[0].num_node_features}; classes: {len(infer_class_names(train))}")


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
