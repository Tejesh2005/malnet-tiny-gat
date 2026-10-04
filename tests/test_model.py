from pathlib import Path

import torch
from torch_geometric.data import Batch, Data

from malnet_gat.config import ExperimentConfig
from malnet_gat.data import (
    DirectedStructuralProfile,
    StructuralProfile,
    make_pre_transform,
    stratified_split_indices,
)
from malnet_gat.model import MalNetGAT


def test_structural_profile_is_finite() -> None:
    graph = Data(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        num_nodes=4,
    )
    transformed = StructuralProfile()(graph)
    assert transformed.x.shape == (4, 5)
    assert torch.isfinite(transformed.x).all()


def test_clean_profile_removes_isolated_nodes_before_features() -> None:
    graph = Data(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        num_nodes=10,
    )
    transformed = make_pre_transform("ldp", remove_isolated_nodes=True)(graph)
    assert transformed.num_nodes == 3
    assert transformed.x.shape == (3, 5)


def test_directed_profile_separates_in_and_out_degree() -> None:
    graph = Data(
        edge_index=torch.tensor([[0, 0, 2], [1, 2, 1]], dtype=torch.long),
        num_nodes=3,
    )
    transformed = DirectedStructuralProfile(log_features=False)(graph)
    assert transformed.x.shape == (3, 11)
    assert transformed.x[:, 0].tolist() == [0.0, 2.0, 1.0]
    assert transformed.x[:, 1].tolist() == [2.0, 0.0, 1.0]
    assert torch.isfinite(transformed.x).all()


def test_model_returns_graph_logits_and_embeddings() -> None:
    graphs = [
        Data(x=torch.ones((3, 5)), edge_index=torch.tensor([[0, 1], [1, 2]]), y=i) for i in range(2)
    ]
    batch = Batch.from_data_list(graphs)
    model = MalNetGAT(5, 5, hidden_channels=8, num_layers=2, heads=2)
    logits, embeddings = model(batch.x, batch.edge_index, batch.batch, return_embedding=True)
    assert logits.shape == (2, 5)
    assert embeddings.shape == (2, 16)


def test_model_can_preserve_concatenated_attention_heads() -> None:
    graphs = [
        Data(x=torch.ones((3, 5)), edge_index=torch.tensor([[0, 1], [1, 2]]), y=i) for i in range(2)
    ]
    batch = Batch.from_data_list(graphs)
    model = MalNetGAT(
        5,
        5,
        hidden_channels=8,
        num_layers=2,
        heads=2,
        concat_heads=True,
    )
    logits, embeddings = model(batch.x, batch.edge_index, batch.batch, return_embedding=True)
    assert logits.shape == (2, 5)
    assert embeddings.shape == (2, 32)


def test_paper_model_returns_128_dimensional_embedding() -> None:
    graphs = [
        Data(x=torch.ones((3, 5)), edge_index=torch.tensor([[0, 1], [1, 2]]), y=i) for i in range(2)
    ]
    batch = Batch.from_data_list(graphs)
    model = MalNetGAT(
        5,
        5,
        hidden_channels=128,
        num_layers=3,
        heads=4,
        concat_heads=False,
        activation="relu",
        layer_norm=False,
        dropout=0.5,
        pooling="mean",
        classifier_hidden=False,
    )
    logits, embeddings = model(batch.x, batch.edge_index, batch.batch, return_embedding=True)
    assert logits.shape == (2, 5)
    assert embeddings.shape == (2, 128)


def test_paper_split_is_deterministic_balanced_80_10_10() -> None:
    labels = [label for label in range(5) for _ in range(100)]
    first = stratified_split_indices(labels, seed=42)
    second = stratified_split_indices(labels, seed=42)
    assert first == second
    assert [len(indices) for indices in first] == [400, 50, 50]
    for indices, expected_per_class in zip(first, [80, 10, 10], strict=True):
        counts = torch.bincount(torch.tensor([labels[index] for index in indices]), minlength=5)
        assert counts.tolist() == [expected_per_class] * 5


def test_exact_paper_configuration_matches_table_three() -> None:
    config = ExperimentConfig.from_yaml(Path("configs/paper_exact.yaml"))
    assert config.feature_profile == "ldp"
    assert config.log_features is False
    assert config.split_strategy == "paper_stratified"
    assert (config.hidden_channels, config.num_layers, config.heads) == (128, 3, 4)
    assert config.concat_heads is False
    assert config.activation == "relu"
    assert config.layer_norm is False
    assert config.pooling == "mean"
    assert config.classifier_hidden is False
    assert (config.dropout, config.batch_size, config.epochs) == (0.5, 128, 50)
    assert config.learning_rate == 0.005
    assert config.lr_scheduler == "none"


def test_concat_paper_interpretation_still_has_128_dimensions() -> None:
    config = ExperimentConfig.from_yaml(Path("configs/paper_exact_concat128.yaml"))
    model = MalNetGAT(
        5,
        5,
        hidden_channels=config.hidden_channels,
        num_layers=config.num_layers,
        heads=config.heads,
        concat_heads=config.concat_heads,
        activation=config.activation,
        layer_norm=config.layer_norm,
        dropout=config.dropout,
        pooling=config.pooling,
        classifier_hidden=config.classifier_hidden,
    )
    graph = Data(x=torch.ones((3, 5)), edge_index=torch.tensor([[0, 1], [1, 2]]), y=0)
    batch = Batch.from_data_list([graph])
    _, embedding = model(batch.x, batch.edge_index, batch.batch, return_embedding=True)
    assert embedding.shape == (1, 128)
