from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch, Data

from malnet_gat.config import ExperimentConfig
from malnet_gat.data import (
    DirectedStructuralProfile,
    EnhancedStructuralProfile,
    StructuralProfile,
    make_pre_transform,
    stratified_split_indices,
)
from malnet_gat.model import MalNetGAT, MalNetGGNN, MalNetMessagePassing, MalNetRGCN
from malnet_gat.robustness import augment_batch, perturb_graph


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


def test_enhanced_profile_adds_pagerank_and_clustering() -> None:
    graph = Data(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        num_nodes=3,
    )
    transformed = EnhancedStructuralProfile(log_features=False)(graph)
    assert transformed.x.shape == (3, 13)
    assert torch.isfinite(transformed.x).all()
    assert transformed.x[:, -2].sum().item() == pytest.approx(1.0, abs=1e-5)
    assert transformed.x[:, -1].tolist() == pytest.approx([1.0, 1.0, 1.0])


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


def test_clean_paper_configuration_removes_identifier_gaps() -> None:
    config = ExperimentConfig.from_yaml(Path("configs/paper_exact_clean.yaml"))
    assert config.feature_profile == "ldp"
    assert config.log_features is False
    assert config.remove_isolated_nodes is True
    assert config.pooling == "mean"
    assert config.hidden_channels * config.heads == 128


def test_paper_dropout_is_only_after_graph_pooling() -> None:
    config = ExperimentConfig.from_yaml(Path("configs/paper_exact_graph_dropout.yaml"))
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
        attention_dropout=config.attention_dropout,
        dropout_location=config.dropout_location,
        pooling=config.pooling,
        classifier_hidden=config.classifier_hidden,
    )
    assert all(conv.dropout == 0.0 for conv in model.convs)
    assert model.node_dropout is False
    assert model.graph_dropout is True


def test_comparison_models_return_matching_graph_shapes() -> None:
    graphs = [
        Data(
            x=torch.ones((3, 5)),
            edge_index=torch.tensor([[0, 1], [1, 2]]),
            y=index,
        )
        for index in range(2)
    ]
    batch = Batch.from_data_list(graphs)
    for model_type in ("gcn", "graphsage", "gin"):
        model = MalNetMessagePassing(
            model_type,
            5,
            5,
            hidden_channels=8,
            num_layers=3,
        )
        logits, embeddings = model(
            batch.x, batch.edge_index, batch.batch, return_embedding=True
        )
        assert logits.shape == (2, 5)
        assert embeddings.shape == (2, 8)


def test_gated_and_relational_models_return_matching_graph_shapes() -> None:
    graphs = [
        Data(
            x=torch.ones((3, 5)),
            edge_index=torch.tensor([[0, 1], [1, 2]]),
            y=index,
        )
        for index in range(2)
    ]
    batch = Batch.from_data_list(graphs)
    for model in (
        MalNetGGNN(5, 5, hidden_channels=8, num_layers=3),
        MalNetRGCN(5, 5, hidden_channels=8, num_layers=3),
    ):
        logits, embeddings = model(
            batch.x, batch.edge_index, batch.batch, return_embedding=True
        )
        assert logits.shape == (2, 5)
        assert embeddings.shape == (2, 8)


def test_graph_perturbations_recompute_features() -> None:
    graph = Data(
        x=torch.ones((3, 5)),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        y=1,
        num_nodes=3,
    )
    for index, perturbation in enumerate(("add_nodes", "add_edges", "rewire", "add_paths")):
        changed, modifications = perturb_graph(
            graph,
            perturbation,
            fraction=0.1,
            feature_profile="ldp",
            log_features=True,
            generator=torch.Generator().manual_seed(index),
        )
        assert changed.x.shape == (changed.num_nodes, 5)
        assert torch.isfinite(changed.x).all()
        assert modifications > 0


def test_robust_augmentation_keeps_originals_and_adds_copies() -> None:
    graphs = [
        Data(
            x=torch.ones((3, 5)),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            y=index,
            num_nodes=3,
        )
        for index in range(2)
    ]
    batch = Batch.from_data_list(graphs)
    augmented = augment_batch(
        batch,
        probability=1.0,
        fraction=0.1,
        perturbation_types=["add_nodes"],
        feature_profile="ldp",
        log_features=True,
        generator=torch.Generator().manual_seed(42),
    )
    assert augmented.num_graphs == 4
    assert augmented.y.tolist() == [0, 1, 0, 1]


def test_model_type_validation() -> None:
    config = ExperimentConfig(model_type="unknown")
    try:
        config.validate()
    except ValueError as error:
        assert "model_type" in str(error)
    else:
        raise AssertionError("Expected invalid model_type to fail validation")


def test_extended_configuration_enables_new_models_and_robust_training() -> None:
    config = ExperimentConfig.from_yaml(Path("configs/robust_gin_enhanced.yaml"))
    assert config.feature_profile == "enhanced"
    assert config.robust_training is True
    assert set(config.robust_perturbation_types) == {
        "add_nodes",
        "add_edges",
        "rewire",
        "add_paths",
    }
    for model_type in ("ggnn", "rgcn"):
        ExperimentConfig(model_type=model_type).validate()
