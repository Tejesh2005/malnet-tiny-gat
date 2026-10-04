import torch
from torch_geometric.data import Batch, Data

from malnet_gat.data import DirectedStructuralProfile, StructuralProfile, make_pre_transform
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
