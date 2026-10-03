import torch
from torch_geometric.data import Batch, Data

from malnet_gat.data import StructuralProfile
from malnet_gat.model import MalNetGAT


def test_structural_profile_is_finite() -> None:
    graph = Data(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        num_nodes=4,
    )
    transformed = StructuralProfile()(graph)
    assert transformed.x.shape == (4, 5)
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
