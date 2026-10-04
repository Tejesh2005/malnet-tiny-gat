from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool


class MalNetGAT(nn.Module):
    """Multi-head GAT graph classifier over topology-only node features."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        concat_heads: bool = False,
        activation: str = "elu",
        layer_norm: bool = True,
        dropout: float = 0.3,
        attention_dropout: float | None = None,
        dropout_location: str = "node",
        pooling: str = "mean_max",
        classifier_hidden: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.dropout = dropout
        self.node_dropout = dropout_location in {"node", "both"}
        self.graph_dropout = dropout_location in {"graph", "both"}
        self.pooling = pooling
        self.activation = activation
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        current = in_channels
        for _ in range(num_layers):
            output_channels = hidden_channels * heads if concat_heads else hidden_channels
            self.convs.append(
                GATConv(
                    current,
                    hidden_channels,
                    heads=heads,
                    concat=concat_heads,
                    dropout=dropout if attention_dropout is None else attention_dropout,
                    add_self_loops=True,
                )
            )
            self.norms.append(nn.LayerNorm(output_channels) if layer_norm else nn.Identity())
            current = output_channels

        pooled_channels = current * (2 if pooling == "mean_max" else 1)
        if classifier_hidden:
            nonlinearity: nn.Module = nn.ELU() if activation == "elu" else nn.ReLU()
            self.classifier = nn.Sequential(
                nn.Linear(pooled_channels, hidden_channels),
                nonlinearity,
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, num_classes),
            )
        else:
            self.classifier = nn.Linear(pooled_channels, num_classes)

    def _activate(self, x: Tensor) -> Tensor:
        if self.activation == "elu":
            return F.elu(x)
        if self.activation == "relu":
            return F.relu(x)
        raise ValueError(f"Unsupported activation: {self.activation}")

    def encode_nodes(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for conv, norm in zip(self.convs, self.norms, strict=True):
            x = conv(x, edge_index)
            x = norm(x)
            x = self._activate(x)
            if self.node_dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def pool(self, x: Tensor, batch: Tensor) -> Tensor:
        if self.pooling == "mean":
            return global_mean_pool(x, batch)
        if self.pooling == "max":
            return global_max_pool(x, batch)
        if self.pooling == "mean_max":
            return torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)
        raise ValueError(f"Unsupported pooling: {self.pooling}")

    def forward(
        self, x: Tensor, edge_index: Tensor, batch: Tensor, return_embedding: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        node_embeddings = self.encode_nodes(x, edge_index)
        graph_embedding = self.pool(node_embeddings, batch)
        if self.graph_dropout:
            graph_embedding = F.dropout(
                graph_embedding, p=self.dropout, training=self.training
            )
        logits = self.classifier(graph_embedding)
        if return_embedding:
            return logits, graph_embedding
        return logits
