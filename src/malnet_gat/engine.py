from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch


def train_epoch(
    model: nn.Module,
    loader: Iterable[Batch],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    count = 0
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        amp_context = torch.amp.autocast("cuda") if use_amp else nullcontext()
        with amp_context:
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        size = int(batch.y.numel())
        loss_sum += float(loss.detach()) * size
        correct += int((logits.argmax(dim=-1) == batch.y).sum())
        count += size
    return loss_sum / count, correct / count


@torch.inference_mode()
def predict(
    model: nn.Module,
    loader: Iterable[Batch],
    criterion: nn.Module,
    device: torch.device,
    include_embeddings: bool = False,
) -> dict[str, np.ndarray | float]:
    model.eval()
    loss_sum = 0.0
    count = 0
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        logits, graph_embedding = model(
            batch.x, batch.edge_index, batch.batch, return_embedding=True
        )
        loss = criterion(logits, batch.y)
        probs = logits.softmax(dim=-1)
        size = int(batch.y.numel())
        loss_sum += float(loss) * size
        count += size
        labels.append(batch.y.cpu().numpy())
        predictions.append(logits.argmax(dim=-1).cpu().numpy())
        probabilities.append(probs.cpu().numpy())
        if include_embeddings:
            embeddings.append(graph_embedding.cpu().numpy())
    result: dict[str, np.ndarray | float] = {
        "loss": loss_sum / count,
        "labels": np.concatenate(labels),
        "predictions": np.concatenate(predictions),
        "probabilities": np.concatenate(probabilities),
    }
    if include_embeddings:
        result["embeddings"] = np.concatenate(embeddings)
    return result
