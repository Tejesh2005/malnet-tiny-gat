# MalNet-Tiny GAT baseline — Phase 1

This repository is a reproducible starting point for topology-only malware-type
classification on the official **MalNet-Tiny** split. It targets the method described
in *Android Malware Family Classification using Graph Attention Networks on Function
Call Graphs* (ICSCCC 2026), which reports **84.60% test accuracy**.

Important terminology note: MalNet-Tiny contains five balanced **types** (`addisplay`,
`adware`, `benign`, `downloader`, and `trojan`), not the 696 fine-grained MalNet
families. This project therefore evaluates five-class malware-type classification even
though the target paper says “family classification.”

## What is reproduced

- Official MalNet-Tiny graphs and official train/validation/test split.
- Topology-only structural node profiles: PyG Local Degree Profile
  `[degree, neighbor-degree min, max, mean, std]`, followed by `log1p`.
- A multi-layer, multi-head `GATConv` encoder, graph mean+max pooling, and an MLP head.
- Validation-loss checkpoint selection, deterministic seeding, early stopping, and a
  held-out test evaluation.
- Accuracy, balanced accuracy, macro precision/recall/F1, weighted F1, class-level
  metrics, confusion matrix, predictions, embeddings, and t-SNE.

The paper's abstract is public, but its complete architecture/hyperparameter table and
reference code were not publicly discoverable when this phase was prepared. The values
in `configs/paper_baseline.yaml` are consequently transparent reproduction choices,
not claims about unpublished author settings. The experiment summary always records the
reported target (`0.846`) beside the measured result.

## Setup

PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

PyTorch will use CUDA automatically when a compatible build/device is available;
otherwise the pipeline runs on CPU.

## Dataset

Download and preprocess the official release (the downloaded graphs are numeric edge
lists, not executable APKs):

```powershell
.\.venv\Scripts\malnet-download --root data/malnet_tiny_ldp
.\.venv\Scripts\malnet-inspect --root data/malnet_tiny_ldp --output outputs/dataset_report.json
```

The inspector records split sizes, graph-size statistics, class order/counts, source
URLs, and SHA-256 hashes of the split manifests. `data/` is gitignored.

## Train and evaluate

```powershell
.\.venv\Scripts\malnet-train --config configs/paper_baseline.yaml
.\.venv\Scripts\malnet-evaluate `
  --checkpoint runs/paper_baseline/best_model.pt `
  --output-dir outputs/evaluation
```

For a quick wiring check:

```powershell
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\malnet-train --config configs/paper_baseline.yaml --epochs 1 --output-dir runs/smoke
```

Do not tune on the test results. Change settings using validation metrics, select one
configuration, and evaluate the test split once for the reported comparison.

## Sources

- Freitas et al., *MalNet: A Large-Scale Cybersecurity Graph Database*, NeurIPS 2021.
- IEEE DOI `10.1109/ICSCCC69031.2026.11600324`, target GAT paper, ICSCCC 2026.
- PyTorch Geometric `MalNetTiny` loader, including the official download URLs and split.

