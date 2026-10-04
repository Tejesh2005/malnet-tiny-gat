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

### Clean-graph controlled follow-up

The first completed run revealed sparse numeric node IDs that make PyG materialize
artificial zero-degree nodes (up to 14,166 tensor rows despite the documented 5,000-node
cap). To test that loader artifact without changing the model, use a separate processed
root and remove isolates before LDP calculation:

```powershell
.\.venv\Scripts\malnet-download --root data/malnet_tiny_ldp_clean --remove-isolated-nodes
.\.venv\Scripts\malnet-train --config configs/clean_graph_baseline.yaml
```

The next controlled experiment replaces only the five-value LDP with an 11-value
direction-aware profile containing in/out/total degree and incoming/outgoing neighbor
degree statistics:

```powershell
.\.venv\Scripts\malnet-download --root data/malnet_tiny_directed_clean `
  --feature-profile directed_ldp --remove-isolated-nodes
.\.venv\Scripts\malnet-train --config configs/directed_clean_baseline.yaml
```

That direction-aware run achieved **75.5% test accuracy** and **76.36% macro-F1**, the
best result so far. The next controlled run preserves each attention head by
concatenating its representation rather than averaging heads at every layer:

```powershell
.\.venv\Scripts\malnet-train --config configs/directed_concat_baseline.yaml
```

The completed concatenated-head run selected epoch 76 by validation loss and achieved
**85.0% test accuracy** and **85.28% test macro-F1**. This is 0.4 percentage points
above the paper's reported 84.6% accuracy. See `outputs/RESULT_ANALYSIS.md` for the
controlled-experiment comparison and per-class findings.

### Multi-seed confirmation

Confirm stability across five predetermined seeds with one resumable command. A
completed seed-42 directory can be reused, so only the other four models are trained:

```powershell
.\.venv\Scripts\malnet-multiseed `
  --config configs/directed_concat_baseline.yaml `
  --seeds 1 7 21 42 84 `
  --reuse-result 42=runs/directed_concat_baseline `
  --output-root runs/directed_concat_multiseed
```

The runner skips complete per-seed directories on restart and writes
`multiseed_summary.json` plus `multiseed_results.csv`, including sample standard
deviations for test accuracy and macro-F1.

The completed five-seed confirmation (`1, 7, 21, 42, 84`) produced **84.18% ± 1.79%**
test accuracy and **84.44% ± 1.80%** macro-F1 (mean ± sample standard deviation). The
paper's 84.6% accuracy lies within this observed run-to-run variation. Individual
accuracies ranged from 81.3% to 85.7%. See `outputs/MULTISEED_REPORT.md` for the full
seed table and interpretation.

### Exact paper configuration

After obtaining the complete paper, its Table III settings were encoded separately in
`configs/paper_exact.yaml`: raw five-value LDP, a deterministic stratified 80/10/10
split, three 128-channel GAT layers, four averaged heads, ReLU, mean pooling, dropout
0.5, a single linear classifier, batch size 128, Adam at 0.005, and 50 epochs.

The paper does not disclose its split seed, and its unequal test supports conflict with
its claim of stratification on five balanced classes. This project therefore uses seed
42 and produces exactly 4,000/500/500 graphs with 100 examples of every class in both
validation and test. Averaged heads are used because the paper specifies four heads but
a 128-dimensional graph embedding.

```powershell
.\.venv\Scripts\malnet-download --root data/malnet_tiny_paper_exact `
  --no-log-features --split-strategy paper_stratified --split-seed 42
.\.venv\Scripts\malnet-train --config configs/paper_exact.yaml
```

The three Table V ablations are also encoded and change only the stated component:

```powershell
.\.venv\Scripts\malnet-train --config configs/paper_exact_one_head.yaml
.\.venv\Scripts\malnet-download --root data/malnet_tiny_paper_constant `
  --feature-profile constant --no-log-features `
  --split-strategy paper_stratified --split-seed 42
.\.venv\Scripts\malnet-train --config configs/paper_exact_constant.yaml
.\.venv\Scripts\malnet-train --config configs/paper_exact_no_dropout.yaml
```

## Sources

- Freitas et al., *MalNet: A Large-Scale Cybersecurity Graph Database*, NeurIPS 2021.
- IEEE DOI `10.1109/ICSCCC69031.2026.11600324`, target GAT paper, ICSCCC 2026.
- PyTorch Geometric `MalNetTiny` loader, including the official download URLs and split.

