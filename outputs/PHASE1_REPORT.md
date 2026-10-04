# Phase 1 report — MalNet-Tiny GAT baseline

Prepared: 4 October 2026 (Asia/Calcutta)

## Outcome

The project is set up, the official MalNet-Tiny archive and official split manifests
have been downloaded, and four controlled topology-only GAT experiments have been run
end to end on the complete official split. A five-seed confirmation of the final frozen
configuration achieved **84.18% ± 1.79% test accuracy** and **84.44% ± 1.80% test
macro-F1** (mean ± sample standard deviation). The paper's reported 84.6% accuracy is
within the observed seed-to-seed variation.

## Reproduction target

- Paper: *Android Malware Family Classification using Graph Attention Networks on
  Function Call Graphs*, ICSCCC 2026.
- DOI: <https://doi.org/10.1109/ICSCCC69031.2026.11600324>
- Publicly reported accuracy: **84.60%**.
- Publicly described ingredients: MalNet-Tiny function-call graphs, topology-derived
  structural node profiles, graph attention, and t-SNE analysis.

The paper's complete methods/hyperparameter table and reference implementation were not
publicly discoverable during this phase. The checked-in configuration is therefore a
close, explicit reconstruction rather than a claim of exact hidden settings.

## Dataset inspection

Source: <http://malnet.cc.gatech.edu/graph-data/malnet-graphs-tiny.tar.gz>

| Split | Graphs | Per class | Mean nodes | Median nodes | Mean edges |
|---|---:|---:|---:|---:|---:|
| Train | 3,500 | 700 | 1,504.672 | 987 | 2,831.067 |
| Validation | 500 | 100 | 1,527.334 | 849 | 2,867.490 |
| Test | 1,000 | 200 | 1,580.284 | 1,036 | 2,957.264 |

The observed label order assigned by the official PyG loader is `adware`, `benign`,
`downloader`, `trojan`, `addisplay`. All splits are exactly balanced.

The processed release occupies approximately 507 MB. Although the MalNet paper describes
Tiny graphs as having at most 5,000 nodes, the official PyG loader derives `num_nodes` as
`max(edge_index) + 1`; sparse numeric node identifiers produce an observed maximum tensor
size of 14,166. The baseline retains this official-loader behavior for reproducibility.
The exact split-manifest hashes and full descriptive statistics are in
`dataset_report.json`.

## Implemented reconstruction

- Direction-aware structural node features: in-degree, out-degree, total degree, and
  incoming/outgoing neighbor-degree minimum, maximum, mean, and standard deviation.
- Isolated rows induced by sparse numeric node identifiers are removed before feature
  calculation.
- `log1p` compression of the non-negative structural values.
- Three GAT layers, 64 channels per head, four concatenated attention heads, ELU, layer
  normalization, 0.30 dropout, mean+max graph pooling, and an MLP classifier.
- Adam (`lr=0.001`, weight decay `0.0005`), cosine schedule, validation-loss checkpoint
  selection, 100-epoch ceiling, and patience of 20 epochs.
- Accuracy, balanced accuracy, macro precision/recall/F1, weighted F1, per-class metrics,
  confusion matrix, per-sample probabilities, graph embeddings, and t-SNE.

No APK bytes, API names, node identities, or hand-authored semantic attributes enter the
model.

## Final result and verification

- Python 3.13.9
- PyTorch 2.14.1 CPU build
- PyTorch Geometric 2.8.0.post1
- Kaggle GPU: NVIDIA T4
- Unit tests: 5 passed
- Lint: passed
- Seeds: 1, 7, 21, 42, and 84
- Test accuracy: **84.18% ± 1.79%** (range 81.3%–85.7%)
- Test macro-F1: **84.44% ± 1.80%**
- Seed-42 test accuracy / macro-F1: 85.0% / 85.28%
- Total runtime across five seeds: 4,187.193 seconds (69.79 minutes)
- Evaluation export: metrics, confusion matrix, predictions, embeddings, and t-SNE

## Commands

Run the successful reconstruction:

```powershell
.\.venv\Scripts\malnet-download --root data\malnet_tiny_directed_clean `
  --feature-profile directed_ldp --remove-isolated-nodes
.\.venv\Scripts\malnet-train --config configs\directed_concat_baseline.yaml
```

Evaluate the selected checkpoint:

```powershell
.\.venv\Scripts\malnet-evaluate `
  --checkpoint runs\directed_concat_baseline\best_model.pt `
  --output-dir outputs\directed_concat_evaluation
```

