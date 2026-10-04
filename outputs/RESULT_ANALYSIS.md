# First full-run analysis

Run source: Kaggle GPU T4, seed 42, `configs/paper_baseline.yaml`.

## Result

- Best validation-loss epoch: 78 of 98 completed epochs
- Validation accuracy at selected epoch: 77.2%
- Validation macro-F1 at selected epoch: 76.66%
- Test accuracy: 74.2%
- Test macro-F1: 74.10%
- Paper-reported target: 84.6%
- Reproduction gap: 10.4 percentage points
- Runtime: 836.956 seconds

The selected checkpoint behaves similarly on validation and test; there is no evidence
that the gap is caused primarily by severe model overfitting. Training accuracy at the
selected epoch was only 77.29%, indicating limited representation/model fit instead.

## Class-level findings

| Class | Precision | Recall | F1 | Main failure |
|---|---:|---:|---:|---|
| adware | 88.77% | 83.0% | 85.79% | 22 predicted benign |
| benign | 47.46% | 70.0% | 56.57% | absorbs other classes |
| downloader | 95.67% | 99.5% | 97.55% | essentially solved |
| trojan | 69.68% | 77.0% | 73.16% | 31 predicted benign |
| addisplay | 93.26% | 41.5% | 57.44% | 102 predicted benign |

The dominant error is `addisplay -> benign`, accounting for 102 errors by itself. The
t-SNE projection agrees: downloader is strongly separated, whereas benign, addisplay,
and parts of trojan overlap.

## Next controlled experiment

The official PyG loader sets `num_nodes = max(edge_index) + 1`. The released edge lists
contain sparse numeric identifiers, so this creates artificial isolated tensor rows and
produces an observed maximum of 14,166 nodes, despite MalNet-Tiny's documented 5,000-node
cap. Those artificial nodes dilute mean pooling, add thousands of GAT self-loops, and
explain the occasional very slow batch.

`configs/clean_graph_baseline.yaml` changes only one factor: it applies PyG
`RemoveIsolatedNodes` before calculating the same Local Degree Profile. It uses a new
dataset root so the preprocessing cache cannot be confused with the first run. Model,
optimizer, split, seed, checkpoint selection, and evaluation remain unchanged.

Select the checkpoint on validation loss exactly as before. Evaluate the test split once
after the run; do not use the existing 74.2% test result to tune this configuration.

## Clean-graph result

The clean-graph run selected epoch 82 and reached 77.4% validation accuracy, 76.63%
validation macro-F1, 73.6% test accuracy, and 73.02% test macro-F1. This is effectively
unchanged from the original run at the aggregate level. Runtime fell only from 836.956
to 813.448 seconds.

The class boundary changed substantially: addisplay recall increased from 41.5% to
87.5%, while benign recall dropped from 70.0% to 42.5%. This confirms that isolated-node
handling is not the missing source of the paper's 84.6%, and that the five LDP values do
not stably separate the overlapping advertising/benign classes.

The next controlled configuration, `configs/directed_clean_baseline.yaml`, keeps the
clean graph and every training/model setting fixed while replacing LDP with 11 explicit
direction-aware values: in-degree, out-degree, total degree, and min/max/mean/std total
degree over outgoing and incoming neighbors. This tests whether caller/callee asymmetry
supplies the missing structural signal without introducing semantic APK features.

## Direction-aware result

The direction-aware run selected epoch 33 and stopped normally after epoch 53. It
reached 79.4% validation accuracy, 79.78% validation macro-F1, 75.5% test accuracy,
and 76.36% test macro-F1 in 446.265 seconds. This is the strongest run so far: relative
to the original baseline it improves test accuracy by 1.3 points and macro-F1 by 2.26
points. It remains 9.1 accuracy points below the paper's reported 84.6%.

Downloader remains essentially solved (99.5% recall). The directional signal makes the
advertising classes more balanced than either five-value LDP run: addisplay recall is
70.5% and adware recall is 61.5%. The largest residual errors all collapse into benign:
73 adware, 39 trojan, and 51 addisplay samples. The t-SNE projection shows the same
pattern: downloader has compact, isolated clusters while benign and the advertising
types overlap heavily.

The 79.4% validation versus 75.5% test result also cautions against choosing the next
configuration from test performance. `configs/directed_concat_baseline.yaml` changes
one architectural factor selected in advance: it uses standard concatenated multi-head
GAT layers instead of averaging the four heads at every layer. This retains distinct
attention-head representations for graph pooling while leaving the data, split,
features, optimizer, stopping rule, and seed unchanged. Select it on validation loss
and evaluate the test set only once after training.

