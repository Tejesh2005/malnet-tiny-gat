# Multi-seed confirmation — MalNet-Tiny GAT

Prepared: 4 October 2026 (Asia/Calcutta)

## Primary result

The frozen `configs/directed_concat_baseline.yaml` configuration was evaluated with
five predetermined random seeds on the official MalNet-Tiny split.

- Test accuracy: **84.18% ± 1.79%**
- Test macro-F1: **84.44% ± 1.80%**
- Accuracy range: **81.3%–85.7%**
- Paper-reported accuracy: **84.6%**
- Difference between means: **-0.42 percentage points**

Values are arithmetic mean ± sample standard deviation. The paper score is a reported
single value rather than a published multi-seed distribution, so no claim of statistical
superiority or equivalence is made. It lies within the reconstruction's observed
seed-to-seed variability.

## Per-seed results

| Seed | Selected epoch | Best validation loss | Test accuracy | Test macro-F1 | Runtime (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 88 | 0.45452 | 81.3% | 81.57% | 943.538 |
| 7 | 49 | 0.39840 | 83.6% | 83.82% | 656.846 |
| 21 | 59 | 0.36439 | 85.3% | 85.55% | 749.837 |
| 42 | 76 | 0.36588 | 85.0% | 85.28% | 902.267 |
| 84 | 78 | 0.35656 | 85.7% | 85.97% | 934.705 |

Each checkpoint was selected independently by minimum validation loss. Test labels were
not used for checkpoint selection or hyperparameter tuning. Configuration and stored
seed values were verified from every archived run.

## Faculty-ready wording

> Our direction-aware, concatenated-head GAT reconstruction achieved 84.18% ± 1.79%
> test accuracy and 84.44% ± 1.80% macro-F1 across five seeds on the official
> MalNet-Tiny split. This is comparable to the paper's reported 84.6% accuracy. The best
> individual run reached 85.7%, but we use the five-seed mean as the primary result.

This is a close reconstruction, not a claim of exact replication, because the paper's
complete implementation and hyperparameter specification were not publicly available.
