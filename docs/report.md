# Technical Report

## Introduction

This project studies session-based next-item recommendation on YooChoose using an experimental hybrid architecture that combines a recurrent session encoder with a KAN-style prediction head. The motivation is to preserve the strong sequential inductive bias of GRU-based session models while exploring a richer nonlinear representation-to-score mapping than a standard linear or MLP head.

## Background and Related Work

Session-based recommendation focuses on predicting the next interaction within a short anonymous session, rather than relying on long-term user profiles. This problem became a major benchmark area through work such as GRU4Rec, which demonstrated that recurrent models can outperform classical nearest-neighbor and factorization baselines in session settings.

Representative benchmark models on YooChoose include:

- POP and S-POP
- Item-KNN
- BPR-MF
- FPMC
- GRU4Rec
- NARM
- STAMP
- SR-GNN

The repository collects literature-reported values for these models from published papers and stores them as structured metadata rather than reimplementing the baselines locally.

## Why GRU-KAN

Direct prior work on applying Kolmogorov-Arnold Networks to session-based recommendation is limited, so the proposed architecture is explicitly framed as an experimental hybrid. The KAN component is used as a **head module after the recurrent session representation**, not as a replacement for the recurrent cell. This design is technically motivated because it:

- keeps the sequence modeling component conventional and interpretable
- isolates the experimental contribution
- enables clean internal ablations against linear and MLP heads

The implemented KAN head uses learnable piecewise linear edge functions as a lightweight approximation of the broader KAN idea.

## Dataset and Preprocessing

### Main protocol

- Dataset: YooChoose clicks
- Variant: deterministic `yoochoose_1_64` session subsample
- Filtering:
  - remove sessions shorter than length 2
  - iteratively remove low-support items with minimum support 5
  - remove newly short sessions again
- Split:
  - training: sessions ending before the last 48 hours
  - validation: sessions ending in the 24-hour window before the final day
  - test: sessions ending in the final 24-hour window
- Evaluation examples:
  - convert each session into prefix-target pairs for next-item prediction

### Current benchmark metadata

The most recent completed preprocessing summary for `yoochoose_1_64` produced:

- raw sampled events: 512,066
- train events: 463,891
- validation events: 825
- test events: 1,033
- train examples: 344,566
- validation examples: 654
- test examples: 791
- vocabulary size: 10,629

These statistics should be interpreted carefully because the current variant is a deterministic raw-session subsample and not a packaged official benchmark release.

## Model Architecture

The implemented family shares the following backbone:

1. item embedding layer
2. GRU encoder
3. masked extraction of the last valid hidden state
4. head module
5. output scorer over the full item vocabulary

The three implemented variants are:

- `gru_linear`
- `gru_mlp`
- `gru_kan`

The KAN head uses two piecewise-linear layers with learnable basis values over a fixed grid and a tanh nonlinearity between them.

## Training Setup

- Framework: PyTorch
- Loss: cross-entropy for next-item classification
- Optimizer: AdamW
- Early stopping target: validation `MRR@20`
- Metrics: `Recall@20`, `MRR@20`
- Determinism:
  - random seeds set for Python, NumPy, and PyTorch
  - optional deterministic CuDNN behavior when CUDA is available
- Runtime logging:
  - resolved config
  - runtime environment metadata
  - training summary
  - checkpoints

## Evaluation Methodology

At evaluation time, the model ranks all candidate items except padding. `Recall@20` measures whether the true next item appears in the top 20. `MRR@20` reflects rank sensitivity by averaging reciprocal ranks for correct hits within the top 20.

## Literature Baselines

The repository stores literature rows in `src/research/literature_baselines.json`. Each row records:

- model name
- paper title
- authors
- venue and year
- dataset variant
- metric values
- citation key
- source URL
- protocol note
- comparability note

Important fairness note: many papers report `P@20` for YooChoose tables, while later work often discusses comparable values as `Recall@20`. This repository preserves the original metric naming and annotates any equivalence claims rather than silently renaming them.

## Current Results Status

The codebase, tests, preprocessing, and literature assets are in place. A long `gru_linear` benchmark run was started and produced checkpoints before interruption, but the full ablation suite and final comparison tables still need to be completed.

Because of that, this report deliberately does **not** claim final model performance yet. The intended next step is to finish the three ablation runs, evaluate the saved checkpoints, export the model-results table, and then compare those measured values against the literature baseline file with explicit protocol caveats.

## Limitations

- The current `yoochoose_1_64` variant is an approximation derived from raw clicks rather than the exact benchmark package used by all papers.
- CPU-only execution makes full ablation sweeps time-consuming.
- Literature tables are useful for orientation, but not perfectly apples-to-apples without exact split reproduction.
- The clicks+buys configuration is exploratory and intentionally separated from the primary comparison workflow.

## Conclusion

This repository already provides a solid, reproducible foundation for an academically honest GRU-KAN session-recommendation study:

- preprocessing is implemented
- the model family is implemented
- evaluation metrics are implemented and tested
- literature baselines are structured and citation-aware

The remaining work is experimental completion and final write-up polish rather than architectural bootstrapping.

## References

- Hidasi, B., Karatzoglou, A., Baltrunas, L., and Tikk, D. Session-based Recommendations with Recurrent Neural Networks.
- Li, J., Ren, P., Chen, Z., Ren, Z., Ma, J., and de Rijke, M. Neural Attentive Session-based Recommendation.
- Liu, Q., Zeng, Y., Mokhosi, R., and Zhang, H. STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation.
- Wu, S., Tang, Y., Zhu, Y., Wang, L., Xie, X., and Tan, T. Session-Based Recommendation with Graph Neural Networks.
- Liu, Z., Wang, Y., Vaidya, S., et al. KAN: Kolmogorov-Arnold Networks.
