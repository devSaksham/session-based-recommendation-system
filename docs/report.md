# Technical Report: Experimental GRU-KAN for Session-Based Recommendation

## 1. Introduction

We study next-item prediction in anonymous sessions using a GRU encoder plus an experimental KAN-style head. The goal is to test head-level nonlinear expressiveness without changing the recurrent backbone.

## 2. Background and Related Work

Session recommendation on YooChoose has historically compared methods such as POP/S-POP, Item-KNN, FPMC, GRU4Rec, NARM, STAMP, and SR-GNN. This repository does **not** reimplement those external baselines; it stores literature-reported values in structured form with citations.

## 3. Motivation for GRU-KAN

KAN-style transformations can represent nonlinear mappings via learnable piecewise functions. We use this idea only in the prediction head to keep:

- sequence modeling stable (`GRU`),
- ablations clean (`linear` vs `MLP` vs `KAN`),
- interpretation focused on representation-to-score mapping.

## 4. Architecture

Shared backbone:

1. item embedding
2. GRU session encoder
3. last valid hidden state extraction
4. head module
5. item-vocabulary scorer

Variants:

- `gru_linear`
- `gru_mlp`
- `gru_kan` (two piecewise-linear KAN layers with tanh + dropout)

## 5. Dataset and Preprocessing

Primary configuration: `configs/data/yoochoose_1_64.yaml`.

Pipeline details:

- clicks-only input
- deterministic session-id subsample with `fraction=1/64` (`session_id % 64 == 0`)
- iterative filtering (`min_session_length=2`, `min_item_support=5`)
- temporal split by session end-time (train / validation / test windows)
- drop unseen validation/test items
- prefix-target construction for next-item learning

### Protocol caveat

This is a deterministic raw-session approximation; it may not exactly match official packaged benchmark variants used in all papers.

## 6. Experiments

Planned ablations:

- `gru_linear`
- `gru_mlp`
- `gru_kan`

Reported metrics:

- Recall@20
- MRR@20
- parameter count
- runtime (aggregated per-epoch train+validation durations when available)

## 7. Evaluation Methodology

All-item ranking is computed (excluding padding id). Recall@20 measures hit presence in top-20, and MRR@20 is reciprocal-rank sensitive over hits.

## 8. Literature Comparison

Baseline rows in `src/research/literature_baselines.json` preserve:

- paper-reported metric naming (`P@20` kept where used)
- citation keys and URLs
- comparability notes

Comparisons against local ablations should be labeled **approximate** unless protocol equivalence is proven.

## 9. Current Results and Analysis

The training pipeline consumes `data/processed/yoochoose_1_64` directly when preprocessed artifacts are available in-repo. Raw ingestion is only required when regenerating processed splits.

What is complete:

- preprocessing, model, training, evaluation pipelines
- tests passing
- literature baseline assets and exports
- run/result auditing and summary tooling

## 10. Limitations

- Full protocol regeneration from raw data still requires local raw YooChoose files, even though training can run directly from committed preprocessed artifacts.
- `yoochoose_1_64` path is an approximation protocol.
- Literature-vs-local comparisons are not strict apples-to-apples.
- GRU-KAN remains exploratory.

## 11. Conclusion

The repository is positioned for reproducible completion from committed processed data: execute the three ablations, aggregate results with `scripts/report_results.py`, and report comparisons with explicit fairness caveats. Re-run preprocessing only when protocol regeneration is needed.

## 12. References

- Hidasi et al., *Session-based Recommendations with Recurrent Neural Networks*.
- Li et al., *Neural Attentive Session-based Recommendation*.
- Liu et al., *STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation*.
- Wu et al., *Session-Based Recommendation with Graph Neural Networks*.
- Liu et al., *KAN: Kolmogorov-Arnold Networks*.
