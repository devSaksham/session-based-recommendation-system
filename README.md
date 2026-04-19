# Session-Based Recommendation System

This repository contains a reproducible PyTorch research codebase for session-based next-item recommendation on YooChoose, centered on an experimental hybrid **GRU-KAN** architecture. The main design keeps the sequential modeling inductive bias in a GRU encoder and replaces the usual dense prediction head with a KAN-style nonlinear transformation block.

## Why This Project

Session-based recommendation predicts the next item a user is likely to interact with when long-term user profiles are unavailable or intentionally ignored. This setting is central to anonymous browsing, cold-start traffic, and short-lived e-commerce sessions. Recurrent models such as GRU4Rec helped establish strong neural baselines in this regime, while later models like NARM, STAMP, and SR-GNN improved session representation learning in different ways.

This repository explores whether a **KAN-based head** can act as a richer nonlinear mapping from recurrent session states to next-item scores without replacing the recurrent backbone itself.

## Current Scope

- Primary protocol: clicks-only YooChoose preprocessing
- Primary dataset variant: deterministic `yoochoose_1_64` approximation from raw clicks
- Model family:
  - `gru_linear`
  - `gru_mlp`
  - `gru_kan`
- Metrics:
  - `Recall@20`
  - `MRR@20`
- Literature comparison:
  - baseline results collected from published papers, not reimplemented here
- Secondary study:
  - optional clicks+buys exploratory preprocessing config is included but intentionally kept separate from the main comparison path

## Repository Layout

```text
session-based-recommendation-system/
  configs/
  data/
  docs/
  notebooks/
  results/
  scripts/
  src/
  tests/
  README.md
  requirements.txt
```

## Preprocessing Protocol

The main preprocessing pipeline:

1. loads YooChoose clicks
2. optionally applies deterministic session-level downsampling for the `1/64` variant
3. sorts by `session_id` and timestamp
4. iteratively filters short sessions and low-support items
5. builds temporal train/validation/test splits using the last 24 hours for test and the previous 24 hours for validation
6. removes validation/test items unseen in training
7. converts sessions into prefix-target examples for next-item prediction

Processed outputs include:

- event-level parquet files
- example-level pickle files
- item encoder / decoder files
- metadata JSON

## Model Design

The shared architecture is:

1. `Embedding(num_items + 1, embedding_dim, padding_idx=0)`
2. `GRU(embedding_dim -> hidden_dim, batch_first=True)`
3. last-valid hidden-state extraction with sequence lengths
4. head module:
   - linear head for `gru_linear`
   - MLP head for `gru_mlp`
   - KAN-style piecewise linear head for `gru_kan`
5. output scorer over the item vocabulary

The KAN integration is intentionally localized to the head so that:

- the recurrent encoder remains standard and interpretable
- the experimental contribution is isolated
- ablations remain fair within the proposed model family

## Reproducible Setup

This workspace did not expose system `python` on `PATH`, so the implementation was validated with a local virtual environment:

```powershell
C:\Users\NS\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Commands

Preprocess the primary benchmark variant:

```powershell
$env:PYTHONPATH='.'
.\.venv\Scripts\python.exe scripts\preprocess.py --config configs\data\yoochoose_1_64.yaml
```

Preprocess the debug variant:

```powershell
$env:PYTHONPATH='.'
.\.venv\Scripts\python.exe scripts\preprocess.py --config configs\data\yoochoose_full_clicks_debug.yaml
```

Train a model:

```powershell
$env:PYTHONPATH='.'
.\.venv\Scripts\python.exe scripts\train.py --data-config configs\data\yoochoose_1_64.yaml --model-config configs\model\gru_kan.yaml
```

Evaluate a checkpoint:

```powershell
$env:PYTHONPATH='.'
.\.venv\Scripts\python.exe scripts\evaluate.py --checkpoint <checkpoint_path> --data-config configs\data\yoochoose_1_64.yaml --model-config configs\model\gru_kan.yaml
```

Export literature baselines:

```powershell
$env:PYTHONPATH='.'
.\.venv\Scripts\python.exe scripts\collect_baselines.py --output results\literature_baselines.csv
```

Assemble model result tables from finished runs:

```powershell
$env:PYTHONPATH='.'
.\.venv\Scripts\python.exe scripts\report_results.py
```

Run tests:

```powershell
$env:PYTHONPATH='.'
.\.venv\Scripts\python.exe -m pytest tests -q
```

## Literature Comparison Methodology

The repository stores literature baselines in `src/research/literature_baselines.json` and exports them to CSV/JSON/Markdown. Reported values are kept citation-aware and protocol-aware:

- the original metric name is preserved when papers use `P@20` rather than `Recall@20`
- source paper and citation key are stored for every row
- comparisons are explicitly treated as approximate when the local preprocessing differs from the original benchmark package

## Current Status

Completed:

- clean repository scaffold
- preprocessing pipeline
- dataset loaders and collator
- metrics
- GRU + Linear / MLP / KAN model family
- training and evaluation CLI
- literature baseline asset and export script
- tests passing locally

Partially completed:

- benchmark training runs were started, and checkpoints exist for an interrupted `gru_linear` run
- polished experiment tables and the final narrative report still need completion after the full ablation runs finish

## Limitations

- The current `yoochoose_1_64` path is a deterministic raw-session subsample, not the official benchmark package used in every paper table. It is useful for controlled local experimentation but must be labeled carefully in comparisons.
- Full ablation runs on CPU are slow; the current defaults are intentionally conservative.
- The clicks+buys path is exploratory and should not be mixed into the main literature table without separate protocol discussion.

## References

- Balázs Hidasi et al. Session-based Recommendations with Recurrent Neural Networks.
- Jing Li et al. Neural Attentive Session-based Recommendation.
- Qiao Liu et al. STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation.
- Shu Wu et al. Session-Based Recommendation with Graph Neural Networks.
- Ziming Liu et al. KAN: Kolmogorov-Arnold Networks.
