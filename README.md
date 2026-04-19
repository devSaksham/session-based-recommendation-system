# Session-Based Recommendation with Experimental GRU-KAN Head

This repository is a research-oriented PyTorch codebase for **session-based next-item recommendation** on YooChoose. It studies an experimental architectural variant where a standard GRU session encoder is paired with a KAN-style prediction head.

## Problem Statement

Session-based recommendation predicts the next item in a short anonymous interaction sequence. This matters in practical settings where persistent user identity is absent or unreliable (e.g., guest users, cold-start traffic, privacy-constrained environments).

## Why Explore GRU-KAN?

The intent is not to replace the recurrent backbone, but to test whether a richer nonlinear head improves mapping from session representation to item scores.

- `gru_linear`: minimal projection head baseline.
- `gru_mlp`: nonlinear dense baseline head.
- `gru_kan`: experimental piecewise-linear KAN-style head.

This keeps the experiment focused and allows fair in-family ablations.

## Dataset and Preprocessing

Primary protocol uses YooChoose clicks only with a deterministic session-id subsample for `yoochoose_1_64`:

1. read clicks
2. keep sessions where `session_id % 64 == 0` (deterministic approximation)
3. iterative filtering (`min_session_length=2`, `min_item_support=5`)
4. temporal split by session end-time:
   - train: older than last 48h
   - validation: next 24h window
   - test: final 24h window
5. drop validation/test items unseen in train
6. convert sessions into prefix-target examples

> **Important comparability caveat:** this `yoochoose_1_64` path is a deterministic raw-session subsample approximation, not guaranteed to be identical to every published benchmark packaging. Literature comparisons are therefore approximate.

## Reproducible Commands

Set environment and run from repository root:

```bash
export PYTHONPATH=.
```

Download raw dataset from Google Drive and place files under `data/raw/`:

```bash
python scripts/fetch_dataset.py --drive-url "https://drive.google.com/file/d/1c5s1ugm-6-xJvLpj5_ibjVkNsGRhJGzO/view?usp=drive_link"
```

Preprocess:

```bash
python scripts/preprocess.py --config configs/data/yoochoose_1_64.yaml
```

Train ablations:

```bash
python scripts/train.py --data-config configs/data/yoochoose_1_64.yaml --model-config configs/model/gru_linear.yaml
python scripts/train.py --data-config configs/data/yoochoose_1_64.yaml --model-config configs/model/gru_mlp.yaml
python scripts/train.py --data-config configs/data/yoochoose_1_64.yaml --model-config configs/model/gru_kan.yaml
```

Evaluate checkpoint:

```bash
python scripts/evaluate.py --checkpoint <checkpoint.pt> --data-config configs/data/yoochoose_1_64.yaml --model-config configs/model/gru_kan.yaml
```

Generate model result summaries (Markdown + CSV + JSON):

```bash
python scripts/report_results.py --runs-root results/runs --output-prefix results/generated/model_results
```

Audit processed data, runs, and literature baseline integrity:

```bash
python scripts/audit_research_state.py --check-urls --output results/generated/research_state_audit.json
```

Export literature baselines:

```bash
python scripts/collect_baselines.py --output results/literature_baselines.csv
```

Run tests:

```bash
PYTHONPATH=. pytest -q
```

## Measured Results (Current Workspace State)

This repository now includes tooling to aggregate finished runs, but this specific environment currently has no local YooChoose raw data at `data/raw` and no completed run artifacts under `results/runs`. Therefore no new model metrics are claimed here.

## Literature Comparison Methodology

Baselines are stored in `src/research/literature_baselines.json` and exported to CSV/JSON/Markdown. Protocol safeguards:

- retain original metric naming from source papers (e.g., `P@20` vs `Recall@20`)
- retain citation key and source URL per row
- annotate `comparability` and protocol notes
- treat local-vs-paper comparison as approximate unless split/package equivalence is established

## Limitations

- No claim of exact canonical YooChoose benchmark reproduction.
- CPU-only environments may make full ablations slow.
- Literature comparison is citation-aware but not fully apples-to-apples without exact benchmark alignment.
- GRU-KAN remains experimental.

## Future Work

- Reproduce official benchmark packaging more exactly and document divergence quantitatively.
- Run multi-seed ablations and report confidence intervals.
- Add rank-cutoff sweeps (`@10`, `@20`, `@50`) and calibration diagnostics.
- Profile head-level compute cost vs. gains.
