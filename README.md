# sc-virtualcell-bench

**Rigorous benchmarking of single-cell perturbation foundation models against strong statistical baselines, aligned with the [Arc Institute Virtual Cell Challenge 2025](https://virtualcellchallenge.org/) evaluation standards.**

> *Does zero-shot scGPT actually beat a mean predictor on perturbation response?*

---

## Overview

This repository benchmarks leading single-cell perturbation models across **7 Cell-Eval metrics** (the same suite used in the 2025 VCC Generalist Prize) and introduces **LitHybrid** — a lightweight hybrid model trainable on a Mac M3 Air that combines ESM-2 protein embeddings with pseudo-bulk expression and cross-attention fusion.

### Models benchmarked

| Model | Type | Status |
|---|---|---|
| MeanPredictor | Statistical baseline | ✅ Day 1 |
| LinearAdditive (+DEG-freq weight) | Statistical baseline | ✅ Day 1 |
| KNN (PCA pseudo-bulk space) | Statistical baseline | ✅ Day 2 |
| GEARS | GO-graph GNN | ✅ Day 2 |
| scGPT (frozen encoder + δ-head) | Foundation model | ✅ Day 2 |
| scPRINT | Causal foundation model | 🔄 Day 3 |
| STATE (Arc Institute) | Set-level foundation model | 🔄 Day 4 |
| **LitHybrid** | **Novel hybrid** | 🔄 Day 5–6 |

### Evaluation metrics (Cell-Eval 7)

| Metric | Measures | Direction |
|---|---|---|
| PDS | Perturbation discrimination (L1-rank) | ↑ |
| DES | DEG set overlap (Jaccard top-50) | ↑ |
| MAE | Transcriptome-level error | ↓ |
| PDC | Pearson Δ correlation | ↑ |
| SLC | Spearman LFC rank correlation | ↑ |
| AUP | AUPRC for DEG detection | ↑ |
| SES | Spearman effect-size correlation | ↑ |

Models are ranked by **mean rank across all 7 metrics** (Arc Generalist Prize logic).

---

## Hypotheses being tested

**H1 — The Mean Predictor Problem:** Zero-shot foundation models (scGPT, Geneformer) will *not* consistently outperform mean-expression baseline on PDS and DES.

**H2 — Pseudo-bulk + Protein Embeddings as the Minimal Viable Hybrid:** LitHybrid (ESM-2 + pseudo-bulk + cross-attention) will outperform fine-tuned scGPT on PDS and DES while remaining trainable on M3 Air hardware.

**H3 — Multi-Metric Divergence:** No single model dominates all 7 metrics — architectural biases are only visible through the full suite.

---

## Repository structure

```
sc-virtual-cell-bench/
├── data/
│   ├── download.py          # fetch Replogle K562 + Norman 2019 via pertpy
│   ├── preprocess.py        # QC → log1p → HVG → pseudo-bulk aggregation
│   └── splits.py            # perturbation-level train/val/test splits
│
├── eval/
│   ├── metrics.py           # all 7 Cell-Eval metrics + mean_rank_score()
│   ├── harness.py           # ModelEvaluator class
│   └── leaderboard.py       # radar chart + ranked table
│
├── models/
│   ├── base.py              # PerturbationModel ABC + PerturbationDataset
│   ├── baselines/           # MeanPredictor, LinearAdditive, KNNPredictor
│   ├── foundation/          # GEARS, scGPT, scPRINT, STATE wrappers
│   └── lithybrid/           # LitHybrid architecture + training (Day 5)
│
├── notebooks/
│   ├── 02_baseline_benchmarks.ipynb
│   └── ...
│
└── scripts/
    ├── run_full_benchmark.sh
    └── download_checkpoints.sh
```

---

## Quick start

```bash
# 1. Create environment (Python 3.10)
conda env create -f environment.yml
conda activate sc-vcell

# Alternatively, use the provided torch-clean env if you have it:
# pip install -e .

# 2. Download data (auto-cached after first run)
python data/download.py

# 3. Run Day 1 baseline benchmark
bash scripts/run_full_benchmark.sh

# 4. Download foundation model checkpoints (for scGPT / STATE)
bash scripts/download_checkpoints.sh
```

---

## Datasets

- **Replogle K562 Essential** (Weissman Lab) — 1,080 CRISPRi perturbations, ~26k cells
- **Norman 2019** (Weissman Lab) — combinatorial CRISPRa, ~111k cells
- **Arc H1 hESC** (VCC 2025 challenge data) — 300 CRISPRi perturbations, ~300k cells *(Phase 2)*

All datasets fetched via `pertpy.data` — no manual download needed for Day 1–3.

---

## Hardware target

All training runs on **Mac M3 Air (8–16 GB unified memory)** using the MPS backend. Foundation models run in inference-only mode via pre-trained checkpoints. LitHybrid trains in < 1h on M3.

---

## References

- **Arc VCC 2025:** [virtualcellchallenge.org](https://virtualcellchallenge.org/)
- **CZI Virtual Cell Models:** [virtualcellmodels.cziscience.com](https://virtualcellmodels.cziscience.com/)
- **GEARS:** Roohani et al., *Nature Biotechnology* 2023
- **scGPT:** Cui et al., *Nature Methods* 2024
- **scPRINT:** Dalmia et al., 2024
- **ESM-2:** Lin et al., *Science* 2023
