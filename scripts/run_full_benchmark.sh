#!/usr/bin/env bash
# run_full_benchmark.sh — Day 1: download data, run baselines, save leaderboard
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Activate the torch-clean conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch-clean

echo "=== sc-virtualcell-bench: Day 1 Baseline Benchmark ==="
echo "Repo: $REPO_ROOT"
echo "Python: $(python --version)"
echo ""

# Install package in editable mode if not already installed
pip install -e . -q

# Run the baseline benchmark pipeline
python - <<'EOF'
import sys
sys.path.insert(0, ".")

from data.download import download_replogle_k562
from data.preprocess import preprocess, make_pseudobulk
from data.splits import build_dataset
from eval.harness import ModelEvaluator
from models.baselines.mean_predictor import MeanPredictor
from models.baselines.linear_additive import LinearAdditivePredictor

# --- Data ---
print("\n[1/4] Downloading data...")
adata_raw = download_replogle_k562()

print("\n[2/4] Preprocessing...")
adata = preprocess(adata_raw, n_hvg=2000)
pseudobulk, pert_col, control_key = make_pseudobulk(adata)

print("\n[3/4] Building dataset + splits...")
dataset = build_dataset(pseudobulk, control_key=control_key)

# Fit models on train split
mean_pred = MeanPredictor()
mean_pred.fit(dataset)

lin_add = LinearAdditivePredictor(use_deg_weights=True)
lin_add.fit(dataset)

lin_add_plain = LinearAdditivePredictor(use_deg_weights=False)
lin_add_plain.fit(dataset)

print("\n[4/4] Evaluating on test split...")
evaluator = ModelEvaluator(dataset, split="test")
models = {
    "MeanPredictor": mean_pred,
    "LinearAdditive+DEGweight": lin_add,
    "LinearAdditive": lin_add_plain,
}
leaderboard_df = evaluator.run_leaderboard(models)
evaluator.save_leaderboard(leaderboard_df)

from eval.leaderboard import generate_leaderboard
generate_leaderboard(save_plots=True)
EOF

echo ""
echo "Done. Results in results/leaderboard.csv and results/figures/"
