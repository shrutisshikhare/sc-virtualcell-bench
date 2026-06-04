#!/usr/bin/env bash
# download_checkpoints.sh — fetch model weights for foundation models
# Run once before Day 2 / Day 3 / Day 4 evaluation
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINTS_DIR="$REPO_ROOT/data/checkpoints"
mkdir -p "$CHECKPOINTS_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch-clean

echo "=== Downloading model checkpoints ==="

# --- scGPT whole-human checkpoint (Day 2) ---
SCGPT_DIR="$CHECKPOINTS_DIR/scGPT_human"
if [ ! -d "$SCGPT_DIR" ]; then
  echo "[scGPT] Downloading whole-human checkpoint from HuggingFace..."
  python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='bowang-lab/scGPT_human',
    local_dir='$SCGPT_DIR',
    ignore_patterns=['*.git*'],
)
print('scGPT checkpoint saved to $SCGPT_DIR')
"
else
  echo "[scGPT] Already downloaded: $SCGPT_DIR"
fi

# --- Arc STATE (Day 4) — check HuggingFace Hub ---
STATE_DIR="$CHECKPOINTS_DIR/STATE"
if [ ! -d "$STATE_DIR" ]; then
  echo "[STATE] Downloading Arc STATE checkpoint..."
  python -c "
from huggingface_hub import snapshot_download
# NOTE: update repo_id when Arc Institute publishes official checkpoint
try:
    snapshot_download(
        repo_id='arc-institute/STATE',
        local_dir='$STATE_DIR',
        ignore_patterns=['*.git*'],
    )
    print('STATE checkpoint saved to $STATE_DIR')
except Exception as e:
    print(f'STATE download failed: {e}')
    print('Check https://virtualcellchallenge.org for the official checkpoint release.')
"
else
  echo "[STATE] Already downloaded: $STATE_DIR"
fi

echo ""
echo "Checkpoint downloads complete."
echo "  scGPT: $SCGPT_DIR"
echo "  STATE: $STATE_DIR"
