"""scGPTWrapper — frozen scGPT encoder + trainable perturbation delta head.

Architecture
------------
scGPT gene embeddings (d_model=512, frozen) encode the identity of the
perturbed gene. A lightweight control-expression projector + gene embedding
are fused and decoded into a delta expression prediction.

Prediction = control_expr + delta_head(gene_emb || ctrl_proj)

This uses scGPT's pre-trained gene-level biological knowledge without
requiring full transformer inference at prediction time — M3 Air compatible.

Checkpoint download
-------------------
Run scripts/download_checkpoints.sh or:
  from huggingface_hub import snapshot_download
  snapshot_download('bowang-lab/scGPT_human', local_dir='data/checkpoints/scGPT_human')

Reference: Cui et al., Nature Methods 2024.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.base import PerturbationDataset, PerturbationModel
from models.baselines.mean_predictor import MeanPredictor

DEFAULT_CKPT = Path(__file__).parent.parent.parent / "data" / "checkpoints" / "scGPT_human"


# ---------------------------------------------------------------------------
# Delta head network
# ---------------------------------------------------------------------------

class _DeltaHead(nn.Module):
    """Maps (gene_emb ∥ ctrl_proj) → Δexpression.

    Inputs
    ------
    gene_emb  : scGPT token embedding for the perturbed gene, shape (d_model,)
    ctrl_proj : linear projection of control expression, shape (proj_dim,)

    Output
    ------
    delta : predicted change from control, shape (n_genes,)
    """

    def __init__(self, d_model: int, ctrl_proj_dim: int, n_genes: int) -> None:
        super().__init__()
        fused_dim = d_model + ctrl_proj_dim
        self.ctrl_proj = nn.Sequential(
            nn.Linear(n_genes, ctrl_proj_dim),
            nn.GELU(),
            nn.LayerNorm(ctrl_proj_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_genes),
        )

    def forward(self, gene_emb: torch.Tensor, ctrl_expr: torch.Tensor) -> torch.Tensor:
        ctrl_feat = self.ctrl_proj(ctrl_expr)
        fused = torch.cat([gene_emb, ctrl_feat], dim=-1)
        return self.decoder(fused)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class scGPTWrapper(PerturbationModel):
    """scGPT frozen gene embeddings + trainable delta head.

    Parameters
    ----------
    checkpoint_dir : path to scGPT_human dir (contains best_model.pt, vocab.json)
    device         : 'mps', 'cpu', or 'cuda'
    n_epochs       : fine-tuning epochs for delta head
    lr             : AdamW learning rate
    batch_size     : training batch size
    ctrl_proj_dim  : hidden dim for control-expression projector
    """

    def __init__(
        self,
        checkpoint_dir: str | Path = DEFAULT_CKPT,
        device: Optional[str] = None,
        n_epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 64,
        ctrl_proj_dim: int = 128,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.ctrl_proj_dim = ctrl_proj_dim

        self._delta_head: Optional[_DeltaHead] = None
        self._gene_emb_table: Optional[torch.Tensor] = None   # (vocab_size, d_model)
        self._vocab: dict[str, int] = {}                       # gene → token id
        self._our_gene_names: list[str] = []
        self._d_model: int = 512
        self._fallback = MeanPredictor()
        self._fitted = False

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def _load_checkpoint(self) -> bool:
        """Load vocab + gene embedding table from scGPT checkpoint.

        Returns True on success, False if checkpoint not found (graceful).
        """
        vocab_path = self.checkpoint_dir / "vocab.json"
        model_path = self.checkpoint_dir / "best_model.pt"
        args_path = self.checkpoint_dir / "args.json"

        if not (vocab_path.exists() and model_path.exists()):
            warnings.warn(
                f"scGPT checkpoint not found at {self.checkpoint_dir}.\n"
                "Run: python -c \"from huggingface_hub import snapshot_download; "
                "snapshot_download('bowang-lab/scGPT_human', "
                "local_dir='data/checkpoints/scGPT_human')\"\n"
                "Falling back to MeanPredictor.",
                RuntimeWarning,
                stacklevel=2,
            )
            return False

        # Load vocabulary
        with open(vocab_path) as f:
            self._vocab = json.load(f)

        # Detect d_model from args if present
        if args_path.exists():
            with open(args_path) as f:
                args = json.load(f)
            self._d_model = args.get("embsize", 512)

        # Load state dict and extract gene embedding table (token embeddings)
        state = torch.load(model_path, map_location="cpu")
        # scGPT stores the embedding table under 'encoder.embedding.weight'
        # or 'gene_encoder.embedding.weight' depending on checkpoint
        emb_key = None
        for k in state.keys():
            if "embedding" in k and "weight" in k and "encoder" in k.lower():
                emb_key = k
                break
        if emb_key is None:
            # Fallback: search for any embedding weight of right size
            for k, v in state.items():
                if "embedding" in k and len(v.shape) == 2 and v.shape[1] == self._d_model:
                    emb_key = k
                    break

        if emb_key is None:
            warnings.warn(
                "Could not locate gene embedding table in checkpoint. "
                "Falling back to random embeddings (d_model=512).",
                RuntimeWarning,
                stacklevel=2,
            )
            vocab_size = max(self._vocab.values()) + 1 if self._vocab else 30000
            self._gene_emb_table = torch.randn(vocab_size, self._d_model) * 0.02
        else:
            self._gene_emb_table = state[emb_key].float()
            print(f"Loaded scGPT gene embeddings: {self._gene_emb_table.shape} from '{emb_key}'")

        return True

    def _get_gene_embedding(self, gene_id: str) -> torch.Tensor:
        """Return scGPT embedding for a gene symbol, shape (d_model,)."""
        if gene_id in self._vocab and self._gene_emb_table is not None:
            idx = self._vocab[gene_id]
            return self._gene_emb_table[idx].to(self.device)
        # Unknown gene → zero embedding
        return torch.zeros(self._d_model, device=self.device)

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------

    def fit(self, dataset: PerturbationDataset) -> None:
        self._our_gene_names = dataset.gene_names
        n_genes = dataset.n_genes
        self._fallback.fit(dataset)

        ckpt_ok = self._load_checkpoint()
        if not ckpt_ok:
            self._fitted = False
            return

        # Build training tensors:
        # X_gene  : (n_train, d_model)  — frozen gene embeddings
        # X_ctrl  : (n_train, n_genes)  — control expression (same for all, but kept for generality)
        # Y_delta : (n_train, n_genes)  — ground-truth delta = perturbed - control
        ctrl = torch.tensor(dataset.control_expr, dtype=torch.float32)
        train_perts = [p for p in dataset.train_perts if p != dataset.control_key]

        gene_embs, ctrl_exprs, deltas = [], [], []
        for pert in train_perts:
            expr = torch.tensor(dataset.get_expr(pert), dtype=torch.float32)
            gene_emb = self._get_gene_embedding(pert).cpu()
            gene_embs.append(gene_emb)
            ctrl_exprs.append(ctrl)
            deltas.append(expr - ctrl)

        if not gene_embs:
            self._fitted = False
            return

        X_gene = torch.stack(gene_embs)   # (N, d_model)
        X_ctrl = torch.stack(ctrl_exprs)  # (N, n_genes)
        Y = torch.stack(deltas)           # (N, n_genes)

        # Build and train delta head (only trainable component)
        self._delta_head = _DeltaHead(
            d_model=self._d_model,
            ctrl_proj_dim=self.ctrl_proj_dim,
            n_genes=n_genes,
        ).to(self.device)

        loader = DataLoader(
            TensorDataset(X_gene, X_ctrl, Y),
            batch_size=self.batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.AdamW(self._delta_head.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs)

        self._delta_head.train()
        for epoch in range(self.n_epochs):
            total_loss = 0.0
            for xg, xc, y in loader:
                xg, xc, y = xg.to(self.device), xc.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                pred_delta = self._delta_head(xg, xc)
                loss = nn.functional.mse_loss(pred_delta, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f"  scGPT δ-head epoch {epoch+1}/{self.n_epochs}  loss={total_loss/len(loader):.4f}")

        self._delta_head.eval()
        self._fitted = True

    def predict(self, control_expr: np.ndarray, gene_id: str) -> np.ndarray:
        if not self._fitted or self._delta_head is None:
            return self._fallback.predict(control_expr, gene_id)

        ctrl_t = torch.tensor(control_expr, dtype=torch.float32, device=self.device).unsqueeze(0)
        gene_t = self._get_gene_embedding(gene_id).unsqueeze(0)

        with torch.no_grad():
            delta = self._delta_head(gene_t, ctrl_t).squeeze(0).cpu().numpy()

        pred = np.asarray(control_expr, dtype=np.float32) + delta
        return np.clip(pred, 0.0, None)

    @property
    def name(self) -> str:
        return "scGPT(frozen+δhead)"
