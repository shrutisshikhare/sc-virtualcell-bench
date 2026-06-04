"""LinearAdditivePredictor — mean delta + DEG-frequency weighting.

Inspired by the key statistical feature from VCC 2025 1st-place solution
(BioMap xTrimoSCPerturb): weight each gene's delta contribution by how
often that gene appears as a differentially expressed gene across all
training perturbations.

Prediction = control + weighted_delta[perturbation]
"""
from __future__ import annotations

import numpy as np
from models.base import PerturbationDataset, PerturbationModel


class LinearAdditivePredictor(PerturbationModel):
    """Control + per-perturbation delta, optionally DEG-frequency weighted."""

    def __init__(self, top_k_deg: int = 50, use_deg_weights: bool = True) -> None:
        self.top_k_deg = top_k_deg
        self.use_deg_weights = use_deg_weights
        self._deltas: dict[str, np.ndarray] = {}
        self._deg_weights: np.ndarray | None = None
        self._control_mean: np.ndarray | None = None

    def fit(self, dataset: PerturbationDataset) -> None:
        ctrl = dataset.control_expr
        self._control_mean = ctrl.copy()
        n_genes = dataset.n_genes

        # accumulate DEG frequency: how often each gene is in top-k by |delta|
        deg_freq = np.zeros(n_genes, dtype=np.float32)
        train_non_ctrl = [p for p in dataset.train_perts if p != dataset.control_key]

        for pert in train_non_ctrl:
            expr = dataset.get_expr(pert)
            delta = expr - ctrl
            top_k = min(self.top_k_deg, n_genes)
            top_idx = np.argsort(np.abs(delta))[-top_k:]
            deg_freq[top_idx] += 1.0
            self._deltas[pert] = delta.copy()

        if len(train_non_ctrl) > 0:
            deg_freq /= len(train_non_ctrl)  # normalise to [0, 1]
        self._deg_weights = deg_freq

    def predict(self, control_expr: np.ndarray, gene_id: str) -> np.ndarray:
        ctrl = np.asarray(control_expr, dtype=np.float32)
        if gene_id not in self._deltas:
            return ctrl.copy()

        delta = self._deltas[gene_id]
        if self.use_deg_weights and self._deg_weights is not None:
            # scale each gene's delta by its DEG frequency weight
            delta = delta * self._deg_weights

        return np.clip(ctrl + delta, 0.0, None)

    @property
    def name(self) -> str:
        suffix = "+DEGweight" if self.use_deg_weights else ""
        return f"LinearAdditive{suffix}"
