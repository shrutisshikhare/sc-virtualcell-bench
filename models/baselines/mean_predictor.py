"""MeanPredictor — per-perturbation mean expression baseline.

The simplest possible baseline: predict the training-set mean expression
for the queried perturbation. Returns control mean for unseen perturbations.
This baseline often beats fine-tuned foundation models on MAE (the 'mean
predictor problem' from the VCC 2025 literature).
"""
from __future__ import annotations

import numpy as np
from models.base import PerturbationDataset, PerturbationModel


class MeanPredictor(PerturbationModel):
    """Stores the mean pseudo-bulk expression per perturbation from train set."""

    def __init__(self) -> None:
        self._means: dict[str, np.ndarray] = {}
        self._control_mean: np.ndarray | None = None

    def fit(self, dataset: PerturbationDataset) -> None:
        self._control_mean = dataset.control_expr.copy()
        for pert in dataset.train_perts:
            self._means[pert] = dataset.get_expr(pert).copy()

    def predict(self, control_expr: np.ndarray, gene_id: str) -> np.ndarray:
        if gene_id in self._means:
            return self._means[gene_id].copy()
        # fallback: return control (predicts no effect)
        return (self._control_mean if self._control_mean is not None
                else np.asarray(control_expr, dtype=np.float32)).copy()

    @property
    def name(self) -> str:
        return "MeanPredictor"
