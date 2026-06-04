"""KNNPredictor — k-nearest-neighbors in PCA-reduced pseudo-bulk space.

Strategy:
  - Fit PCA on training perturbation profiles (delta vectors).
  - For each test perturbation gene, find the k closest training perturbations
    in that PCA space using cosine distance on the delta embedding.
  - For an unseen gene, fall back to the weighted average of the top-k
    most similar training perturbations (pure expression-profile similarity).
  - For a seen gene (in train split), return that gene's mean expression.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from models.base import PerturbationDataset, PerturbationModel


class KNNPredictor(PerturbationModel):
    """kNN in PCA-reduced delta space over training perturbations.

    Parameters
    ----------
    k          : number of nearest neighbours
    n_pca      : PCA components for the delta embedding
    metric     : sklearn NearestNeighbors metric
    """

    def __init__(self, k: int = 5, n_pca: int = 50, metric: str = "cosine") -> None:
        self.k = k
        self.n_pca = n_pca
        self.metric = metric

        self._train_exprs: dict[str, np.ndarray] = {}
        self._train_deltas: np.ndarray | None = None   # (n_train, n_pca)
        self._train_perts: list[str] = []
        self._control_mean: np.ndarray | None = None
        self._pca: PCA | None = None
        self._nn: NearestNeighbors | None = None

    def fit(self, dataset: PerturbationDataset) -> None:
        ctrl = dataset.control_expr
        self._control_mean = ctrl.copy()

        non_ctrl = [p for p in dataset.train_perts if p != dataset.control_key]
        for pert in non_ctrl:
            self._train_exprs[pert] = dataset.get_expr(pert).copy()

        if not non_ctrl:
            return

        # build delta matrix and PCA-embed it
        delta_matrix = np.stack(
            [self._train_exprs[p] - ctrl for p in non_ctrl], axis=0
        )  # (n_train, n_genes)

        n_components = min(self.n_pca, delta_matrix.shape[0] - 1, delta_matrix.shape[1])
        self._pca = PCA(n_components=n_components, random_state=42)
        emb = self._pca.fit_transform(delta_matrix)   # (n_train, n_pca)
        emb_normed = normalize(emb, norm="l2")

        self._train_perts = non_ctrl
        self._train_deltas = emb_normed

        n_neighbors = min(self.k, len(non_ctrl))
        self._nn = NearestNeighbors(n_neighbors=n_neighbors, metric=self.metric, algorithm="brute")
        self._nn.fit(emb_normed)

    def predict(self, control_expr: np.ndarray, gene_id: str) -> np.ndarray:
        ctrl = np.asarray(control_expr, dtype=np.float32)

        # seen perturbation → return memorised mean directly
        if gene_id in self._train_exprs:
            return self._train_exprs[gene_id].copy()

        # unseen → project control delta=0 direction and find neighbours
        if self._pca is None or self._nn is None:
            return (self._control_mean if self._control_mean is not None else ctrl).copy()

        # use zero-delta as query (no prior signal on direction)
        zero_delta = np.zeros((1, len(ctrl)), dtype=np.float64)
        query_emb = self._pca.transform(zero_delta)
        query_normed = normalize(query_emb, norm="l2")

        distances, indices = self._nn.kneighbors(query_normed)
        weights = 1.0 / (distances[0] + 1e-8)
        weights /= weights.sum()

        pred = np.zeros_like(ctrl)
        for w, idx in zip(weights, indices[0]):
            neighbour_pert = self._train_perts[idx]
            pred += w * self._train_exprs[neighbour_pert]
        return pred

    @property
    def name(self) -> str:
        return f"KNN(k={self.k},pca={self.n_pca})"
