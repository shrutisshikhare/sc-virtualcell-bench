"""GEARSWrapper — wraps snap-stanford GEARS into the PerturbationModel interface.

GEARS (Graph-Enhanced Activation Regulation with Subgraph prediction) models
gene–gene interactions via a GO-graph GNN to predict perturbation outcomes.

Reference: Roohani et al., Nature Biotechnology 2023.
GitHub: https://github.com/snap-stanford/GEARS

Design notes
------------
- GEARS expects raw single-cell AnnData (not pseudo-bulk), so this wrapper
  accepts the preprocessed AnnData alongside the PerturbationDataset.
- GEARS uses its own internal 'simulation' split; we align its predictions
  to our dataset's test perturbations for evaluation.
- For genes not in GEARS's perturbation graph, we fall back to MeanPredictor.
- GEARS outputs are log1p-normalised, consistent with our harness.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from models.base import PerturbationDataset, PerturbationModel
from models.baselines.mean_predictor import MeanPredictor

GEARS_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "cache" / "gears"


def _check_gears_import() -> None:
    try:
        from gears import PertData, GEARS  # noqa: F401
    except ImportError:
        raise ImportError(
            "GEARS not installed. Run:\n"
            "  pip install git+https://github.com/snap-stanford/GEARS.git"
        )


def _prepare_adata_for_gears(adata, pert_col: str, control_key: str):
    """Add required obs/var columns for GEARS PertData.new_data_process."""
    import anndata as ad

    adata = adata.copy()
    # GEARS requires: condition, cell_type
    adata.obs["condition"] = adata.obs[pert_col].astype(str).copy()
    # normalise control label to 'ctrl'
    adata.obs["condition"] = adata.obs["condition"].replace({control_key: "ctrl"})
    adata.obs["cell_type"] = "K562"
    # GEARS requires gene_name in var
    adata.var["gene_name"] = adata.var_names.tolist()
    return adata


class GEARSWrapper(PerturbationModel):
    """GEARS graph perturbation model wrapped in the PerturbationModel interface.

    Parameters
    ----------
    adata_raw      : preprocessed AnnData (cell-level, log1p HVG); needed for
                     GEARS's internal graph construction and cell-level training.
    pert_col       : obs column holding perturbation labels in adata_raw.
    control_key    : obs value for unperturbed control cells.
    data_dir       : directory GEARS uses to cache its processed data + graphs.
    device         : 'cpu' (safe for M3 Air) or 'mps' (Apple Silicon).
    epochs         : training epochs (20 is GEARS default; ~1–2h on M3 CPU).
    hidden_size    : GNN hidden dimension (64 = default; reduce for speed).
    """

    def __init__(
        self,
        adata_raw,
        pert_col: str = "perturbation",
        control_key: str = "control",
        data_dir: str | Path = GEARS_DATA_DIR,
        device: Optional[str] = None,
        epochs: int = 20,
        hidden_size: int = 64,
    ) -> None:
        _check_gears_import()
        self.adata_raw = adata_raw
        self.pert_col = pert_col
        self.control_key = control_key
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            import torch
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.epochs = epochs
        self.hidden_size = hidden_size

        self._gears = None
        self._gears_gene_order: list[str] = []
        self._our_gene_order: list[str] = []
        self._reindex: Optional[np.ndarray] = None  # mapping from GEARS→our genes
        self._fallback = MeanPredictor()

    def fit(self, dataset: PerturbationDataset) -> None:
        from gears import PertData, GEARS

        self._our_gene_order = dataset.gene_names
        self._fallback.fit(dataset)

        # 1. Prepare AnnData in GEARS format
        adata_g = _prepare_adata_for_gears(self.adata_raw, self.pert_col, self.control_key)

        # 2. Build PertData and process
        pert_data = PertData(str(self.data_dir))
        dataset_name = "k562_hvg"
        pert_data.new_data_process(
            dataset_name=dataset_name,
            adata=adata_g,
            skip_calc_de=False,
        )
        pert_data.prepare_split(split="simulation", seed=1)
        pert_data.get_dataloader(batch_size=32, test_batch_size=128)

        # Store GEARS gene ordering for output reindexing
        self._gears_gene_order = list(pert_data.gene_names)
        self._build_reindex()

        # 3. Initialise + train GEARS
        gears_model = GEARS(pert_data, device=self.device)
        gears_model.model_initialize(hidden_size=self.hidden_size)
        print(f"Training GEARS for {self.epochs} epochs on {self.device}...")
        gears_model.train(epochs=self.epochs)
        self._gears = gears_model

    def _build_reindex(self) -> None:
        """Build index map: GEARS gene order → our dataset gene order."""
        our_set = {g: i for i, g in enumerate(self._our_gene_order)}
        gears_set = {g: i for i, g in enumerate(self._gears_gene_order)}
        # intersection only (both must have the gene)
        common = [g for g in self._gears_gene_order if g in our_set]
        if not common:
            raise ValueError(
                "No gene name overlap between GEARS and our dataset. "
                "Check that gene symbols match."
            )
        self._reindex = np.array([our_set[g] for g in common])
        self._gears_reindex = np.array([gears_set[g] for g in common])

    def predict(self, control_expr: np.ndarray, gene_id: str) -> np.ndarray:
        if self._gears is None:
            raise RuntimeError("Call fit() before predict().")

        # Check if gene is in GEARS perturbation graph
        if gene_id not in self._gears.pert_list:
            return self._fallback.predict(control_expr, gene_id)

        try:
            results = self._gears.predict([[gene_id]])
            gears_pred = results[gene_id]  # shape: (n_gears_genes,)
        except Exception:
            return self._fallback.predict(control_expr, gene_id)

        # Reindex GEARS output to our gene ordering
        control_expr = np.asarray(control_expr, dtype=np.float32)
        pred = control_expr.copy()  # start from control, fill in overlapping genes
        if self._reindex is not None:
            pred[self._reindex] = gears_pred[self._gears_reindex].astype(np.float32)
        return pred

    @property
    def name(self) -> str:
        return f"GEARS(h={self.hidden_size})"
