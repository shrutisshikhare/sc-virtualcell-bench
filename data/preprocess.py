"""QC, normalisation, HVG selection, and pseudo-bulk aggregation."""
from __future__ import annotations

from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


# Column name patterns tried in order to locate the perturbation label
_PERT_COL_CANDIDATES = [
    "perturbation", "gene_target", "gene", "condition",
    "perturbation_name", "target_gene", "pert_name",
]

# Values that indicate a control/unperturbed cell
_CONTROL_ALIASES = {"control", "ctrl", "non-targeting", "non_targeting", "nt", "unperturbed"}


def _find_pert_col(adata: ad.AnnData) -> str:
    for col in _PERT_COL_CANDIDATES:
        if col in adata.obs.columns:
            return col
    raise KeyError(
        f"Cannot find perturbation column. Tried: {_PERT_COL_CANDIDATES}. "
        f"Available obs columns: {adata.obs.columns.tolist()}"
    )


def _find_control_key(series: pd.Series) -> str:
    vals = series.unique()
    for v in vals:
        if str(v).lower().strip() in _CONTROL_ALIASES:
            return str(v)
    raise KeyError(
        f"Cannot identify control label. Found values (sample): {vals[:10]}. "
        f"Expected one of: {_CONTROL_ALIASES}"
    )


def preprocess(
    adata: ad.AnnData,
    n_hvg: int = 2000,
    min_cells: int = 3,
    min_genes: int = 200,
    target_sum: float = 1e4,
    pert_col: Optional[str] = None,
) -> ad.AnnData:
    """Run standard QC + normalisation pipeline.

    Steps
    -----
    1. QC filter (min cells per gene, min genes per cell)
    2. normalize_total → 10k counts
    3. log1p
    4. Select top `n_hvg` highly variable genes (seurat_v3 flavor)

    Returns a new AnnData with .raw preserved and .X = HVG log-normalised counts.
    """
    adata = adata.copy()
    pert_col = pert_col or _find_pert_col(adata)

    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_cells(adata, min_genes=min_genes)

    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    adata.raw = adata  # preserve full normalised matrix

    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat_v3", subset=True)

    print(
        f"After preprocessing: {adata.n_obs} cells × {adata.n_vars} HVGs "
        f"(pert_col='{pert_col}')"
    )
    return adata


def make_pseudobulk(
    adata: ad.AnnData,
    pert_col: Optional[str] = None,
    min_cells_per_pert: int = 5,
) -> tuple[pd.DataFrame, str, str]:
    """Aggregate single-cell data to pseudo-bulk by averaging per perturbation.

    Parameters
    ----------
    adata               : preprocessed AnnData (.X = log1p HVG)
    pert_col            : obs column holding perturbation labels
    min_cells_per_pert  : drop perturbations with fewer cells than this

    Returns
    -------
    pseudobulk   : DataFrame (n_perturbations × n_genes), index = pert label
    pert_col     : column name used
    control_key  : the string label for control cells
    """
    pert_col = pert_col or _find_pert_col(adata)
    control_key = _find_control_key(adata.obs[pert_col])

    import scipy.sparse as sp
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    df = pd.DataFrame(X, columns=adata.var_names, index=adata.obs[pert_col].values)
    counts = df.index.value_counts()
    keep = counts[counts >= min_cells_per_pert].index
    df = df[df.index.isin(keep)]

    pseudobulk = df.groupby(df.index).mean()
    pseudobulk.index.name = "perturbation"

    n_perts = len(pseudobulk)
    print(
        f"Pseudo-bulk: {n_perts} perturbations × {pseudobulk.shape[1]} genes "
        f"(control='{control_key}', min_cells={min_cells_per_pert})"
    )
    return pseudobulk, pert_col, control_key
