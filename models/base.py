from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PerturbationDataset:
    """Container for a pseudo-bulk perturbation dataset.

    Attributes
    ----------
    pseudobulk : DataFrame of shape (n_perturbations, n_genes)
    gene_names : list of gene symbols in column order
    control_key : perturbation label for the unperturbed control
    train_perts / val_perts / test_perts : perturbation-level split labels
    """
    pseudobulk: pd.DataFrame
    gene_names: list[str]
    control_key: str = "control"
    train_perts: list[str] = field(default_factory=list)
    val_perts: list[str] = field(default_factory=list)
    test_perts: list[str] = field(default_factory=list)

    @property
    def control_expr(self) -> np.ndarray:
        return self.pseudobulk.loc[self.control_key].values.astype(np.float32)

    def get_expr(self, perturbation: str) -> np.ndarray:
        return self.pseudobulk.loc[perturbation].values.astype(np.float32)

    @property
    def n_genes(self) -> int:
        return len(self.gene_names)


class PerturbationModel(ABC):
    """Interface every model in this benchmark must satisfy.

    Subclasses implement fit() and predict(). All models operate on
    pseudo-bulk expression vectors (float32, log1p-normalised, n_genes dims).
    """

    @abstractmethod
    def fit(self, dataset: PerturbationDataset) -> None:
        """Train on dataset.train_perts using dataset.pseudobulk."""
        ...

    @abstractmethod
    def predict(self, control_expr: np.ndarray, gene_id: str) -> np.ndarray:
        """Return predicted post-perturbation expression vector.

        Parameters
        ----------
        control_expr : mean control expression, shape (n_genes,)
        gene_id      : HGNC symbol of the perturbed gene

        Returns
        -------
        np.ndarray of shape (n_genes,), log1p-normalised expression
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
