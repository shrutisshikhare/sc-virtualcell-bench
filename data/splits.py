"""Perturbation-level train/val/test splits (no cell-level leakage)."""
from __future__ import annotations

import random
from typing import Optional

import numpy as np
import pandas as pd

from models.base import PerturbationDataset


def make_splits(
    pseudobulk: pd.DataFrame,
    control_key: str = "control",
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Split perturbation labels into train/val/test with no leakage.

    Control is always placed in train. Splits are by perturbation identity,
    not by cell, so the model never sees test perturbations during training.

    Returns
    -------
    (train_perts, val_perts, test_perts) — lists of perturbation label strings
    """
    assert abs(train + val + test - 1.0) < 1e-6, "Split fractions must sum to 1."

    all_perts = [p for p in pseudobulk.index.tolist() if p != control_key]
    rng = random.Random(seed)
    rng.shuffle(all_perts)

    n = len(all_perts)
    n_val = max(1, round(n * val))
    n_test = max(1, round(n * test))
    n_train = n - n_val - n_test

    train_perts = [control_key] + all_perts[:n_train]
    val_perts = all_perts[n_train: n_train + n_val]
    test_perts = all_perts[n_train + n_val:]

    print(
        f"Split: {len(train_perts)} train / {len(val_perts)} val / {len(test_perts)} test perturbations"
    )
    return train_perts, val_perts, test_perts


def make_combinatorial_splits(
    pseudobulk: pd.DataFrame,
    control_key: str = "control",
    sep: str = "+",
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Splits for combinatorial datasets (Norman-style).

    Single-gene perturbations → train/val.
    Combinatorial perturbations → test (OOD generalization evaluation).
    """
    all_perts = [p for p in pseudobulk.index.tolist() if p != control_key]
    single = [p for p in all_perts if sep not in p]
    combo = [p for p in all_perts if sep in p]

    rng = random.Random(seed)
    rng.shuffle(single)
    n_val = max(1, round(len(single) * 0.1))

    train_perts = [control_key] + single[n_val:]
    val_perts = single[:n_val]
    test_perts = combo

    print(
        f"Combinatorial split: {len(train_perts)} train (single-gene) / "
        f"{len(val_perts)} val / {len(test_perts)} test (combos)"
    )
    return train_perts, val_perts, test_perts


def build_dataset(
    pseudobulk: pd.DataFrame,
    control_key: str = "control",
    combinatorial: bool = False,
    seed: int = 42,
) -> PerturbationDataset:
    """One-shot: split + wrap in PerturbationDataset."""
    if combinatorial:
        train_perts, val_perts, test_perts = make_combinatorial_splits(
            pseudobulk, control_key=control_key, seed=seed
        )
    else:
        train_perts, val_perts, test_perts = make_splits(
            pseudobulk, control_key=control_key, seed=seed
        )
    return PerturbationDataset(
        pseudobulk=pseudobulk,
        gene_names=pseudobulk.columns.tolist(),
        control_key=control_key,
        train_perts=train_perts,
        val_perts=val_perts,
        test_perts=test_perts,
    )
