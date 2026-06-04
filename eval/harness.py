"""ModelEvaluator — runs any PerturbationModel against any PerturbationDataset
and returns the full 7-metric Cell-Eval scorecard.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from eval.metrics import compute_all_metrics, mean_rank_score
from models.base import PerturbationDataset, PerturbationModel

RESULTS_DIR = Path(__file__).parent.parent / "results"
LEADERBOARD_CSV = RESULTS_DIR / "leaderboard.csv"


class ModelEvaluator:
    """Evaluate one or many models on a PerturbationDataset.

    Usage
    -----
    evaluator = ModelEvaluator(dataset)
    scores = evaluator.evaluate(model)          # dict of 7 metrics
    df = evaluator.run_leaderboard({"Mean": m1, "LinAdd": m2})
    evaluator.save_leaderboard(df)
    """

    def __init__(self, dataset: PerturbationDataset, split: str = "test") -> None:
        self.dataset = dataset
        self.split = split
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def _get_perturbations(self) -> list[str]:
        if self.split == "test":
            return self.dataset.test_perts
        if self.split == "val":
            return self.dataset.val_perts
        return self.dataset.train_perts

    def evaluate(
        self,
        model: PerturbationModel,
        verbose: bool = True,
    ) -> dict[str, float]:
        """Evaluate model on the chosen split, return macro-averaged metrics."""
        perts = self._get_perturbations()
        if not perts:
            raise ValueError(f"No perturbations in split='{self.split}'. Check splits.")

        control = self.dataset.control_expr
        all_trues = np.stack(
            [self.dataset.get_expr(p) for p in perts], axis=0
        )

        per_pert: list[dict[str, float]] = []
        for pert in tqdm(perts, desc=f"Evaluating {model.name}", disable=not verbose):
            true = self.dataset.get_expr(pert)
            pred = model.predict(control, pert)
            metrics = compute_all_metrics(
                pred=pred,
                true=true,
                control=control,
                all_trues=all_trues,
            )
            per_pert.append(metrics)

        df = pd.DataFrame(per_pert)
        return df.mean(skipna=True).to_dict()

    def run_leaderboard(
        self,
        models: dict[str, PerturbationModel],
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Evaluate multiple models, return DataFrame indexed by model name."""
        rows: dict[str, dict[str, float]] = {}
        for name, model in models.items():
            print(f"\n=== {name} ===")
            rows[name] = self.evaluate(model, verbose=verbose)
        df = pd.DataFrame(rows).T
        df.index.name = "model"
        df["mean_rank"] = mean_rank_score(df[["PDS", "DES", "MAE", "PDC", "SLC", "AUP", "SES"]])
        return df.sort_values("mean_rank")

    def save_leaderboard(
        self,
        df: pd.DataFrame,
        path: Optional[Path] = None,
        append: bool = True,
    ) -> Path:
        """Write leaderboard DataFrame to CSV, optionally merging with existing."""
        path = Path(path) if path else LEADERBOARD_CSV
        if append and path.exists():
            existing = pd.read_csv(path, index_col="model")
            df = pd.concat([existing, df[~df.index.isin(existing.index)]])
        df.to_csv(path, index=True)
        print(f"Leaderboard saved → {path}")
        return path
