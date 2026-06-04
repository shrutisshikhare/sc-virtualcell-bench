"""Cell-Eval metrics aligned with the Arc Institute Virtual Cell Challenge 2025.

All functions accept float32 numpy arrays of shape (n_genes,).
Each returns a single scalar float.

Metric suite (7 total):
  PDS  — Perturbation Discrimination Score
  DES  — Differential Expression Score
  MAE  — Mean Absolute Error
  PDC  — Pearson Delta Correlation
  SLC  — Spearman LFC (log-fold-change rank correlation)
  AUP  — AUPRC for DEG detection
  SES  — Spearman Effect Size
"""
from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def perturbation_discrimination_score(
    pred: np.ndarray,
    true: np.ndarray,
    all_trues: np.ndarray,
) -> float:
    """PDS: fraction of other perturbations further (L1) from true than pred.

    Parameters
    ----------
    pred      : predicted expression for this perturbation, shape (n_genes,)
    true      : ground-truth expression for this perturbation, shape (n_genes,)
    all_trues : ground-truth matrix for ALL test perturbations, shape (n_perts, n_genes)
                Should include the current perturbation's row.

    Returns
    -------
    float in [0, 1]; higher is better.
    """
    pred = np.asarray(pred, dtype=np.float32)
    true = np.asarray(true, dtype=np.float32)
    all_trues = np.asarray(all_trues, dtype=np.float32)

    dist_pred = np.sum(np.abs(pred - true))
    dists_all = np.sum(np.abs(all_trues - true), axis=1)
    # fraction of rows where our prediction is closer than theirs
    return float(np.mean(dists_all > dist_pred))


def differential_expression_score(
    pred_delta: np.ndarray,
    true_delta: np.ndarray,
    top_k: int = 50,
) -> float:
    """DES: Jaccard overlap of top-k DEGs by absolute delta magnitude.

    Parameters
    ----------
    pred_delta : pred - control, shape (n_genes,)
    true_delta : true - control, shape (n_genes,)
    top_k      : number of top genes to compare

    Returns
    -------
    float in [0, 1]; higher is better.
    """
    pred_delta = np.asarray(pred_delta, dtype=np.float32)
    true_delta = np.asarray(true_delta, dtype=np.float32)
    top_k = min(top_k, len(pred_delta))

    pred_top = set(np.argsort(np.abs(pred_delta))[-top_k:])
    true_top = set(np.argsort(np.abs(true_delta))[-top_k:])
    intersection = len(pred_top & true_top)
    union = len(pred_top | true_top)
    return float(intersection / union) if union > 0 else 0.0


def mean_absolute_error(pred: np.ndarray, true: np.ndarray) -> float:
    """MAE over the full transcriptome. Lower is better."""
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(true))))


def pearson_delta_correlation(
    pred: np.ndarray,
    true: np.ndarray,
    control: np.ndarray,
) -> float:
    """Pearson r between predicted delta and true delta.

    delta = perturbed - control (in log1p space).
    Returns NaN if either delta is constant.
    """
    pred_delta = np.asarray(pred, dtype=np.float64) - np.asarray(control, dtype=np.float64)
    true_delta = np.asarray(true, dtype=np.float64) - np.asarray(control, dtype=np.float64)
    if np.std(pred_delta) < 1e-10 or np.std(true_delta) < 1e-10:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, _ = stats.pearsonr(pred_delta, true_delta)
    return float(r)


def spearman_lfc(
    pred: np.ndarray,
    true: np.ndarray,
    control: np.ndarray,
    pseudo: float = 1e-3,
) -> float:
    """Spearman rank correlation of log-fold-change vectors.

    LFC = log2((expr + pseudo) / (control + pseudo))
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    ctrl = np.asarray(control, dtype=np.float64)

    pred_lfc = np.log2((pred + pseudo) / (ctrl + pseudo))
    true_lfc = np.log2((true + pseudo) / (ctrl + pseudo))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, _ = stats.spearmanr(pred_lfc, true_lfc)
    return float(r)


def auprc_deg(
    pred_delta: np.ndarray,
    deg_labels: np.ndarray,
) -> float:
    """AUPRC for DEG detection: treats |pred_delta| as the score.

    Parameters
    ----------
    pred_delta : predicted delta from control, shape (n_genes,)
    deg_labels : binary array, 1 = differentially expressed gene, shape (n_genes,)

    Returns
    -------
    float in [0, 1]; higher is better.  Returns NaN if no positive labels.
    """
    pred_delta = np.asarray(pred_delta, dtype=np.float32)
    deg_labels = np.asarray(deg_labels, dtype=np.int32)
    if deg_labels.sum() == 0:
        return float("nan")
    scores = np.abs(pred_delta)
    return float(average_precision_score(deg_labels, scores))


def spearman_effect_size(
    pred_delta: np.ndarray,
    true_delta: np.ndarray,
) -> float:
    """Spearman r between per-gene effect magnitudes (|delta|).

    Captures whether predicted and true effect sizes are rank-correlated
    across genes, regardless of sign.
    """
    pred_mag = np.abs(np.asarray(pred_delta, dtype=np.float64))
    true_mag = np.abs(np.asarray(true_delta, dtype=np.float64))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, _ = stats.spearmanr(pred_mag, true_mag)
    return float(r)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def compute_all_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    control: np.ndarray,
    all_trues: Optional[np.ndarray] = None,
    deg_labels: Optional[np.ndarray] = None,
    top_k_des: int = 50,
) -> dict[str, float]:
    """Compute all 7 Cell-Eval metrics and return as a dict.

    Parameters
    ----------
    pred       : predicted expression, shape (n_genes,)
    true       : ground-truth expression, shape (n_genes,)
    control    : control (unperturbed) expression, shape (n_genes,)
    all_trues  : (n_perts, n_genes) matrix of all test perturbations, for PDS.
                 If None, PDS is computed as pred vs. true only (score = 0.5 sentinel).
    deg_labels : binary DEG labels, shape (n_genes,). If None, derived from
                 top-50 genes by |true_delta| magnitude.
    top_k_des  : number of top genes for DES Jaccard.
    """
    pred = np.asarray(pred, dtype=np.float32)
    true = np.asarray(true, dtype=np.float32)
    control = np.asarray(control, dtype=np.float32)
    pred_delta = pred - control
    true_delta = true - control

    if all_trues is None:
        pds = float("nan")
    else:
        pds = perturbation_discrimination_score(pred, true, all_trues)

    if deg_labels is None:
        top_k = min(top_k_des, len(true_delta))
        deg_labels = np.zeros(len(true_delta), dtype=np.int32)
        deg_labels[np.argsort(np.abs(true_delta))[-top_k:]] = 1

    return {
        "PDS":  pds,
        "DES":  differential_expression_score(pred_delta, true_delta, top_k=top_k_des),
        "MAE":  mean_absolute_error(pred, true),
        "PDC":  pearson_delta_correlation(pred, true, control),
        "SLC":  spearman_lfc(pred, true, control),
        "AUP":  auprc_deg(pred_delta, deg_labels),
        "SES":  spearman_effect_size(pred_delta, true_delta),
    }


# ---------------------------------------------------------------------------
# Arc Generalist Prize ranking logic
# ---------------------------------------------------------------------------

def mean_rank_score(leaderboard_df) -> "pd.Series":
    """Compute mean rank across all 7 metrics (lower = better generalist).

    Parameters
    ----------
    leaderboard_df : DataFrame with model names as index, metric columns as values.
                     MAE is ranked ascending; all others descending.

    Returns
    -------
    pd.Series of mean rank per model, sorted ascending.
    """
    import pandas as pd

    ascending_metrics = {"MAE"}
    ranks = pd.DataFrame(index=leaderboard_df.index)
    for col in leaderboard_df.columns:
        asc = col in ascending_metrics
        ranks[col] = leaderboard_df[col].rank(ascending=asc, na_option="bottom")
    return ranks.mean(axis=1).sort_values()
