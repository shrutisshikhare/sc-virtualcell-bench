"""Download perturbation datasets via pertpy, caching locally as .h5ad."""
from __future__ import annotations

import os
from pathlib import Path

import anndata as ad

DEFAULT_CACHE = Path(__file__).parent / "cache"


def _cache_path(name: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{name}.h5ad"


def download_replogle_k562(cache_dir: str | Path = DEFAULT_CACHE) -> ad.AnnData:
    """Download Replogle 2022 K562 Essential (scPerturb standard benchmark).

    ~26k cells, ~8k genes, 1,080 single-gene CRISPRi perturbations.
    """
    cache_dir = Path(cache_dir)
    cached = _cache_path("replogle_k562_essential", cache_dir)
    if cached.exists():
        print(f"Loading from cache: {cached}")
        return ad.read_h5ad(cached)

    print("Downloading Replogle K562 Essential via pertpy...")
    import pertpy as pt
    adata = pt.data.replogle_2022_k562_essential()
    adata.write_h5ad(cached)
    print(f"Saved: {cached} | {adata.n_obs} cells × {adata.n_vars} genes")
    return adata


def download_norman_2019(cache_dir: str | Path = DEFAULT_CACHE) -> ad.AnnData:
    """Download Norman 2019 combinatorial CRISPRa K562 dataset.

    ~111k cells, combinatorial gene activation perturbations.
    """
    cache_dir = Path(cache_dir)
    cached = _cache_path("norman_2019", cache_dir)
    if cached.exists():
        print(f"Loading from cache: {cached}")
        return ad.read_h5ad(cached)

    print("Downloading Norman 2019 combinatorial perturbations via pertpy...")
    import pertpy as pt
    adata = pt.data.norman_2019()
    adata.write_h5ad(cached)
    print(f"Saved: {cached} | {adata.n_obs} cells × {adata.n_vars} genes")
    return adata


def list_cached(cache_dir: str | Path = DEFAULT_CACHE) -> list[str]:
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return []
    return [f.stem for f in cache_dir.glob("*.h5ad")]


if __name__ == "__main__":
    adata = download_replogle_k562()
    print(adata)
    print("Perturbation key sample:", adata.obs.columns.tolist()[:10])
