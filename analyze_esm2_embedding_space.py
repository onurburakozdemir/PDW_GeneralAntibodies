#!/usr/bin/env python3
"""Visualize and sanity-check heavy-chain ESM2 embedding space for FLAb Tm dataset."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances


def load_seq2emb(cache_npz: Path, cache_map: Path) -> dict[str, np.ndarray]:
    npz = np.load(cache_npz, allow_pickle=True)
    key_to_seq = json.loads(cache_map.read_text())
    return {seq: np.asarray(npz[k], dtype=np.float32).ravel() for k, seq in key_to_seq.items()}


def build_table(df: pd.DataFrame, seq2emb: dict[str, np.ndarray]) -> tuple[np.ndarray, pd.DataFrame]:
    rows = []
    X = []
    for _, r in df.iterrows():
        v = seq2emb.get(str(r["heavy"]))
        if v is None:
            continue
        X.append(v)
        rows.append({"source_file": r["source_file"], "assay_name": r.get("assay_name", ""), "y": float(r["y"])})
    if not X:
        raise RuntimeError("No rows mapped to embeddings")
    return np.stack(X).astype(np.float32), pd.DataFrame(rows)


def same_study_knn_rate(X: np.ndarray, study: np.ndarray, k: int = 10) -> float:
    D = pairwise_distances(X, metric="euclidean")
    np.fill_diagonal(D, np.inf)
    nn = np.argpartition(D, kth=k, axis=1)[:, :k]
    hits = (study[nn] == study[:, None]).mean()
    return float(hits)


def knn_y_smoothness(X: np.ndarray, y: np.ndarray, k: int = 10) -> float:
    D = pairwise_distances(X, metric="euclidean")
    np.fill_diagonal(D, np.inf)
    nn = np.argpartition(D, kth=k, axis=1)[:, :k]
    y_hat = y[nn].mean(axis=1)
    corr = np.corrcoef(y, y_hat)[0, 1]
    return float(corr)


def main() -> None:
    data_csv = Path("flab_thermo_unified_ml_tm_only.csv")
    cache_npz = Path("output/esm_cache/esm2_space_cache_separate_hl.npz")
    cache_map = Path("output/esm_cache/esm2_space_cache_separate_hl_map.json")
    out_dir = Path("output/embedding_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_csv)
    df = df[["source_file", "assay_name", "heavy", "y"]].copy()
    df["y"] = pd.to_numeric(df["y"], errors="raise")

    seq2emb = load_seq2emb(cache_npz, cache_map)
    X, tbl = build_table(df, seq2emb)

    # Reduce dimensions with PCA first for denoising / speed.
    pca = PCA(n_components=min(50, X.shape[1], X.shape[0] - 1), random_state=42)
    Xp = pca.fit_transform(X)

    # 2D projections.
    pca2 = PCA(n_components=2, random_state=42).fit_transform(X)
    tsne2 = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42).fit_transform(Xp)

    # Quantitative checks.
    study = tbl["source_file"].to_numpy()
    y = tbl["y"].to_numpy()
    rate = same_study_knn_rate(Xp, study, k=10)
    corr = knn_y_smoothness(Xp, y, k=10)

    # Save metrics.
    pd.DataFrame(
        [{
            "n_rows": len(tbl),
            "n_studies": int(tbl["source_file"].nunique()),
            "same_study_knn10_rate": rate,
            "knn10_y_smoothness_corr": corr,
            "pca_explained_var_50": float(pca.explained_variance_ratio_.sum()),
        }]
    ).to_csv(out_dir / "embedding_diagnostic_metrics.csv", index=False)

    # Plot 1: PCA colored by y.
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(pca2[:, 0], pca2[:, 1], c=y, s=10, cmap="viridis", alpha=0.8)
    plt.colorbar(sc, label="Tm (y)")
    plt.title("ESM2 Heavy Embeddings: PCA (color=y)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_dir / "pca2_by_y.png", dpi=180)
    plt.close()

    # Plot 2: t-SNE colored by y.
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(tsne2[:, 0], tsne2[:, 1], c=y, s=10, cmap="viridis", alpha=0.8)
    plt.colorbar(sc, label="Tm (y)")
    plt.title("ESM2 Heavy Embeddings: t-SNE (color=y)")
    plt.xlabel("t-SNE1")
    plt.ylabel("t-SNE2")
    plt.tight_layout()
    plt.savefig(out_dir / "tsne2_by_y.png", dpi=180)
    plt.close()

    # Plot 3: t-SNE colored by study (top 8 studies + other).
    top = tbl["source_file"].value_counts().head(8).index.tolist()
    grp = tbl["source_file"].where(tbl["source_file"].isin(top), "other")
    labs = grp.astype("category")
    colors = labs.cat.codes.to_numpy()

    plt.figure(figsize=(6, 5))
    plt.scatter(tsne2[:, 0], tsne2[:, 1], c=colors, s=10, cmap="tab10", alpha=0.8)
    plt.title("ESM2 Heavy Embeddings: t-SNE (study groups)")
    plt.xlabel("t-SNE1")
    plt.ylabel("t-SNE2")
    plt.tight_layout()
    plt.savefig(out_dir / "tsne2_by_study_top8.png", dpi=180)
    plt.close()

    print("Saved diagnostics to", out_dir)
    print(f"rows={len(tbl)}, studies={tbl['source_file'].nunique()}")
    print(f"same-study kNN@10 rate={rate:.4f}")
    print(f"kNN@10 y-smoothness corr={corr:.4f}")


if __name__ == "__main__":
    main()
