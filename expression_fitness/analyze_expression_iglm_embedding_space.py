#!/usr/bin/env python3
"""Visualize IgLM embedding space for FLAb expression data (heavy+light)."""

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
    return {str(seq).strip().upper(): np.asarray(npz[k], dtype=np.float32).ravel() for k, seq in key_to_seq.items()}


def l2norm(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / max(n, eps)


def build_hl_features(df: pd.DataFrame, seq2emb: dict[str, np.ndarray]) -> tuple[np.ndarray, pd.DataFrame]:
    xs = []
    rows = []
    for _, r in df.iterrows():
        h = seq2emb.get(str(r["heavy"]).strip().upper())
        l = seq2emb.get(str(r["light"]).strip().upper())
        if h is None or l is None:
            continue
        x = np.concatenate([l2norm(h), l2norm(l)]).astype(np.float32)
        xs.append(x)
        rows.append({"source_file": r["source_file"], "y": float(r["y"])})

    if not xs:
        raise RuntimeError("No rows mapped to chain embeddings")
    return np.stack(xs), pd.DataFrame(rows)


def same_study_knn_rate(X: np.ndarray, study: np.ndarray, k: int = 10) -> float:
    D = pairwise_distances(X, metric="euclidean")
    np.fill_diagonal(D, np.inf)
    nn = np.argpartition(D, kth=k, axis=1)[:, :k]
    return float((study[nn] == study[:, None]).mean())


def knn_y_smoothness(X: np.ndarray, y: np.ndarray, k: int = 10) -> float:
    D = pairwise_distances(X, metric="euclidean")
    np.fill_diagonal(D, np.inf)
    nn = np.argpartition(D, kth=k, axis=1)[:, :k]
    y_hat = y[nn].mean(axis=1)
    return float(np.corrcoef(y, y_hat)[0, 1])


def main() -> None:
    data_csv = Path("expression_fitness/expression_unified_fitness.csv")
    cache_npz = Path("expression_fitness/iglm_chain_cache.npz")
    cache_map = Path("expression_fitness/iglm_chain_cache_map.json")
    out_dir = Path("expression_fitness/output/embedding_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (data_csv.exists() and cache_npz.exists() and cache_map.exists()):
        raise FileNotFoundError("Need expression unified CSV and chain cache first")

    df = pd.read_csv(data_csv)
    df = df[["source_file", "heavy", "light", "y"]].copy()
    df["y"] = pd.to_numeric(df["y"], errors="raise")

    seq2emb = load_seq2emb(cache_npz, cache_map)
    X, tbl = build_hl_features(df, seq2emb)

    pca = PCA(n_components=min(50, X.shape[1], X.shape[0] - 1), random_state=42)
    Xp = pca.fit_transform(X)

    pca2 = PCA(n_components=2, random_state=42).fit_transform(X)
    tsne2 = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42).fit_transform(Xp)

    study = tbl["source_file"].to_numpy()
    y = tbl["y"].to_numpy()
    study_knn = same_study_knn_rate(Xp, study, k=10)
    y_smooth = knn_y_smoothness(Xp, y, k=10)

    pd.DataFrame(
        [{
            "n_rows": len(tbl),
            "n_studies": int(tbl["source_file"].nunique()),
            "same_study_knn10_rate": study_knn,
            "knn10_y_smoothness_corr": y_smooth,
            "pca_explained_var_50": float(pca.explained_variance_ratio_.sum()),
        }]
    ).to_csv(out_dir / "embedding_diagnostic_metrics.csv", index=False)

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(pca2[:, 0], pca2[:, 1], c=y, s=6, cmap="viridis", alpha=0.7)
    plt.colorbar(sc, label="Expression fitness")
    plt.title("Expression IgLM(H+L): PCA (color=y)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_dir / "pca2_by_y.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(tsne2[:, 0], tsne2[:, 1], c=y, s=6, cmap="viridis", alpha=0.7)
    plt.colorbar(sc, label="Expression fitness")
    plt.title("Expression IgLM(H+L): t-SNE (color=y)")
    plt.xlabel("t-SNE1")
    plt.ylabel("t-SNE2")
    plt.tight_layout()
    plt.savefig(out_dir / "tsne2_by_y.png", dpi=180)
    plt.close()

    labs = tbl["source_file"].astype("category")
    colors = labs.cat.codes.to_numpy()

    plt.figure(figsize=(6, 5))
    plt.scatter(tsne2[:, 0], tsne2[:, 1], c=colors, s=6, cmap="tab10", alpha=0.7)
    plt.title("Expression IgLM(H+L): t-SNE (by study)")
    plt.xlabel("t-SNE1")
    plt.ylabel("t-SNE2")
    plt.tight_layout()
    plt.savefig(out_dir / "tsne2_by_study.png", dpi=180)
    plt.close()

    print("Saved diagnostics to", out_dir)
    print(f"rows={len(tbl)}, studies={tbl['source_file'].nunique()}")
    print(f"same-study kNN@10 rate={study_knn:.4f}")
    print(f"kNN@10 y-smoothness corr={y_smooth:.4f}")


if __name__ == "__main__":
    main()
