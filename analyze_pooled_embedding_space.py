#!/usr/bin/env python3
"""Visualize and diagnose pooled ESM2+IgLM heavy-chain embedding space."""

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


def build_pooled_map(esm: dict[str, np.ndarray], iglm: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    common = set(esm.keys()) & set(iglm.keys())
    pooled: dict[str, np.ndarray] = {}
    for s in common:
        pooled[s] = np.concatenate([l2norm(esm[s]), l2norm(iglm[s])]).astype(np.float32)
    if not pooled:
        raise RuntimeError("No overlapping sequences between ESM2 and IgLM caches")
    return pooled


def build_table(df: pd.DataFrame, seq2emb: dict[str, np.ndarray]) -> tuple[np.ndarray, pd.DataFrame]:
    rows = []
    X = []
    for _, r in df.iterrows():
        seq = str(r["heavy"]).strip().upper()
        v = seq2emb.get(seq)
        if v is None:
            continue
        X.append(v)
        rows.append({"source_file": r["source_file"], "assay_name": r.get("assay_name", ""), "y": float(r["y"])})
    if not X:
        raise RuntimeError("No rows mapped to pooled embeddings")
    return np.stack(X).astype(np.float32), pd.DataFrame(rows)


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
    data_csv = Path("flab_thermo_unified_ml_tm_only.csv")
    esm_npz = Path("output/esm_cache/esm2_space_cache_separate_hl.npz")
    esm_map = Path("output/esm_cache/esm2_space_cache_separate_hl_map.json")
    iglm_npz = Path("output/iglm_cache/iglm_heavy_meanpool.npz")
    iglm_map = Path("output/iglm_cache/iglm_heavy_meanpool_map.json")

    out_dir = Path("output/embedding_diagnostics_pooled")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (esm_npz.exists() and esm_map.exists() and iglm_npz.exists() and iglm_map.exists()):
        raise FileNotFoundError("Need both ESM2 and IgLM caches to build pooled embeddings")

    df = pd.read_csv(data_csv)
    df = df[["source_file", "assay_name", "heavy", "y"]].copy()
    df["y"] = pd.to_numeric(df["y"], errors="raise")

    esm = load_seq2emb(esm_npz, esm_map)
    iglm = load_seq2emb(iglm_npz, iglm_map)
    pooled = build_pooled_map(esm, iglm)

    X, tbl = build_table(df, pooled)

    pca = PCA(n_components=min(50, X.shape[1], X.shape[0] - 1), random_state=42)
    Xp = pca.fit_transform(X)

    pca2 = PCA(n_components=2, random_state=42).fit_transform(X)
    tsne2 = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42).fit_transform(Xp)

    study = tbl["source_file"].to_numpy()
    y = tbl["y"].to_numpy()
    rate = same_study_knn_rate(Xp, study, k=10)
    corr = knn_y_smoothness(Xp, y, k=10)

    pd.DataFrame([
        {
            "n_rows": len(tbl),
            "n_studies": int(tbl["source_file"].nunique()),
            "same_study_knn10_rate": rate,
            "knn10_y_smoothness_corr": corr,
            "pca_explained_var_50": float(pca.explained_variance_ratio_.sum()),
        }
    ]).to_csv(out_dir / "embedding_diagnostic_metrics.csv", index=False)

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(pca2[:, 0], pca2[:, 1], c=y, s=10, cmap="viridis", alpha=0.8)
    plt.colorbar(sc, label="Tm (y)")
    plt.title("Pooled ESM2+IgLM: PCA (color=y)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_dir / "pca2_by_y.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(tsne2[:, 0], tsne2[:, 1], c=y, s=10, cmap="viridis", alpha=0.8)
    plt.colorbar(sc, label="Tm (y)")
    plt.title("Pooled ESM2+IgLM: t-SNE (color=y)")
    plt.xlabel("t-SNE1")
    plt.ylabel("t-SNE2")
    plt.tight_layout()
    plt.savefig(out_dir / "tsne2_by_y.png", dpi=180)
    plt.close()

    top = tbl["source_file"].value_counts().head(8).index.tolist()
    grp = tbl["source_file"].where(tbl["source_file"].isin(top), "other")
    labs = grp.astype("category")
    colors = labs.cat.codes.to_numpy()

    plt.figure(figsize=(6, 5))
    plt.scatter(tsne2[:, 0], tsne2[:, 1], c=colors, s=10, cmap="tab10", alpha=0.8)
    plt.title("Pooled ESM2+IgLM: t-SNE (study groups)")
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
