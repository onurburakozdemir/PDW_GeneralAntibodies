#!/usr/bin/env python3
"""Train a heavy-chain-only ESM2+MLP model for FLAb thermostability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_csv", type=Path, default=Path("flab_thermo_unified_ml_tm_only.csv"))
    p.add_argument("--cache_npz", type=Path, default=Path("output/esm_cache/esm2_space_cache_separate_hl.npz"))
    p.add_argument("--cache_map", type=Path, default=Path("output/esm_cache/esm2_space_cache_separate_hl_map.json"))
    p.add_argument("--holdout_experiment", type=str, default="rosace2023automated_tm1_golimumab.csv")
    p.add_argument("--train_frac", type=float, default=0.80)
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--test_frac", type=float, default=0.10)
    p.add_argument("--save_plots", action="store_true")
    p.add_argument("--out_dir", type=Path, default=Path("output/esm_results_heavy_only"))
    return p.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"source_file", "heavy", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.copy()
    df["y"] = pd.to_numeric(df["y"], errors="raise")
    df["heavy"] = df["heavy"].astype(str)
    return df


def load_embedding_cache(cache_npz: Path, cache_map: Path) -> dict[str, np.ndarray]:
    if not cache_npz.exists() or not cache_map.exists():
        raise FileNotFoundError(f"Missing cache files: {cache_npz} / {cache_map}")
    npz = np.load(cache_npz, allow_pickle=True)
    key_to_seq = json.loads(cache_map.read_text())
    return {seq: np.asarray(npz[k], dtype=np.float32) for k, seq in key_to_seq.items()}


def build_heavy_features(df: pd.DataFrame, seq2emb: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rows, ys, idxs = [], [], []
    for i, r in df.iterrows():
        h = r["heavy"]
        v = seq2emb.get(h)
        if v is None:
            continue
        rows.append(v.astype(np.float32))
        ys.append(float(r["y"]))
        idxs.append(i)
    if not rows:
        raise RuntimeError("No rows could be mapped to heavy-chain embeddings.")
    X = np.stack(rows)
    y = np.array(ys, dtype=np.float32)
    used = df.loc[idxs].reset_index(drop=True)
    return X, y, used


def split_per_experiment_by_heavy(
    used: pd.DataFrame,
    holdout_experiment: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.")
    if holdout_experiment not in set(used["source_file"]):
        raise ValueError(f"Holdout experiment not found: {holdout_experiment}")

    work = used.copy()
    work["seq_key"] = work["heavy"]
    holdout_mask = work["source_file"].eq(holdout_experiment)

    train_idx, val_idx, test_idx = [], [], []
    rng = np.random.default_rng(42)

    for _, g in work[~holdout_mask].groupby("source_file"):
        uniq = g["seq_key"].drop_duplicates().to_numpy()
        rng.shuffle(uniq)
        n = len(uniq)

        n_train = int(np.floor(train_frac * n))
        n_val = int(np.floor(val_frac * n))
        n_test = n - n_train - n_val

        # Ensure every non-holdout experiment contributes to training.
        if n > 0 and n_train == 0:
            n_train = 1
            if n_test > 0:
                n_test -= 1
            elif n_val > 0:
                n_val -= 1

        tr_keys = set(uniq[:n_train])
        va_keys = set(uniq[n_train : n_train + n_val])
        te_keys = set(uniq[n_train + n_val : n_train + n_val + n_test])

        train_idx.extend(g[g["seq_key"].isin(tr_keys)].index.tolist())
        val_idx.extend(g[g["seq_key"].isin(va_keys)].index.tolist())
        test_idx.extend(g[g["seq_key"].isin(te_keys)].index.tolist())

    return (
        work.index.isin(train_idx),
        work.index.isin(val_idx),
        work.index.isin(test_idx),
        holdout_mask.to_numpy(),
    )


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def r2(a: np.ndarray, b: np.ndarray) -> float:
    ss_res = float(np.sum((a - b) ** 2))
    ss_tot = float(np.sum((a - np.mean(a)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


class MLPRegressorNP:
    def __init__(self, d_in: int, d_h: int = 128, lr: float = 1e-3, weight_decay: float = 1e-4, seed: int = 42):
        rg = np.random.default_rng(seed)
        self.W1 = (rg.normal(0, 1, (d_in, d_h)) * np.sqrt(2.0 / d_in)).astype(np.float32)
        self.b1 = np.zeros((1, d_h), dtype=np.float32)
        self.W2 = (rg.normal(0, 1, (d_h, 1)) * np.sqrt(2.0 / d_h)).astype(np.float32)
        self.b2 = np.zeros((1, 1), dtype=np.float32)
        self.lr = lr
        self.weight_decay = weight_decay

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(z1, 0.0)
        y = a1 @ self.W2 + self.b2
        return z1, a1, y

    def pred(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X.astype(np.float32))[2].ravel()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        Xv: np.ndarray,
        yv: np.ndarray,
        epochs: int = 300,
        batch_size: int = 64,
        patience: int = 25,
        seed: int = 123,
    ) -> tuple[list[float], list[float]]:
        y2 = y.reshape(-1, 1).astype(np.float32)
        n = len(X)
        rg = np.random.default_rng(seed)

        best = None
        best_val = np.inf
        bad = 0
        train_hist, val_hist = [], []

        for _ in range(epochs):
            idx = rg.permutation(n)
            Xs, ys = X[idx], y2[idx]
            for i in range(0, n, batch_size):
                xb = Xs[i : i + batch_size]
                yb = ys[i : i + batch_size]
                z1, a1, yp = self._forward(xb)
                dY = (2.0 / len(xb)) * (yp - yb)
                dW2 = a1.T @ dY + self.weight_decay * self.W2
                db2 = dY.sum(axis=0, keepdims=True)
                dA1 = dY @ self.W2.T
                dZ1 = dA1 * (z1 > 0)
                dW1 = xb.T @ dZ1 + self.weight_decay * self.W1
                db1 = dZ1.sum(axis=0, keepdims=True)
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1

            tr_mse = float(np.mean((self.pred(X) - y) ** 2))
            va_mse = float(np.mean((self.pred(Xv) - yv) ** 2))
            train_hist.append(tr_mse)
            val_hist.append(va_mse)

            if va_mse < best_val:
                best_val = va_mse
                bad = 0
                best = (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())
            else:
                bad += 1
                if bad >= patience:
                    break

        self.W1, self.b1, self.W2, self.b2 = best
        return train_hist, val_hist


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data_csv)
    seq2emb = load_embedding_cache(args.cache_npz, args.cache_map)
    X, y, used = build_heavy_features(df, seq2emb)

    m_tr, m_va, m_te, m_ho = split_per_experiment_by_heavy(
        used=used,
        holdout_experiment=args.holdout_experiment,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )

    X_train, y_train = X[m_tr], y[m_tr]
    X_val, y_val = X[m_va], y[m_va]
    X_test, y_test = X[m_te], y[m_te]
    X_hold, y_hold = X[m_ho], y[m_ho]

    # Train-only normalization.
    xm = X_train.mean(axis=0, keepdims=True)
    xs = X_train.std(axis=0, keepdims=True) + 1e-8
    Xtr = (X_train - xm) / xs
    Xva = (X_val - xm) / xs
    Xte = (X_test - xm) / xs
    Xho = (X_hold - xm) / xs

    ym = y_train.mean()
    ys = y_train.std() + 1e-8
    ytr = (y_train - ym) / ys
    yva = (y_val - ym) / ys

    # Conservative hyperparameter search to reduce overfitting on small data.
    search_space = [
        {"d_h": 32, "lr": 1e-3, "weight_decay": 1e-3},
        {"d_h": 64, "lr": 1e-3, "weight_decay": 1e-3},
        {"d_h": 128, "lr": 1e-3, "weight_decay": 1e-3},
        {"d_h": 32, "lr": 5e-4, "weight_decay": 1e-3},
        {"d_h": 64, "lr": 5e-4, "weight_decay": 1e-3},
        {"d_h": 128, "lr": 5e-4, "weight_decay": 1e-3},
        {"d_h": 32, "lr": 5e-4, "weight_decay": 1e-2},
        {"d_h": 64, "lr": 5e-4, "weight_decay": 1e-2},
    ]
    val_rank = []
    for hp in search_space:
        m = MLPRegressorNP(d_in=Xtr.shape[1], seed=42, **hp)
        tr_hist, va_hist = m.fit(Xtr, ytr, Xva, yva, seed=123)
        tr_rmse = rmse(ytr, m.pred(Xtr))
        va_rmse = rmse(yva, m.pred(Xva))
        # Tie-break toward lower overfit gap when validation scores are close.
        model_score = va_rmse + 0.10 * max(0.0, va_rmse - tr_rmse)
        val_rank.append(
            {
                **hp,
                "train_rmse_z": tr_rmse,
                "val_rmse_z": va_rmse,
                "model_score": model_score,
                "train_hist": tr_hist,
                "val_hist": va_hist,
            }
        )
    val_rank.sort(key=lambda x: x["model_score"])
    best = val_rank[0]

    seeds = [1, 2, 3, 4, 5]
    rows = []
    first_payload = None
    for sd in seeds:
        m = MLPRegressorNP(
            d_in=Xtr.shape[1],
            d_h=best["d_h"],
            lr=best["lr"],
            weight_decay=best["weight_decay"],
            seed=sd,
        )
        tr_hist, va_hist = m.fit(Xtr, ytr, Xva, yva, seed=1000 + sd)
        pred_test = m.pred(Xte) * ys + ym
        pred_hold = m.pred(Xho) * ys + ym
        rows.append(
            {
                "seed": sd,
                "test_mae": mae(y_test, pred_test),
                "test_rmse": rmse(y_test, pred_test),
                "test_r2": r2(y_test, pred_test),
                "hold_mae": mae(y_hold, pred_hold),
                "hold_rmse": rmse(y_hold, pred_hold),
                "hold_r2": r2(y_hold, pred_hold),
            }
        )
        if first_payload is None:
            first_payload = (pred_test, pred_hold, tr_hist, va_hist)

    results = pd.DataFrame(rows)
    results.to_csv(args.out_dir / "metrics_by_seed.csv", index=False)
    summary = results.agg(["mean", "std"]).T
    summary.to_csv(args.out_dir / "metrics_summary.csv")

    print("Data:")
    print(f"  total used rows={len(used)} train={len(y_train)} val={len(y_val)} test={len(y_test)} holdout={len(y_hold)}")
    print("Best hyperparameters:", {k: best[k] for k in ["d_h", "lr", "weight_decay"]})
    print(f"Best model train/val RMSE (z-scale): {best['train_rmse_z']:.4f} / {best['val_rmse_z']:.4f}")
    print("\nMetrics mean ± std across seeds:")
    for col in ["test_mae", "test_rmse", "test_r2", "hold_mae", "hold_rmse", "hold_r2"]:
        print(f"  {col}: {results[col].mean():.4f} ± {results[col].std(ddof=1):.4f}")

    if float(results["test_r2"].mean()) <= 0.0:
        raise RuntimeError("Non-holdout mean test R2 <= 0. Model is not learning useful signal.")

    if len(y_hold) < 20:
        print(
            f"\nNote: holdout n={len(y_hold)} is very small; holdout R2 is unstable. "
            "Use MAE/RMSE and/or pick a larger holdout experiment for reliable R2."
        )

    if args.save_plots and first_payload is not None:
        mpl_dir = args.out_dir / ".mplconfig"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        import os

        os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
        import matplotlib.pyplot as plt

        pred_test, pred_hold, tr_hist, va_hist = first_payload
        plt.figure(figsize=(7, 4))
        plt.plot(tr_hist, label="train MSE")
        plt.plot(va_hist, label="val MSE")
        plt.xlabel("Epoch")
        plt.ylabel("MSE (z-scale)")
        plt.title("Training vs Validation Loss (seed 1)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out_dir / "loss_curve.png", dpi=180)
        plt.close()

        plt.figure(figsize=(5, 5))
        plt.scatter(y_test, pred_test, alpha=0.5, label="non-holdout test")
        plt.scatter(y_hold, pred_hold, alpha=0.8, label="holdout exp")
        lo = min(np.min(y_test), np.min(y_hold), np.min(pred_test), np.min(pred_hold))
        hi = max(np.max(y_test), np.max(y_hold), np.max(pred_test), np.max(pred_hold))
        plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        plt.xlabel("True Tm")
        plt.ylabel("Predicted Tm")
        plt.title("Predicted vs True (seed 1)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out_dir / "pred_vs_true.png", dpi=180)
        plt.close()
        print(f"\nSaved plots to {args.out_dir}")


if __name__ == "__main__":
    main()
