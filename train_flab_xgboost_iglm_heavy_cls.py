#!/usr/bin/env python3
"""Heavy-only IgLM + XGBoost classification for FLAb thermostability.

Task:
- Convert y (Tm) to binary label with threshold (default 70C).
- Evaluate with strict leave-one-study-out protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_csv", type=Path, default=Path("flab_thermo_unified_ml_tm_only.csv"))

    p.add_argument("--iglm_npz", type=Path, default=Path("output/iglm_cache/iglm_heavy_meanpool.npz"))
    p.add_argument("--iglm_map", type=Path, default=Path("output/iglm_cache/iglm_heavy_meanpool_map.json"))
    p.add_argument("--iglm_dim", type=int, default=512)

    p.add_argument("--cutoff", type=float, default=70.0)

    p.add_argument("--holdout_domain", type=str, default="tresanco2023nbthermo_tm.csv")
    p.add_argument("--run_all_holdouts", action="store_true")
    p.add_argument("--min_holdout_rows", type=int, default=20)
    p.add_argument("--max_holdouts", type=int, default=0)

    p.add_argument("--val_frac", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--out_dir", type=Path, default=Path("output/xgboost_iglm_heavy_cls"))
    return p.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"source_file", "heavy", "y"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns: {sorted(miss)}")

    out = df.copy()
    out["source_file"] = out["source_file"].astype(str)
    out["heavy"] = out["heavy"].astype(str).str.strip().str.upper()
    out["y"] = pd.to_numeric(out["y"], errors="raise")
    out = out[out["heavy"] != ""].reset_index(drop=True)
    if out.empty:
        raise RuntimeError("Empty dataset after filtering")
    return out


def load_seq2emb(npz_path: Path, map_path: Path, expected_dim: int) -> dict[str, np.ndarray]:
    if not npz_path.exists() or not map_path.exists():
        raise FileNotFoundError(f"Missing cache files: {npz_path} / {map_path}")
    npz = np.load(npz_path, allow_pickle=True)
    key_to_seq = json.loads(map_path.read_text())
    seq2emb = {str(seq).strip().upper(): np.asarray(npz[k], dtype=np.float32).ravel() for k, seq in key_to_seq.items()}
    if not seq2emb:
        raise RuntimeError("Empty embedding cache")
    d = int(next(iter(seq2emb.values())).shape[0])
    if d != int(expected_dim):
        raise ValueError(f"IgLM dim mismatch: got {d}, expected {expected_dim}")
    return seq2emb


def map_df_to_cache(df: pd.DataFrame, seq2emb: dict[str, np.ndarray]) -> pd.DataFrame:
    out = df[df["heavy"].isin(seq2emb.keys())].reset_index(drop=True)
    if out.empty:
        raise RuntimeError("No rows matched IgLM cache")
    return out


def split_source_domain_by_heavy(df_domain: pd.DataFrame, val_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = df_domain.reset_index(drop=True)
    uniq = g["heavy"].drop_duplicates().to_numpy()
    if len(uniq) < 2:
        return pd.DataFrame(columns=g.columns), pd.DataFrame(columns=g.columns)

    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)

    n_val = int(np.floor(len(uniq) * val_frac))
    n_val = min(max(n_val, 1), len(uniq) - 1)

    val_keys = set(uniq[:n_val])
    tr_keys = set(uniq[n_val:])

    tr = g[g["heavy"].isin(tr_keys)].reset_index(drop=True)
    va = g[g["heavy"].isin(val_keys)].reset_index(drop=True)
    return tr, va


def build_xy(df: pd.DataFrame, seq2emb: dict[str, np.ndarray], cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for _, r in df.iterrows():
        v = seq2emb.get(r["heavy"])
        if v is None:
            continue
        xs.append(v)
        ys.append(1 if float(r["y"]) >= cutoff else 0)
    if not xs:
        raise RuntimeError("No rows mapped to embeddings")
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64)


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def safe_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= thr).astype(int)
    return {
        "roc_auc": safe_auc(y_true, y_prob),
        "auprc": safe_auprc(y_true, y_prob),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "pos_rate": float(y_true.mean()),
        "pred_pos_rate": float(y_pred.mean()),
    }


def choose_hparams(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray, seed: int) -> dict[str, Any]:
    grid = [
        {"max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
        {"max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 2.0},
        {"max_depth": 3, "learning_rate": 0.1, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0},
        {"max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5.0},
    ]

    # Class weight from train distribution.
    n_pos = int(y_tr.sum())
    n_neg = int(len(y_tr) - n_pos)
    spw = float(n_neg / max(n_pos, 1))

    best = None
    best_score = -np.inf

    for hp in grid:
        model = XGBClassifier(
            n_estimators=2000,
            objective="binary:logistic",
            tree_method="hist",
            random_state=seed,
            eval_metric="aucpr" if np.unique(y_va).size >= 2 else "logloss",
            early_stopping_rounds=50,
            scale_pos_weight=spw,
            **hp,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        prob = model.predict_proba(X_va)[:, 1]
        m = classification_metrics(y_va, prob)

        # Primary selection metric: AUPRC if valid, else balanced accuracy.
        score = m["auprc"] if not np.isnan(m["auprc"]) else m["balanced_acc"]
        if score > best_score:
            best_score = score
            best = {**hp, "best_iteration": int(getattr(model, "best_iteration", model.n_estimators)), "scale_pos_weight": spw}

    if best is None:
        raise RuntimeError("Hyperparameter search failed")
    return best


def run_single_holdout(df: pd.DataFrame, seq2emb: dict[str, np.ndarray], holdout_domain: str, val_frac: float, seed: int, cutoff: float) -> dict[str, Any]:
    if holdout_domain not in set(df["source_file"]):
        raise ValueError(f"Holdout not found: {holdout_domain}")

    holdout_df = df[df["source_file"].eq(holdout_domain)].reset_index(drop=True)
    source_df = df[~df["source_file"].eq(holdout_domain)].reset_index(drop=True)

    tr_parts, va_parts = [], []
    for i, (_, g) in enumerate(source_df.groupby("source_file")):
        tr, va = split_source_domain_by_heavy(g, val_frac=val_frac, seed=seed + i)
        if not tr.empty and not va.empty:
            tr_parts.append(tr)
            va_parts.append(va)

    if not tr_parts or not va_parts:
        raise RuntimeError("No usable source train/val splits")

    src_tr = pd.concat(tr_parts, ignore_index=True)
    src_va = pd.concat(va_parts, ignore_index=True)

    X_tr, y_tr = build_xy(src_tr, seq2emb, cutoff=cutoff)
    X_va, y_va = build_xy(src_va, seq2emb, cutoff=cutoff)
    X_ho, y_ho = build_xy(holdout_df, seq2emb, cutoff=cutoff)

    best = choose_hparams(X_tr, y_tr, X_va, y_va, seed=seed)

    src_full = pd.concat([src_tr, src_va], ignore_index=True)
    X_full, y_full = build_xy(src_full, seq2emb, cutoff=cutoff)

    model = XGBClassifier(
        n_estimators=max(50, int(best["best_iteration"]) + 1),
        objective="binary:logistic",
        tree_method="hist",
        random_state=seed,
        eval_metric="logloss",
        max_depth=int(best["max_depth"]),
        learning_rate=float(best["learning_rate"]),
        subsample=float(best["subsample"]),
        colsample_bytree=float(best["colsample_bytree"]),
        reg_lambda=float(best["reg_lambda"]),
        scale_pos_weight=float(best["scale_pos_weight"]),
    )
    model.fit(X_full, y_full, verbose=False)

    prob_ho = model.predict_proba(X_ho)[:, 1]
    m = classification_metrics(y_ho, prob_ho)

    return {
        "domain": holdout_domain,
        "n_holdout": int(len(y_ho)),
        "n_source_train": int(len(y_tr)),
        "n_source_val": int(len(y_va)),
        **m,
        "best_max_depth": int(best["max_depth"]),
        "best_learning_rate": float(best["learning_rate"]),
        "best_subsample": float(best["subsample"]),
        "best_colsample_bytree": float(best["colsample_bytree"]),
        "best_reg_lambda": float(best["reg_lambda"]),
        "best_iteration": int(best["best_iteration"]),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data_csv)
    seq2emb = load_seq2emb(args.iglm_npz, args.iglm_map, args.iglm_dim)
    df = map_df_to_cache(df, seq2emb)

    sizes = df.groupby("source_file").size().sort_values(ascending=False)
    if args.run_all_holdouts:
        holdouts = [d for d, n in sizes.items() if int(n) >= int(args.min_holdout_rows)]
        if args.max_holdouts > 0:
            holdouts = holdouts[: args.max_holdouts]
        if not holdouts:
            raise RuntimeError("No holdouts passed min_holdout_rows")
    else:
        holdouts = [args.holdout_domain]

    print("IgLM + XGBoost classification (heavy-only)")
    print(f"  cutoff={args.cutoff}")
    print(f"  run_all_holdouts={args.run_all_holdouts}")
    print(f"  holdouts={len(holdouts)}")

    rows = []
    for i, h in enumerate(holdouts, start=1):
        print(f"\n[{i}/{len(holdouts)}] holdout={h}")
        r = run_single_holdout(df=df, seq2emb=seq2emb, holdout_domain=h, val_frac=args.val_frac, seed=args.seed, cutoff=args.cutoff)
        rows.append(r)
        print(f"  n={r['n_holdout']}  AUC={r['roc_auc']:.4f}  AUPRC={r['auprc']:.4f}  BalAcc={r['balanced_acc']:.4f}  F1={r['f1']:.4f}")

    out = pd.DataFrame(rows).sort_values("domain").reset_index(drop=True)
    out_path = args.out_dir / f"summary_xgboost_iglm_heavy_cls_cutoff{int(args.cutoff)}.csv"
    out.to_csv(out_path, index=False)

    if len(out) > 1:
        print("\nLOSO summary")
        for c in ["roc_auc", "auprc", "balanced_acc", "f1", "accuracy"]:
            valid = out[c].dropna()
            if len(valid) > 0:
                std = valid.std(ddof=1) if len(valid) > 1 else 0.0
                print(f"  mean {c}: {valid.mean():.4f} ± {std:.4f}")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
