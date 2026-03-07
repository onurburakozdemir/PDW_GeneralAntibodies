#!/usr/bin/env python3
"""IgLM heavy-only XGBoost classification with pooled stratified train/val and held-out study test.

Split protocol:
- Test set: one full holdout study (`source_file`)
- Source pool: all remaining studies merged together
- Train/val: stratified split on binary label from pooled source rows
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
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_csv", type=Path, default=Path("flab_thermo_unified_ml_tm_only.csv"))

    p.add_argument("--iglm_npz", type=Path, default=Path("output/iglm_cache/iglm_heavy_meanpool.npz"))
    p.add_argument("--iglm_map", type=Path, default=Path("output/iglm_cache/iglm_heavy_meanpool_map.json"))
    p.add_argument("--iglm_dim", type=int, default=512)

    p.add_argument("--cutoff", type=float, default=70.0)
    p.add_argument("--holdout_domain", type=str, default="tresanco2023nbthermo_tm.csv")
    p.add_argument("--val_frac", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--out_dir", type=Path, default=Path("output/xgboost_iglm_heavy_cls_pooled_trainval"))
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
        raise RuntimeError("Dataset empty after filtering heavy chains")
    return out


def load_seq2emb(npz_path: Path, map_path: Path, expected_dim: int) -> dict[str, np.ndarray]:
    if not npz_path.exists() or not map_path.exists():
        raise FileNotFoundError(f"Missing cache files: {npz_path} / {map_path}")

    npz = np.load(npz_path, allow_pickle=True)
    key_to_seq = json.loads(map_path.read_text())
    seq2emb = {str(seq).strip().upper(): np.asarray(npz[k], dtype=np.float32).ravel() for k, seq in key_to_seq.items()}

    if not seq2emb:
        raise RuntimeError("Empty IgLM cache")
    dim = int(next(iter(seq2emb.values())).shape[0])
    if dim != int(expected_dim):
        raise ValueError(f"IgLM dim mismatch: got {dim}, expected {expected_dim}")
    return seq2emb


def map_df_to_cache(df: pd.DataFrame, seq2emb: dict[str, np.ndarray]) -> pd.DataFrame:
    out = df[df["heavy"].isin(seq2emb.keys())].reset_index(drop=True)
    if out.empty:
        raise RuntimeError("No rows matched IgLM cache")
    return out


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


def cls_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> dict[str, float]:
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
        m = cls_metrics(y_va, prob)
        score = m["auprc"] if not np.isnan(m["auprc"]) else m["balanced_acc"]
        if score > best_score:
            best_score = score
            best = {**hp, "best_iteration": int(getattr(model, "best_iteration", model.n_estimators)), "scale_pos_weight": spw}

    if best is None:
        raise RuntimeError("Hyperparameter search failed")
    return best


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data_csv)
    seq2emb = load_seq2emb(args.iglm_npz, args.iglm_map, args.iglm_dim)
    df = map_df_to_cache(df, seq2emb)

    if args.holdout_domain not in set(df["source_file"]):
        raise ValueError(f"holdout_domain not found: {args.holdout_domain}")

    test_df = df[df["source_file"].eq(args.holdout_domain)].reset_index(drop=True)
    source_df = df[~df["source_file"].eq(args.holdout_domain)].reset_index(drop=True)

    X_src, y_src = build_xy(source_df, seq2emb, cutoff=args.cutoff)
    X_test, y_test = build_xy(test_df, seq2emb, cutoff=args.cutoff)

    # Pooled, stratified split for train/val (as requested).
    idx = np.arange(len(y_src))
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=args.val_frac,
        random_state=args.seed,
        stratify=y_src,
    )

    X_tr, y_tr = X_src[tr_idx], y_src[tr_idx]
    X_va, y_va = X_src[va_idx], y_src[va_idx]

    best = choose_hparams(X_tr, y_tr, X_va, y_va, seed=args.seed)

    # Refit on full pooled source data.
    model = XGBClassifier(
        n_estimators=max(50, int(best["best_iteration"]) + 1),
        objective="binary:logistic",
        tree_method="hist",
        random_state=args.seed,
        eval_metric="logloss",
        max_depth=int(best["max_depth"]),
        learning_rate=float(best["learning_rate"]),
        subsample=float(best["subsample"]),
        colsample_bytree=float(best["colsample_bytree"]),
        reg_lambda=float(best["reg_lambda"]),
        scale_pos_weight=float(best["scale_pos_weight"]),
    )
    model.fit(X_src, y_src, verbose=False)

    prob_test = model.predict_proba(X_test)[:, 1]
    m_test = cls_metrics(y_test, prob_test)

    out_row = {
        "holdout_domain": args.holdout_domain,
        "cutoff": float(args.cutoff),
        "n_source": int(len(y_src)),
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_va)),
        "n_test": int(len(y_test)),
        **m_test,
        "best_max_depth": int(best["max_depth"]),
        "best_learning_rate": float(best["learning_rate"]),
        "best_subsample": float(best["subsample"]),
        "best_colsample_bytree": float(best["colsample_bytree"]),
        "best_reg_lambda": float(best["reg_lambda"]),
        "best_iteration": int(best["best_iteration"]),
    }

    out = pd.DataFrame([out_row])
    out_path = args.out_dir / f"summary_pooled_trainval_holdout_{Path(args.holdout_domain).stem}_cutoff{int(args.cutoff)}.csv"
    out.to_csv(out_path, index=False)

    print("IgLM + XGBoost classification (pooled stratified train/val, held-out study test)")
    print(f"holdout: {args.holdout_domain}")
    print(f"n_source={len(y_src)}  n_train={len(y_tr)}  n_val={len(y_va)}  n_test={len(y_test)}")
    print(f"test AUC={m_test['roc_auc']:.4f}  AUPRC={m_test['auprc']:.4f}  BalAcc={m_test['balanced_acc']:.4f}  F1={m_test['f1']:.4f}  Acc={m_test['accuracy']:.4f}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
