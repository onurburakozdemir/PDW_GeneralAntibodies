#!/usr/bin/env python3
"""Heavy-only FLAb thermostability benchmark using IgLM embeddings + XGBoost.

This script is designed to be directly comparable to the ESM2+XGBoost baseline:
- Same strict study-wise split logic.
- Same per-source-study validation split by unique heavy sequence.
- Only embedding model changes: ESM2 -> IgLM.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import transformers
from xgboost import XGBRegressor

# Workaround for occasional OpenMP duplication on macOS environments.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import iglm  # noqa: E402
from iglm.model.IgLM import CHECKPOINT_DICT, VOCAB_FILE  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    # Input data.
    p.add_argument("--data_csv", type=Path, default=Path("flab_thermo_unified_ml_tm_only.csv"))

    # IgLM embedding cache paths.
    p.add_argument("--cache_npz", type=Path, default=Path("output/iglm_cache/iglm_heavy_meanpool.npz"))
    p.add_argument("--cache_map", type=Path, default=Path("output/iglm_cache/iglm_heavy_meanpool_map.json"))
    p.add_argument("--model_name", type=str, default="IgLM", choices=["IgLM", "IgLM-S"])
    p.add_argument("--chain_token", type=str, default="[HEAVY]")
    p.add_argument("--species_token", type=str, default="[HUMAN]")
    p.add_argument("--recompute_embeddings", action="store_true")

    # Holdout control.
    p.add_argument("--holdout_domain", type=str, default="tresanco2023nbthermo_tm.csv")
    p.add_argument("--run_all_holdouts", action="store_true")
    p.add_argument("--min_holdout_rows", type=int, default=20)
    p.add_argument("--max_holdouts", type=int, default=0)

    # Split/tuning control.
    p.add_argument("--val_frac", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)

    # Output.
    p.add_argument("--out_dir", type=Path, default=Path("output/xgboost_iglm_heavy"))

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
        raise RuntimeError("Dataset empty after heavy filtering")
    return out


def load_vocab_map(vocab_file: str) -> dict[str, int]:
    """Read IgLM vocab.txt into token->id mapping."""
    tok2id: dict[str, int] = {}
    with open(vocab_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            tok = line.strip()
            if tok:
                tok2id[tok] = i
    return tok2id


def build_iglm_embedding(model: torch.nn.Module, device: torch.device, tok2id: dict[str, int], seq: str, chain_token: str, species_token: str) -> np.ndarray:
    """Build one sequence embedding from IgLM last hidden states (mean pooled over sequence positions)."""
    tokens = [chain_token, species_token] + list(seq) + ["[SEP]"]
    unk_id = tok2id.get("[UNK]", 1)
    ids = [tok2id.get(t, unk_id) for t in tokens]
    if any(int(t) == int(unk_id) for t in ids):
        raise ValueError("Sequence has tokens unknown to IgLM vocab")

    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(x, output_hidden_states=True, return_dict=True)

    hs = out.hidden_states[-1][0]  # shape [L, D]
    # sequence positions exclude [HEAVY], [HUMAN], and final [SEP]
    seq_hs = hs[2:-1]
    if seq_hs.shape[0] == 0:
        raise ValueError("Empty sequence after tokenization")

    emb = seq_hs.mean(dim=0).detach().cpu().numpy().astype(np.float32)
    return emb


def compute_or_load_iglm_embeddings(unique_seqs: list[str], args: argparse.Namespace) -> dict[str, np.ndarray]:
    """Load cached IgLM embeddings or compute them if missing/forced."""
    args.cache_npz.parent.mkdir(parents=True, exist_ok=True)

    # If cache exists and recompute is not forced, load and return.
    if args.cache_npz.exists() and args.cache_map.exists() and not args.recompute_embeddings:
        npz = np.load(args.cache_npz, allow_pickle=True)
        key_to_seq = json.loads(args.cache_map.read_text())
        seq2emb = {seq: np.asarray(npz[k], dtype=np.float32).ravel() for k, seq in key_to_seq.items()}
        # Ensure all needed sequences are covered.
        if set(unique_seqs).issubset(set(seq2emb.keys())):
            print(f"Loaded IgLM cache: {len(seq2emb)} sequences")
            return seq2emb
        print("Existing IgLM cache does not cover all required sequences; recomputing.")

    print(f"Computing IgLM embeddings for {len(unique_seqs)} unique heavy sequences...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = transformers.GPT2LMHeadModel.from_pretrained(CHECKPOINT_DICT[args.model_name]).to(device)
    model.eval()
    tok2id = load_vocab_map(VOCAB_FILE)
    seq2emb: dict[str, np.ndarray] = {}

    for i, seq in enumerate(unique_seqs, start=1):
        try:
            seq2emb[seq] = build_iglm_embedding(
                model=model,
                device=device,
                tok2id=tok2id,
                seq=seq,
                chain_token=args.chain_token,
                species_token=args.species_token,
            )
        except Exception:
            # Keep conservative behavior: skip only failing sequences.
            continue
        if i % 50 == 0 or i == len(unique_seqs):
            print(f"  embedded {i}/{len(unique_seqs)}")

    if not seq2emb:
        raise RuntimeError("No IgLM embeddings computed successfully")

    # Save cache.
    key_to_seq = {}
    payload = {}
    for j, (seq, emb) in enumerate(seq2emb.items()):
        key = f"emb_{j:07d}"
        key_to_seq[key] = seq
        payload[key] = np.asarray(emb, dtype=np.float32)

    np.savez_compressed(args.cache_npz, **payload)
    args.cache_map.write_text(json.dumps(key_to_seq))
    print(f"Saved IgLM cache: {args.cache_npz}")

    return seq2emb


def map_df_to_cache(df: pd.DataFrame, seq2emb: dict[str, np.ndarray]) -> pd.DataFrame:
    out = df[df["heavy"].isin(seq2emb.keys())].reset_index(drop=True)
    if out.empty:
        raise RuntimeError("No rows mapped to IgLM cache")
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


def build_xy(df: pd.DataFrame, seq2emb: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for _, r in df.iterrows():
        v = seq2emb.get(r["heavy"])
        if v is None:
            continue
        xs.append(v)
        ys.append(float(r["y"]))

    if not xs:
        raise RuntimeError("No rows mapped to embeddings")

    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32)


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def r2(y: np.ndarray, p: np.ndarray) -> float:
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def choose_hparams(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray, seed: int) -> dict[str, Any]:
    grid = [
        {"max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
        {"max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 2.0},
        {"max_depth": 3, "learning_rate": 0.1, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0},
        {"max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5.0},
    ]

    best = None
    best_rmse = np.inf
    for hp in grid:
        model = XGBRegressor(
            n_estimators=2000,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=seed,
            eval_metric="rmse",
            early_stopping_rounds=50,
            **hp,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        pred = model.predict(X_va)
        sc = rmse(y_va, pred)
        if sc < best_rmse:
            best_rmse = sc
            best = {**hp, "best_iteration": int(getattr(model, "best_iteration", model.n_estimators))}

    if best is None:
        raise RuntimeError("Hyperparameter search failed")
    return best


def run_single_holdout(df: pd.DataFrame, seq2emb: dict[str, np.ndarray], holdout_domain: str, val_frac: float, seed: int) -> dict[str, Any]:
    if holdout_domain not in set(df["source_file"]):
        raise ValueError(f"Holdout not found: {holdout_domain}")

    holdout_df = df[df["source_file"].eq(holdout_domain)].reset_index(drop=True)
    source_df = df[~df["source_file"].eq(holdout_domain)].reset_index(drop=True)

    tr_parts = []
    va_parts = []
    for i, (_, g) in enumerate(source_df.groupby("source_file")):
        tr, va = split_source_domain_by_heavy(g, val_frac=val_frac, seed=seed + i)
        if not tr.empty and not va.empty:
            tr_parts.append(tr)
            va_parts.append(va)

    if not tr_parts or not va_parts:
        raise RuntimeError("No usable source splits")

    src_tr = pd.concat(tr_parts, ignore_index=True)
    src_va = pd.concat(va_parts, ignore_index=True)

    X_tr, y_tr = build_xy(src_tr, seq2emb)
    X_va, y_va = build_xy(src_va, seq2emb)
    X_ho, y_ho = build_xy(holdout_df, seq2emb)

    best = choose_hparams(X_tr, y_tr, X_va, y_va, seed=seed)

    src_full = pd.concat([src_tr, src_va], ignore_index=True)
    X_full, y_full = build_xy(src_full, seq2emb)

    model = XGBRegressor(
        n_estimators=max(50, int(best["best_iteration"]) + 1),
        objective="reg:squarederror",
        tree_method="hist",
        random_state=seed,
        eval_metric="rmse",
        max_depth=int(best["max_depth"]),
        learning_rate=float(best["learning_rate"]),
        subsample=float(best["subsample"]),
        colsample_bytree=float(best["colsample_bytree"]),
        reg_lambda=float(best["reg_lambda"]),
    )
    model.fit(X_full, y_full, verbose=False)

    pred = model.predict(X_ho)

    return {
        "domain": holdout_domain,
        "n_holdout": int(len(holdout_df)),
        "n_source_train": int(len(src_tr)),
        "n_source_val": int(len(src_va)),
        "mae": mae(y_ho, pred),
        "rmse": rmse(y_ho, pred),
        "r2": r2(y_ho, pred),
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
    unique_seqs = df["heavy"].drop_duplicates().tolist()
    seq2emb = compute_or_load_iglm_embeddings(unique_seqs, args)
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

    print("IgLM + XGBoost heavy-only run")
    print(f"  run_all_holdouts={args.run_all_holdouts}")
    print(f"  holdouts={len(holdouts)}")
    print(f"  embeddings_available={len(seq2emb)}")

    rows = []
    for i, h in enumerate(holdouts, start=1):
        print(f"\n[{i}/{len(holdouts)}] holdout={h}")
        r = run_single_holdout(df=df, seq2emb=seq2emb, holdout_domain=h, val_frac=args.val_frac, seed=args.seed)
        rows.append(r)
        print(f"  n_holdout={r['n_holdout']}  MAE={r['mae']:.4f}  RMSE={r['rmse']:.4f}  R2={r['r2']:.4f}")

    summary = pd.DataFrame(rows).sort_values("domain").reset_index(drop=True)
    out_path = args.out_dir / "summary_xgboost_iglm_heavy.csv"
    summary.to_csv(out_path, index=False)

    if len(summary) > 1:
        print("\nLOSO summary")
        print(f"  mean MAE:  {summary['mae'].mean():.4f}")
        print(f"  mean RMSE: {summary['rmse'].mean():.4f}")
        print(f"  mean R2:   {summary['r2'].mean():.4f} ± {summary['r2'].std(ddof=1):.4f}")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
