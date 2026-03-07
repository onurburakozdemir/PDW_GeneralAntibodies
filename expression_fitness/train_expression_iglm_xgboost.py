#!/usr/bin/env python3
"""Train XGBoost regressor on FLAb expression fitness using IgLM heavy+light embeddings.

Design choices for correctness:
- Uses unified expression CSV with columns heavy/light/y.
- Embeds heavy and light chains separately with IgLM.
- Feature for each sample = concat(heavy_emb, light_emb).
- Splits by unique heavy|light pair to avoid leakage across train/val/test.
- Hyperparameters tuned on validation only.
- Reports final test metrics in original y units.
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
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Avoid macOS OpenMP duplication crashes in mixed libraries.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from iglm.model.IgLM import CHECKPOINT_DICT, VOCAB_FILE  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data_csv",
        type=Path,
        default=Path("expression_fitness/expression_unified_fitness.csv"),
        help="Unified expression CSV produced by build_expression_unified_csv.py",
    )
    p.add_argument("--model_name", type=str, default="IgLM", choices=["IgLM", "IgLM-S"])
    p.add_argument("--chain_token_heavy", type=str, default="[HEAVY]")
    p.add_argument("--chain_token_light", type=str, default="[LIGHT]")
    p.add_argument("--species_token", type=str, default="[HUMAN]")

    p.add_argument("--train_frac", type=float, default=0.70)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--holdout_assay",
        type=str,
        default="",
        help="If set, hold out this source_file entirely for final test.",
    )

    p.add_argument("--cache_npz", type=Path, default=Path("expression_fitness/iglm_chain_cache.npz"))
    p.add_argument("--cache_map", type=Path, default=Path("expression_fitness/iglm_chain_cache_map.json"))
    p.add_argument("--recompute_embeddings", action="store_true")

    p.add_argument("--out_dir", type=Path, default=Path("expression_fitness/output"))
    return p.parse_args()


def check_split_fracs(train_frac: float, val_frac: float, test_frac: float) -> None:
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-8:
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)

    need = {"source_file", "heavy", "light", "y", "pair_id"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns: {sorted(miss)}")

    out = df.copy()
    out["heavy"] = out["heavy"].astype(str).str.strip().str.upper()
    out["light"] = out["light"].astype(str).str.strip().str.upper()
    out["pair_id"] = out["pair_id"].astype(str)
    out["y"] = pd.to_numeric(out["y"], errors="raise")

    out = out[(out["heavy"] != "") & (out["light"] != "")].reset_index(drop=True)
    if out.empty:
        raise RuntimeError("No valid rows after sequence cleaning")
    return out


def load_vocab_map(vocab_file: str) -> dict[str, int]:
    tok2id: dict[str, int] = {}
    with open(vocab_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            t = line.strip()
            if t:
                tok2id[t] = i
    return tok2id


def embed_seq(model: torch.nn.Module, device: torch.device, tok2id: dict[str, int], seq: str, chain_token: str, species_token: str) -> np.ndarray:
    tokens = [chain_token, species_token] + list(seq) + ["[SEP]"]
    unk_id = tok2id.get("[UNK]", 1)
    ids = [tok2id.get(t, unk_id) for t in tokens]
    if any(i == unk_id for i in ids):
        raise ValueError("Unknown token in sequence")

    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(x, output_hidden_states=True, return_dict=True)
    hs = out.hidden_states[-1][0]
    seq_hs = hs[2:-1]
    if seq_hs.shape[0] == 0:
        raise ValueError("Empty sequence embedding")
    return seq_hs.mean(dim=0).cpu().numpy().astype(np.float32)


def compute_or_load_chain_cache(df: pd.DataFrame, args: argparse.Namespace) -> dict[str, np.ndarray]:
    all_chains = pd.concat([df["heavy"], df["light"]], ignore_index=True).drop_duplicates().tolist()

    if args.cache_npz.exists() and args.cache_map.exists() and not args.recompute_embeddings:
        npz = np.load(args.cache_npz, allow_pickle=True)
        key_to_seq = json.loads(args.cache_map.read_text())
        seq2emb = {seq: np.asarray(npz[k], dtype=np.float32) for k, seq in key_to_seq.items()}
        if set(all_chains).issubset(set(seq2emb.keys())):
            print(f"Loaded chain cache: {len(seq2emb)} chains")
            return seq2emb
        print("Existing cache incomplete for current data; recomputing cache.")

    print(f"Computing IgLM embeddings for {len(all_chains)} unique chains...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = transformers.GPT2LMHeadModel.from_pretrained(CHECKPOINT_DICT[args.model_name]).to(device)
    model.eval()
    tok2id = load_vocab_map(VOCAB_FILE)

    seq2emb: dict[str, np.ndarray] = {}
    failures = 0
    for i, seq in enumerate(all_chains, start=1):
        # Heuristic: most heavy chains start with EVQ/QVQ/VQL; otherwise try light token.
        likely_heavy = seq.startswith(("EVQ", "QVQ", "VQL", "EAQ"))
        token_order = [args.chain_token_heavy, args.chain_token_light] if likely_heavy else [args.chain_token_light, args.chain_token_heavy]

        ok = False
        for chain_token in token_order:
            try:
                seq2emb[seq] = embed_seq(model, device, tok2id, seq, chain_token, args.species_token)
                ok = True
                break
            except Exception:
                continue

        if not ok:
            failures += 1

        if i % 100 == 0 or i == len(all_chains):
            print(f"  embedded {i}/{len(all_chains)} (failures={failures})")

    if not seq2emb:
        raise RuntimeError("No embeddings were computed")

    args.cache_npz.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    key_to_seq = {}
    for j, (seq, emb) in enumerate(seq2emb.items()):
        k = f"emb_{j:07d}"
        payload[k] = np.asarray(emb, dtype=np.float32)
        key_to_seq[k] = seq
    np.savez_compressed(args.cache_npz, **payload)
    args.cache_map.write_text(json.dumps(key_to_seq))
    print(f"Saved chain cache: {args.cache_npz}")

    return seq2emb


def build_features(df: pd.DataFrame, seq2emb: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    xs, ys, idx = [], [], []
    for i, r in df.iterrows():
        h = seq2emb.get(r["heavy"])
        l = seq2emb.get(r["light"])
        if h is None or l is None:
            continue
        xs.append(np.concatenate([h, l]).astype(np.float32))
        ys.append(float(r["y"]))
        idx.append(i)

    if not xs:
        raise RuntimeError("No rows could be featurized")

    used = df.loc[idx].reset_index(drop=True)
    return np.stack(xs), np.asarray(ys, dtype=np.float32), used


def split_by_pair_ids(used: pd.DataFrame, train_frac: float, val_frac: float, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Leakage-safe split by unique pair_id with y-bin stratification."""
    check_split_fracs(train_frac, val_frac, test_frac)

    pair_tbl = used.groupby("pair_id", as_index=False)["y"].mean()
    # Use quantile bins for approximate stratification in regression.
    nbins = min(10, max(2, pair_tbl["y"].nunique()))
    pair_tbl["y_bin"] = pd.qcut(pair_tbl["y"], q=nbins, duplicates="drop")

    if test_frac <= 1e-12:
        train_pairs, val_pairs = train_test_split(
            pair_tbl,
            test_size=val_frac,
            random_state=seed,
            stratify=pair_tbl["y_bin"].astype(str),
        )
        test_pairs = pair_tbl.iloc[0:0].copy()
    else:
        train_pairs, temp_pairs = train_test_split(
            pair_tbl,
            test_size=(1.0 - train_frac),
            random_state=seed,
            stratify=pair_tbl["y_bin"].astype(str),
        )

        rel_test = test_frac / (val_frac + test_frac)
        temp_bins = temp_pairs["y_bin"].astype(str)
        val_pairs, test_pairs = train_test_split(
            temp_pairs,
            test_size=rel_test,
            random_state=seed,
            stratify=temp_bins,
        )

    tr_set = set(train_pairs["pair_id"])
    va_set = set(val_pairs["pair_id"])
    te_set = set(test_pairs["pair_id"])

    m_tr = used["pair_id"].isin(tr_set).to_numpy()
    m_va = used["pair_id"].isin(va_set).to_numpy()
    m_te = used["pair_id"].isin(te_set).to_numpy()

    if test_frac <= 1e-12:
        if not (m_tr.any() and m_va.any()):
            raise RuntimeError("Invalid split: train or val is empty")
    else:
        if not (m_tr.any() and m_va.any() and m_te.any()):
            raise RuntimeError("Invalid split: one split is empty")

    return m_tr, m_va, m_te


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
        {"max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
        {"max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 2.0},
        {"max_depth": 4, "learning_rate": 0.1, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0},
    ]

    best = None
    best_rmse = np.inf

    for hp in grid:
        model = XGBRegressor(
            n_estimators=2500,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=seed,
            eval_metric="rmse",
            early_stopping_rounds=80,
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


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data_csv)
    seq2emb = compute_or_load_chain_cache(df, args)
    X, y, used = build_features(df, seq2emb)

    if args.holdout_assay:
        if args.holdout_assay not in set(used["source_file"]):
            raise ValueError(f"holdout_assay not found: {args.holdout_assay}")
        src = used[~used["source_file"].eq(args.holdout_assay)].reset_index(drop=True)
        te = used[used["source_file"].eq(args.holdout_assay)].reset_index(drop=True)
        X_src, y_src, src_used = build_features(src, seq2emb)
        X_te, y_te, _ = build_features(te, seq2emb)
        m_tr, m_va, _ = split_by_pair_ids(
            used=src_used,
            train_frac=args.train_frac / (args.train_frac + args.val_frac),
            val_frac=args.val_frac / (args.train_frac + args.val_frac),
            test_frac=0.0,
            seed=args.seed,
        )
        X_tr, y_tr = X_src[m_tr], y_src[m_tr]
        X_va, y_va = X_src[m_va], y_src[m_va]
    else:
        m_tr, m_va, m_te = split_by_pair_ids(
            used=used,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
        )
        X_tr, y_tr = X[m_tr], y[m_tr]
        X_va, y_va = X[m_va], y[m_va]
        X_te, y_te = X[m_te], y[m_te]

    best = choose_hparams(X_tr, y_tr, X_va, y_va, seed=args.seed)

    # Refit on train+val with tuned number of trees.
    X_trva = np.concatenate([X_tr, X_va], axis=0)
    y_trva = np.concatenate([y_tr, y_va], axis=0)

    model = XGBRegressor(
        n_estimators=max(100, int(best["best_iteration"]) + 1),
        objective="reg:squarederror",
        tree_method="hist",
        random_state=args.seed,
        eval_metric="rmse",
        max_depth=int(best["max_depth"]),
        learning_rate=float(best["learning_rate"]),
        subsample=float(best["subsample"]),
        colsample_bytree=float(best["colsample_bytree"]),
        reg_lambda=float(best["reg_lambda"]),
    )
    model.fit(X_trva, y_trva, verbose=False)

    p_te = model.predict(X_te)

    results = {
        "n_total": int(len(used)),
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_va)),
        "n_test": int(len(y_te)),
        "holdout_assay": args.holdout_assay if args.holdout_assay else "",
        "test_mae": mae(y_te, p_te),
        "test_rmse": rmse(y_te, p_te),
        "test_r2": r2(y_te, p_te),
        "best_max_depth": int(best["max_depth"]),
        "best_learning_rate": float(best["learning_rate"]),
        "best_subsample": float(best["subsample"]),
        "best_colsample_bytree": float(best["colsample_bytree"]),
        "best_reg_lambda": float(best["reg_lambda"]),
        "best_iteration": int(best["best_iteration"]),
    }

    out_metrics = args.out_dir / "metrics_expression_iglm_xgboost.csv"
    pd.DataFrame([results]).to_csv(out_metrics, index=False)

    print("Expression IgLM+XGBoost run complete")
    print(f"rows: total={results['n_total']} train={results['n_train']} val={results['n_val']} test={results['n_test']}")
    print(f"test: MAE={results['test_mae']:.4f} RMSE={results['test_rmse']:.4f} R2={results['test_r2']:.4f}")
    print(f"best hp: depth={results['best_max_depth']} lr={results['best_learning_rate']} subsample={results['best_subsample']} colsample={results['best_colsample_bytree']} reg_lambda={results['best_reg_lambda']} trees={results['best_iteration']+1}")
    print(f"saved: {out_metrics}")


if __name__ == "__main__":
    main()
