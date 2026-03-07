#!/usr/bin/env python3
"""DomainBed-style heavy-chain regression on FLAb thermostability.

What this script does:
1) Uses studies (`source_file`) as domains.
2) Trains a model on source domains only.
3) Validates on per-domain validation splits (source domains only).
4) Tests on a held-out target domain (OOD).
5) Supports ERM / IRM / CORAL objectives.
6) Can run one holdout or full leave-one-study-out (LOSO).

Notes:
- Heavy chain only.
- Uses cached ESM2 embeddings (sequence -> vector).
- Uses train-only scaling to avoid leakage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reproducibility settings for deterministic behavior where possible.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    """CLI arguments for data, model, and evaluation mode."""
    p = argparse.ArgumentParser(description=__doc__)

    # Input dataset and ESM2 cache.
    p.add_argument("--data_csv", type=Path, default=Path("flab_thermo_unified_ml_tm_only.csv"))
    p.add_argument("--cache_npz", type=Path, default=Path("output/esm_cache/esm2_space_cache_separate_hl.npz"))
    p.add_argument("--cache_map", type=Path, default=Path("output/esm_cache/esm2_space_cache_separate_hl_map.json"))
    p.add_argument("--expected_dim", type=int, default=320)

    # DomainBed-style algorithm choice.
    p.add_argument("--algorithm", type=str, default="ERM", choices=["ERM", "IRM", "CORAL"])

    # Holdout control:
    # - if run_all_holdouts is False => only holdout_domain is used
    # - if run_all_holdouts is True  => leave-one-study-out on eligible domains
    p.add_argument("--holdout_domain", type=str, default="tresanco2023nbthermo_tm.csv")
    p.add_argument("--run_all_holdouts", action="store_true")
    p.add_argument("--min_holdout_rows", type=int, default=20)
    p.add_argument("--max_holdouts", type=int, default=0)

    # Split/training hyperparameters.
    p.add_argument("--val_frac", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.20)

    # Penalty controls.
    p.add_argument("--irm_lambda", type=float, default=1.0)
    p.add_argument("--irm_anneal_steps", type=int, default=500)
    p.add_argument("--coral_lambda", type=float, default=1.0)

    # Output.
    p.add_argument("--out_dir", type=Path, default=Path("output/domainbed_heavy"))

    return p.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    """Load harmonized CSV and enforce minimal columns."""
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)

    need = {"source_file", "heavy", "y"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns: {sorted(miss)}")

    out = df.copy()
    out["source_file"] = out["source_file"].astype(str)
    out["heavy"] = out["heavy"].astype(str).str.strip()
    out["y"] = pd.to_numeric(out["y"], errors="raise")

    # Drop empty heavy rows.
    out = out[out["heavy"] != ""].reset_index(drop=True)
    if out.empty:
        raise RuntimeError("Dataset is empty after filtering heavy chains.")
    return out


def load_seq2emb(cache_npz: Path, cache_map: Path, expected_dim: int) -> dict[str, np.ndarray]:
    """Load embedding cache as sequence -> vector mapping."""
    if not cache_npz.exists() or not cache_map.exists():
        raise FileNotFoundError(f"Missing cache files: {cache_npz} / {cache_map}")

    npz = np.load(cache_npz, allow_pickle=True)
    key_to_seq = json.loads(cache_map.read_text())

    seq2emb: dict[str, np.ndarray] = {}
    for key, seq in key_to_seq.items():
        vec = np.asarray(npz[key], dtype=np.float32).ravel()
        seq2emb[str(seq)] = vec

    if not seq2emb:
        raise RuntimeError("Empty embedding cache.")

    d = int(next(iter(seq2emb.values())).shape[0])
    if d != int(expected_dim):
        raise ValueError(f"Embedding dim mismatch: got {d}, expected {expected_dim}")

    return seq2emb


def map_df_to_cache(df: pd.DataFrame, seq2emb: dict[str, np.ndarray]) -> pd.DataFrame:
    """Keep only rows where heavy sequence exists in embedding cache."""
    m = df["heavy"].isin(seq2emb.keys())
    out = df[m].reset_index(drop=True)
    if out.empty:
        raise RuntimeError("No rows matched embedding cache.")
    return out


def split_source_domain_indices(df_domain: pd.DataFrame, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split one source domain by unique heavy sequence to avoid leakage."""
    rng = np.random.default_rng(seed)
    uniq = df_domain["heavy"].drop_duplicates().to_numpy()
    rng.shuffle(uniq)

    if len(uniq) < 2:
        raise ValueError("Need at least 2 unique heavy sequences for train/val split.")

    n_val = int(np.floor(len(uniq) * val_frac))
    n_val = min(max(n_val, 1), len(uniq) - 1)

    val_keys = set(uniq[:n_val])
    tr_keys = set(uniq[n_val:])

    m_tr = df_domain["heavy"].isin(tr_keys).to_numpy()
    m_va = df_domain["heavy"].isin(val_keys).to_numpy()
    return m_tr, m_va


def build_domain_splits(df: pd.DataFrame, holdout_domain: str, val_frac: float, seed: int) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    """Return source-domain train/val dicts and target holdout dataframe."""
    if holdout_domain not in set(df["source_file"]):
        raise ValueError(f"Holdout domain not found: {holdout_domain}")

    holdout_df = df[df["source_file"].eq(holdout_domain)].reset_index(drop=True)
    source_df = df[~df["source_file"].eq(holdout_domain)].reset_index(drop=True)

    tr_domains: dict[str, pd.DataFrame] = {}
    va_domains: dict[str, pd.DataFrame] = {}

    for i, (dom, g) in enumerate(source_df.groupby("source_file")):
        g = g.reset_index(drop=True)
        if g["heavy"].nunique() < 2:
            continue
        try:
            m_tr, m_va = split_source_domain_indices(g, val_frac=val_frac, seed=seed + i)
        except ValueError:
            continue
        g_tr = g[m_tr].reset_index(drop=True)
        g_va = g[m_va].reset_index(drop=True)
        if g_tr.empty or g_va.empty:
            continue
        tr_domains[dom] = g_tr
        va_domains[dom] = g_va

    if len(tr_domains) < 1 or len(va_domains) < 1:
        raise RuntimeError("No usable source domains after splitting.")

    return tr_domains, va_domains, holdout_df


def build_xy(df: pd.DataFrame, seq2emb: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Convert dataframe to (X,y) using heavy embeddings."""
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for _, row in df.iterrows():
        vec = seq2emb.get(row["heavy"])
        if vec is None:
            continue
        xs.append(np.asarray(vec, dtype=np.float32))
        ys.append(float(row["y"]))
    if not xs:
        raise RuntimeError("No rows in this split mapped to embeddings.")
    return np.stack(xs), np.asarray(ys, dtype=np.float32)


def stack_domains(domains: dict[str, pd.DataFrame], seq2emb: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Map each domain dataframe to arrays, skipping impossible domains."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for dom, d in domains.items():
        try:
            out[dom] = build_xy(d, seq2emb)
        except RuntimeError:
            continue
    if not out:
        raise RuntimeError("No source domains remained after embedding mapping.")
    return out


def fit_scalers(train_domains: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fit feature/target scaling on source-domain training only."""
    X_all = np.concatenate([v[0] for v in train_domains.values()], axis=0)
    y_all = np.concatenate([v[1] for v in train_domains.values()], axis=0)
    x_mean = X_all.mean(axis=0, keepdims=True)
    x_std = X_all.std(axis=0, keepdims=True) + 1e-8
    y_mean = float(y_all.mean())
    y_std = float(y_all.std() + 1e-8)
    return x_mean, x_std, y_mean, y_std


def apply_scaling(X: np.ndarray, y: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray, y_mean: float, y_std: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply scaling computed from source train set."""
    Xs = ((X - x_mean) / x_std).astype(np.float32)
    ys = ((y - y_mean) / y_std).astype(np.float32)
    return Xs, ys


class Featurizer(nn.Module):
    """Simple MLP featurizer."""

    def __init__(self, d_in: int, d_hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Regressor(nn.Module):
    """Linear regression head on top of learned features."""

    def __init__(self, d_hidden: int) -> None:
        super().__init__()
        self.out = nn.Linear(d_hidden, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.out(z)


class DomainBedRegressor(nn.Module):
    """Featurizer + scalar regressor."""

    def __init__(self, d_in: int, d_hidden: int, dropout: float) -> None:
        super().__init__()
        self.featurizer = Featurizer(d_in, d_hidden, dropout)
        self.regressor = Regressor(d_hidden)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.featurizer(x)
        y_hat = self.regressor(z).squeeze(-1)
        return z, y_hat


def irm_penalty(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """IRM penalty via scale parameter gradient."""
    scale = torch.tensor(1.0, device=y_hat.device, requires_grad=True)
    loss = F.mse_loss(y_hat * scale, y)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)


def coral_penalty(z_list: list[torch.Tensor]) -> torch.Tensor:
    """CORAL penalty: align feature means and covariances across domains."""
    if len(z_list) < 2:
        return torch.tensor(0.0, device=z_list[0].device)

    p = torch.tensor(0.0, device=z_list[0].device)
    n_pairs = 0
    for i in range(len(z_list)):
        for j in range(i + 1, len(z_list)):
            zi = z_list[i]
            zj = z_list[j]
            mi = zi.mean(dim=0)
            mj = zj.mean(dim=0)
            ci = zi - mi
            cj = zj - mj
            cov_i = (ci.T @ ci) / max(zi.shape[0] - 1, 1)
            cov_j = (cj.T @ cj) / max(zj.shape[0] - 1, 1)
            p = p + torch.mean((mi - mj) ** 2) + torch.mean((cov_i - cov_j) ** 2)
            n_pairs += 1
    return p / max(n_pairs, 1)


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def r2(a: np.ndarray, b: np.ndarray) -> float:
    ss_res = float(np.sum((a - b) ** 2))
    ss_tot = float(np.sum((a - np.mean(a)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def evaluate(model: DomainBedRegressor, X: np.ndarray, y_scaled: np.ndarray, y_mean: float, y_std: float, device: torch.device) -> dict[str, float]:
    """Evaluate in original target units."""
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X, dtype=torch.float32, device=device)
        _, y_hat_scaled = model(xt)
        y_pred = (y_hat_scaled.cpu().numpy() * y_std) + y_mean
    y_true = (y_scaled * y_std) + y_mean
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "r2": r2(y_true, y_pred)}


def run_single_holdout(df: pd.DataFrame, seq2emb: dict[str, np.ndarray], args: argparse.Namespace, holdout_domain: str) -> dict[str, Any]:
    """Train/evaluate one target holdout domain."""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tr_df, va_df, holdout_df = build_domain_splits(df, holdout_domain=holdout_domain, val_frac=args.val_frac, seed=args.seed)

    tr_np = stack_domains(tr_df, seq2emb)
    va_np = stack_domains(va_df, seq2emb)
    X_hold, y_hold = build_xy(holdout_df, seq2emb)

    x_mean, x_std, y_mean, y_std = fit_scalers(tr_np)

    tr_scaled = {d: apply_scaling(X, y, x_mean, x_std, y_mean, y_std) for d, (X, y) in tr_np.items()}
    va_scaled = {d: apply_scaling(X, y, x_mean, x_std, y_mean, y_std) for d, (X, y) in va_np.items()}
    Xh, yh = apply_scaling(X_hold, y_hold, x_mean, x_std, y_mean, y_std)

    if not va_scaled:
        raise RuntimeError("No validation domains available after scaling.")

    device = torch.device("cpu")
    d_in = int(next(iter(tr_scaled.values()))[0].shape[1])

    model = DomainBedRegressor(d_in=d_in, d_hidden=args.hidden_dim, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    tr_tensors = {
        d: (torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(y, dtype=torch.float32, device=device))
        for d, (X, y) in tr_scaled.items()
    }
    va_tensors = {
        d: (torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(y, dtype=torch.float32, device=device))
        for d, (X, y) in va_scaled.items()
    }

    rng = np.random.default_rng(args.seed)

    def sample_idx(n: int, b: int) -> np.ndarray:
        return rng.choice(n, size=b, replace=(n < b))

    best_state = None
    best_val_rmse = np.inf

    for step in range(1, args.steps + 1):
        model.train()
        opt.zero_grad()

        losses = []
        irm_ps = []
        z_list = []

        for dom, (X_dom, y_dom) in tr_tensors.items():
            idx = sample_idx(X_dom.shape[0], args.batch_size)
            xb = X_dom[idx]
            yb = y_dom[idx]

            z, y_hat = model(xb)
            loss_dom = F.mse_loss(y_hat, yb)
            losses.append(loss_dom)
            z_list.append(z)

            if args.algorithm == "IRM":
                irm_ps.append(irm_penalty(y_hat, yb))

        erm = torch.stack(losses).mean()

        if args.algorithm == "ERM":
            loss = erm
        elif args.algorithm == "IRM":
            irm_p = torch.stack(irm_ps).mean()
            irm_w = args.irm_lambda if step >= args.irm_anneal_steps else 0.0
            loss = erm + irm_w * irm_p
        elif args.algorithm == "CORAL":
            c_p = coral_penalty(z_list)
            loss = erm + args.coral_lambda * c_p
        else:
            raise ValueError(f"Unknown algorithm: {args.algorithm}")

        loss.backward()
        opt.step()

        if step % 50 == 0 or step == args.steps:
            model.eval()
            rmses = []
            with torch.no_grad():
                for _, (Xv, yv) in va_tensors.items():
                    _, pred = model(Xv)
                    rmses.append(torch.sqrt(torch.mean((pred - yv) ** 2)).item())
            val_rmse = float(np.mean(rmses))
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    hold_metrics = evaluate(model, Xh, yh, y_mean=y_mean, y_std=y_std, device=device)

    # Per-source validation metrics in original units.
    val_rows = []
    for dom, (Xv, yv) in va_scaled.items():
        m = evaluate(model, Xv, yv, y_mean=y_mean, y_std=y_std, device=device)
        val_rows.append({"domain": dom, **m})
    val_df = pd.DataFrame(val_rows)

    holdout_tag = Path(holdout_domain).stem
    pd.DataFrame([{"split": "holdout", "domain": holdout_domain, **hold_metrics}]).to_csv(
        args.out_dir / f"metrics_{args.algorithm.lower()}_{holdout_tag}.csv", index=False
    )
    val_df.to_csv(args.out_dir / f"source_val_metrics_{args.algorithm.lower()}_{holdout_tag}.csv", index=False)

    print(f"\nHoldout done: {holdout_domain}")
    print(f"  source domains used: {len(tr_scaled)}")
    print(f"  best source-val RMSE (z): {best_val_rmse:.4f}")
    print(f"  holdout MAE/RMSE/R2: {hold_metrics['mae']:.4f} / {hold_metrics['rmse']:.4f} / {hold_metrics['r2']:.4f}")

    return {
        "domain": holdout_domain,
        "n_holdout": int(len(holdout_df)),
        "source_domains": int(len(tr_scaled)),
        "best_source_val_rmse_z": float(best_val_rmse),
        **hold_metrics,
    }


def main() -> None:
    """Entry point."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data_csv)
    seq2emb = load_seq2emb(args.cache_npz, args.cache_map, expected_dim=args.expected_dim)
    df = map_df_to_cache(df, seq2emb)

    domain_sizes = df.groupby("source_file").size().sort_values(ascending=False)

    if args.run_all_holdouts:
        holdouts = [dom for dom, n in domain_sizes.items() if int(n) >= int(args.min_holdout_rows)]
        if args.max_holdouts > 0:
            holdouts = holdouts[: args.max_holdouts]
        if not holdouts:
            raise RuntimeError("No holdouts passed min_holdout_rows filter.")
    else:
        holdouts = [args.holdout_domain]

    print("Run configuration")
    print(f"  algorithm: {args.algorithm}")
    print(f"  run_all_holdouts: {args.run_all_holdouts}")
    print(f"  number of holdouts: {len(holdouts)}")

    rows: list[dict[str, Any]] = []
    for i, holdout in enumerate(holdouts, start=1):
        print(f"\n[{i}/{len(holdouts)}] holdout={holdout}")
        row = run_single_holdout(df=df, seq2emb=seq2emb, args=args, holdout_domain=holdout)
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("domain").reset_index(drop=True)
    summary_path = args.out_dir / f"summary_{args.algorithm.lower()}.csv"
    summary.to_csv(summary_path, index=False)

    if len(summary) > 1:
        print("\nLOSO summary")
        print(f"  holdouts: {len(summary)}")
        print(f"  mean MAE:  {summary['mae'].mean():.4f}")
        print(f"  mean RMSE: {summary['rmse'].mean():.4f}")
        print(f"  mean R2:   {summary['r2'].mean():.4f} ± {summary['r2'].std(ddof=1):.4f}")

    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
