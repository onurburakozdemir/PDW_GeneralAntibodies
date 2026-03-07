#!/usr/bin/env python3
"""Train heavy-only ESM2+MLP on Tresanco and test in/out of study."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from gradio_client import Client, handle_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_csv", type=Path, default=Path("flab_thermo_unified_ml_tm_only.csv"))
    p.add_argument("--train_source", type=str, default="tresanco2023nbthermo_tm.csv")
    p.add_argument("--esm_model", type=str, default="facebook/esm2_t6_8M_UR50D")
    p.add_argument("--space_id", type=str, default="hugging-science/ESM2")
    p.add_argument("--test_frac", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_duration", type=int, default=900)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--cache_npz", type=Path, default=Path("output/esm_cache/tresanco_heavy_fresh_t6.npz"))
    p.add_argument("--cache_map", type=Path, default=Path("output/esm_cache/tresanco_heavy_fresh_t6_map.json"))
    p.add_argument("--out_dir", type=Path, default=Path("output/tresanco_heavy_only_fresh"))
    return p.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"source_file", "heavy", "y"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns: {sorted(miss)}")
    out = df.copy()
    out["heavy"] = out["heavy"].astype(str).str.strip()
    out["y"] = pd.to_numeric(out["y"], errors="raise")
    out = out[out["heavy"] != ""].reset_index(drop=True)
    return out


def split_tresanco(df: pd.DataFrame, train_source: str, test_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    in_exp = df[df["source_file"].eq(train_source)].copy()
    out_exp = df[~df["source_file"].eq(train_source)].copy()
    if in_exp.empty:
        raise ValueError(f"No rows for source_file={train_source}")

    rng = np.random.default_rng(seed)
    uniq = in_exp["heavy"].drop_duplicates().to_numpy()
    rng.shuffle(uniq)
    n_test = int(np.floor(test_frac * len(uniq)))
    n_test = min(max(n_test, 1), len(uniq) - 1)
    test_keys = set(uniq[:n_test])
    train_keys = set(uniq[n_test:])

    train_df = in_exp[in_exp["heavy"].isin(train_keys)].reset_index(drop=True)
    test_df = in_exp[in_exp["heavy"].isin(test_keys)].reset_index(drop=True)
    return train_df, test_df, out_exp.reset_index(drop=True)


def write_fasta_chunk(ids: list[str], seqs: list[str]) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="esm_space_"))
    fp = tmp / "chunk.fasta"
    with fp.open("w") as f:
        for sid, seq in zip(ids, seqs):
            f.write(f">{sid}\n{seq}\n")
    return fp


def parse_npz_embedding_file(npz_path: Path) -> dict[str, np.ndarray]:
    z = np.load(npz_path, allow_pickle=True)
    out: dict[str, np.ndarray] = {}

    if "embeddings" in z.files and "sequence_ids" in z.files:
        embs = np.asarray(z["embeddings"])
        ids = np.asarray(z["sequence_ids"])
        for sid, emb in zip(ids.tolist(), embs):
            out[str(sid)] = np.asarray(emb, dtype=np.float32).ravel()
        return out

    for k in z.files:
        arr = np.asarray(z[k], dtype=object)
        if arr.ndim == 0:
            continue
        if arr.dtype.kind in {"f", "i"}:
            out[str(k)] = np.asarray(arr, dtype=np.float32).ravel()
    return out


def call_space_embeddings(
    client: Client,
    seq_items: list[tuple[str, str]],
    model_key: str,
    infer_batch_size: int,
    max_duration: int,
    retries: int,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for start in range(0, len(seq_items), infer_batch_size):
        chunk = seq_items[start : start + infer_batch_size]
        ids = [sid for sid, _ in chunk]
        seqs = [seq for _, seq in chunk]
        fasta_path = write_fasta_chunk(ids, seqs)

        done = False
        for attempt in range(1, retries + 1):
            try:
                out = client.predict(
                    fasta_files=[handle_file(str(fasta_path))],
                    model_key=model_key,
                    batch_size_value=min(32, len(chunk)),
                    max_duration=max_duration,
                    api_name="/run_pipeline_with_selected_model",
                )
                files = out[0] if isinstance(out[0], list) else [out[0]]
                local_map: dict[str, np.ndarray] = {}
                for fp in files:
                    p = Path(fp)
                    if p.suffix == ".npz":
                        local_map.update(parse_npz_embedding_file(p))

                for sid in ids:
                    if sid in local_map:
                        result[sid] = local_map[sid]
                done = True
                break
            except Exception:
                if attempt == retries:
                    raise
                time.sleep(2 * attempt)

        if not done:
            raise RuntimeError(f"Failed chunk at start={start}")
        print(f"embedded {min(start + infer_batch_size, len(seq_items))}/{len(seq_items)}")

    return result


def recompute_embeddings(unique_seqs: list[str], args: argparse.Namespace) -> dict[str, np.ndarray]:
    args.cache_npz.parent.mkdir(parents=True, exist_ok=True)
    sid_to_seq = {f"seq_{i:07d}": s for i, s in enumerate(unique_seqs)}
    seq_items = list(sid_to_seq.items())
    client = Client(args.space_id)
    sid_to_emb = call_space_embeddings(
        client=client,
        seq_items=seq_items,
        model_key=args.esm_model,
        infer_batch_size=args.batch_size,
        max_duration=args.max_duration,
        retries=args.retries,
    )

    seq_to_emb = {sid_to_seq[sid]: emb for sid, emb in sid_to_emb.items() if sid in sid_to_seq}
    if len(seq_to_emb) == 0:
        raise RuntimeError("No embeddings were returned from the Space.")

    key_to_seq = {}
    payload = {}
    for i, (seq, emb) in enumerate(seq_to_emb.items()):
        key = f"emb_{i:07d}"
        key_to_seq[key] = seq
        payload[key] = np.asarray(emb, dtype=np.float32)

    np.savez_compressed(args.cache_npz, **payload)
    args.cache_map.write_text(json.dumps(key_to_seq))
    return seq_to_emb


def load_embedding_cache(cache_npz: Path, cache_map: Path) -> dict[str, np.ndarray]:
    npz = np.load(cache_npz, allow_pickle=True)
    key_to_seq = json.loads(cache_map.read_text())
    return {seq: np.asarray(npz[k], dtype=np.float32) for k, seq in key_to_seq.items()}


def build_xy(df: pd.DataFrame, seq2emb: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for _, r in df.iterrows():
        v = seq2emb.get(r["heavy"])
        if v is None:
            continue
        xs.append(np.asarray(v, dtype=np.float32).ravel())
        ys.append(float(r["y"]))
    if not xs:
        raise RuntimeError("No rows mapped to embeddings.")
    return np.stack(xs), np.asarray(ys, dtype=np.float32)


class MLPRegressorNP:
    def __init__(self, d_in: int, d_h: int = 64, lr: float = 5e-4, weight_decay: float = 1e-2, seed: int = 42):
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

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 250, batch_size: int = 64, seed: int = 123) -> None:
        y2 = y.reshape(-1, 1).astype(np.float32)
        n = len(X)
        rg = np.random.default_rng(seed)
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


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def r2(a: np.ndarray, b: np.ndarray) -> float:
    ss_res = float(np.sum((a - b) ** 2))
    ss_tot = float(np.sum((a - np.mean(a)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "r2": r2(y_true, y_pred)}


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data_csv)
    train_df, test_df, external_df = split_tresanco(df, args.train_source, args.test_frac, args.seed)
    all_needed = pd.concat([train_df, test_df, external_df], ignore_index=True)
    unique_seqs = all_needed["heavy"].drop_duplicates().tolist()

    print(f"Recomputing embeddings for unique heavy sequences: {len(unique_seqs)}")
    seq2emb = recompute_embeddings(unique_seqs, args)
    print(f"Embeddings available: {len(seq2emb)}")

    X_train, y_train = build_xy(train_df, seq2emb)
    X_test, y_test = build_xy(test_df, seq2emb)
    X_ext, y_ext = build_xy(external_df, seq2emb)

    xm = X_train.mean(axis=0, keepdims=True)
    xs = X_train.std(axis=0, keepdims=True) + 1e-8
    Xtr = (X_train - xm) / xs
    Xte = (X_test - xm) / xs
    Xex = (X_ext - xm) / xs

    ym = y_train.mean()
    ys = y_train.std() + 1e-8
    ytr = (y_train - ym) / ys

    model = MLPRegressorNP(d_in=Xtr.shape[1], seed=args.seed)
    model.fit(Xtr, ytr, epochs=250, batch_size=64, seed=args.seed + 1)

    pred_test = model.pred(Xte) * ys + ym
    pred_ext = model.pred(Xex) * ys + ym
    m_test = metrics(y_test, pred_test)
    m_ext = metrics(y_ext, pred_ext)

    print("\nSplit sizes:")
    print(f"  train ({args.train_source}): {len(y_train)}")
    print(f"  test  ({args.train_source}): {len(y_test)}")
    print(f"  external (all non-{args.train_source}): {len(y_ext)}")

    print("\nWithin-study test metrics:")
    print(f"  MAE:  {m_test['mae']:.4f}")
    print(f"  RMSE: {m_test['rmse']:.4f}")
    print(f"  R2:   {m_test['r2']:.4f}")

    print("\nOut-of-study external metrics:")
    print(f"  MAE:  {m_ext['mae']:.4f}")
    print(f"  RMSE: {m_ext['rmse']:.4f}")
    print(f"  R2:   {m_ext['r2']:.4f}")

    out_pred = external_df[["source_file", "heavy", "y"]].copy()
    out_pred["y_pred"] = pred_ext
    out_pred.to_csv(args.out_dir / "external_predictions.csv", index=False)
    pd.DataFrame(
        [
            {"split": "within_test", "n": len(y_test), **m_test},
            {"split": "external_all_other_experiments", "n": len(y_ext), **m_ext},
        ]
    ).to_csv(args.out_dir / "metrics.csv", index=False)
    print(f"\nSaved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
