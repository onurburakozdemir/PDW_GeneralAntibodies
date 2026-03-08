#!/usr/bin/env python3
"""Strict overlap-experiment holdout pipeline (heavy-only).

Same core pipeline as final_step, but combined evaluation is harder:
- pick the largest thermo-expression overlap experiment pair,
- hide those entire two experiments from training,
- evaluate developability only on overlap heavies from that hidden pair.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import transformers

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplconfig").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib.pyplot as plt

from iglm.model.IgLM import CHECKPOINT_DICT, VOCAB_FILE

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_vocab() -> dict[str, int]:
    tok2id = {}
    with open(VOCAB_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            t = line.strip()
            if t:
                tok2id[t] = i
    return tok2id


def embed_heavy(model, device, tok2id: dict[str, int], seq: str) -> np.ndarray:
    tokens = ["[HEAVY]", "[HUMAN]"] + list(seq) + ["[SEP]"]
    unk = tok2id.get("[UNK]", 1)
    ids = [tok2id.get(t, unk) for t in tokens]
    if any(i == unk for i in ids):
        raise ValueError("Unknown token")
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(x, output_hidden_states=True, return_dict=True)
    hs = out.hidden_states[-1][0]
    aa = hs[2:-1]
    return aa.mean(dim=0).cpu().numpy().astype(np.float32)


def load_cache(cache_npz: Path, cache_map: Path) -> Dict[str, np.ndarray]:
    if not (cache_npz.exists() and cache_map.exists()):
        return {}
    npz = np.load(cache_npz, allow_pickle=True)
    key2seq = json.loads(cache_map.read_text())
    return {seq: np.asarray(npz[k], dtype=np.float32) for k, seq in key2seq.items()}


def save_cache(seq2emb: Dict[str, np.ndarray], cache_npz: Path, cache_map: Path) -> None:
    payload, key2seq = {}, {}
    for i, (seq, emb) in enumerate(seq2emb.items()):
        k = f"emb_{i:07d}"
        payload[k] = np.asarray(emb, dtype=np.float32)
        key2seq[k] = seq
    cache_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_npz, **payload)
    cache_map.write_text(json.dumps(key2seq))


def import_existing_cache(seq2emb: Dict[str, np.ndarray], root: Path) -> Dict[str, np.ndarray]:
    candidates = [
        (root / "final_step" / "cache" / "iglm_heavy_cache.npz", root / "final_step" / "cache" / "iglm_heavy_cache_map.json"),
        (root / "sunday_work" / "iglm_heavy_cache.npz", root / "sunday_work" / "iglm_heavy_cache_map.json"),
    ]
    # expression cache has H:: keys
    expr_npz = root / "sunday_work" / "iglm_expression_hl_cache.npz"
    expr_map = root / "sunday_work" / "iglm_expression_hl_cache_map.json"
    if expr_npz.exists() and expr_map.exists():
        npz = np.load(expr_npz, allow_pickle=True)
        key2seq = json.loads(expr_map.read_text())
        for k, sk in key2seq.items():
            if isinstance(sk, str) and sk.startswith("H::"):
                seq = sk.split("::", 1)[1]
                if seq not in seq2emb:
                    seq2emb[seq] = np.asarray(npz[k], dtype=np.float32)

    for npz_p, map_p in candidates:
        if npz_p.exists() and map_p.exists():
            npz = np.load(npz_p, allow_pickle=True)
            key2seq = json.loads(map_p.read_text())
            for k, seq in key2seq.items():
                if seq not in seq2emb:
                    seq2emb[seq] = np.asarray(npz[k], dtype=np.float32)
    return seq2emb


def ensure_embeddings(all_heavy: np.ndarray, seq2emb: Dict[str, np.ndarray], cache_npz: Path, cache_map: Path) -> Dict[str, np.ndarray]:
    missing = [s for s in all_heavy if s not in seq2emb]
    print(f"Embedding cache: have={len(seq2emb)} need={len(all_heavy)} missing={len(missing)}")
    if not missing:
        return seq2emb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = transformers.GPT2LMHeadModel.from_pretrained(CHECKPOINT_DICT["IgLM"]).to(device)
    model.eval()
    tok2id = load_vocab()

    fail = 0
    for i, seq in enumerate(missing, start=1):
        try:
            seq2emb[seq] = embed_heavy(model, device, tok2id, seq)
        except Exception:
            fail += 1
        if i % 100 == 0 or i == len(missing):
            print(f"  embedded {i}/{len(missing)} (fail={fail})")

    save_cache(seq2emb, cache_npz, cache_map)
    return seq2emb


def build_Xy(df: pd.DataFrame, seq2emb: Dict[str, np.ndarray], label_col: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    X, y, idx = [], [], []
    for i, r in df.iterrows():
        emb = seq2emb.get(r["heavy"])
        if emb is None:
            continue
        X.append(emb)
        y.append(int(r[label_col]))
        idx.append(i)
    used = df.loc[idx].reset_index(drop=True)
    return np.stack(X), np.asarray(y, dtype=np.int32), used


def train_eval_logreg(X_tr, y_tr, X_te, y_te):
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_tr)
    Xte = sc.transform(X_te)
    clf = LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced", random_state=SEED)
    clf.fit(Xtr, y_tr)
    prob = clf.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "acc": float(accuracy_score(y_te, pred)),
        "auc": float(roc_auc_score(y_te, prob)) if len(np.unique(y_te)) > 1 else float("nan"),
        "fpr": roc_curve(y_te, prob)[0],
        "tpr": roc_curve(y_te, prob)[1],
        "clf": clf,
        "sc": sc,
    }


def choose_largest_overlap_pair(thermo: pd.DataFrame, expr: pd.DataFrame):
    h = set(thermo["heavy"]).intersection(set(expr["heavy"]))
    thermo_o = thermo[thermo["heavy"].isin(h)]
    expr_o = expr[expr["heavy"].isin(h)]

    best = None
    for ts, td in thermo_o.groupby("source_file"):
        hs = set(td["heavy"])
        for es, ed in expr_o.groupby("source_file"):
            he = set(ed["heavy"])
            n = len(hs & he)
            if n == 0:
                continue
            if best is None or n > best[2]:
                best = (ts, es, n)
    if best is None:
        raise RuntimeError("No overlap experiment pair found")
    return best


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "final_step_holdout_experiment" / "data"
    out_dir = root / "final_step_holdout_experiment" / "output"
    cache_npz = root / "final_step_holdout_experiment" / "cache" / "iglm_heavy_cache.npz"
    cache_map = root / "final_step_holdout_experiment" / "cache" / "iglm_heavy_cache_map.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    thermo = pd.read_csv(data_dir / "thermo_binary_heavy.csv")
    expr = pd.read_csv(data_dir / "expression_binary_heavy.csv")

    all_heavy = np.unique(np.concatenate([thermo["heavy"].values, expr["heavy"].values]))
    seq2emb = load_cache(cache_npz, cache_map)
    seq2emb = import_existing_cache(seq2emb, root)
    seq2emb = ensure_embeddings(all_heavy, seq2emb, cache_npz, cache_map)

    hold_tm, hold_ex, n_inter = choose_largest_overlap_pair(thermo, expr)

    tm_train = thermo[~thermo["source_file"].eq(hold_tm)].reset_index(drop=True)
    ex_train = expr[~expr["source_file"].eq(hold_ex)].reset_index(drop=True)

    tm_hold = thermo[thermo["source_file"].eq(hold_tm)].reset_index(drop=True)
    ex_hold = expr[expr["source_file"].eq(hold_ex)].reset_index(drop=True)

    # Per-task holdout experiment performance
    Xtm_tr, ytm_tr, _ = build_Xy(tm_train[["heavy", "label_tm", "source_file"]], seq2emb, "label_tm")
    Xtm_te, ytm_te, _ = build_Xy(tm_hold[["heavy", "label_tm", "source_file"]], seq2emb, "label_tm")
    res_tm = train_eval_logreg(Xtm_tr, ytm_tr, Xtm_te, ytm_te)

    Xex_tr, yex_tr, _ = build_Xy(ex_train[["heavy", "label_expr", "source_file"]], seq2emb, "label_expr")
    Xex_te, yex_te, _ = build_Xy(ex_hold[["heavy", "label_expr", "source_file"]], seq2emb, "label_expr")
    res_ex = train_eval_logreg(Xex_tr, yex_tr, Xex_te, yex_te)

    # Combined on overlap of hidden experiments
    h_test = sorted(set(tm_hold["heavy"]).intersection(set(ex_hold["heavy"])))
    tm_h = tm_hold[tm_hold["heavy"].isin(h_test)].groupby("heavy", as_index=False)["label_tm"].mean()
    tm_h["label_tm"] = (tm_h["label_tm"] >= 0.5).astype(int)
    ex_h = ex_hold[ex_hold["heavy"].isin(h_test)].groupby("heavy", as_index=False)["label_expr"].mean()
    ex_h["label_expr"] = (ex_h["label_expr"] >= 0.5).astype(int)
    both = tm_h.merge(ex_h, on="heavy", how="inner")

    Xb, _, _ = build_Xy(both[["heavy", "label_tm"]], seq2emb, "label_tm")
    p_tm = res_tm["clf"].predict_proba(res_tm["sc"].transform(Xb))[:, 1]
    p_ex = res_ex["clf"].predict_proba(res_ex["sc"].transform(Xb))[:, 1]
    p_dev = p_tm * p_ex
    y_dev = ((both["label_tm"].values == 1) & (both["label_expr"].values == 1)).astype(int)
    pred_dev = (p_dev >= 0.5).astype(int)

    comb_acc = float(accuracy_score(y_dev, pred_dev))
    comb_auc = float(roc_auc_score(y_dev, p_dev)) if len(np.unique(y_dev)) > 1 else float("nan")

    # Save metrics
    metrics = {
        "holdout_pair": {"thermo_source": hold_tm, "expr_source": hold_ex, "shared_heavy": int(n_inter)},
        "thermo_holdout_experiment": {"acc": res_tm["acc"], "auc": res_tm["auc"], "n_train": int(len(ytm_tr)), "n_test": int(len(ytm_te))},
        "expression_holdout_experiment": {"acc": res_ex["acc"], "auc": res_ex["auc"], "n_train": int(len(yex_tr)), "n_test": int(len(yex_te))},
        "combined_hidden_pair": {"acc": comb_acc, "auc": comb_auc, "n_test": int(len(y_dev))},
    }
    (out_dir / "metrics_summary.json").write_text(json.dumps(metrics, indent=2))

    pd.DataFrame([
        {"task": "thermostability_holdout_experiment", "acc": res_tm["acc"], "auc": res_tm["auc"]},
        {"task": "expression_holdout_experiment", "acc": res_ex["acc"], "auc": res_ex["auc"]},
        {"task": "combined_hidden_pair", "acc": comb_acc, "auc": comb_auc},
    ]).to_csv(out_dir / "task_metrics.csv", index=False)

    # Simple presentation figure
    labels = ["Thermo", "Expression", "Combined"]
    accs = [res_tm["acc"], res_ex["acc"], comb_acc]
    aucs = [res_tm["auc"], res_ex["auc"], comb_auc]
    x = np.arange(len(labels))
    w = 0.35
    plt.figure(figsize=(7, 4.5))
    plt.bar(x - w / 2, accs, w, label="Accuracy")
    plt.bar(x + w / 2, aucs, w, label="AUC")
    for i, v in enumerate(accs):
        plt.text(i - w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    for i, v in enumerate(aucs):
        plt.text(i + w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    plt.ylim(0, 1.0)
    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.title("Strict Hidden-Experiment Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_hidden_experiment_metrics.png", dpi=180)
    plt.close()

    both.assign(p_tm=p_tm, p_expr=p_ex, p_dev=p_dev, label_dev=y_dev, pred_dev=pred_dev).to_csv(
        out_dir / "combined_hidden_pair_predictions.csv", index=False
    )

    print("Done: strict hidden overlap experiment pipeline")
    print(json.dumps(metrics, indent=2))
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
