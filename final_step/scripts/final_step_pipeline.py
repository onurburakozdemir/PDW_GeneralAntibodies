#!/usr/bin/env python3
"""Final step: heavy-only developability pipeline.

1) Load prepared binary datasets.
2) Build/reuse IgLM heavy embeddings.
3) Train/test thermostability classifier.
4) Train/test expression classifier.
5) Evaluate combined developability on overlap holdout.
6) Save presentation-ready plots and metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import transformers

from iglm.model.IgLM import CHECKPOINT_DICT, VOCAB_FILE

import os


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplconfig").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt


def load_vocab() -> dict[str, int]:
    tok2id: dict[str, int] = {}
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
        raise ValueError("Unknown token in heavy sequence")

    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(x, output_hidden_states=True, return_dict=True)
    hs = out.hidden_states[-1][0]
    aa = hs[2:-1]
    if aa.shape[0] == 0:
        raise ValueError("Empty heavy sequence")
    return aa.mean(dim=0).cpu().numpy().astype(np.float32)


def load_cache(cache_npz: Path, cache_map: Path) -> Dict[str, np.ndarray]:
    if not (cache_npz.exists() and cache_map.exists()):
        return {}
    npz = np.load(cache_npz, allow_pickle=True)
    key2seq = json.loads(cache_map.read_text())
    return {seq: np.asarray(npz[k], dtype=np.float32) for k, seq in key2seq.items()}


def save_cache(seq2emb: Dict[str, np.ndarray], cache_npz: Path, cache_map: Path) -> None:
    payload = {}
    key2seq = {}
    for i, (seq, emb) in enumerate(seq2emb.items()):
        k = f"emb_{i:07d}"
        payload[k] = np.asarray(emb, dtype=np.float32)
        key2seq[k] = seq
    cache_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_npz, **payload)
    cache_map.write_text(json.dumps(key2seq))


def import_existing_cache(seq2emb: Dict[str, np.ndarray], project_root: Path) -> Dict[str, np.ndarray]:
    # Import from previous caches to avoid recomputation.
    expr_map = project_root / "sunday_work" / "iglm_expression_hl_cache_map.json"
    expr_npz = project_root / "sunday_work" / "iglm_expression_hl_cache.npz"
    if expr_map.exists() and expr_npz.exists():
        npz = np.load(expr_npz, allow_pickle=True)
        key2seq = json.loads(expr_map.read_text())
        for k, seq_key in key2seq.items():
            if isinstance(seq_key, str) and seq_key.startswith("H::"):
                seq = seq_key.split("::", 1)[1]
                if seq not in seq2emb:
                    seq2emb[seq] = np.asarray(npz[k], dtype=np.float32)

    thermo_map = project_root / "sunday_work" / "iglm_heavy_cache_map.json"
    thermo_npz = project_root / "sunday_work" / "iglm_heavy_cache.npz"
    if thermo_map.exists() and thermo_npz.exists():
        npz = np.load(thermo_npz, allow_pickle=True)
        key2seq = json.loads(thermo_map.read_text())
        for k, seq in key2seq.items():
            if seq not in seq2emb:
                seq2emb[seq] = np.asarray(npz[k], dtype=np.float32)

    return seq2emb


def compute_embeddings_if_needed(all_heavy: np.ndarray, seq2emb: Dict[str, np.ndarray], cache_npz: Path, cache_map: Path) -> Dict[str, np.ndarray]:
    missing = [s for s in all_heavy if s not in seq2emb]
    print(f"Embedding cache status: have={len(seq2emb)} need={len(all_heavy)} missing={len(missing)}")
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
    idx = []
    X = []
    y = []
    for i, r in df.iterrows():
        emb = seq2emb.get(r["heavy"])
        if emb is None:
            continue
        idx.append(i)
        X.append(emb)
        y.append(int(r[label_col]))

    used = df.loc[idx].reset_index(drop=True)
    return np.stack(X), np.asarray(y, dtype=np.int32), used


def split_by_heavy(used: pd.DataFrame, y: np.ndarray, test_size: float = 0.2, seed: int = SEED):
    heavy_tbl = used.groupby("heavy", as_index=False).agg(label=(used.columns[-1], "mean"))
    heavy_tbl["label_bin"] = (heavy_tbl["label"] >= 0.5).astype(int)
    h_tr, h_te = train_test_split(
        heavy_tbl["heavy"].values,
        test_size=test_size,
        random_state=seed,
        stratify=heavy_tbl["label_bin"].values,
    )
    m_tr = used["heavy"].isin(set(h_tr)).to_numpy()
    m_te = used["heavy"].isin(set(h_te)).to_numpy()
    return m_tr, m_te


def train_eval_logreg(X_tr, y_tr, X_te, y_te):
    scaler = StandardScaler()
    X_trs = scaler.fit_transform(X_tr)
    X_tes = scaler.transform(X_te)

    clf = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        class_weight="balanced",
        random_state=SEED,
    )
    clf.fit(X_trs, y_tr)

    prob = clf.predict_proba(X_tes)[:, 1]
    pred = (prob >= 0.5).astype(int)

    acc = float(accuracy_score(y_te, pred))
    roc = float(roc_auc_score(y_te, prob)) if len(np.unique(y_te)) > 1 else float("nan")
    cm = confusion_matrix(y_te, pred)
    fpr, tpr, _ = roc_curve(y_te, prob)

    return {
        "acc": acc,
        "auc": roc,
        "cm": cm,
        "fpr": fpr,
        "tpr": tpr,
        "prob": prob,
        "pred": pred,
        "y_true": y_te,
        "model": clf,
        "scaler": scaler,
    }


def plot_class_balance(thermo: pd.DataFrame, expr: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    t_counts = thermo["label_tm"].value_counts().sort_index()
    e_counts = expr["label_expr"].value_counts().sort_index()

    ax[0].bar(["tm=0", "tm=1"], [t_counts.get(0, 0), t_counts.get(1, 0)])
    ax[0].set_title("Thermostability Label Balance")
    ax[0].set_ylabel("Count")

    ax[1].bar(["expr=0", "expr=1"], [e_counts.get(0, 0), e_counts.get(1, 0)])
    ax[1].set_title("Expression Label Balance")

    fig.suptitle("Dataset Class Balance", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_task_roc(res_tm: dict, res_ex: dict, out_png: Path) -> None:
    plt.figure(figsize=(6, 5))
    plt.plot(res_tm["fpr"], res_tm["tpr"], label=f"Thermo AUC={res_tm['auc']:.2f}")
    plt.plot(res_ex["fpr"], res_ex["tpr"], label=f"Expr AUC={res_ex['auc']:.2f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Task ROC Curves (Heavy-only)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_task_bars(res_tm: dict, res_ex: dict, out_png: Path) -> None:
    labels = ["Accuracy", "AUC"]
    tm_vals = [res_tm["acc"], res_tm["auc"]]
    ex_vals = [res_ex["acc"], res_ex["auc"]]
    x = np.arange(len(labels))
    w = 0.36

    plt.figure(figsize=(6.5, 4.2))
    plt.bar(x - w / 2, tm_vals, w, label="Thermostability")
    plt.bar(x + w / 2, ex_vals, w, label="Expression")
    for i, v in enumerate(tm_vals):
        plt.text(i - w / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=9)
    for i, v in enumerate(ex_vals):
        plt.text(i + w / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=9)
    plt.ylim(0, 1.0)
    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.title("Per-task Test Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def overlap_combined_eval(
    thermo: pd.DataFrame,
    expr: pd.DataFrame,
    seq2emb: Dict[str, np.ndarray],
) -> dict:
    h_overlap = sorted(set(thermo["heavy"]).intersection(set(expr["heavy"])))
    if len(h_overlap) < 20:
        return {"ok": False, "reason": "Overlap too small"}

    h_tr, h_te = train_test_split(h_overlap, test_size=0.3, random_state=SEED)

    # Exclude overlap test heavy from both training sets to avoid leakage.
    tm_train = thermo[~thermo["heavy"].isin(set(h_te))].copy()
    ex_train = expr[~expr["heavy"].isin(set(h_te))].copy()

    tm_test = thermo[thermo["heavy"].isin(set(h_te))].copy()
    ex_test = expr[expr["heavy"].isin(set(h_te))].copy()

    X_tm_tr, y_tm_tr, _ = build_Xy(tm_train[["heavy", "label_tm"]], seq2emb, "label_tm")
    X_ex_tr, y_ex_tr, _ = build_Xy(ex_train[["heavy", "label_expr"]], seq2emb, "label_expr")

    # Aggregate overlap test to one row per heavy for clean combine.
    tm_h = tm_test.groupby("heavy", as_index=False)["label_tm"].mean()
    tm_h["label_tm"] = (tm_h["label_tm"] >= 0.5).astype(int)
    ex_h = ex_test.groupby("heavy", as_index=False)["label_expr"].mean()
    ex_h["label_expr"] = (ex_h["label_expr"] >= 0.5).astype(int)

    both = tm_h.merge(ex_h, on="heavy", how="inner")
    if both.empty:
        return {"ok": False, "reason": "No overlap rows after aggregation"}

    X_both, _, used_both = build_Xy(both[["heavy", "label_tm"]], seq2emb, "label_tm")

    # Train two task models on separate full training data.
    sc_tm = StandardScaler().fit(X_tm_tr)
    clf_tm = LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced", random_state=SEED)
    clf_tm.fit(sc_tm.transform(X_tm_tr), y_tm_tr)

    sc_ex = StandardScaler().fit(X_ex_tr)
    clf_ex = LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced", random_state=SEED)
    clf_ex.fit(sc_ex.transform(X_ex_tr), y_ex_tr)

    p_tm = clf_tm.predict_proba(sc_tm.transform(X_both))[:, 1]
    p_ex = clf_ex.predict_proba(sc_ex.transform(X_both))[:, 1]

    table = used_both[["heavy"]].copy()
    table = table.merge(both, on="heavy", how="left")
    table["p_tm"] = p_tm
    table["p_expr"] = p_ex
    table["p_dev"] = table["p_tm"] * table["p_expr"]
    table["label_dev"] = ((table["label_tm"] == 1) & (table["label_expr"] == 1)).astype(int)

    y = table["label_dev"].values
    prob = table["p_dev"].values
    pred = (prob >= 0.5).astype(int)

    return {
        "ok": True,
        "n_overlap_test": int(len(table)),
        "dev_acc": float(accuracy_score(y, pred)),
        "dev_auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan"),
        "table": table,
    }


def plot_combined(comb: dict, out_roc: Path, out_hist: Path) -> None:
    tbl = comb["table"]
    y = tbl["label_dev"].values
    p = tbl["p_dev"].values
    if len(np.unique(y)) > 1:
        fpr, tpr, _ = roc_curve(y, p)
        rauc = auc(fpr, tpr)
        plt.figure(figsize=(5.8, 4.8))
        plt.plot(fpr, tpr, label=f"Developability AUC={rauc:.2f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Combined Developability ROC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_roc, dpi=180)
        plt.close()

    plt.figure(figsize=(6.2, 4.6))
    plt.hist(tbl.loc[tbl["label_dev"] == 1, "p_dev"], bins=20, alpha=0.7, label="dev=1")
    plt.hist(tbl.loc[tbl["label_dev"] == 0, "p_dev"], bins=20, alpha=0.7, label="dev=0")
    plt.xlabel("P(developable) = P(tm) * P(expr)")
    plt.ylabel("Count")
    plt.title("Developability Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_hist, dpi=180)
    plt.close()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "final_step" / "data"
    out_dir = root / "final_step" / "output"
    cache_npz = root / "final_step" / "cache" / "iglm_heavy_cache.npz"
    cache_map = root / "final_step" / "cache" / "iglm_heavy_cache_map.json"

    out_dir.mkdir(parents=True, exist_ok=True)

    thermo = pd.read_csv(data_dir / "thermo_binary_heavy.csv")
    expr = pd.read_csv(data_dir / "expression_binary_heavy.csv")

    # Build heavy embeddings once.
    all_heavy = np.unique(np.concatenate([thermo["heavy"].values, expr["heavy"].values]))
    seq2emb = load_cache(cache_npz, cache_map)
    seq2emb = import_existing_cache(seq2emb, root)
    seq2emb = compute_embeddings_if_needed(all_heavy, seq2emb, cache_npz, cache_map)

    # Thermo train/test.
    Xt, yt, ut = build_Xy(thermo[["heavy", "label_tm"]], seq2emb, "label_tm")
    ut = ut.assign(label_tm=yt)
    mt_tr, mt_te = split_by_heavy(ut, yt)
    res_tm = train_eval_logreg(Xt[mt_tr], yt[mt_tr], Xt[mt_te], yt[mt_te])

    # Expression train/test.
    Xe, ye, ue = build_Xy(expr[["heavy", "label_expr"]], seq2emb, "label_expr")
    ue = ue.assign(label_expr=ye)
    me_tr, me_te = split_by_heavy(ue, ye)
    res_ex = train_eval_logreg(Xe[me_tr], ye[me_tr], Xe[me_te], ye[me_te])

    # Combined overlap developability check.
    comb = overlap_combined_eval(thermo, expr, seq2emb)

    # Save metrics.
    metrics = {
        "thermo": {"acc": res_tm["acc"], "auc": res_tm["auc"], "n_train": int(mt_tr.sum()), "n_test": int(mt_te.sum())},
        "expression": {"acc": res_ex["acc"], "auc": res_ex["auc"], "n_train": int(me_tr.sum()), "n_test": int(me_te.sum())},
        "combined": {k: v for k, v in comb.items() if k != "table"},
    }
    (out_dir / "metrics_summary.json").write_text(json.dumps(metrics, indent=2))

    pd.DataFrame([
        {"task": "thermostability", "acc": res_tm["acc"], "auc": res_tm["auc"]},
        {"task": "expression", "acc": res_ex["acc"], "auc": res_ex["auc"]},
    ]).to_csv(out_dir / "task_metrics.csv", index=False)

    if comb.get("ok", False):
        comb["table"].to_csv(out_dir / "combined_overlap_predictions.csv", index=False)

    # Plots for presentation.
    plot_class_balance(thermo, expr, out_dir / "fig01_class_balance.png")
    plot_task_roc(res_tm, res_ex, out_dir / "fig02_task_roc.png")
    plot_task_bars(res_tm, res_ex, out_dir / "fig03_task_metrics.png")
    if comb.get("ok", False):
        plot_combined(comb, out_dir / "fig04_combined_dev_roc.png", out_dir / "fig05_combined_dev_hist.png")

    print("Done: final_step pipeline")
    print(json.dumps(metrics, indent=2))
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
