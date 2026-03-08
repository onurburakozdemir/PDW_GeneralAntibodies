#!/usr/bin/env python3
"""Prepare binary heavy-only datasets for thermostability and expression."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def prep_thermo(src_csv: Path, out_csv: Path, stats_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(src_csv)
    need = {"source_file", "heavy", "y"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Thermo CSV missing columns: {sorted(miss)}")

    out = df.copy()
    out["source_file"] = out["source_file"].astype(str)
    out["heavy"] = out["heavy"].fillna("").astype(str).str.strip().str.upper()
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out = out[(out["heavy"] != "") & out["y"].notna()].reset_index(drop=True)

    # Study-aware threshold to handle assay/domain offsets.
    med = out.groupby("source_file")["y"].median().rename("tm_median")
    out = out.join(med, on="source_file")
    out["label_tm"] = (out["y"] >= out["tm_median"]).astype(int)

    keep = out[["source_file", "heavy", "y", "tm_median", "label_tm"]].copy()
    keep.to_csv(out_csv, index=False)

    stats = keep.groupby("source_file", as_index=False).agg(
        rows=("label_tm", "size"),
        pos=("label_tm", "sum"),
        tm_median=("tm_median", "first"),
        y_mean=("y", "mean"),
        y_std=("y", "std"),
    )
    stats["pos_rate"] = stats["pos"] / stats["rows"]
    stats.to_csv(stats_csv, index=False)
    return keep


def prep_expression(src_csv: Path, out_csv: Path, stats_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(src_csv)
    need = {"dataset", "heavy", "expression_label", "fitness"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Expression CSV missing columns: {sorted(miss)}")

    out = df.copy()
    out["source_file"] = out["dataset"].astype(str)
    out["heavy"] = out["heavy"].fillna("").astype(str).str.strip().str.upper()
    out["fitness"] = pd.to_numeric(out["fitness"], errors="coerce")
    out["label_expr"] = pd.to_numeric(out["expression_label"], errors="coerce")

    out = out[(out["heavy"] != "") & out["label_expr"].isin([0, 1])].copy()
    out["label_expr"] = out["label_expr"].astype(int)

    keep = out[["source_file", "heavy", "fitness", "label_expr"]].reset_index(drop=True)
    keep.to_csv(out_csv, index=False)

    stats = keep.groupby("source_file", as_index=False).agg(
        rows=("label_expr", "size"),
        pos=("label_expr", "sum"),
        fitness_mean=("fitness", "mean"),
        fitness_std=("fitness", "std"),
    )
    stats["pos_rate"] = stats["pos"] / stats["rows"]
    stats.to_csv(stats_csv, index=False)
    return keep


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "final_step" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    thermo_src = root / "flab_thermo_unified_ml_tm_only.csv"
    expr_src = root / "data" / "raw" / "expression_data.csv"

    thermo_out = data_dir / "thermo_binary_heavy.csv"
    thermo_stats = data_dir / "thermo_binary_stats.csv"
    expr_out = data_dir / "expression_binary_heavy.csv"
    expr_stats = data_dir / "expression_binary_stats.csv"

    t = prep_thermo(thermo_src, thermo_out, thermo_stats)
    e = prep_expression(expr_src, expr_out, expr_stats)

    print("Prepared datasets")
    print(f"thermo rows={len(t)} studies={t['source_file'].nunique()} -> {thermo_out}")
    print(f"expression rows={len(e)} studies={e['source_file'].nunique()} -> {expr_out}")


if __name__ == "__main__":
    main()
