#!/usr/bin/env python3
"""Unify FLAb thermostability CSV files into one ML-ready CSV."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def clean_sequence(s: object) -> str | None:
    if pd.isna(s):
        return None
    seq = re.sub(r"[^A-Za-z]", "", str(s)).upper()
    return seq if seq else None


def clean_numeric(s: object) -> float | None:
    if pd.isna(s):
        return None
    txt = re.sub(r"[^0-9eE+\-.]", "", str(s))
    if not txt:
        return None
    try:
        return float(txt)
    except ValueError:
        return None


def pick_target_column(columns: list[str]) -> str | None:
    for c in columns:
        lc = c.lower()
        if c == "fitness":
            continue
        if any(k in lc for k in ("tm", "tm1", "tm2", "dsf", "dls", "melt", "temperature", "thermal")):
            return c
    return "fitness" if "fitness" in columns else None


def main() -> None:
    root = Path("data/raw/flab_thermostability")
    out_path = Path("flab_thermo_unified_ml.csv")
    csv_files = sorted(root.glob("*.csv"))

    rows: list[dict] = []
    for path in csv_files:
        df = pd.read_csv(path)
        cols = [str(c) for c in df.columns]

        heavy_col = "heavy" if "heavy" in cols else None
        light_col = "light" if "light" in cols else None
        target_col = pick_target_column(cols)

        if heavy_col is None or target_col is None:
            continue

        for _, rec in df.iterrows():
            heavy = clean_sequence(rec.get(heavy_col))
            light = clean_sequence(rec.get(light_col)) if light_col else None
            y = clean_numeric(rec.get(target_col))

            if heavy is None or y is None:
                continue

            rows.append(
                {
                    "source_file": path.name,
                    "assay_name": target_col,
                    "heavy": heavy,
                    "light": light,
                    "y": y,
                }
            )

    out = pd.DataFrame(rows, columns=["source_file", "assay_name", "heavy", "light", "y"])
    out.to_csv(out_path, index=False)

    n_missing_light = int(out["light"].isna().sum()) if not out.empty else 0
    print(f"Wrote: {out_path}")
    print(f"Rows: {len(out)}")
    print(f"Files used: {out['source_file'].nunique() if not out.empty else 0}")
    print(f"Rows with missing light chain: {n_missing_light}")


if __name__ == "__main__":
    main()
