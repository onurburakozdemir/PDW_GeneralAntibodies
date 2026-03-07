#!/usr/bin/env python3
"""Build a unified FLAb expression-fitness CSV from raw expression studies."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--expression_dir",
        type=Path,
        default=Path("data/raw/FLAb/data/expression"),
        help="Directory containing expression CSV files.",
    )
    p.add_argument(
        "--out_csv",
        type=Path,
        default=Path("expression_fitness/expression_unified_fitness.csv"),
        help="Output unified CSV path.",
    )
    p.add_argument(
        "--out_profile",
        type=Path,
        default=Path("expression_fitness/expression_unified_profile.csv"),
        help="Output profile/summary CSV path.",
    )
    return p.parse_args()


def find_target_column(df: pd.DataFrame) -> str:
    """Find best target column; prefer explicit fitness if present."""
    cols = {c.lower(): c for c in df.columns}
    if "fitness" in cols:
        return cols["fitness"]

    # fallback heuristics for expression readouts
    candidates = [c for c in df.columns if any(k in c.lower() for k in ["expression", "titer", "enrichment", "er"])]
    if not candidates:
        raise ValueError("Could not find target column (fitness/expression/titer/enrichment)")
    return candidates[0]


def normalize_sequences(s: pd.Series) -> pd.Series:
    """Uppercase and strip spaces; keep amino-acid letters only where possible."""
    s = s.fillna("").astype(str).str.strip().str.upper()
    # Keep letters only; remove separators or unexpected punctuation.
    s = s.str.replace(r"[^A-Z]", "", regex=True)
    return s


def load_one_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    need = {"heavy", "light"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"{path.name}: missing required sequence columns: {sorted(miss)}")

    y_col = find_target_column(df)

    out = pd.DataFrame(
        {
            "source_file": path.name,
            "heavy": normalize_sequences(df["heavy"]),
            "light": normalize_sequences(df["light"]),
            "y": pd.to_numeric(df[y_col], errors="coerce"),
            "y_column": y_col,
            "assay_name": y_col,
            "format": df["format"].astype(str) if "format" in df.columns else np.nan,
        }
    )

    # Keep only rows with valid sequences and numeric target.
    out = out[(out["heavy"] != "") & (out["light"] != "") & out["y"].notna()].reset_index(drop=True)
    return out


def main() -> None:
    args = parse_args()

    files = sorted([p for p in args.expression_dir.glob("*.csv") if p.name.lower() != "readme.md"])
    if not files:
        raise FileNotFoundError(f"No CSV files found under {args.expression_dir}")

    frames = []
    for p in files:
        frames.append(load_one_csv(p))

    unified = pd.concat(frames, ignore_index=True)
    unified["pair_id"] = unified["heavy"] + "|" + unified["light"]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    unified.to_csv(args.out_csv, index=False)

    profile = (
        unified.groupby("source_file")
        .agg(
            n_rows=("y", "size"),
            n_unique_heavy=("heavy", "nunique"),
            n_unique_light=("light", "nunique"),
            n_unique_pairs=("pair_id", "nunique"),
            y_mean=("y", "mean"),
            y_std=("y", "std"),
            y_min=("y", "min"),
            y_max=("y", "max"),
        )
        .reset_index()
        .sort_values("n_rows", ascending=False)
    )
    profile.to_csv(args.out_profile, index=False)

    print(f"Wrote unified CSV: {args.out_csv}")
    print(f"Wrote profile CSV: {args.out_profile}")
    print(f"Rows: {len(unified)}")
    print(f"Studies: {unified['source_file'].nunique()}")
    print(f"Unique heavy: {unified['heavy'].nunique()}")
    print(f"Unique light: {unified['light'].nunique()}")
    print(f"Unique pairs: {unified['pair_id'].nunique()}")


if __name__ == "__main__":
    main()
