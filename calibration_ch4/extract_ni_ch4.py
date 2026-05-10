#!/usr/bin/env python3
"""
extract_ni_ch4.py

Read the BICEFS chamber flux dataset (edi.1828.3) and produce two clean CSVs:

  data/monthly_ch4_ni.csv
      Monthly mean CH4 flux by chamber type (μmol CH4 m-2 s-1).
      Columns: month, chamber_type, ch4_mean, ch4_std, ch4_se, n_chambers,
               co2_mean, co2_std, orpN_mean (N = 10,20,30,40 cm)

  data/plant_transport_ni.csv
      Per-month ratio of root+shoot to root-only CH4 flux.
      Constrains ch4_plant_transport_factor (= ratio - 1).
      Columns: month, flux_total, flux_root, ratio, n_pairs

Usage
-----
  python extract_ni_ch4.py                  # all defaults
  python extract_ni_ch4.py --r2-min 0.85   # looser R2 filter
  python extract_ni_ch4.py --no-filter      # keep all measurements
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

_LTER_ROOT = Path(__file__).parent.parent / "lter_data"
_FLUX_FILE = _LTER_ROOT / "edi.1828.3" / "241206_BICEFS_flux_data.csv"
_OUT_DIR = Path(__file__).parent / "data"

# Raw column names in the BICEFS file (strip BOM if present)
_COL_SITE = "site"
_COL_DATE = "date"
_COL_TYPE = "chamber type"
_COL_CH4  = "CH4 flux (\U0001d707mol CH4 m-2 s-1)"   # Unicode mu
_COL_CH4R = "R2 value for CH4 flux"
_COL_CO2  = "CO2 flux (\U0001d707mol CO2 m-2 s-1)"
_COL_ORP  = {10: "oxidation reduction potential at 10 cm depth (mv)",
             20: "oxidation reduction potential at 20 cm depth (mv)",
             30: "oxidation reduction potential at 30 cm depth (mv)",
             40: "oxidation reduction potential at 40 cm depth (mv)"}


def _load_raw(flux_file: Path) -> pd.DataFrame:
    df = pd.read_csv(flux_file, encoding="utf-8-sig")
    # Normalise column names — strip BOM, leading/trailing spaces
    df.columns = [c.strip() for c in df.columns]
    return df


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first column name in df that matches any candidate."""
    for col in candidates:
        if col in df.columns:
            return col
    # Fall back: case-insensitive substring search
    for col in df.columns:
        for cand in candidates:
            if cand.lower() in col.lower():
                return col
    raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")


def load_bicefs(flux_file: Path = _FLUX_FILE, r2_min: float = 0.9,
                apply_filter: bool = True) -> pd.DataFrame:
    """Load and clean BICEFS flux data.

    Returns DataFrame with standardised column names and only salt-marsh rows.
    """
    df = _load_raw(flux_file)

    # Locate the CH4 column — the μ character can be encoded several ways
    ch4_col = _find_column(df, [
        "CH4 flux",
        "μmol CH4",
        "\U0001d707mol CH4",
        "CH4 flux (μmol CH4 m-2 s-1)",
    ])
    r2_col = _find_column(df, ["R2 value for CH4 flux", "R2 value for CH4"])
    co2_col = _find_column(df, ["CO2 flux", "μmol CO2", "CO2"])
    type_col = _find_column(df, ["chamber type", "Chamber type"])

    # ORP columns — try both exact and partial matches
    orp_cols = {}
    for depth in (10, 20, 30, 40):
        try:
            orp_cols[depth] = _find_column(df, [
                f"oxidation reduction potential at {depth} cm depth (mv)",
                f"ORP {depth}",
                f"oxidation reduction potential at {depth}",
            ])
        except KeyError:
            orp_cols[depth] = None

    # Parse date (YYYYMMDD integer → datetime)
    date_col = _find_column(df, ["date", "Date"])
    df["date_dt"] = pd.to_datetime(df[date_col].astype(str), format="%Y%m%d",
                                   errors="coerce")
    df["month"] = df["date_dt"].dt.month

    # Normalise chamber type labels
    df["chamber_type"] = (
        df[type_col].str.strip().str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )
    # Canonical labels: "root_shoot" and "root_only"
    df["chamber_type"] = df["chamber_type"].map(
        lambda s: "root_shoot" if "shoot" in s else "root_only"
    )

    # Rename key flux columns
    df = df.rename(columns={
        ch4_col: "ch4_flux",
        r2_col: "r2_ch4",
        co2_col: "co2_flux",
    })
    for depth, col in orp_cols.items():
        if col is not None:
            df = df.rename(columns={col: f"orp_{depth}cm"})
        else:
            df[f"orp_{depth}cm"] = np.nan

    # Coerce numeric
    for c in ["ch4_flux", "r2_ch4", "co2_flux",
              "orp_10cm", "orp_20cm", "orp_30cm", "orp_40cm"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Filter to salt marsh site only
    site_col = _find_column(df, ["site", "Site"])
    df = df[df[site_col].str.lower().str.contains("salt marsh")].copy()

    # Quality filter
    if apply_filter:
        df = df[df["r2_ch4"] >= r2_min].copy()

    return df


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Return monthly mean CH4 flux by chamber type."""
    groups = []
    for (month, ctype), grp in df.groupby(["month", "chamber_type"]):
        ch4 = grp["ch4_flux"].dropna()
        co2 = grp["co2_flux"].dropna()
        rec = {
            "month": month,
            "chamber_type": ctype,
            "ch4_mean": ch4.mean(),
            "ch4_std":  ch4.std(),
            "ch4_se":   ch4.sem(),
            "ch4_min":  ch4.min(),
            "ch4_max":  ch4.max(),
            "n_chambers": len(ch4),
            "co2_mean": co2.mean(),
            "co2_std":  co2.std(),
        }
        for depth in (10, 20, 30, 40):
            col = f"orp_{depth}cm"
            if col in grp.columns:
                rec[f"orp{depth}_mean"] = grp[col].mean()
                rec[f"orp{depth}_std"] = grp[col].std()
        groups.append(rec)

    return pd.DataFrame(groups).sort_values(["month", "chamber_type"]).reset_index(drop=True)


def plant_transport_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the root+shoot / root-only ratio per month.

    This directly constrains ch4_plant_transport_factor = ratio - 1,
    since the model defines: total_flux = diffusive × (1 + beta_plant).
    """
    rs = (df[df["chamber_type"] == "root_shoot"]
          .groupby("month")["ch4_flux"].agg(flux_total="mean", n="count")
          .reset_index())
    ro = (df[df["chamber_type"] == "root_only"]
          .groupby("month")["ch4_flux"].agg(flux_root="mean")
          .reset_index())

    merged = rs.merge(ro, on="month", how="inner")
    merged["ratio"] = merged["flux_total"] / merged["flux_root"].replace(0, np.nan)
    merged["beta_plant"] = merged["ratio"] - 1.0

    return merged.rename(columns={"n": "n_pairs"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--r2-min", type=float, default=0.90,
                        help="Minimum R2 for CH4 flux acceptance (default: 0.90)")
    parser.add_argument("--no-filter", action="store_true",
                        help="Skip R2 quality filter")
    parser.add_argument("--out-dir", default=str(_OUT_DIR),
                        help=f"Output directory (default: {_OUT_DIR})")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {_FLUX_FILE}")
    df = load_bicefs(r2_min=args.r2_min, apply_filter=not args.no_filter)
    print(f"  {len(df)} rows retained after filtering  "
          f"(site=Salt marsh, R2>={args.r2_min if not args.no_filter else 'off'})")
    print(f"  Chamber types: {df['chamber_type'].value_counts().to_dict()}")
    print(f"  Months with data: {sorted(df['month'].unique())}")

    monthly = aggregate_monthly(df)
    out_monthly = out_dir / "monthly_ch4_ni.csv"
    monthly.to_csv(out_monthly, index=False)
    print(f"\nMonthly CH4 summary written to {out_monthly}")
    print(monthly.to_string(index=False))

    ratio = plant_transport_ratio(df)
    out_ratio = out_dir / "plant_transport_ni.csv"
    ratio.to_csv(out_ratio, index=False)
    print(f"\nPlant transport ratio written to {out_ratio}")
    if not ratio.empty:
        mean_beta = ratio["beta_plant"].mean()
        print(f"  Mean ch4_plant_transport_factor (beta_plant): {mean_beta:.2f}")
        print(f"  (= total/diffusive ratio {ratio['ratio'].mean():.2f} minus 1)")
        print(ratio[["month","flux_total","flux_root","ratio","beta_plant"]].to_string(index=False))
    else:
        print("  No months have both root+shoot and root-only measurements.")


if __name__ == "__main__":
    main()
