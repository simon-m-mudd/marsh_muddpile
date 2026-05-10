#!/usr/bin/env python3
"""
extract_ni_porewater.py

Read the NILTREB porewater dataset (edi.136.11) and produce:

  data/porewater_profiles_ni.csv
      Mean ± SD porewater chemistry by DEPTH for each site/location
      combination. Columns: site, location, depth_cm, nh4_mean, nh4_std,
      s2_mean, s2_std, salinity_mean, salinity_std, n_obs.

  data/smtz_depth_ni.csv
      Estimated sulfate-methane transition zone (SMTZ) depth (cm) per
      site/location as the shallowest depth where mean S2- exceeds a
      detection threshold. Constrains ch4_flushing_depth_scale_m and
      ch4_km_so4_umol_L.

Usage
-----
  python extract_ni_porewater.py
  python extract_ni_porewater.py --s2-threshold 2.0  # μmol/L
  python extract_ni_porewater.py --sites OL GI        # subset of sites
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_LTER_ROOT = Path(__file__).parent.parent / "lter_data"
_PW_FILE = _LTER_ROOT / "edi.136.11" / "NILTREB_porewater.csv"
_OUT_DIR = Path(__file__).parent / "data"

# S2- above this threshold (μmol/L) indicates SO4 depletion (active SRB)
_S2_THRESHOLD_DEFAULT = 2.0

# Sites and locations to include in calibration target
_CAL_SITES = ["OL", "GI"]
_CAL_LOCATIONS = ["HM"]   # high marsh matches BICEFS chamber plots


def load_porewater(pw_file: Path = _PW_FILE,
                   sites: list[str] | None = None,
                   locations: list[str] | None = None) -> pd.DataFrame:
    """Load and clean NILTREB porewater data.

    Returns DataFrame with numeric NH4, S2, SALINITY columns. Flags and
    'NM' (not measured) entries are replaced with NaN.
    """
    df = pd.read_csv(pw_file, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["DEPTH"] = pd.to_numeric(df["DEPTH"], errors="coerce")

    # Remove flag columns (e.g. NH4_flag) — keep only the measurement columns
    flag_cols = [c for c in df.columns if c.endswith("_flag")]
    df = df.drop(columns=flag_cols, errors="ignore")

    # Coerce 'NM' and other non-numeric strings to NaN
    for col in ["NH4", "S2", "SALINITY", "PO4", "Fe2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter to requested sites/locations
    if sites:
        df = df[df["SITE"].isin(sites)].copy()
    if locations:
        df = df[df["LOCATION"].isin(locations)].copy()

    # Drop rows with no valid depth or date
    df = df.dropna(subset=["DEPTH", "DATE"])

    return df.reset_index(drop=True)


def depth_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ± SD per SITE × LOCATION × DEPTH."""
    records = []
    for (site, loc, depth), grp in df.groupby(["SITE", "LOCATION", "DEPTH"]):
        rec = {"site": site, "location": loc, "depth_cm": depth}
        for col, label in [("NH4", "nh4"), ("S2", "s2"), ("SALINITY", "salinity")]:
            vals = grp[col].dropna() if col in grp.columns else pd.Series([], dtype=float)
            rec[f"{label}_mean"] = vals.mean() if len(vals) else np.nan
            rec[f"{label}_std"]  = vals.std()  if len(vals) else np.nan
            rec[f"{label}_se"]   = vals.sem()  if len(vals) else np.nan
            rec[f"{label}_n"]    = len(vals)
        records.append(rec)

    out = pd.DataFrame(records).sort_values(["site", "location", "depth_cm"])
    return out.reset_index(drop=True)


def smtz_depths(profiles: pd.DataFrame, s2_threshold: float = _S2_THRESHOLD_DEFAULT) -> pd.DataFrame:
    """Estimate SMTZ depth from S2- profiles.

    SMTZ depth = shallowest depth_cm at which mean S2- > threshold.
    If S2- never exceeds threshold the SMTZ is deeper than the measurement range.

    The model should reproduce SO4 ≈ 0 at this depth (modeled SMTZ
    matches observed SMTZ when modeled SO4 drops below ~10% of creek).
    """
    records = []
    for (site, loc), grp in profiles.groupby(["site", "location"]):
        grp_sorted = grp.sort_values("depth_cm")
        above = grp_sorted[grp_sorted["s2_mean"] > s2_threshold]
        smtz = float(above["depth_cm"].min()) if not above.empty else np.nan

        # Deepest depth with S2- data for context
        measured_to = grp_sorted.dropna(subset=["s2_mean"])["depth_cm"].max()

        records.append({
            "site": site,
            "location": loc,
            "smtz_depth_cm": smtz,
            "s2_threshold_umol_L": s2_threshold,
            "measured_to_cm": measured_to,
            "smtz_below_range": np.isnan(smtz),
        })

    return pd.DataFrame(records)


def seasonal_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean depth profiles by calendar month for seasonal comparison."""
    df = df.copy()
    df["month"] = df["DATE"].dt.month
    records = []
    for (site, loc, month, depth), grp in df.groupby(
            ["SITE", "LOCATION", "month", "DEPTH"]):
        rec = {"site": site, "location": loc, "month": month, "depth_cm": depth}
        for col, label in [("NH4", "nh4"), ("S2", "s2"), ("SALINITY", "salinity")]:
            vals = grp[col].dropna() if col in grp.columns else pd.Series([], dtype=float)
            rec[f"{label}_mean"] = vals.mean() if len(vals) else np.nan
            rec[f"{label}_n"] = len(vals)
        records.append(rec)
    out = pd.DataFrame(records).sort_values(["site", "location", "month", "depth_cm"])
    return out.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--s2-threshold", type=float, default=_S2_THRESHOLD_DEFAULT,
                        help=f"S2- threshold for SMTZ detection (μmol/L, default: {_S2_THRESHOLD_DEFAULT})")
    parser.add_argument("--sites", nargs="+", default=_CAL_SITES,
                        help=f"Sites to include (default: {_CAL_SITES})")
    parser.add_argument("--locations", nargs="+", default=_CAL_LOCATIONS,
                        help=f"Locations to include (default: {_CAL_LOCATIONS})")
    parser.add_argument("--out-dir", default=str(_OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {_PW_FILE}")
    df = load_porewater(sites=args.sites, locations=args.locations)
    print(f"  {len(df)} rows  ({df['SITE'].nunique()} sites, "
          f"{df['DATE'].dt.year.nunique()} years, "
          f"{df['DEPTH'].nunique()} depths: {sorted(df['DEPTH'].unique())} cm)")

    # ---- Overall depth profiles ----
    profiles = depth_profiles(df)
    out_prof = out_dir / "porewater_profiles_ni.csv"
    profiles.to_csv(out_prof, index=False)
    print(f"\nDepth profiles written to {out_prof}")
    print(profiles[["site","location","depth_cm",
                     "nh4_mean","s2_mean","salinity_mean","nh4_n"]].to_string(index=False))

    # ---- SMTZ depth ----
    smtz = smtz_depths(profiles, s2_threshold=args.s2_threshold)
    out_smtz = out_dir / "smtz_depth_ni.csv"
    smtz.to_csv(out_smtz, index=False)
    print(f"\nSMTZ depths written to {out_smtz}")
    print(smtz.to_string(index=False))

    # ---- Seasonal profiles ----
    seas = seasonal_profiles(df)
    out_seas = out_dir / "seasonal_porewater_ni.csv"
    seas.to_csv(out_seas, index=False)
    print(f"\nSeasonal profiles written to {out_seas}")

    # ---- Summary for calibration ----
    print("\nCalibration summary:")
    for _, row in smtz.iterrows():
        if not row["smtz_below_range"]:
            print(f"  {row['site']} {row['location']}: "
                  f"SMTZ at {row['smtz_depth_cm']:.0f} cm  "
                  f"(S2- > {args.s2_threshold} μmol/L)")
        else:
            print(f"  {row['site']} {row['location']}: "
                  f"SMTZ deeper than {row['measured_to_cm']:.0f} cm "
                  f"(S2- never exceeds threshold)")


if __name__ == "__main__":
    main()
