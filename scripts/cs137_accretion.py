"""
cs137_accretion.py

Screen every marsh core in the CCN synthesis that has ¹³⁷Cs depth-profile
data for a distinct sub-surface peak, then compute a mean accretion rate
under the assumption that the peak records the 1963 maximum atmospheric
nuclear-weapons fallout.

Reference year
--------------
REFERENCE_YEAR = 1963
  The global maximum of ¹³⁷Cs deposition from atmospheric testing occurred
  in 1963 (after the 1963 Partial Nuclear Test Ban Treaty). The depth of the
  peak in a continuous sediment record therefore equals the thickness
  accumulated between 1963 and the core collection year.

  accretion_rate_mm_yr = depth_of_peak_midpoint_cm × 10
                         ─────────────────────────────────
                            collection_year − 1963

Peak-detection criteria (all must be satisfied)
------------------------------------------------
MIN_LAYERS = 4
  Minimum number of non-NaN ¹³⁷Cs measurements required to attempt
  peak detection. Fewer layers cannot reliably distinguish a peak from noise.

MIN_PEAK_BACKGROUND_RATIO = 5.0
  peak_activity / background_activity ≥ 5
  Background is the median activity of the deepest 25 % of layers
  (at least 2 layers). A factor-of-five elevation above background
  is the standard criterion used in the geochronology literature (e.g.
  Sanchez-Cabeza & Ruiz-Fernández 2012, Science of the Total Environment).

MIN_PEAK_SURFACE_RATIO = 2.0
  peak_activity / surface_activity ≥ 2
  The peak must stand at least twice the near-surface activity so that a
  monotonically decreasing profile (no true sub-surface peak) is rejected.
  Surface activity = median of the shallowest 2 layers.

PEAK_MUST_BE_SUBSURFACE = True
  The peak layer must NOT be the shallowest layer. If the maximum occurs at
  the surface the 1963 horizon cannot be resolved from the data.

Notes
-----
- ¹³⁷Cs activities are reported in many different units across CCN studies
  (Bq/kg, dpm/g, pCi/g, etc.). All thresholds are expressed as dimensionless
  ratios, so no unit conversion is needed.
- Some cores in the CCN dataset already carry a cs137_peak_age = 1963 flag
  set by the original investigators. Where available, this is preserved in
  the output as `investigator_peak_age` for comparison with the automated
  result.
- Cores collected in or before 1963 are excluded (negative or zero elapsed
  time). Cores with no collection year are also excluded.
- Depth values are in cm. Where depth_min / depth_max are missing the row
  is skipped for peak-location purposes.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REFERENCE_YEAR = 1963  # peak atmospheric ¹³⁷Cs fallout (weapons testing)

# Peak-detection thresholds
MIN_LAYERS                = 4    # minimum non-NaN ¹³⁷Cs measurements per core
MIN_PEAK_BACKGROUND_RATIO = 5.0  # peak / deepest-25%-median must exceed this
MIN_PEAK_SURFACE_RATIO    = 2.0  # peak / shallowest-2-layer-median must exceed this
PEAK_MUST_BE_SUBSURFACE   = True # True → peak cannot be the shallowest layer

# Marsh filter (matches plot_core_depth_profiles.py)
_MARSH_VEG_CLASSES  = {"emergent", "emergent marsh", "marsh"}
_EXCLUDED_SAL       = {"fresh", "oligohaline"}

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

_CCN_DIR = Path(__file__).parent.parent / "core_data" / "CCN_synthesis"


def _load_data(ccn_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cores = pd.read_csv(ccn_dir / "CCN_cores.csv", low_memory=False)
    ds    = pd.read_csv(ccn_dir / "CCN_depthseries.csv", low_memory=False)
    return cores, ds


def _filter_marsh_cores(cores: pd.DataFrame) -> pd.DataFrame:
    veg = cores["vegetation_class"].str.strip().str.lower()
    sal = cores["salinity_class"].str.strip().str.lower()
    veg_ok = veg.isin(_MARSH_VEG_CLASSES)
    sal_ok = sal.isna() | (~sal.isin(_EXCLUDED_SAL))
    return cores[veg_ok & sal_ok].copy()


# ---------------------------------------------------------------------------
# Peak detection
# ---------------------------------------------------------------------------

def _detect_peak(profile: pd.DataFrame,
                 peak_bg_ratio: float = MIN_PEAK_BACKGROUND_RATIO,
                 peak_surf_ratio: float = MIN_PEAK_SURFACE_RATIO) -> dict | None:
    """Return peak info dict or None if no distinct peak found.

    `profile` must be sorted by depth_midpoint and have columns:
        depth_midpoint, cs137_activity
    """
    p = profile.dropna(subset=["cs137_activity", "depth_midpoint"]).copy()
    p = p.sort_values("depth_midpoint").reset_index(drop=True)

    if len(p) < MIN_LAYERS:
        return None

    acts = p["cs137_activity"].values
    depths = p["depth_midpoint"].values

    # Background: median of deepest 25 % (at least 2 layers)
    n_bg = max(2, int(np.ceil(len(p) * 0.25)))
    background = np.median(acts[-n_bg:])
    if background <= 0:
        background = 1e-30  # near-zero floor avoids divide-by-zero

    # Surface: median of shallowest 2 layers
    surface = np.median(acts[:2])
    if surface <= 0:
        surface = 1e-30

    # Peak
    peak_idx = int(np.argmax(acts))
    peak_act = acts[peak_idx]
    peak_depth_cm = depths[peak_idx]

    # Criteria
    if PEAK_MUST_BE_SUBSURFACE and peak_idx == 0:
        return None
    if peak_act / background < peak_bg_ratio:
        return None
    if peak_act / surface < peak_surf_ratio:
        return None

    return {
        "peak_depth_cm":      peak_depth_cm,
        "peak_activity":      peak_act,
        "background_activity": background,
        "surface_activity":   surface,
        "peak_background_ratio": peak_act / background,
        "peak_surface_ratio":    peak_act / surface,
        "n_layers":           len(p),
        "peak_layer_index":   peak_idx,
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyse(ccn_dir: Path, outdir: Path,
            peak_bg_ratio: float = MIN_PEAK_BACKGROUND_RATIO,
            peak_surf_ratio: float = MIN_PEAK_SURFACE_RATIO) -> pd.DataFrame:
    outdir.mkdir(parents=True, exist_ok=True)

    cores, ds = _load_data(ccn_dir)
    marsh_cores = _filter_marsh_cores(cores)
    marsh_ids   = set(marsh_cores["core_id"])

    # Keep only marsh cores with ≥1 cs137 measurement
    cs_ds = ds[ds["core_id"].isin(marsh_ids) & ds["cs137_activity"].notna()].copy()
    print(f"Marsh cores with ≥1 ¹³⁷Cs measurement: {cs_ds['core_id'].nunique()}")

    # Depth midpoints
    cs_ds["depth_midpoint"] = (
        cs_ds["depth_min"].fillna(cs_ds["depth_max"]) +
        cs_ds["depth_max"].fillna(cs_ds["depth_min"])
    ) / 2.0

    # Metadata joined from cores table
    meta_cols = ["core_id", "latitude", "longitude", "year",
                 "vegetation_class", "salinity_class", "study_id"]
    meta_cols = [c for c in meta_cols if c in marsh_cores.columns]
    meta = marsh_cores[meta_cols].drop_duplicates("core_id")

    # Investigator-assigned peak age (where present)
    inv_peak = (
        ds[ds["cs137_peak_age"].notna()]
        .groupby("core_id")["cs137_peak_age"]
        .first()
        .reset_index()
        .rename(columns={"cs137_peak_age": "investigator_peak_age"})
    )

    results = []
    skipped_no_year = 0
    skipped_year_le_ref = 0
    skipped_no_peak = 0

    for core_id, grp in cs_ds.groupby("core_id"):
        core_meta = meta[meta["core_id"] == core_id]
        if core_meta.empty:
            skipped_no_year += 1
            continue

        collection_year = core_meta["year"].values[0]
        if pd.isna(collection_year):
            skipped_no_year += 1
            continue
        collection_year = int(collection_year)
        if collection_year <= REFERENCE_YEAR:
            skipped_year_le_ref += 1
            continue

        elapsed_yr = collection_year - REFERENCE_YEAR

        peak = _detect_peak(grp, peak_bg_ratio, peak_surf_ratio)
        if peak is None:
            skipped_no_peak += 1
            results.append({
                "core_id":            core_id,
                "latitude":           core_meta["latitude"].values[0] if "latitude" in core_meta else np.nan,
                "longitude":          core_meta["longitude"].values[0] if "longitude" in core_meta else np.nan,
                "collection_year":    collection_year,
                "distinct_peak":      False,
                "peak_depth_cm":      np.nan,
                "accretion_mm_yr":    np.nan,
                "peak_background_ratio": np.nan,
                "peak_surface_ratio":    np.nan,
                "n_cs137_layers":     grp["cs137_activity"].notna().sum(),
                "study_id":           core_meta["study_id"].values[0] if "study_id" in core_meta else "",
            })
            continue

        accretion_mm_yr = peak["peak_depth_cm"] * 10.0 / elapsed_yr

        results.append({
            "core_id":               core_id,
            "latitude":              core_meta["latitude"].values[0] if "latitude" in core_meta else np.nan,
            "longitude":             core_meta["longitude"].values[0] if "longitude" in core_meta else np.nan,
            "collection_year":       collection_year,
            "distinct_peak":         True,
            "peak_depth_cm":         peak["peak_depth_cm"],
            "accretion_mm_yr":       accretion_mm_yr,
            "peak_background_ratio": peak["peak_background_ratio"],
            "peak_surface_ratio":    peak["peak_surface_ratio"],
            "n_cs137_layers":        peak["n_layers"],
            "study_id":              core_meta["study_id"].values[0] if "study_id" in core_meta else "",
        })

    df = pd.DataFrame(results)
    df = df.merge(inv_peak, on="core_id", how="left")

    n_total   = len(df)
    n_peaked  = df["distinct_peak"].sum()
    n_no_peak = n_total - n_peaked

    print(f"\nCores analysed:          {n_total}")
    print(f"  Distinct peak found:   {n_peaked}")
    print(f"  No distinct peak:      {n_no_peak}")
    print(f"  Skipped (no year):     {skipped_no_year}")
    print(f"  Skipped (year ≤ 1963): {skipped_year_le_ref}")

    peaked = df[df["distinct_peak"]].copy()
    if not peaked.empty:
        print(f"\nAccretion rates (mm/yr) for cores with distinct peak:")
        print(f"  Median:  {peaked['accretion_mm_yr'].median():.1f}")
        print(f"  Mean:    {peaked['accretion_mm_yr'].mean():.1f}")
        print(f"  Std dev: {peaked['accretion_mm_yr'].std():.1f}")
        print(f"  5th pct: {peaked['accretion_mm_yr'].quantile(0.05):.1f}")
        print(f"  95th pct:{peaked['accretion_mm_yr'].quantile(0.95):.1f}")

    # Save CSV
    csv_path = outdir / "cs137_accretion_rates.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Histogram
    _plot_histogram(peaked, outdir)

    return df


def _plot_histogram(peaked: pd.DataFrame, outdir: Path):
    if peaked.empty:
        return

    # Clip extreme outliers for display (keep 1–99th pct)
    lo, hi = peaked["accretion_mm_yr"].quantile([0.01, 0.99])
    clipped = peaked["accretion_mm_yr"].clip(lo, hi)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: histogram
    ax = axes[0]
    ax.hist(clipped, bins=40, color="#2a6496", edgecolor="white", linewidth=0.4)
    ax.axvline(peaked["accretion_mm_yr"].median(), color="k", linewidth=1.2,
               linestyle="--", label=f"Median {peaked['accretion_mm_yr'].median():.1f} mm/yr")
    ax.set_xlabel("Accretion rate (mm yr⁻¹)")
    ax.set_ylabel("Number of cores")
    ax.set_title(f"¹³⁷Cs accretion rates\n"
                 f"n = {len(peaked)} cores with distinct peak (of {len(peaked) + (~peaked['distinct_peak'].values.sum() if False else 0)} total)")
    ax.legend(fontsize=8)

    # Right: accretion vs. collection year
    ax2 = axes[1]
    sc = ax2.scatter(peaked["collection_year"], peaked["accretion_mm_yr"],
                     c=peaked["peak_depth_cm"], cmap="viridis", s=15, alpha=0.6,
                     vmin=0, vmax=peaked["peak_depth_cm"].quantile(0.95))
    cbar = fig.colorbar(sc, ax=ax2)
    cbar.set_label("Peak depth (cm)")
    ax2.set_xlabel("Core collection year")
    ax2.set_ylabel("Accretion rate (mm yr⁻¹)")
    ax2.set_title("Accretion rate vs. collection year")
    ax2.set_ylim(0, max(hi * 1.05, 5))

    fig.tight_layout()
    fig_path = outdir / "cs137_accretion_rates.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute ¹³⁷Cs-based accretion rates for CCN marsh cores."
    )
    parser.add_argument(
        "--ccn-dir",
        default=str(_CCN_DIR),
        help="Directory containing CCN_cores.csv and CCN_depthseries.csv "
             f"(default: {_CCN_DIR})",
    )
    parser.add_argument(
        "--outdir",
        default=str(Path(__file__).parent.parent / "scripts" / "figures"),
        help="Output directory for CSV and figure (default: scripts/figures/)",
    )
    parser.add_argument(
        "--min-peak-background-ratio",
        type=float, default=MIN_PEAK_BACKGROUND_RATIO,
        help=f"Peak/background threshold (default: {MIN_PEAK_BACKGROUND_RATIO})",
    )
    parser.add_argument(
        "--min-peak-surface-ratio",
        type=float, default=MIN_PEAK_SURFACE_RATIO,
        help=f"Peak/surface threshold (default: {MIN_PEAK_SURFACE_RATIO})",
    )
    args = parser.parse_args()

    analyse(
        Path(args.ccn_dir),
        Path(args.outdir),
        peak_bg_ratio=args.min_peak_background_ratio,
        peak_surf_ratio=args.min_peak_surface_ratio,
    )


if __name__ == "__main__":
    main()
