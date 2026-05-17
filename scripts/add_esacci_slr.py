#!/usr/bin/env python3
"""
add_esacci_slr.py

Adds ESA CCI altimetry sea-level-rise trend to core_inundation.fgb and plots
inundation fraction against SLR.

For each core:
  - If the nearest ESA CCI grid point is within 1 km → use it directly.
  - Otherwise → interpolate using scipy linear triangulation of the full grid.

New columns added to core_inundation.fgb:
  esacci_slr_mmyr     — SLR trend from ESA CCI (mm yr⁻¹)
  esacci_slr_err_mmyr — formal trend uncertainty (mm yr⁻¹)

Output plots:
  scripts/figures/datum_comparison/inundation_vs_esacci_slr.png

Usage
-----
python3 scripts/add_esacci_slr.py
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree

_THIS = Path(__file__).parent
_ROOT = _THIS.parent

_SLR_CSV   = _ROOT / "core_data" / "SLR_summary.csv"
_INUND_FGB = _THIS / "core_inundation.fgb"
_FIG_DIR   = _THIS / "figures" / "datum_comparison"

_DIRECT_KM = 1.0          # use nearest point directly if within this distance


# ── helpers ──────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised haversine — all arrays must broadcast."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def assign_esacci_slr(cores_df, slr_df, direct_km=1.0):
    """
    Return arrays (slr_mmyr, slr_err_mmyr, method) for each core.

    method values: 'direct' | 'interpolated'
    """
    core_lats = cores_df["latitude"].values
    core_lons = cores_df["longitude"].values

    slr_lats = slr_df["Latitude"].values
    slr_lons = slr_df["Longitude"].values
    slr_vals = slr_df["SLA Trend"].values
    slr_errs = slr_df["Trend Error"].values

    n_cores = len(core_lats)
    out_slr  = np.full(n_cores, np.nan)
    out_err  = np.full(n_cores, np.nan)
    out_meth = np.full(n_cores, "", dtype=object)

    # KD-tree in lat/lon degrees — find nearest neighbour per core
    tree = cKDTree(np.column_stack([slr_lats, slr_lons]))
    _, nn_idx = tree.query(np.column_stack([core_lats, core_lons]))

    nn_dist_km = haversine_km(
        core_lats, core_lons,
        slr_lats[nn_idx], slr_lons[nn_idx],
    )

    # Direct assignment for close cores
    direct_mask = nn_dist_km <= direct_km
    out_slr[direct_mask]  = slr_vals[nn_idx[direct_mask]]
    out_err[direct_mask]  = slr_errs[nn_idx[direct_mask]]
    out_meth[direct_mask] = "direct"

    # Linear interpolation for the rest
    interp_mask = ~direct_mask
    if interp_mask.any():
        print(f"  Building linear interpolator for {interp_mask.sum()} cores …")
        # Use projected coords (simple equirectangular) for interpolation
        # weights are correct near-enough for regional scale
        interp_slr = LinearNDInterpolator(
            list(zip(slr_lats, slr_lons)), slr_vals
        )
        interp_err = LinearNDInterpolator(
            list(zip(slr_lats, slr_lons)), slr_errs
        )
        q_pts = np.column_stack([core_lats[interp_mask], core_lons[interp_mask]])
        out_slr[interp_mask]  = interp_slr(q_pts)
        out_err[interp_mask]  = interp_err(q_pts)
        out_meth[interp_mask] = "interpolated"

        # Fallback: any NaN from interpolation (outside convex hull) → nearest
        still_nan = interp_mask & np.isnan(out_slr)
        if still_nan.any():
            print(f"  Nearest-neighbour fallback for {still_nan.sum()} hull-exterior cores")
            out_slr[still_nan]  = slr_vals[nn_idx[still_nan]]
            out_err[still_nan]  = slr_errs[nn_idx[still_nan]]
            out_meth[still_nan] = "nearest"

    return out_slr, out_err, out_meth


# ── plot ──────────────────────────────────────────────────────────────────────

_SALINITY_COLORS = {
    "fresh":       "#1f77b4",
    "oligohaline": "#aec7e8",
    "mesohaline":  "#ff7f0e",
    "polyhaline":  "#2ca02c",
    "saline":      "#d62728",
    "unknown":     "#7f7f7f",
}


def plot_inundation_vs_slr(gdf, out_path):
    has = gdf.dropna(subset=["esacci_slr_mmyr", "inund_tuned"]).copy()
    print(f"  Plotting {len(has)} cores with both ESA CCI SLR and tuned inundation")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Inundation fraction vs ESA CCI sea-level rise — CCN US marsh cores", fontsize=13)

    # ── left: scatter coloured by salinity class ──────────────────────────────
    ax = axes[0]
    sal_classes = has["salinity_class"].fillna("unknown")
    for sal, grp in has.groupby(sal_classes):
        col = _SALINITY_COLORS.get(sal, "#999999")
        ax.scatter(grp["esacci_slr_mmyr"], grp["inund_tuned"],
                   c=col, s=18, alpha=0.6, label=sal, linewidths=0)

    ax.set_xlabel("ESA CCI SLR trend (mm yr⁻¹)")
    ax.set_ylabel("Inundation fraction (tuned)")
    ax.set_title("Coloured by salinity class")
    ax.legend(title="Salinity class", fontsize=8, markerscale=1.5)
    ax.grid(True, alpha=0.3)

    # thin error bars (x-axis SLR uncertainty) — sample 20% to reduce clutter
    rng = np.random.default_rng(42)
    sample = has.sample(frac=0.20, random_state=42) if len(has) > 500 else has
    ax.errorbar(
        sample["esacci_slr_mmyr"], sample["inund_tuned"],
        xerr=sample["esacci_slr_err_mmyr"],
        fmt="none", ecolor="black", elinewidth=0.4, alpha=0.25, zorder=0
    )

    # ── right: 2-D hexbin density ─────────────────────────────────────────────
    ax2 = axes[1]
    hb = ax2.hexbin(has["esacci_slr_mmyr"], has["inund_tuned"],
                    gridsize=40, cmap="YlOrRd", mincnt=1)
    fig.colorbar(hb, ax=ax2, label="Core count")
    ax2.set_xlabel("ESA CCI SLR trend (mm yr⁻¹)")
    ax2.set_ylabel("Inundation fraction (tuned)")
    ax2.set_title("Density")
    ax2.grid(True, alpha=0.3)

    # overall stats annotation
    r = np.corrcoef(has["esacci_slr_mmyr"], has["inund_tuned"])[0, 1]
    axes[1].text(
        0.97, 0.97, f"n = {len(has)}\nr = {r:.3f}",
        transform=axes[1].transAxes, ha="right", va="top",
        fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_inundation_vs_slr_byhabitat(gdf, out_path):
    """Secondary plot broken out by habitat type."""
    has = gdf.dropna(subset=["esacci_slr_mmyr", "inund_tuned"]).copy()
    habitats = sorted(has["habitat"].dropna().unique())

    ncols = 3
    nrows = -(-len(habitats) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             sharex=True, sharey=True)
    axes_flat = axes.flatten() if nrows > 1 else axes

    for i, hab in enumerate(habitats):
        ax = axes_flat[i]
        sub = has[has["habitat"] == hab]
        ax.scatter(sub["esacci_slr_mmyr"], sub["inund_tuned"],
                   s=14, alpha=0.5, linewidths=0)
        r = np.corrcoef(sub["esacci_slr_mmyr"], sub["inund_tuned"])[0, 1] if len(sub) > 2 else np.nan
        ax.set_title(f"{hab}  (n={len(sub)}, r={r:.2f})", fontsize=9)
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.supxlabel("ESA CCI SLR trend (mm yr⁻¹)", fontsize=11)
    fig.supylabel("Inundation fraction (tuned)", fontsize=11)
    fig.suptitle("Inundation vs ESA CCI SLR by habitat", fontsize=13)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 60)
    print("  ESA CCI SLR → core_inundation.fgb")
    print("=" * 60)

    # Load data
    print("\n[1] Loading data …")
    slr_df = pd.read_csv(_SLR_CSV)
    print(f"  ESA CCI grid: {len(slr_df)} points")
    print(f"  SLR range: {slr_df['SLA Trend'].min():.2f} to {slr_df['SLA Trend'].max():.2f} mm yr⁻¹")

    gdf = gpd.read_file(_INUND_FGB)
    print(f"  Cores: {len(gdf)}")

    # Assign SLR
    print(f"\n[2] Assigning ESA CCI SLR (direct threshold: {_DIRECT_KM} km) …")
    slr_vals, slr_errs, methods = assign_esacci_slr(gdf, slr_df, _DIRECT_KM)

    n_direct = (methods == "direct").sum()
    n_interp = (methods == "interpolated").sum()
    n_near   = (methods == "nearest").sum()
    n_nan    = np.isnan(slr_vals).sum()
    print(f"  Direct (≤{_DIRECT_KM} km):  {n_direct}")
    print(f"  Interpolated:    {n_interp}")
    print(f"  Nearest fallback:{n_near}")
    print(f"  NaN (no data):   {n_nan}")

    # Add / overwrite columns
    print("\n[3] Updating FGB …")
    gdf["esacci_slr_mmyr"]     = slr_vals
    gdf["esacci_slr_err_mmyr"] = slr_errs
    gdf["esacci_slr_method"]   = methods

    gdf.to_file(_INUND_FGB, driver="FlatGeobuf")
    print(f"  Saved {_INUND_FGB}")

    # Stats
    valid = gdf.dropna(subset=["esacci_slr_mmyr"])
    print(f"\n  ESA CCI SLR stats ({len(valid)} cores with valid SLR):")
    print(f"    mean  {valid['esacci_slr_mmyr'].mean():.2f} mm yr⁻¹")
    print(f"    median {valid['esacci_slr_mmyr'].median():.2f} mm yr⁻¹")
    print(f"    range  {valid['esacci_slr_mmyr'].min():.2f}  to  {valid['esacci_slr_mmyr'].max():.2f} mm yr⁻¹")

    # Plots
    print("\n[4] Plots …")
    plot_inundation_vs_slr(
        gdf, _FIG_DIR / "inundation_vs_esacci_slr.png"
    )
    plot_inundation_vs_slr_byhabitat(
        gdf, _FIG_DIR / "inundation_vs_esacci_slr_byhabitat.png"
    )

    print(f"\n  Total time: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
