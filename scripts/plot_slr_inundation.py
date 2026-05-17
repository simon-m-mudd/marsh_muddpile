#!/usr/bin/env python3
"""
plot_slr_inundation.py

ESA CCI SLR vs tuned inundation fraction, broken out by:
  1. Latitude bands (25–30, 30–35, 35–40, 40–45, 45–50 °N)
  2. Primary plant species (species with ≥ MIN_CORES cores)

Outputs
-------
  scripts/figures/datum_comparison/slr_inundation_latbands.png
  scripts/figures/datum_comparison/slr_inundation_species.png

Usage
-----
python3 scripts/plot_slr_inundation.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress

_THIS    = Path(__file__).parent
_ROOT    = _THIS.parent
_FGB     = _THIS / "core_inundation.fgb"
_SP_CSV  = _ROOT / "core_data" / "CCN_synthesis" / "CCN_species.csv"
_FIG_DIR = _THIS / "figures" / "datum_comparison"

_MIN_CORES = 20          # minimum cores for a species panel
_LAT_BINS  = [25, 30, 35, 40, 45, 50]
_LAT_LABELS = ["25–30 °N", "30–35 °N", "35–40 °N", "40–45 °N", "45–50 °N"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _panel(ax, x, y, title, color="steelblue"):
    """Scatter + OLS line on a single axis. Returns r, p, n."""
    ax.scatter(x, y, s=12, alpha=0.45, color=color, linewidths=0)
    n = len(x)
    r = p = float("nan")
    if n >= 5:
        r, p = pearsonr(x, y)
        sl, ic, *_ = linregress(x, y)
        xr = np.linspace(x.min(), x.max(), 100)
        ax.plot(xr, sl * xr + ic, color="k", lw=1.2, zorder=5)
    stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    ax.set_title(f"{title}\nn={n}  r={r:.2f}{stars}", fontsize=9)
    ax.grid(True, alpha=0.25)
    return r, p, n


def _fig_grid(n_panels, ncols=3):
    nrows = -(-n_panels // ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.2 * ncols, 4.2 * nrows),
                             sharex=False, sharey=True)
    return fig, axes.flatten()


# ── latitude-band plot ────────────────────────────────────────────────────────

def plot_by_latband(df):
    df = df.copy()
    df["latband"] = pd.cut(df["latitude"], bins=_LAT_BINS, labels=_LAT_LABELS)

    n_bands = len(_LAT_LABELS)
    band_colors = plt.cm.plasma(np.linspace(0.15, 0.85, n_bands))

    fig, axes = _fig_grid(n_bands, ncols=3)

    for i, (band, color) in enumerate(zip(_LAT_LABELS, band_colors)):
        sub = df[df["latband"] == band].dropna(subset=["esacci_slr_mmyr", "inund_tuned"])
        _panel(axes[i], sub["esacci_slr_mmyr"], sub["inund_tuned"],
               title=band, color=color)

    # hide unused panels
    for j in range(n_bands, len(axes)):
        axes[j].set_visible(False)

    fig.supxlabel("ESA CCI SLR trend (mm yr⁻¹)", fontsize=12)
    fig.supylabel("Inundation fraction (tuned)", fontsize=12)
    fig.suptitle("Inundation fraction vs ESA CCI SLR — by latitude band", fontsize=13, y=1.01)
    plt.tight_layout()

    out = _FIG_DIR / "slr_inundation_latbands.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ── species plot ──────────────────────────────────────────────────────────────

_SPECIES_REMAP = {
    "Spartina":          "Spartina alterniflora",
    "Juncus":            "Juncus roemerianus",
}
_PLOT_SPECIES = [
    "Spartina alterniflora",
    "Spartina patens",
    "Juncus roemerianus",
    "no species data",
]
_SPECIES_COLORS = {
    "Spartina alterniflora": "#2ca02c",
    "Spartina patens":       "#98df8a",
    "Juncus roemerianus":    "#8c6d31",
    "no species data":       "#aaaaaa",
}


def plot_by_species(df):
    # Assign primary species (first unique listing per core)
    sp = pd.read_csv(_SP_CSV)
    sp_primary = (sp.drop_duplicates(subset=["core_id", "species_code"])
                    .groupby("core_id")["species_code"]
                    .first()
                    .reset_index()
                    .rename(columns={"species_code": "primary_species"}))

    df = df.merge(sp_primary, on="core_id", how="left")
    df["primary_species"] = df["primary_species"].fillna("no species data")

    # Lump generic genus-level entries into the target species
    df["primary_species"] = df["primary_species"].replace(_SPECIES_REMAP)

    # Keep only the four groups of interest
    df = df[df["primary_species"].isin(_PLOT_SPECIES)].copy()

    counts = df["primary_species"].value_counts()
    print("  Species group counts:")
    for sp_name in _PLOT_SPECIES:
        print(f"    {sp_name}: {counts.get(sp_name, 0)}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharey=True)
    axes_flat = axes.flatten()

    for i, sp_name in enumerate(_PLOT_SPECIES):
        sub = df[df["primary_species"] == sp_name].dropna(
            subset=["esacci_slr_mmyr", "inund_tuned"])
        color = _SPECIES_COLORS[sp_name]
        _panel(axes_flat[i], sub["esacci_slr_mmyr"], sub["inund_tuned"],
               title=sp_name, color=color)

    fig.supxlabel("ESA CCI SLR trend (mm yr⁻¹)", fontsize=12)
    fig.supylabel("Inundation fraction (tuned)", fontsize=12)
    fig.suptitle("Inundation fraction vs ESA CCI SLR — by plant species", fontsize=13)
    plt.tight_layout()

    out = _FIG_DIR / "slr_inundation_species.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading core_inundation.fgb …")
    gdf = gpd.read_file(_FGB)
    df  = pd.DataFrame(gdf.drop(columns="geometry"))
    print(f"  {len(df)} cores, {df['inund_tuned'].notna().sum()} with tuned inundation")

    print("\n[1] Latitude-band plots …")
    plot_by_latband(df)

    print("\n[2] Species plots …")
    plot_by_species(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
