"""
deep_om_vs_site_properties.py

For all east-coast US marsh sediment cores with a well-constrained OM depth
profile (best-fit R² ≥ 0.75, ≥ 6 layers) and a measurable deep organic
fraction (OMinf > DEEP_OM_THRESHOLD), plot deep_om_frac as a function of:

  1. inundation fraction
  2. latitude
  3. distance from nearest flowline / tidal creek (m)

Data source
-----------
scripts/core_inundation.fgb — one row per core, produced by
  fit_carbon_profiles.py  (adds best_model, deep_om_frac, …)
  add_noaa_gauge_inundation.py  (adds inund_frac, inund_tuned, …)
  add_water_distance.py         (adds dist_flowline_m, dist_waterbody_m)
  tuned_inundation.py           (adds inund_tuned, inund_tuned_c)

Definitions
-----------
well-fit     best_model is not NaN  (R² ≥ 0.75, ≥ 6 depth layers)
deep OM      deep_om_frac > DEEP_OM_THRESHOLD (default 0.02, i.e. 2 wt%)
             OMinf values effectively zero (< 1e-6) are filtered out
             because they are exponential tails with no real plateau
inundation   inund_tuned where available, inund_frac otherwise

East coast   lat 24–47 °N, lon -85–-60 °E

Output
------
  scripts/figures/deep_om_vs_inundation.png
  scripts/figures/deep_om_vs_latitude.png
  scripts/figures/deep_om_vs_creek_distance.png
  scripts/figures/deep_om_vs_site_properties.png   (3-panel combined)
"""

from pathlib import Path

import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEEP_OM_THRESHOLD = 0.02   # minimum OMinf to be considered "deep organic"
LAT_LO, LAT_HI   = 24.0, 47.0
LON_LO, LON_HI   = -85.0, -60.0

_ROOT    = Path(__file__).parent.parent
_FGB     = _ROOT / "scripts" / "core_inundation.fgb"
_FIG_DIR = _ROOT / "scripts" / "figures"

# Vegetation class colour map
VEG_COLORS = {
    "emergent":   "#2ca02c",
    "scrub/shrub":"#8c6d31",
    "forested":   "#17becf",
    "seagrass":   "#9467bd",
    "other":      "#7f7f7f",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(str(_FGB))

    # east coast filter
    ec = gdf[
        (gdf["latitude"]  >= LAT_LO) & (gdf["latitude"]  <= LAT_HI) &
        (gdf["longitude"] >= LON_LO) & (gdf["longitude"] <= LON_HI)
    ].copy()

    # well-fit profiles only
    ec = ec[ec["best_model"].notna()]

    # deep organic fraction present
    ec = ec[ec["deep_om_frac"].notna() & (ec["deep_om_frac"] > DEEP_OM_THRESHOLD)]

    # inundation: prefer tuned, fall back to raw
    ec["inund"] = ec["inund_tuned"].where(
        ec["inund_tuned"].notna(), ec["inund_frac"]
    )

    # distance from creek: prefer flowline
    ec["dist_creek_m"] = ec["dist_flowline_m"].where(
        ec["dist_flowline_m"].notna(), ec["dist_waterbody_m"]
    )

    # vegetation colour
    ec["veg_color"] = ec["vegetation_class"].str.lower().map(
        lambda v: VEG_COLORS.get(str(v).split("/")[0].strip(), VEG_COLORS["other"])
    )

    return ec


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _add_legend(ax, data):
    seen = {}
    for _, row in data.iterrows():
        vc = str(row.get("vegetation_class", "other")).lower()
        key = vc.split("/")[0].strip()
        label = key.capitalize() if key in VEG_COLORS else "Other"
        col   = VEG_COLORS.get(key, VEG_COLORS["other"])
        if label not in seen:
            ax.scatter([], [], c=col, s=30, label=label)
            seen[label] = True
    ax.legend(fontsize=8, title="Vegetation", title_fontsize=8,
               loc="upper right", markerscale=1.2)


def _scatter(ax, x, y, colors, xlabel, title, xlog=False):
    ax.scatter(x, y, c=colors, s=18, alpha=0.7, edgecolors="none")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Deep OM fraction (OMinf)", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.tick_params(labelsize=9)
    if xlog:
        ax.set_xscale("log")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _FIG_DIR.mkdir(parents=True, exist_ok=True)

    ec = load_data()
    print(f"East-coast well-fit cores with deep OM > {DEEP_OM_THRESHOLD:.0%}: {len(ec)}")
    print(f"  inundation available: {ec['inund'].notna().sum()}")
    print(f"  dist_creek available: {ec['dist_creek_m'].notna().sum()}")

    deep  = ec["deep_om_frac"].values
    inund = ec["inund"].values
    lat   = ec["latitude"].values
    dist  = ec["dist_creek_m"].values
    cols  = ec["veg_color"].values

    # --- combined 3-panel figure ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax_if, ax_lat, ax_dist = axes

    mask_if   = ~np.isnan(inund)
    mask_dist = ~np.isnan(dist)

    _scatter(ax_if,   inund[mask_if],  deep[mask_if],  cols[mask_if],
             "Inundation fraction", "Deep OM vs inundation fraction")
    _scatter(ax_lat,  lat,             deep,           cols,
             "Latitude (°N)",       "Deep OM vs latitude")
    _scatter(ax_dist, dist[mask_dist], deep[mask_dist], cols[mask_dist],
             "Distance from creek (m)", "Deep OM vs creek distance", xlog=True)

    # add vegetation legend to rightmost panel only
    _add_legend(ax_dist, ec[mask_dist])

    # latitude-panel: annotate n
    ax_lat.text(0.03, 0.97, f"n = {len(ec)}", transform=ax_lat.transAxes,
                va="top", fontsize=8)
    ax_if.text(0.03, 0.97, f"n = {mask_if.sum()}", transform=ax_if.transAxes,
               va="top", fontsize=8)
    ax_dist.text(0.03, 0.97, f"n = {mask_dist.sum()}", transform=ax_dist.transAxes,
                 va="top", fontsize=8)

    fig.suptitle(
        f"East-coast US marsh cores  |  well-fit OM profiles (R² ≥ 0.75)  |  "
        f"deep OM fraction > {DEEP_OM_THRESHOLD:.0%}",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    combined = _FIG_DIR / "deep_om_vs_site_properties.png"
    fig.savefig(combined, dpi=150, bbox_inches="tight")
    print(f"Saved: {combined}")
    plt.close(fig)

    # --- individual figures ---
    for x_arr, mask, xlab, xlog, fname in [
        (inund, mask_if,   "Inundation fraction",      False, "deep_om_vs_inundation.png"),
        (lat,   np.ones(len(ec), dtype=bool),
                            "Latitude (°N)",           False, "deep_om_vs_latitude.png"),
        (dist,  mask_dist, "Distance from creek (m)",  True,  "deep_om_vs_creek_distance.png"),
    ]:
        fig2, ax = plt.subplots(figsize=(6, 5))
        _scatter(ax, x_arr[mask], deep[mask], cols[mask], xlab,
                 xlab.replace("(", "vs deep OM (").replace(")", ")"), xlog=xlog)
        _add_legend(ax, ec[mask])
        ax.text(0.03, 0.97, f"n = {mask.sum()}", transform=ax.transAxes, va="top", fontsize=8)
        plt.tight_layout()
        out = _FIG_DIR / fname
        fig2.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
        plt.close(fig2)


if __name__ == "__main__":
    main()
