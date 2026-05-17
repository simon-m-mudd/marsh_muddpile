"""
Plot east coast USA salt marsh aboveground biomass vs latitude and elevation.

Data sources:
  - North Inlet, SC (33.3°N): NI LTER edi.135.12 (biomass) + edi.134.12 (elevation)
  - Altamaha River, GA (~31.3°N): GCE-LTER knb-lter-gce.759.27
  - Plum Island, MA (42.7°N): PIE LTER knb-lter-pie.32.15

All east coast data is Spartina alterniflora (or S. cynosuroides at GCE transition sites).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_THIS_DIR = Path(__file__).parent
_LTER_DIR = _THIS_DIR.parent / "lter_data"
_OUT_DIR = _THIS_DIR / "figures"
_OUT_DIR.mkdir(exist_ok=True)

# ── site coordinate registry ──────────────────────────────────────────────────
# (lat, lon, label, species)
_GCE_SITES = {
    "SCSA": (31.323, -81.374, "GA: Broughton Is.\n(S. alt./S. cyn. transition)"),
    "ZSC1": (31.328, -81.451, "GA: Altamaha Marsh 1\n(S. alterniflora)"),
    "ZSC2": (31.341, -81.454, "GA: Altamaha Marsh 2\n(S. alterniflora)"),
}

_PIE_SITE = dict(
    lat=42.731, lon=-70.843,
    label="MA: Plum Island\n(S. alterniflora)"
)

_NI_SITES = {
    # (lat, lon) from edi.134.12 centroids
    ("GI", "HM"): (33.3316, -79.1982, "SC: North Inlet GI-HM"),
    ("GI", "LM"): (33.3316, -79.1979, "SC: North Inlet GI-LM"),
    ("OL", "HM"): (33.348,  -79.182,  "SC: North Inlet OL-HM"),
    ("OL", "LM"): (33.348,  -79.182,  "SC: North Inlet OL-LM"),
}


# ── loaders ──────────────────────────────────────────────────────────────────

def _load_ni_biomass_elevation():
    """Return plot-level mean summer biomass and elevation for NI LTER GI site."""
    bio = pd.read_csv(
        _LTER_DIR / "edi.135.12/NILTREB_plants_aboveground_biomass_density.csv"
    )
    bio["DATE"] = pd.to_datetime(bio["DATE"])
    bio["Month"] = bio["DATE"].dt.month
    bio["Year"] = bio["DATE"].dt.year
    bio["ABOVEGROUND_BIOMASS"] = pd.to_numeric(bio["ABOVEGROUND_BIOMASS"], errors="coerce")

    elev = pd.read_csv(
        _LTER_DIR / "edi.134.12/NILTREB_marsh_surface_elevation_init.csv"
    )
    mean_elev = (
        elev.groupby(["SITE", "LOCATION", "PLOT"])["MARSH_SURFACE_ELEVATION"]
        .mean()
        .reset_index()
        .rename(columns={"MARSH_SURFACE_ELEVATION": "elevation_m"})
    )

    # Peak summer biomass (Jul–Aug) per site/location/plot/year, then mean over years
    # Filter to control plots only (exclude NP fertilized plots)
    summer = bio[(bio["Month"].between(7, 8)) & (bio["TREATMENT"] == "C")]
    annual_peak = (
        summer.groupby(["SITE", "LOCATION", "PLOT", "Year"])["ABOVEGROUND_BIOMASS"]
        .mean()
        .reset_index()
    )
    plot_mean = (
        annual_peak.groupby(["SITE", "LOCATION", "PLOT"])["ABOVEGROUND_BIOMASS"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "biomass_g_m2", "std": "biomass_std", "count": "n_years"})
    )

    # Join elevation — only GI plots have matching elevation records
    merged = plot_mean.merge(mean_elev, on=["SITE", "LOCATION", "PLOT"], how="inner")
    merged["biomass_se"] = merged["biomass_std"] / np.sqrt(merged["n_years"])
    return merged


def _load_ni_site_biomass():
    """Return site/location-level summer peak biomass for all NI plots (control only)."""
    bio = pd.read_csv(
        _LTER_DIR / "edi.135.12/NILTREB_plants_aboveground_biomass_density.csv"
    )
    bio["DATE"] = pd.to_datetime(bio["DATE"])
    bio["Month"] = bio["DATE"].dt.month
    bio["Year"] = bio["DATE"].dt.year
    bio["ABOVEGROUND_BIOMASS"] = pd.to_numeric(bio["ABOVEGROUND_BIOMASS"], errors="coerce")

    # Control plots only
    summer = bio[(bio["Month"].between(7, 8)) & (bio["TREATMENT"] == "C")]
    annual_peak = (
        summer.groupby(["SITE", "LOCATION", "Year"])["ABOVEGROUND_BIOMASS"]
        .mean()
        .reset_index()
    )
    site_stats = (
        annual_peak.groupby(["SITE", "LOCATION"])["ABOVEGROUND_BIOMASS"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "biomass_g_m2", "std": "biomass_std", "count": "n"})
    )
    site_stats["biomass_se"] = site_stats["biomass_std"] / np.sqrt(site_stats["n"])
    return site_stats


def _load_ni_location_elevation():
    """Mean elevation per NI site/location from edi.134.12 (all measurement dates)."""
    elev = pd.read_csv(
        _LTER_DIR / "edi.134.12/NILTREB_marsh_surface_elevation_init.csv"
    )
    return (
        elev.groupby(["SITE", "LOCATION"])["MARSH_SURFACE_ELEVATION"]
        .mean()
        .reset_index()
        .rename(columns={"MARSH_SURFACE_ELEVATION": "elevation_m"})
    )


def _load_gce_biomass():
    """Return mean biomass per GCE site."""
    df = pd.read_csv(
        _LTER_DIR
        / "knb-lter-gce.759.27/PLT-GCES-1609c_Biomass_Stats_5_0.CSV",
        skiprows=2,
        header=0,
    )
    df = df.iloc[2:].reset_index(drop=True)
    df.columns = [
        "Year", "Site", "Zone", "Plot", "Quadrat_Area", "Plot_Disturbance",
        "Pct_Calc", "Num_plants", "Min_biomass", "Max_biomass", "Total_biomass",
        "Mean_biomass", "SD_biomass", "SE_biomass",
    ]
    df["Total_biomass"] = pd.to_numeric(df["Total_biomass"], errors="coerce")

    # Total_biomass is quadrat-level aboveground biomass in g/m² (sum over all plants)
    stats = (
        df.groupby("Site")["Total_biomass"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "biomass_g_m2", "std": "biomass_std", "count": "n"})
    )
    stats["biomass_se"] = stats["biomass_std"] / np.sqrt(stats["n"])
    return stats


def _load_pie_biomass():
    """Return summer peak biomass for PIE MA LPA site."""
    df = pd.read_csv(
        _LTER_DIR / "knb-lter-pie.32.15/LTE-MP-LPA-biomass.csv"
    )
    df["MEAN BIOMASS"] = pd.to_numeric(df["MEAN BIOMASS"], errors="coerce")
    summer = df[df["MONTH"].between(7, 8)]
    annual = summer.groupby("YEAR")["MEAN BIOMASS"].mean().reset_index()
    return {
        "biomass_g_m2": annual["MEAN BIOMASS"].mean(),
        "biomass_std": annual["MEAN BIOMASS"].std(),
        "n": len(annual),
        "biomass_se": annual["MEAN BIOMASS"].std() / np.sqrt(len(annual)),
    }


# ── colour scheme ─────────────────────────────────────────────────────────────
_REGION_COLORS = {
    "GA": "#e6701a",  # orange for Georgia
    "SC": "#2a9d8f",  # teal for South Carolina
    "MA": "#457b9d",  # blue for Massachusetts
}
_LOC_MARKERS = {"HM": "^", "LM": "o", "MM": "s"}


# ── figure ───────────────────────────────────────────────────────────────────

def make_figure():
    ni_plot = _load_ni_biomass_elevation()
    ni_site = _load_ni_site_biomass()
    ni_loc_elev = _load_ni_location_elevation()
    gce = _load_gce_biomass()
    pie = _load_pie_biomass()

    # Build location-level elevation + biomass table for NI (for elevation panel)
    # Only include site/location combos in BOTH datasets; elevations are representative
    ni_loc_bio = ni_site.merge(ni_loc_elev, on=["SITE", "LOCATION"], how="inner")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        "US East Coast Salt Marsh Aboveground Biomass\n"
        "Spartina alterniflora (and S. cynosuroides at GA transition sites)",
        fontsize=12, y=1.01
    )

    # ── Panel A: Biomass vs Latitude ──────────────────────────────────────────
    ax1 = axes[0]

    # GA – GCE sites
    for site_code, row in gce.iterrows():
        site = row["Site"]
        if site not in _GCE_SITES:
            continue
        lat, lon, label = _GCE_SITES[site]
        ax1.errorbar(
            lat, row["biomass_g_m2"], yerr=row["biomass_se"],
            fmt="D", color=_REGION_COLORS["GA"],
            markersize=7, capsize=3, elinewidth=1, label=f"GA: {site}",
            zorder=3
        )
        ax1.annotate(
            site, (lat, row["biomass_g_m2"]),
            xytext=(4, 4), textcoords="offset points", fontsize=7
        )

    # SC – NI LTER site-level
    for _, row in ni_site.iterrows():
        key = (row["SITE"], row["LOCATION"])
        if key not in _NI_SITES:
            continue
        lat, lon, label = _NI_SITES[key]
        marker = _LOC_MARKERS.get(row["LOCATION"], "o")
        lbl = f"SC: NI {row['SITE']}-{row['LOCATION']}"
        ax1.errorbar(
            lat, row["biomass_g_m2"], yerr=row["biomass_se"],
            fmt=marker, color=_REGION_COLORS["SC"],
            markersize=8, capsize=3, elinewidth=1, label=lbl,
            zorder=3
        )
        ax1.annotate(
            f"{row['SITE']}-{row['LOCATION']}",
            (lat, row["biomass_g_m2"]),
            xytext=(4, 4), textcoords="offset points", fontsize=7
        )

    # MA – PIE LTER
    ax1.errorbar(
        _PIE_SITE["lat"], pie["biomass_g_m2"], yerr=pie["biomass_se"],
        fmt="s", color=_REGION_COLORS["MA"],
        markersize=8, capsize=3, elinewidth=1, label="MA: PIE LPA",
        zorder=3
    )
    ax1.annotate(
        "PIE-MA", (_PIE_SITE["lat"], pie["biomass_g_m2"]),
        xytext=(4, 4), textcoords="offset points", fontsize=7
    )

    ax1.set_xlabel("Latitude (°N)", fontsize=11)
    ax1.set_ylabel("Aboveground biomass (g m⁻²)", fontsize=11)
    ax1.set_title("(a) Biomass vs Latitude", fontsize=11)
    ax1.set_xlim(30.5, 44)

    # Region colour patches for legend
    legend_patches = [
        mpatches.Patch(color=_REGION_COLORS["GA"], label="Georgia (GCE-LTER)"),
        mpatches.Patch(color=_REGION_COLORS["SC"], label="South Carolina (NI-LTER)"),
        mpatches.Patch(color=_REGION_COLORS["MA"], label="Massachusetts (PIE-LTER)"),
    ]
    marker_patches = [
        plt.Line2D([0], [0], marker="^", color="k", linestyle="None",
                   markersize=7, label="High marsh (HM)"),
        plt.Line2D([0], [0], marker="o", color="k", linestyle="None",
                   markersize=7, label="Low marsh (LM)"),
        plt.Line2D([0], [0], marker="D", color="k", linestyle="None",
                   markersize=7, label="Transition zone (GA)"),
    ]
    ax1.legend(
        handles=legend_patches + marker_patches,
        fontsize=8, loc="upper left"
    )
    ax1.grid(True, alpha=0.3)

    # ── Panel B: Biomass vs Elevation ────────────────────────────────────────
    ax2 = axes[1]

    # Background: location-level means (elevation from edi.134.12, not exact
    # same plots as biomass — indicative of marsh zone elevation)
    for _, row in ni_loc_bio.iterrows():
        marker = _LOC_MARKERS.get(row["LOCATION"], "o")
        ax2.errorbar(
            row["elevation_m"], row["biomass_g_m2"], yerr=row["biomass_se"],
            fmt=marker, color=_REGION_COLORS["SC"],
            markersize=11, capsize=4, elinewidth=1.2,
            markerfacecolor="none", markeredgewidth=1.5,
            zorder=3
        )
        lbl_text = f"{row['SITE']}-{row['LOCATION']}"
        ax2.annotate(
            lbl_text,
            (row["elevation_m"], row["biomass_g_m2"]),
            xytext=(5, 5), textcoords="offset points", fontsize=7
        )

    # Foreground: matched plot-level data (GI HM plots 5, 9, 12)
    for _, row in ni_plot.iterrows():
        marker = _LOC_MARKERS.get(row["LOCATION"], "o")
        ax2.errorbar(
            row["elevation_m"], row["biomass_g_m2"], yerr=row["biomass_se"],
            fmt=marker, color=_REGION_COLORS["SC"],
            markersize=8, capsize=3, elinewidth=1,
            zorder=4
        )
        ax2.annotate(
            f"P{int(row['PLOT'])}",
            (row["elevation_m"], row["biomass_g_m2"]),
            xytext=(4, -10), textcoords="offset points", fontsize=7
        )

    loc_patches = [
        plt.Line2D([0], [0], marker="^", color=_REGION_COLORS["SC"],
                   linestyle="None", markersize=8, label="High marsh (HM)"),
        plt.Line2D([0], [0], marker="o", color=_REGION_COLORS["SC"],
                   linestyle="None", markersize=8, label="Low marsh (LM)"),
        plt.Line2D([0], [0], marker="o", color=_REGION_COLORS["SC"],
                   linestyle="None", markersize=10,
                   markerfacecolor="none", markeredgewidth=1.5,
                   label="Location mean (elev. not plot-matched)"),
        plt.Line2D([0], [0], marker="o", color=_REGION_COLORS["SC"],
                   linestyle="None", markersize=8,
                   label="Plot-matched (GI HM only)"),
    ]
    ax2.set_xlabel("Marsh surface elevation (m NAVD88)", fontsize=11)
    ax2.set_ylabel("Aboveground biomass (g m⁻²)", fontsize=11)
    ax2.set_title(
        "(b) Biomass vs Elevation — North Inlet, SC\nS. alterniflora, control plots only\n"
        "(Jul–Aug peak biomass; error bars = SE over years)",
        fontsize=10
    )
    ax2.legend(handles=loc_patches, fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = _OUT_DIR / "biomass_latitude_elevation_US_east_coast.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    return fig


if __name__ == "__main__":
    make_figure()
