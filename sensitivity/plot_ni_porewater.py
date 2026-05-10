"""
plot_ni_porewater.py

Plots porewater chemistry from the North Inlet LTREB dataset (edi.136.11).

Three figure types are produced for every site / location combination
(control plots only):

  1. depth_profiles_<site>_<loc>_<year>.png
     Monthly depth profiles for the most recent complete year.
     Each month is one line; depth is on the y-axis (inverted).
     Produced for SALINITY, NH4, and S2 (sulfide).

  2. timeseries_<site>_<loc>.png
     Full-record time series at the shallowest (10 cm) and deepest
     (100 cm) depths for SALINITY, NH4, and S2.
     Mean of three replicates is plotted; shaded band shows min–max range.

  3. salinity_temperature_correlation.png  (one figure, all sites)
     Left:  scatter of monthly mean porewater salinity vs monthly mean
            air temperature (ERA5), one colour per site, with linear fits.
     Right: mean seasonal cycle of salinity and temperature overlaid,
            illustrating the phase relationship.

Usage
-----
    python sensitivity/plot_ni_porewater.py
    python sensitivity/plot_ni_porewater.py --outdir viz/porewater
    python sensitivity/plot_ni_porewater.py --year 2023
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

matplotlib.rcParams.update({"font.size": 9})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT  = Path(__file__).parent.parent
_DATA_PATH  = _REPO_ROOT / "lter_data" / "edi.136.11" / "NILTREB_porewater.csv"
_ERA5_PATH  = _REPO_ROOT / "era5_data" / "era5_land_ni_1985_2024_daily.csv"

# ---------------------------------------------------------------------------
# Variable display config
# ---------------------------------------------------------------------------

VAR_CONFIG = {
    "SALINITY": {
        "label": "Porewater salinity (ppt)",
        "color_shallow": "#2166ac",
        "color_deep":    "#d73027",
    },
    "NH4": {
        "label": "Porewater NH₄ (µM)",
        "color_shallow": "#1b7837",
        "color_deep":    "#762a83",
    },
    "S2": {
        "label": "Porewater sulfide S²⁻ (µM)",
        "color_shallow": "#bf812d",
        "color_deep":    "#35978f",
    },
}

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

SITE_LABELS = {
    "GI": "Goat Island",
    "OL": "Oyster Landing",
    "SB": "Sage Bluff",
}

LOC_LABELS = {
    "HM": "High Marsh",
    "LM": "Low Marsh",
}

DEPTHS_CM = [10, 25, 50, 75, 100]
SHALLOW_CM = 10
DEEP_CM    = 100


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: Path = _DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["DATE"])
    df["YEAR"]  = df["DATE"].dt.year
    df["MONTH"] = df["DATE"].dt.month
    # Numeric coercion — 'NM' flag rows already have NaN in value columns
    for col in ["SALINITY", "NH4", "S2", "Fe2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _rep_mean(df: pd.DataFrame, groupby: list[str], var: str) -> pd.DataFrame:
    """Aggregate replicates: mean, min, max per group."""
    agg = (
        df.dropna(subset=[var])
          .groupby(groupby)[var]
          .agg(mean="mean", lo="min", hi="max")
          .reset_index()
    )
    return agg


# ---------------------------------------------------------------------------
# Figure 1 — depth profiles for one year
# ---------------------------------------------------------------------------

def plot_depth_profiles(
    df: pd.DataFrame,
    site: str,
    location: str,
    year: int,
    outdir: Path,
) -> None:
    """Monthly depth profiles of SALINITY, NH4, S2 for one site/location/year."""
    sub = df[
        (df["SITE"] == site) &
        (df["LOCATION"] == location) &
        (df["TREATMENT"] == "C") &
        (df["YEAR"] == year)
    ].copy()

    if sub.empty:
        print(f"  No data for {site} {location} {year} — skipping depth profiles")
        return

    months_present = sorted(sub["MONTH"].unique())
    nvars = len(VAR_CONFIG)
    fig, axes = plt.subplots(1, nvars, figsize=(5 * nvars, 6), sharey=True)

    # Month colours from a cyclic colormap
    cmap = matplotlib.colormaps.get_cmap("hsv").resampled(13)
    colors = {m: cmap(m / 12) for m in range(1, 13)}

    for ax, (var, vcfg) in zip(axes, VAR_CONFIG.items()):
        agg = _rep_mean(sub, ["MONTH", "DEPTH"], var)
        plotted_months = []
        for month in months_present:
            mdf = agg[agg["MONTH"] == month].sort_values("DEPTH")
            if mdf.empty or mdf["mean"].isna().all():
                continue
            color = colors[month]
            ax.plot(
                mdf["mean"], mdf["DEPTH"],
                "o-", color=color, linewidth=1.2, markersize=4,
                label=MONTH_NAMES[month - 1],
            )
            ax.fill_betweenx(
                mdf["DEPTH"], mdf["lo"], mdf["hi"],
                color=color, alpha=0.12,
            )
            plotted_months.append(month)

        ax.invert_yaxis()
        ax.set_yticks(DEPTHS_CM)
        ax.set_xlabel(vcfg["label"])
        ax.set_title(var.replace("S2", "S²⁻ (sulfide)").replace("NH4", "NH₄"))
        ax.axhline(0, color="k", linewidth=0.4, linestyle="--")
        if ax is axes[0]:
            ax.set_ylabel("Depth (cm)")
        ax.legend(fontsize=7, ncol=2, loc="lower right", framealpha=0.6)

    site_label = SITE_LABELS.get(site, site)
    loc_label  = LOC_LABELS.get(location, location)
    fig.suptitle(
        f"{site_label} — {loc_label}   |   {year} monthly depth profiles",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    out = outdir / f"depth_profiles_{site}_{location}_{year}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — full-record time series at top and bottom depths
# ---------------------------------------------------------------------------

def plot_timeseries(
    df: pd.DataFrame,
    site: str,
    location: str,
    outdir: Path,
) -> None:
    """Full-record time series of SALINITY, NH4, S2 at 10 cm and 100 cm."""
    sub = df[
        (df["SITE"] == site) &
        (df["LOCATION"] == location) &
        (df["TREATMENT"] == "C") &
        (df["DEPTH"].isin([SHALLOW_CM, DEEP_CM]))
    ].copy()

    if sub.empty:
        print(f"  No data for {site} {location} — skipping time series")
        return

    nvars = len(VAR_CONFIG)
    fig, axes = plt.subplots(nvars, 1, figsize=(14, 3.5 * nvars), sharex=True)

    for ax, (var, vcfg) in zip(axes, VAR_CONFIG.items()):
        agg = _rep_mean(sub, ["DATE", "DEPTH"], var)

        for depth, color, lw, label_suffix in [
            (SHALLOW_CM, vcfg["color_shallow"], 1.0, f"{SHALLOW_CM} cm"),
            (DEEP_CM,    vcfg["color_deep"],    1.0, f"{DEEP_CM} cm"),
        ]:
            ddf = agg[agg["DEPTH"] == depth].sort_values("DATE")
            if ddf.empty:
                continue
            ax.fill_between(
                ddf["DATE"], ddf["lo"], ddf["hi"],
                color=color, alpha=0.20,
            )
            ax.plot(
                ddf["DATE"], ddf["mean"],
                color=color, linewidth=lw,
                label=label_suffix,
            )

        ax.set_ylabel(vcfg["label"])
        ax.legend(fontsize=8, loc="upper right", framealpha=0.7)
        ax.grid(axis="x", linewidth=0.4, color="0.85")
        # Mark decade boundaries
        for yr in range(1990, 2030, 10):
            ax.axvline(
                pd.Timestamp(f"{yr}-01-01"),
                color="0.6", linewidth=0.6, linestyle=":",
            )

    site_label = SITE_LABELS.get(site, site)
    loc_label  = LOC_LABELS.get(location, location)
    fig.suptitle(
        f"{site_label} — {loc_label}   |   Porewater record 1993–2025\n"
        f"solid = mean of 3 replicates,  shading = min–max range",
        fontsize=10, fontweight="bold",
    )
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    out = outdir / f"timeseries_{site}_{location}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — salinity vs temperature correlation (all sites combined)
# ---------------------------------------------------------------------------

# Distinct colours for each (site, location) combo
_COMBO_COLORS = {
    ("GI", "HM"): "#e41a1c",
    ("GI", "LM"): "#ff7f00",
    ("OL", "HM"): "#377eb8",
    ("OL", "LM"): "#4daf4a",
    ("SB", "HM"): "#984ea3",
}


def _load_era5_monthly(path: Path = _ERA5_PATH) -> pd.DataFrame | None:
    """Return monthly mean temperature from ERA5 daily CSV, or None if missing."""
    if not path.exists():
        print(f"  ERA5 file not found: {path} — skipping correlation plot")
        return None
    era5 = pd.read_csv(path, parse_dates=["date"])
    era5["YEAR"]  = era5["date"].dt.year
    era5["MONTH"] = era5["date"].dt.month
    return (
        era5.groupby(["YEAR", "MONTH"])["temperature_mean_c"]
            .mean()
            .reset_index()
    )


def plot_salinity_temperature_correlation(
    df: pd.DataFrame,
    outdir: Path,
    era5_path: Path = _ERA5_PATH,
) -> None:
    """Scatter and seasonal-cycle panels for salinity–temperature correlation."""
    t_monthly = _load_era5_monthly(era5_path)
    if t_monthly is None:
        return

    # Monthly mean salinity per site/location (all depths, control only)
    sal_monthly = (
        df[df["TREATMENT"] == "C"]
          .dropna(subset=["SALINITY"])
          .assign(YEAR=lambda d: d["DATE"].dt.year,
                  MONTH=lambda d: d["DATE"].dt.month)
          .groupby(["SITE", "LOCATION", "YEAR", "MONTH"])["SALINITY"]
          .mean()
          .reset_index()
    )
    merged = sal_monthly.merge(t_monthly, on=["YEAR", "MONTH"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ---- left: scatter with linear fits ----
    ax = axes[0]
    for (site, loc), grp in merged.groupby(["SITE", "LOCATION"]):
        color = _COMBO_COLORS.get((site, loc), "grey")
        label_str = f"{SITE_LABELS.get(site, site)} {LOC_LABELS.get(loc, loc)}"
        ax.scatter(
            grp["temperature_mean_c"], grp["SALINITY"],
            color=color, s=8, alpha=0.45, label=label_str,
        )
        # Linear fit
        if len(grp) > 4:
            m, b = np.polyfit(grp["temperature_mean_c"], grp["SALINITY"], 1)
            t_range = np.linspace(
                grp["temperature_mean_c"].min(),
                grp["temperature_mean_c"].max(), 60
            )
            r = grp["SALINITY"].corr(grp["temperature_mean_c"])
            ax.plot(t_range, m * t_range + b, color=color, linewidth=1.5,
                    label=f"  r = {r:.2f}")

    ax.set_xlabel("Monthly mean air temperature (°C)")
    ax.set_ylabel("Monthly mean porewater salinity (ppt)")
    ax.set_title("Salinity vs temperature — all sites")
    ax.legend(fontsize=7, ncol=2, framealpha=0.7)
    ax.grid(linewidth=0.3, color="0.85")

    # ---- right: mean seasonal cycles ----
    ax2 = axes[1]
    ax2_t = ax2.twinx()

    months = np.arange(1, 13)
    t_seasonal = (
        t_monthly.groupby("MONTH")["temperature_mean_c"].mean()
    )
    sal_seasonal = (
        merged.groupby(["SITE", "LOCATION", "MONTH"])["SALINITY"]
              .mean()
              .reset_index()
    )

    for (site, loc), grp in sal_seasonal.groupby(["SITE", "LOCATION"]):
        color = _COMBO_COLORS.get((site, loc), "grey")
        label_str = f"{SITE_LABELS.get(site, site)} {LOC_LABELS.get(loc, loc)}"
        grp_sorted = grp.set_index("MONTH").reindex(months)
        ax2.plot(months, grp_sorted["SALINITY"],
                 "o-", color=color, linewidth=1.4, markersize=4, label=label_str)

    ax2_t.plot(months, [t_seasonal.get(m, np.nan) for m in months],
               "k--", linewidth=1.8, label="Air temperature")
    ax2_t.set_ylabel("Mean air temperature (°C)", color="black")

    ax2.set_xlabel("Month")
    ax2.set_ylabel("Mean porewater salinity (ppt)")
    ax2.set_title("Mean seasonal cycle")
    ax2.set_xticks(months)
    ax2.set_xticklabels(MONTH_NAMES, fontsize=8)
    ax2.grid(axis="x", linewidth=0.3, color="0.85")

    # Combined legend
    lines1, labs1 = ax2.get_legend_handles_labels()
    lines2, labs2 = ax2_t.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labs1 + labs2, fontsize=7, framealpha=0.7)

    fig.suptitle(
        "North Inlet porewater salinity — correlation with air temperature\n"
        "(monthly means, control plots, all depths averaged)",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    out = outdir / "salinity_temperature_correlation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _latest_well_sampled_year(df: pd.DataFrame, site: str, location: str,
                               min_months: int = 6) -> int:
    """Return the most recent year with at least min_months of data."""
    sub = df[(df["SITE"] == site) & (df["LOCATION"] == location) &
             (df["TREATMENT"] == "C")]
    counts = (
        sub.dropna(subset=["SALINITY"])
           .groupby("YEAR")["MONTH"].nunique()
    )
    valid = counts[counts >= min_months]
    return int(valid.index.max()) if not valid.empty else int(sub["YEAR"].max())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot North Inlet LTREB porewater chemistry."
    )
    parser.add_argument(
        "--outdir",
        default=str(_REPO_ROOT / "viz" / "porewater"),
        help="Output directory for figures (default: viz/porewater/)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Year for depth-profile plots (default: most recent well-sampled year).",
    )
    parser.add_argument(
        "--data",
        default=str(_DATA_PATH),
        help="Path to NILTREB_porewater.csv",
    )
    parser.add_argument(
        "--era5",
        default=str(_ERA5_PATH),
        help="Path to ERA5 daily CSV for temperature correlation plot",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.data} ...")
    df = load_data(Path(args.data))
    print(f"  {len(df):,} rows,  {df['DATE'].min().year}–{df['DATE'].max().year}")

    # Sites to plot: all (site, location) combos with control data
    combos = (
        df[df["TREATMENT"] == "C"][["SITE", "LOCATION"]]
        .drop_duplicates()
        .sort_values(["SITE", "LOCATION"])
        .values.tolist()
    )

    for site, location in combos:
        label = f"{SITE_LABELS.get(site, site)} {LOC_LABELS.get(location, location)}"
        print(f"\n{label}")

        profile_year = args.year if args.year else \
            _latest_well_sampled_year(df, site, location)
        print(f"  Depth profiles for year: {profile_year}")

        plot_depth_profiles(df, site, location, profile_year, outdir)
        plot_timeseries(df, site, location, outdir)

    print("\nSalinity–temperature correlation")
    plot_salinity_temperature_correlation(df, outdir, era5_path=Path(args.era5))

    print(f"\nDone. Figures written to {outdir}")


if __name__ == "__main__":
    main()
