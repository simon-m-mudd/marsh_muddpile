"""
porewater_chemistry_north_inlet.py

Investigate drivers of porewater chemistry at North Inlet LTREB sites.

Combines three data sources:
  - Porewater chemistry (edi.136.11): monthly depth profiles 1993-2025
  - Aboveground biomass  (edi.135.12): monthly biomass 1984-2025
  - NOAA tidal harmonics (calibration/site_data_ni.json): used to reconstruct
    monthly inundation fractions from tidal harmonic constituents

For each site/location the top-layer (10 cm) mean of SALINITY, NH4, and S2
(sulfide) is correlated against:
  (a) modelled monthly inundation fraction
  (b) observed monthly mean aboveground biomass

Tidal inundation model
----------------------
Hourly water levels are reconstructed for each calendar month using M2, S2,
N2, K1, O1 (the five dominant semi-diurnal and diurnal constituents) plus SA
and SSA (seasonal mean-water-level corrections).  Constituents are phase-
advanced from a fixed epoch (1900-01-01 00:00 UTC) using published angular
speeds and the NOAA Greenwich phase lags.  The 18.6-year nodal modulation is
ignored (small for monthly statistics).

Inundation fraction = fraction of hours in a 30-day window where
reconstructed water level > site elevation above MSL.

Datum conversion (North Inlet, station 8661070)
  NAVD88 datum offset = 9.890 m (station datum)
  MSL in NAVD88       = MSL_station - NAVD88_station = 9.754 - 9.890 = −0.136 m
  Site elevation above MSL = elev_NAVD88 − MSL_NAVD88

Usage
-----
    python sensitivity/porewater_chemistry_north_inlet.py
    python sensitivity/porewater_chemistry_north_inlet.py --outdir viz/porewater
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.rcParams.update({"font.size": 9})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO = Path(__file__).parent.parent
_PW_PATH    = _REPO / "lter_data" / "edi.136.11" / "NILTREB_porewater.csv"
_BIO_PATH   = _REPO / "lter_data" / "edi.135.12" / "NILTREB_plants_aboveground_biomass_density.csv"
_TIDES_PATH = _REPO / "calibration" / "site_data_ni.json"

# ---------------------------------------------------------------------------
# Site definitions — elevations measured at each plot (NAVD88)
# ---------------------------------------------------------------------------

SITES = {
    ("GI", "HM"): {"elev_navd88": 0.50, "label": "Goat Island High Marsh"},
    ("GI", "LM"): {"elev_navd88": 0.20, "label": "Goat Island Low Marsh"},
    ("OL", "HM"): {"elev_navd88": 0.45, "label": "Oyster Landing High Marsh"},
    ("OL", "LM"): {"elev_navd88": 0.25, "label": "Oyster Landing Low Marsh"},
}

# North Inlet datum: MSL in NAVD88 and MHW, from NOAA station 8661070
MSL_NAVD88 = -0.136   # m  (MSL_station 9.754 − NAVD88_offset 9.890)
MHW_NAVD88 =  0.626   # m  reference
MLW_NAVD88 = -0.905   # m  reference

# Porewater top layer depth
TOP_DEPTH_CM = 10

# Tidal constituent angular speeds (degrees per hour)
SPEEDS_DEG_PER_HR = {
    "M2": 28.9841, "S2": 30.0000, "N2": 28.4397,
    "K1": 15.0411, "O1": 13.9430,
    "SA":  0.0411, "SSA": 0.0823,
}

# Reference epoch for harmonic reconstruction
_EPOCH = datetime(1900, 1, 1, 0, 0, 0)

CHEM_VARS = {
    "SALINITY": "Porewater salinity (ppt)",
    "NH4":      "Porewater NH₄ (µM)",
    "S2":       "Porewater S²⁻ sulfide (µM)",
}

SITE_COLORS = {
    ("GI", "HM"): "#e41a1c",
    ("GI", "LM"): "#ff7f00",
    ("OL", "HM"): "#377eb8",
    ("OL", "LM"): "#4daf4a",
}

# ---------------------------------------------------------------------------
# Tidal inundation model
# ---------------------------------------------------------------------------

def _hours_from_epoch(year: int, month: int) -> float:
    """Hours elapsed from _EPOCH to the first of year/month."""
    t = datetime(year, month, 1)
    return (t - _EPOCH).total_seconds() / 3600.0


def _monthly_inundation(
    site_elev_navd88: float,
    year: int,
    month: int,
    harmonics: dict,
    n_days: int = 30,
) -> float:
    """
    Reconstruct hourly water levels for n_days starting on year/month day 1
    and return the fraction of hours that exceed site elevation (above MSL).

    Water level is computed relative to MSL.  Site elevation above MSL is:
        z = site_elev_navd88 - MSL_NAVD88

    SA and SSA are treated as slowly-varying seasonal MSL corrections:
    their contribution is computed at the mid-month time and added as a
    constant offset, avoiding beating with the short-period constituents.
    """
    z_msl = site_elev_navd88 - MSL_NAVD88  # site height above MSL (m)

    t0_hrs = _hours_from_epoch(year, month)
    t_local = np.arange(n_days * 24, dtype=float)  # 0, 1, ..., n_days*24-1 hours

    eta = np.zeros(len(t_local))

    # Short-period constituents: reconstruct hourly
    for name in ("M2", "S2", "N2", "K1", "O1"):
        if name not in harmonics:
            continue
        amp   = harmonics[name]["amplitude_m"]
        g_deg = harmonics[name]["phase_deg"]
        speed = SPEEDS_DEG_PER_HR[name]
        # Phase at first hour of month (degrees)
        phase0 = (speed * t0_hrs - g_deg) % 360.0
        eta += amp * np.cos(np.radians(speed * t_local + phase0))

    # Long-period constituents (SA, SSA): use mid-month value as DC offset
    t_mid = t0_hrs + n_days * 12.0
    for name in ("SA", "SSA"):
        if name not in harmonics:
            continue
        amp   = harmonics[name]["amplitude_m"]
        g_deg = harmonics[name]["phase_deg"]
        speed = SPEEDS_DEG_PER_HR[name]
        phase_mid = (speed * t_mid - g_deg) % 360.0
        eta += amp * np.cos(np.radians(phase_mid))

    return float(np.mean(eta > z_msl))


def compute_inundation_table(harmonics: dict) -> pd.DataFrame:
    """
    Compute monthly inundation fractions for all sites and all year/months
    present in the porewater record (1993-2025).
    Returns a DataFrame with columns:
        SITE, LOCATION, YEAR, MONTH, inundation_fraction
    """
    rows = []
    for (site, loc), sinfo in SITES.items():
        elev = sinfo["elev_navd88"]
        for year in range(1993, 2026):
            for month in range(1, 13):
                f = _monthly_inundation(elev, year, month, harmonics)
                rows.append({
                    "SITE": site, "LOCATION": loc,
                    "YEAR": year, "MONTH": month,
                    "inundation_fraction": f,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_porewater() -> pd.DataFrame:
    df = pd.read_csv(_PW_PATH, parse_dates=["DATE"])
    df["YEAR"]  = df["DATE"].dt.year
    df["MONTH"] = df["DATE"].dt.month
    for col in ("SALINITY", "NH4", "S2"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Top layer, control plots only
    top = df[
        (df["TREATMENT"] == "C") &
        (df["DEPTH"] == TOP_DEPTH_CM)
    ].copy()
    return (
        top.groupby(["SITE", "LOCATION", "YEAR", "MONTH"])[list(CHEM_VARS)]
           .mean()
           .reset_index()
    )


def load_biomass() -> pd.DataFrame:
    df = pd.read_csv(_BIO_PATH, parse_dates=["DATE"])
    df["YEAR"]  = df["DATE"].dt.year
    df["MONTH"] = df["DATE"].dt.month
    ctrl = df[df["TREATMENT"] == "C"]
    return (
        ctrl.groupby(["SITE", "LOCATION", "YEAR", "MONTH"])["ABOVEGROUND_BIOMASS"]
            .mean()
            .reset_index()
    )


def load_harmonics() -> dict:
    with open(_TIDES_PATH) as fh:
        return json.load(fh)["harmonics"]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _scatter_with_fit(ax, x, y, color, xlabel, ylabel, title):
    """Scatter + linear regression with r and p annotation."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 6:
        ax.text(0.5, 0.5, "insufficient data", transform=ax.transAxes,
                ha="center", va="center", color="grey")
        ax.set_title(title)
        return

    ax.scatter(x, y, c=color, s=8, alpha=0.45, linewidths=0)

    slope, intercept, r, p, _ = stats.linregress(x, y)
    x_fit = np.linspace(x.min(), x.max(), 80)
    ax.plot(x_fit, slope * x_fit + intercept, color=color, linewidth=1.8)

    pstr = f"p={p:.2e}" if p < 0.001 else f"p={p:.3f}"
    ax.text(0.05, 0.95, f"r = {r:.2f}\n{pstr}\nn = {len(x)}",
            transform=ax.transAxes, va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=8)
    ax.grid(linewidth=0.3, color="0.88")


# ---------------------------------------------------------------------------
# Figure 1 — site-by-site dashboards (inundation + biomass vs chemistry)
# ---------------------------------------------------------------------------

def plot_site_dashboard(merged: pd.DataFrame, site: str, location: str,
                        outdir: Path) -> None:
    """
    3 rows (SALINITY, NH4, S2) × 2 cols (vs inundation, vs biomass) for one site.
    """
    sub = merged[
        (merged["SITE"] == site) &
        (merged["LOCATION"] == location)
    ].copy()

    if sub.empty:
        return

    color = SITE_COLORS.get((site, location), "#555555")
    label = SITES[(site, location)]["label"]

    fig, axes = plt.subplots(3, 2, figsize=(9, 10))
    fig.suptitle(
        f"{label}\n"
        f"Porewater chemistry ({TOP_DEPTH_CM} cm depth, control plots) vs "
        f"tidal inundation and aboveground biomass",
        fontsize=10, fontweight="bold",
    )

    for row, (var, vlabel) in enumerate(CHEM_VARS.items()):
        ax_inn  = axes[row, 0]
        ax_bio  = axes[row, 1]

        _scatter_with_fit(
            ax_inn,
            sub["inundation_fraction"].values,
            sub[var].values,
            color=color,
            xlabel="Monthly inundation fraction",
            ylabel=vlabel,
            title=f"{var} vs inundation",
        )
        _scatter_with_fit(
            ax_bio,
            sub["ABOVEGROUND_BIOMASS"].values,
            sub[var].values,
            color=color,
            xlabel="Aboveground biomass (g m⁻²)",
            ylabel=vlabel,
            title=f"{var} vs biomass",
        )

    fig.tight_layout()
    out = outdir / f"pw_drivers_{site}_{location}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — all-sites summary: one panel per chemical variable
# ---------------------------------------------------------------------------

def plot_all_sites_summary(merged: pd.DataFrame, outdir: Path) -> None:
    """
    Two summary figures (vs inundation, vs biomass), each with 3 rows
    (one per chemical) and 4 columns (one per site/location).
    """
    combos = [("GI", "HM"), ("GI", "LM"), ("OL", "HM"), ("OL", "LM")]

    for predictor, xlabel, fname_tag in [
        ("inundation_fraction", "Monthly inundation fraction", "inundation"),
        ("ABOVEGROUND_BIOMASS", "Aboveground biomass (g m⁻²)", "biomass"),
    ]:
        fig, axes = plt.subplots(
            len(CHEM_VARS), len(combos),
            figsize=(4 * len(combos), 3.5 * len(CHEM_VARS)),
            sharey="row",
        )

        for col, (site, loc) in enumerate(combos):
            sub = merged[(merged["SITE"] == site) & (merged["LOCATION"] == loc)]
            color = SITE_COLORS[(site, loc)]
            label = SITES[(site, loc)]["label"]

            for row, (var, vlabel) in enumerate(CHEM_VARS.items()):
                ax = axes[row, col]
                _scatter_with_fit(
                    ax,
                    sub[predictor].values,
                    sub[var].values,
                    color=color,
                    xlabel=xlabel if row == len(CHEM_VARS) - 1 else "",
                    ylabel=vlabel if col == 0 else "",
                    title=label if row == 0 else "",
                )

        fig.suptitle(
            f"North Inlet porewater chemistry ({TOP_DEPTH_CM} cm) vs {xlabel.lower()}\n"
            f"(monthly means, control plots)",
            fontsize=10, fontweight="bold",
        )
        fig.tight_layout()
        out = outdir / f"pw_vs_{fname_tag}_all_sites.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved {out}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — seasonal climatology: mean inundation, biomass, chemistry by month
# ---------------------------------------------------------------------------

def plot_seasonal_climatology(merged: pd.DataFrame, outdir: Path) -> None:
    """
    Mean seasonal cycle (months 1–12) of inundation fraction, biomass, and
    porewater chemistry for each site.  Four panels in a 2×2 grid.
    """
    months = np.arange(1, 13)
    month_labels = ["J","F","M","A","M","J","J","A","S","O","N","D"]
    combos = [("GI","HM"), ("GI","LM"), ("OL","HM"), ("OL","LM")]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes_flat = axes.flatten()

    for ax, (site, loc) in zip(axes_flat, combos):
        sub = merged[(merged["SITE"] == site) & (merged["LOCATION"] == loc)]
        color = SITE_COLORS[(site, loc)]
        label = SITES[(site, loc)]["label"]

        clim = sub.groupby("MONTH")[
            ["inundation_fraction", "ABOVEGROUND_BIOMASS"] + list(CHEM_VARS)
        ].mean()

        ax2 = ax.twinx()
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("outward", 55))

        # Inundation — left axis
        ax.bar(months, clim.reindex(months)["inundation_fraction"],
               color=color, alpha=0.30, width=0.8, label="Inundation fraction")
        ax.set_ylabel("Inundation fraction", color=color)
        ax.set_ylim(0, min(1.0, clim["inundation_fraction"].max() * 2.5 + 0.05))
        ax.tick_params(axis="y", labelcolor=color)

        # Salinity — right axis 1
        sal = clim.reindex(months)["SALINITY"]
        ax2.plot(months, sal, "o-", color="navy", linewidth=1.5,
                 markersize=4, label="Salinity (ppt)")
        ax2.set_ylabel("Salinity (ppt)", color="navy")
        ax2.tick_params(axis="y", labelcolor="navy")

        # Biomass — right axis 2
        bio = clim.reindex(months)["ABOVEGROUND_BIOMASS"]
        ax3.plot(months, bio, "s--", color="darkgreen", linewidth=1.3,
                 markersize=4, label="Biomass (g m⁻²)")
        ax3.set_ylabel("Biomass (g m⁻²)", color="darkgreen")
        ax3.tick_params(axis="y", labelcolor="darkgreen")

        ax.set_xticks(months)
        ax.set_xticklabels(month_labels)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.grid(axis="x", linewidth=0.3, color="0.88")

        # Combined legend
        lines = (
            [matplotlib.patches.Patch(color=color, alpha=0.4, label="Inundation fraction")] +
            [matplotlib.lines.Line2D([0],[0], color="navy", marker="o", label="Salinity (ppt)")] +
            [matplotlib.lines.Line2D([0],[0], color="darkgreen", marker="s",
                                     linestyle="--", label="Biomass (g m⁻²)")]
        )
        ax.legend(handles=lines, fontsize=7, loc="upper left", framealpha=0.7)

    fig.suptitle(
        "North Inlet — mean seasonal cycle of inundation, biomass, and porewater salinity\n"
        "(climatological monthly means, control plots)",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    out = outdir / "seasonal_climatology_all_sites.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — inundation fraction time series (diagnostic)
# ---------------------------------------------------------------------------

def plot_inundation_timeseries(inund: pd.DataFrame, outdir: Path) -> None:
    """
    Monthly inundation fractions over the full record for each site,
    showing the seasonal and spring-neap driven variability.
    """
    combos = [("GI","HM"), ("GI","LM"), ("OL","HM"), ("OL","LM")]
    fig, axes = plt.subplots(len(combos), 1, figsize=(14, 8), sharex=True)

    for ax, (site, loc) in zip(axes, combos):
        sub = inund[(inund["SITE"] == site) & (inund["LOCATION"] == loc)].copy()
        sub["date"] = pd.to_datetime(
            sub["YEAR"].astype(str) + "-" + sub["MONTH"].astype(str).str.zfill(2) + "-15"
        )
        color = SITE_COLORS[(site, loc)]
        label = SITES[(site, loc)]["label"]
        elev  = SITES[(site, loc)]["elev_navd88"]

        ax.plot(sub["date"], sub["inundation_fraction"], color=color,
                linewidth=0.8, label=label)
        ax.fill_between(sub["date"], 0, sub["inundation_fraction"],
                        color=color, alpha=0.18)

        ax.set_ylabel("Inundation\nfraction")
        ax.text(0.01, 0.93, f"{label}  (elev {elev:.2f} m NAVD88)",
                transform=ax.transAxes, va="top", fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(axis="x", linewidth=0.3, color="0.88")

    axes[-1].set_xlabel("Date")
    fig.suptitle(
        "North Inlet — reconstructed monthly inundation fractions\n"
        "(harmonic tidal reconstruction, M2+S2+N2+K1+O1+SA+SSA)",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    out = outdir / "inundation_timeseries.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Porewater chemistry drivers at North Inlet LTREB."
    )
    parser.add_argument(
        "--outdir",
        default=str(_REPO / "viz" / "porewater"),
        help="Output directory for figures (default: viz/porewater/)",
    )
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading harmonic constituents ...")
    harmonics = load_harmonics()

    print("Computing monthly inundation fractions (1993–2025) ...")
    inund = compute_inundation_table(harmonics)

    # Print summary inundation by site
    clim = inund.groupby(["SITE", "LOCATION"])["inundation_fraction"].agg(
        ["mean", "min", "max"]
    )
    print(clim.to_string())

    print("\nLoading porewater chemistry ...")
    pw = load_porewater()

    print("Loading biomass ...")
    bio = load_biomass()

    print("Merging datasets ...")
    merged = (
        pw.merge(inund, on=["SITE", "LOCATION", "YEAR", "MONTH"], how="left")
          .merge(bio,   on=["SITE", "LOCATION", "YEAR", "MONTH"], how="left")
    )
    # Keep only rows with at least one chemistry observation
    merged = merged.dropna(subset=["SALINITY", "NH4", "S2"], how="all")
    print(f"  Merged rows: {len(merged)}")

    print("\nPlotting inundation time series ...")
    plot_inundation_timeseries(inund, outdir)

    print("Plotting seasonal climatology ...")
    plot_seasonal_climatology(merged, outdir)

    print("Plotting all-sites summary ...")
    plot_all_sites_summary(merged, outdir)

    print("Plotting per-site dashboards ...")
    for site, loc in SITES:
        plot_site_dashboard(merged, site, loc, outdir)

    print(f"\nDone.  Figures written to {outdir}")


if __name__ == "__main__":
    main()
