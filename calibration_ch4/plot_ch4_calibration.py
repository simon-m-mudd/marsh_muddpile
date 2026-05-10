#!/usr/bin/env python3
"""
plot_ch4_calibration.py

Visualise CH4 calibration results against North Inlet LTER observations.

Functions
---------
  plot_ch4_flux_comparison    Monthly CH4 flux: model vs BICEFS chamber data
  plot_porewater_profiles     Depth profiles of SO4/CH4/NH4 vs NILTREB data
  plot_smtz_comparison        Seasonal SO4/CH4 profiles at key months
  plot_plant_transport        Model CH4 flux breakdown vs root/shoot chamber ratio
  plot_ch4_timeseries         CH4 flux time series with spin-up shading
  plot_ch4_dashboard          Multi-panel summary figure

All functions accept an optional ``ax`` (or ``axes``) argument.  When called
without an axis they create and show their own figure.

Usage
-----
  # Full dashboard:
  python plot_ch4_calibration.py --nc calibration_runs/ni_ch4_best.nc \\
                                  --obs-dir data/

  # Single panel:
  python plot_ch4_calibration.py --nc calibration_runs/ni_ch4_best.nc \\
                                  --obs-dir data/ --panel flux

  # List available model variables:
  python plot_ch4_calibration.py --nc calibration_runs/ni_ch4_best.nc --list-vars
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "calibration"))

from ch4_output_reader import (
    mean_ch4_flux_monthly,
    mean_porewater_profile,
    monthly_porewater_profiles,
    smtz_depth_m,
    read_layer_variables,
)
from output_reader import read_time_series

# ---------------------------------------------------------------------------
# Colour palette (consistent across panels)
# ---------------------------------------------------------------------------
_MOD_COLOUR  = "#2171b5"   # blue  — model
_OBS_COLOUR  = "#cb181d"   # red   — observations
_SO4_COLOUR  = "#41ab5d"   # green — SO4
_CH4_COLOUR  = "#fd8d3c"   # orange — CH4
_NH4_COLOUR  = "#9e9ac8"   # purple — NH4
_BAND_ALPHA  = 0.20
_SPINUP_ALPHA = 0.10

_MONTH_LABELS = ["J","F","M","A","M","J","J","A","S","O","N","D"]


def _require_mpl():
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        return plt, ticker
    except ImportError:
        raise ImportError("matplotlib is required: pip install matplotlib")


def _load_monthly_obs(obs_dir: str | Path, chamber_type: str = "root_shoot"):
    """Load monthly_ch4_ni.csv; return (months, means, ses) arrays."""
    import pandas as pd
    path = Path(obs_dir) / "monthly_ch4_ni.csv"
    if not path.exists():
        return None, None, None
    df = pd.read_csv(path)
    df = df[df["chamber_type"] == chamber_type].copy()
    months = df["month"].values
    means  = df["ch4_mean"].values
    ses    = df["ch4_se"].values if "ch4_se" in df.columns else np.zeros_like(means)
    return months, means, ses


def _load_porewater_obs(obs_dir: str | Path, variable: str = "s2"):
    """Load porewater_profiles_ni.csv; return (depth_cm, mean, std) for `variable`."""
    import pandas as pd
    path = Path(obs_dir) / "porewater_profiles_ni.csv"
    if not path.exists():
        return None, None, None
    df = pd.read_csv(path)
    col_mean = f"{variable}_mean"
    col_std  = f"{variable}_std"
    if col_mean not in df.columns:
        return None, None, None
    df = df.dropna(subset=[col_mean])
    grp = df.groupby("depth_cm")[[col_mean, col_std]].mean().reset_index()
    return grp["depth_cm"].values, grp[col_mean].values, grp[col_std].values


def _load_smtz_obs(obs_dir: str | Path):
    """Load smtz_depth_ni.csv; return mean and (min, max) range in m."""
    import pandas as pd
    path = Path(obs_dir) / "smtz_depth_ni.csv"
    if not path.exists():
        return None, None
    df = pd.read_csv(path)
    df = df.dropna(subset=["smtz_depth_cm"])
    if df.empty:
        return None, None
    vals_m = df["smtz_depth_cm"].values / 100.0
    return float(np.mean(vals_m)), (float(vals_m.min()), float(vals_m.max()))


# ---------------------------------------------------------------------------
# 1. Monthly CH4 flux comparison
# ---------------------------------------------------------------------------

def plot_ch4_flux_comparison(
    nc_path: str,
    obs_dir: Optional[str] = None,
    skip_spinup_years: int = 3,
    chamber_type: str = "root_shoot",
    ax=None,
    title: Optional[str] = None,
) -> Tuple:
    """Bar chart: monthly CH4 flux — model vs BICEFS observations.

    Parameters
    ----------
    nc_path : path to model NetCDF output
    obs_dir : directory containing monthly_ch4_ni.csv (from extract_ni_ch4.py)
    chamber_type : "root_shoot" (default) or "root_only"

    Returns
    -------
    (fig, ax)
    """
    plt, _ = _require_mpl()

    mod_means, mod_stds, _ = mean_ch4_flux_monthly(nc_path, skip_spinup_years)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = ax.get_figure()

    x = np.arange(1, 13)
    width = 0.35

    ax.bar(x - width/2, mod_means, width,
           color=_MOD_COLOUR, alpha=0.8, label="Model",
           yerr=mod_stds, error_kw={"ecolor": _MOD_COLOUR, "elinewidth": 1.0,
                                    "capsize": 3})

    if obs_dir is not None:
        obs_months, obs_means, obs_ses = _load_monthly_obs(obs_dir, chamber_type)
        if obs_months is not None:
            # Fill full 12-month array (obs may be missing some months)
            om = np.full(12, np.nan)
            oe = np.full(12, np.nan)
            for i, m in enumerate(obs_months):
                om[m - 1] = obs_means[i]
                oe[m - 1] = obs_ses[i]
            ax.bar(x + width/2, om, width,
                   color=_OBS_COLOUR, alpha=0.8, label=f"Obs ({chamber_type})",
                   yerr=oe, error_kw={"ecolor": _OBS_COLOUR, "elinewidth": 1.0,
                                      "capsize": 3})

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(_MONTH_LABELS)
    ax.set_xlabel("Month")
    ax.set_ylabel(r"CH$_4$ flux ($\mu$mol m$^{-2}$ s$^{-1}$)")
    ax.set_title(title or "Monthly CH₄ flux")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)

    if own_fig:
        plt.tight_layout()
        plt.show()

    return fig, ax


# ---------------------------------------------------------------------------
# 2. Porewater depth profiles
# ---------------------------------------------------------------------------

def plot_porewater_profiles(
    nc_path: str,
    obs_dir: Optional[str] = None,
    skip_spinup_years: int = 3,
    axes=None,
    title: Optional[str] = None,
) -> Tuple:
    """Three-panel depth profiles: SO4, CH4, NH4 vs NILTREB observations.

    Parameters
    ----------
    axes : sequence of 3 Axes, or None to create a new figure

    Returns
    -------
    (fig, axes)
    """
    plt, _ = _require_mpl()

    own_fig = axes is None
    if own_fig:
        fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharey=True)
    else:
        fig = axes[0].get_figure()

    specs = [
        ("layer_porewater_so4", "SO₄ (μmol L⁻¹)",  _SO4_COLOUR, "s2",       "S²⁻ (μmol L⁻¹)"),
        ("layer_porewater_ch4", "CH₄ (μmol L⁻¹)",  _CH4_COLOUR, None,       None),
        ("layer_porewater_nh4", "NH₄ (μmol L⁻¹)",  _NH4_COLOUR, "nh4",      "NH₄ (μmol L⁻¹)"),
    ]

    for ax, (mod_var, xlabel, colour, obs_var, obs_xlabel) in zip(axes, specs):
        try:
            depth_m, conc = mean_porewater_profile(nc_path, mod_var, skip_spinup_years)
        except KeyError:
            ax.text(0.5, 0.5, f"{mod_var}\nnot in output",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8)
            continue

        ax.plot(conc, depth_m * 100.0, color=colour, linewidth=2.0, label="Model")
        ax.fill_betweenx(depth_m * 100.0, 0, conc, color=colour, alpha=0.15)

        if obs_dir is not None and obs_var is not None:
            od, om, os_ = _load_porewater_obs(obs_dir, obs_var)
            if od is not None:
                ax.errorbar(om, od, xerr=os_, fmt="o", color=_OBS_COLOUR,
                            markersize=5, linewidth=1.2, capsize=3,
                            label=f"Obs ({obs_var})")

        ax.set_xlabel(xlabel, fontsize=9)
        ax.invert_yaxis()
        ax.set_ylim(bottom=0)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(fontsize=7)

    axes[0].set_ylabel("Depth below surface (cm)")
    if title:
        fig.suptitle(title, fontsize=11)

    if own_fig:
        plt.tight_layout()
        plt.show()

    return fig, axes


# ---------------------------------------------------------------------------
# 3. Seasonal SO4/CH4 profiles
# ---------------------------------------------------------------------------

def plot_smtz_comparison(
    nc_path: str,
    obs_dir: Optional[str] = None,
    skip_spinup_years: int = 3,
    months_shown: tuple = (1, 4, 7, 10),
    axes=None,
    title: Optional[str] = None,
) -> Tuple:
    """Seasonal SO4 and CH4 profiles at four months, with SMTZ marker.

    Parameters
    ----------
    months_shown : tuple of 1-based month indices to plot (default: Jan/Apr/Jul/Oct)
    axes : sequence of 4 Axes (one per month), or None

    Returns
    -------
    (fig, axes)
    """
    plt, _ = _require_mpl()

    own_fig = axes is None
    if own_fig:
        fig, axes = plt.subplots(1, len(months_shown), figsize=(13, 5), sharey=True)
    else:
        fig = axes[0].get_figure()

    try:
        _, depth_m, so4_monthly = monthly_porewater_profiles(
            nc_path, "layer_porewater_so4", skip_spinup_years)
        _, _,       ch4_monthly = monthly_porewater_profiles(
            nc_path, "layer_porewater_ch4", skip_spinup_years)
    except KeyError as e:
        for ax in axes:
            ax.text(0.5, 0.5, str(e), ha="center", va="center",
                    transform=ax.transAxes, fontsize=7)
        return fig, axes

    smtz_m = smtz_depth_m(nc_path, skip_spinup_years)

    obs_smtz_mean, obs_smtz_range = None, None
    if obs_dir is not None:
        obs_smtz_mean, obs_smtz_range = _load_smtz_obs(obs_dir)

    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    for ax, m in zip(axes, months_shown):
        idx = m - 1
        depth_cm = depth_m * 100.0

        ax.plot(so4_monthly[idx], depth_cm, color=_SO4_COLOUR,
                linewidth=1.8, label="SO₄")
        ax.plot(ch4_monthly[idx], depth_cm, color=_CH4_COLOUR,
                linewidth=1.8, label="CH₄")

        if not np.isnan(smtz_m):
            ax.axhline(smtz_m * 100.0, color="grey", linewidth=1.2,
                       linestyle="--", alpha=0.8, label="SMTZ (mod)")
        if obs_smtz_mean is not None:
            ax.axhline(obs_smtz_mean * 100.0, color=_OBS_COLOUR,
                       linewidth=1.2, linestyle=":", label="SMTZ (obs)")
            if obs_smtz_range is not None:
                ax.axhspan(obs_smtz_range[0] * 100.0, obs_smtz_range[1] * 100.0,
                           color=_OBS_COLOUR, alpha=0.10)

        ax.set_title(month_names[idx], fontsize=10)
        ax.set_xlabel("Concentration\n(μmol L⁻¹)", fontsize=8)
        ax.invert_yaxis()
        ax.set_ylim(bottom=0)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        if m == months_shown[0]:
            ax.legend(fontsize=7)

    axes[0].set_ylabel("Depth below surface (cm)")
    if title:
        fig.suptitle(title, fontsize=11)

    if own_fig:
        plt.tight_layout()
        plt.show()

    return fig, axes


# ---------------------------------------------------------------------------
# 4. Plant transport contribution
# ---------------------------------------------------------------------------

def plot_plant_transport(
    nc_path: str,
    obs_dir: Optional[str] = None,
    skip_spinup_years: int = 3,
    ax=None,
    title: Optional[str] = None,
) -> Tuple:
    """Monthly ratio of total / diffusive CH4 flux vs BICEFS root+shoot / root-only ratio.

    The model does not separately output diffusive and plant-transport fluxes;
    this panel instead shows modelled total flux alongside the observed
    root_shoot / root_only ratio (right axis) to check qualitative seasonality.

    Returns
    -------
    (fig, ax)
    """
    plt, _ = _require_mpl()

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    mod_means, _, _ = mean_ch4_flux_monthly(nc_path, skip_spinup_years)
    x = np.arange(1, 13)

    ax.bar(x, mod_means, color=_MOD_COLOUR, alpha=0.7, label="Model total flux")
    ax.set_ylabel(r"CH$_4$ flux ($\mu$mol m$^{-2}$ s$^{-1}$)", color=_MOD_COLOUR)
    ax.tick_params(axis="y", labelcolor=_MOD_COLOUR)

    if obs_dir is not None:
        import pandas as pd
        ratio_path = Path(obs_dir) / "plant_transport_ni.csv"
        if ratio_path.exists():
            df = pd.read_csv(ratio_path)
            ax2 = ax.twinx()
            ax2.plot(df["month"], df["ratio"], "o-", color=_OBS_COLOUR,
                     markersize=5, linewidth=1.5, label="Obs shoot/root ratio")
            ax2.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
            ax2.set_ylabel("Root+shoot / root-only ratio", color=_OBS_COLOUR)
            ax2.tick_params(axis="y", labelcolor=_OBS_COLOUR)
            ax2.legend(fontsize=8, loc="upper right")

    ax.set_xticks(x)
    ax.set_xticklabels(_MONTH_LABELS)
    ax.set_xlabel("Month")
    ax.set_title(title or "Plant-mediated CH₄ transport")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)

    if own_fig:
        plt.tight_layout()
        plt.show()

    return fig, ax


# ---------------------------------------------------------------------------
# 5. CH4 flux time series
# ---------------------------------------------------------------------------

def plot_ch4_timeseries(
    nc_path: str,
    skip_spinup_years: int = 3,
    ax=None,
    title: Optional[str] = None,
) -> Tuple:
    """Full CH4 flux time series with spin-up shading.

    Returns
    -------
    (fig, ax)
    """
    plt, _ = _require_mpl()

    ts = read_time_series(nc_path)
    time_days = ts["model_time_days"]
    time_years = time_days / 365.25
    flux = ts.get("surface_ch4_flux_umol_m2_s")

    if flux is None:
        raise KeyError("surface_ch4_flux_umol_m2_s not in NetCDF. "
                       "Run with methane_model_name: sulfate_methane.")

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(11, 3.5))
    else:
        fig = ax.get_figure()

    cutoff = skip_spinup_years
    ax.axvspan(0, cutoff, color="grey", alpha=_SPINUP_ALPHA, label="spin-up")
    ax.plot(time_years, flux, color=_MOD_COLOUR, linewidth=0.9, label="CH₄ flux")
    ax.axhline(0, color="black", linewidth=0.6)

    ax.set_xlabel("Model time (years)")
    ax.set_ylabel(r"CH$_4$ flux ($\mu$mol m$^{-2}$ s$^{-1}$)")
    ax.set_title(title or "Surface CH₄ flux — model time series")
    ax.set_xlim(0, time_years[-1])
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    if own_fig:
        plt.tight_layout()
        plt.show()

    return fig, ax


# ---------------------------------------------------------------------------
# 6. Multi-panel dashboard
# ---------------------------------------------------------------------------

def plot_ch4_dashboard(
    nc_path: str,
    site_key: str = "NI",
    obs_dir: Optional[str] = None,
    skip_spinup_years: int = 3,
    out_path: Optional[str] = None,
) -> Tuple:
    """Six-panel CH4 calibration summary figure.

    Layout
    ------
    Row 0:  [CH4 time series                            ]
    Row 1:  [Monthly flux vs obs] [SO4 profile] [CH4 profile] [NH4 profile]
    Row 2:  [Seasonal SMTZ (Jan / Apr / Jul / Oct)          ] [Plant transport]

    Parameters
    ----------
    out_path : if provided, save figure to this path instead of showing

    Returns
    -------
    (fig, axes_dict)  — axes_dict keyed by panel name
    """
    plt, _ = _require_mpl()
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(16, 13))
    fig.suptitle(f"CH₄ calibration — {site_key}   [{os.path.basename(nc_path)}]",
                 fontsize=12, y=0.99)

    gs = gridspec.GridSpec(3, 6, figure=fig,
                           hspace=0.52, wspace=0.55)

    ax_ts      = fig.add_subplot(gs[0, :])
    ax_flux    = fig.add_subplot(gs[1, 0:2])
    ax_so4     = fig.add_subplot(gs[1, 2])
    ax_ch4p    = fig.add_subplot(gs[1, 3], sharey=ax_so4)
    ax_nh4     = fig.add_subplot(gs[1, 4:], sharey=ax_so4)
    ax_s1      = fig.add_subplot(gs[2, 0])
    ax_s2      = fig.add_subplot(gs[2, 1], sharey=ax_s1)
    ax_s3      = fig.add_subplot(gs[2, 2], sharey=ax_s1)
    ax_s4      = fig.add_subplot(gs[2, 3], sharey=ax_s1)
    ax_plant   = fig.add_subplot(gs[2, 4:])

    # --- Panel 1: time series ---
    try:
        plot_ch4_timeseries(nc_path, skip_spinup_years, ax=ax_ts)
        ax_ts.set_title("CH₄ flux time series", fontsize=9)
    except KeyError as e:
        ax_ts.text(0.5, 0.5, str(e), ha="center", va="center",
                   transform=ax_ts.transAxes, fontsize=8)

    # --- Panel 2: monthly flux ---
    try:
        plot_ch4_flux_comparison(nc_path, obs_dir, skip_spinup_years,
                                  ax=ax_flux, title="Monthly CH₄ flux")
    except Exception as e:
        ax_flux.text(0.5, 0.5, str(e), ha="center", va="center",
                     transform=ax_flux.transAxes, fontsize=7)

    # --- Panels 3–5: porewater profiles ---
    try:
        plot_porewater_profiles(nc_path, obs_dir, skip_spinup_years,
                                 axes=[ax_so4, ax_ch4p, ax_nh4])
        for ax, t in zip([ax_so4, ax_ch4p, ax_nh4], ["SO₄", "CH₄ (pore)", "NH₄"]):
            ax.set_title(t, fontsize=9)
            ax.set_xlabel("μmol L⁻¹", fontsize=8)
    except Exception as e:
        ax_so4.text(0.5, 0.5, str(e), ha="center", va="center",
                    transform=ax_so4.transAxes, fontsize=7)

    # --- Panels 6–9: seasonal SMTZ ---
    try:
        plot_smtz_comparison(nc_path, obs_dir, skip_spinup_years,
                              months_shown=(1, 4, 7, 10),
                              axes=[ax_s1, ax_s2, ax_s3, ax_s4])
    except Exception as e:
        ax_s1.text(0.5, 0.5, str(e), ha="center", va="center",
                   transform=ax_s1.transAxes, fontsize=7)

    # --- Panel 10: plant transport ---
    try:
        plot_plant_transport(nc_path, obs_dir, skip_spinup_years,
                              ax=ax_plant, title="Plant transport")
    except Exception as e:
        ax_plant.text(0.5, 0.5, str(e), ha="center", va="center",
                      transform=ax_plant.transAxes, fontsize=7)

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Dashboard saved to {out_path}")
    else:
        plt.show()

    axes_dict = {
        "timeseries": ax_ts,
        "monthly_flux": ax_flux,
        "so4_profile": ax_so4,
        "ch4_profile": ax_ch4p,
        "nh4_profile": ax_nh4,
        "smtz_jan": ax_s1,
        "smtz_apr": ax_s2,
        "smtz_jul": ax_s3,
        "smtz_oct": ax_s4,
        "plant_transport": ax_plant,
    }
    return fig, axes_dict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--nc", required=True,
                        help="Path to model NetCDF output file")
    parser.add_argument("--obs-dir", default=None,
                        help="Directory with extract_ni_*.py output CSVs "
                             "(default: calibration_ch4/data/)")
    parser.add_argument("--site", default="NI",
                        help="Site label for figure titles (default: NI)")
    parser.add_argument("--spinup", type=int, default=3,
                        help="Spin-up years to skip (default: 3)")
    parser.add_argument("--out", default=None,
                        help="Save figure to this path instead of showing")
    parser.add_argument("--panel",
                        choices=["flux", "profiles", "smtz",
                                 "plant", "timeseries", "dashboard"],
                        default="dashboard",
                        help="Which panel to draw (default: dashboard)")
    parser.add_argument("--list-vars", action="store_true",
                        help="Print available variables in the NetCDF and exit")
    return parser.parse_args()


def main():
    args = _parse_args()
    nc_path = args.nc

    if args.list_vars:
        ts = read_time_series(nc_path)
        lv = read_layer_variables(nc_path)
        print("Time-series variables:")
        for k in sorted(ts):
            print(f"  {k}")
        print("Layer (2-D) variables:")
        for k in sorted(lv):
            print(f"  {k}")
        return

    obs_dir = args.obs_dir
    if obs_dir is None:
        candidate = os.path.join(os.path.dirname(__file__), "data")
        obs_dir = candidate if os.path.isdir(candidate) else None

    panel = args.panel

    if panel == "dashboard":
        plot_ch4_dashboard(nc_path, site_key=args.site, obs_dir=obs_dir,
                            skip_spinup_years=args.spinup, out_path=args.out)
    elif panel == "flux":
        plot_ch4_flux_comparison(nc_path, obs_dir, args.spinup)
    elif panel == "profiles":
        plot_porewater_profiles(nc_path, obs_dir, args.spinup)
    elif panel == "smtz":
        plot_smtz_comparison(nc_path, obs_dir, args.spinup)
    elif panel == "plant":
        plot_plant_transport(nc_path, obs_dir, args.spinup)
    elif panel == "timeseries":
        plot_ch4_timeseries(nc_path, args.spinup)


if __name__ == "__main__":
    main()
