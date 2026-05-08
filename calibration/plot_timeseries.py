#!/usr/bin/env python3
"""
plot_timeseries.py

Visualise marsh_muddpile model output against North Inlet LTER observations.

Functions
---------
  plot_variable_timeseries    Raw model variable(s) vs model time
  plot_anpp_comparison        Annual ANPP: model time series + observed mean band
  plot_seasonal_biomass       Mean seasonal cycle: model vs monthly obs
  plot_site_dashboard         Four-panel summary for one NI site

All functions accept an optional ``ax`` (or ``axes``) argument so they can be
embedded in larger figures.  When called without an axis they create and show
their own figure.

Usage
-----
  # Quick dashboard for a best-fit run:
  python plot_timeseries.py --nc calibration_runs/ni_gi_hm_c_best.nc \\
                             --site GI --location HM --treatment C

  # Single-variable time series:
  python plot_timeseries.py --nc calibration_runs/ni_gi_hm_c_best.nc \\
                             --variable aboveground_biomass_kg_m2

  # Show available variable names in a NetCDF file:
  python plot_timeseries.py --nc calibration_runs/ni_gi_hm_c_best.nc --list-vars
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from lter_reader import read_ni_annual_productivity, read_ni_monthly_biomass
from output_reader import annual_anpp_g_m2_yr, read_time_series

# ---------------------------------------------------------------------------
# Colour palette (consistent across panels)
# ---------------------------------------------------------------------------
_MOD_COLOUR = "#2171b5"   # blue for model
_OBS_COLOUR = "#cb181d"   # red for observations
_BAND_ALPHA = 0.20
_SPINUP_ALPHA = 0.10


def _require_mpl():
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        return plt, ticker
    except ImportError:
        raise ImportError("matplotlib is required for plotting: pip install matplotlib")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_month_index(time_days: np.ndarray) -> np.ndarray:
    """Return 1-based month-of-year for each model timestep."""
    month_float = (time_days % 365.25) / (365.25 / 12.0)
    return (month_float.astype(int) % 12) + 1  # 1–12


def _model_year_index(time_days: np.ndarray) -> np.ndarray:
    return (time_days / 365.25).astype(int)


def _spinup_cutoff(time_days: np.ndarray, skip_spinup_years: int) -> float:
    return skip_spinup_years * 365.25


# ---------------------------------------------------------------------------
# 1. Raw model variable(s) time series
# ---------------------------------------------------------------------------

def plot_variable_timeseries(
    nc_path: str,
    variables: Sequence[str],
    skip_spinup_years: int = 3,
    ax=None,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> Tuple:
    """Plot one or more model variables against model time (years).

    Parameters
    ----------
    nc_path:
        Path to the marsh_muddpile NetCDF output file.
    variables:
        Variable names to plot (must exist in the NetCDF).
    skip_spinup_years:
        Shade this many years at the start as spin-up.
    ax:
        Existing matplotlib Axes.  If None a new figure is created.
    title, ylabel:
        Optional axis labels.

    Returns
    -------
    (fig, ax)
    """
    plt, ticker = _require_mpl()

    series = read_time_series(nc_path)
    time_days = series["model_time_days"]
    time_years = time_days / 365.25
    cutoff_y = skip_spinup_years

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    # Spin-up shading
    ax.axvspan(0, cutoff_y, color="grey", alpha=_SPINUP_ALPHA, label="spin-up")

    for i, var in enumerate(variables):
        if var not in series:
            raise KeyError(f"Variable '{var}' not found in {nc_path}. "
                           f"Available: {list(series)}")
        ax.plot(time_years, series[var],
                label=var.replace("_", " "),
                linewidth=1.2,
                color=f"C{i}")

    ax.set_xlabel("Model time (years)")
    ax.set_ylabel(ylabel or "")
    ax.set_title(title or os.path.basename(nc_path))
    ax.set_xlim(0, time_years[-1])
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    if own_fig:
        plt.tight_layout()
        plt.show()

    return fig, ax


# ---------------------------------------------------------------------------
# 2. Annual ANPP comparison
# ---------------------------------------------------------------------------

def plot_anpp_comparison(
    nc_path: str,
    site: str,
    location: str,
    treatment: str,
    skip_spinup_years: int = 3,
    ax=None,
) -> Tuple:
    """Plot modelled annual ANPP time series alongside the observed distribution.

    The model uses synthetic sinusoidal forcing that repeats identically every
    year, so modelled ANPP has near-zero inter-annual variability — it converges
    to a deterministic steady state.  The observed record spans decades of real
    weather-driven variability.  Plotting both on the same time axis would imply
    temporal alignment that does not exist.

    When called without an ``ax``, a split figure is produced: the left panel
    shows the model time series (with the observed mean ± 1σ band for reference)
    and the right panel shows the full observed distribution as a boxplot with
    individual year values overlaid.  When an ``ax`` is supplied (e.g. from
    ``plot_site_dashboard``), a compact single-panel version is drawn instead.

    Parameters
    ----------
    nc_path:
        Path to model NetCDF output.
    site, location, treatment:
        NI LTER filter values ('GI'/'OL', 'HM'/'LM', 'C'/'NP').

    Returns
    -------
    (fig, ax_model)
    """
    plt, _ = _require_mpl()
    import matplotlib.gridspec as gridspec

    # --- Model ---
    anpp_mod = annual_anpp_g_m2_yr(nc_path)
    n_years = len(anpp_mod)
    years_mod = np.arange(n_years)
    mask_post = years_mod >= skip_spinup_years
    mean_mod = float(anpp_mod[mask_post].mean())

    # --- Observations ---
    obs = read_ni_annual_productivity(treatment=treatment)
    obs = obs[(obs["SITE"] == site) & (obs["LOCATION"] == location)]
    obs_vals = obs["PRODUCTIVITY"].values
    mean_obs = float(obs_vals.mean())
    std_obs = float(obs_vals.std())

    own_fig = ax is None
    if own_fig:
        fig = plt.figure(figsize=(12, 4))
        gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)
        ax_mod = fig.add_subplot(gs[0])
        ax_dist = fig.add_subplot(gs[1], sharey=ax_mod)
    else:
        fig = ax.get_figure()
        ax_mod = ax
        ax_dist = None

    # --- Model time-series panel ---
    ax_mod.axvspan(0, skip_spinup_years, color="grey", alpha=_SPINUP_ALPHA,
                   label="spin-up")
    ax_mod.axhspan(
        mean_obs - std_obs, mean_obs + std_obs,
        color=_OBS_COLOUR, alpha=_BAND_ALPHA,
        label=f"Obs mean ± 1σ  ({mean_obs:.0f} ± {std_obs:.0f} g m⁻² yr⁻¹)",
    )
    ax_mod.axhline(mean_obs, color=_OBS_COLOUR, linewidth=1.5, linestyle="--")
    ax_mod.plot(years_mod, anpp_mod, color=_MOD_COLOUR, linewidth=1.4,
                label="Model ANPP")
    ax_mod.axhline(mean_mod, color=_MOD_COLOUR, linewidth=1.5, linestyle="--",
                   label=f"Model mean (post-spinup): {mean_mod:.0f} g m⁻² yr⁻¹")
    ax_mod.set_xlabel("Model year  (synthetic repeating annual forcing)")
    ax_mod.set_ylabel("ANPP (g m⁻² yr⁻¹)")
    ax_mod.set_title(
        f"Annual ANPP — {site} {location} {treatment}"
        f"   mod/obs = {mean_mod / mean_obs:.2f}"
    )
    ax_mod.set_xlim(0, n_years - 1)
    ax_mod.legend(fontsize=8)
    ax_mod.grid(True, linewidth=0.4, alpha=0.5)

    # --- Observed distribution panel (standalone only) ---
    if ax_dist is not None:
        rng = np.random.default_rng(0)
        jitter = rng.uniform(-0.15, 0.15, len(obs_vals))
        ax_dist.scatter(jitter, obs_vals,
                        color=_OBS_COLOUR, alpha=0.5, s=16, zorder=3)
        ax_dist.boxplot(
            obs_vals, positions=[0], widths=0.4,
            patch_artist=True,
            boxprops=dict(facecolor=_OBS_COLOUR, alpha=0.25),
            medianprops=dict(color=_OBS_COLOUR, linewidth=2),
            whiskerprops=dict(color=_OBS_COLOUR),
            capprops=dict(color=_OBS_COLOUR),
            flierprops=dict(marker=""),
            zorder=2,
        )
        ax_dist.axhline(mean_obs, color=_OBS_COLOUR, linewidth=1.5, linestyle="--")
        ax_dist.axhspan(mean_obs - std_obs, mean_obs + std_obs,
                        color=_OBS_COLOUR, alpha=_BAND_ALPHA)
        ax_dist.set_xticks([0])
        ax_dist.set_xticklabels(
            [f"Obs\n{obs['YEAR'].min()}–{obs['YEAR'].max()}\n(n={len(obs_vals)})"],
            fontsize=8,
        )
        ax_dist.tick_params(left=False)
        plt.setp(ax_dist.get_yticklabels(), visible=False)
        ax_dist.grid(axis="y", linewidth=0.4, alpha=0.5)
        ax_dist.set_xlim(-0.5, 0.5)

    if own_fig:
        plt.tight_layout()
        plt.show()

    return fig, ax_mod


# ---------------------------------------------------------------------------
# 3. Seasonal biomass cycle
# ---------------------------------------------------------------------------

_MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def plot_seasonal_biomass(
    nc_path: str,
    site: str,
    location: str,
    treatment: str,
    skip_spinup_years: int = 3,
    ax=None,
) -> Tuple:
    """Plot mean seasonal cycle of aboveground biomass: model vs LTER observations.

    Model biomass is averaged over post-spinup years by calendar month.
    Observed biomass is averaged over all available years by calendar month,
    with ± 1 SE error bars.

    Parameters
    ----------
    nc_path:
        Path to model NetCDF output.
    site, location, treatment:
        NI LTER filter values.

    Returns
    -------
    (fig, ax)
    """
    plt, _ = _require_mpl()

    # --- Model ---
    series = read_time_series(nc_path)
    time_days = series["model_time_days"]
    biomass_g_m2 = series["aboveground_biomass_kg_m2"] * 1000.0
    year_idx = _model_year_index(time_days)
    month_idx = _model_month_index(time_days)
    mask_post = year_idx >= skip_spinup_years

    mod_monthly_mean = np.array([
        biomass_g_m2[mask_post & (month_idx == m)].mean()
        for m in range(1, 13)
    ])

    # --- Observations ---
    obs = read_ni_monthly_biomass(treatment=treatment, location=location)
    obs = obs[obs["SITE"] == site].copy()
    obs["month"] = obs["DATE"].dt.month
    obs_grouped = obs.groupby("month")["ABOVEGROUND_BIOMASS"].agg(
        mean="mean", sem=lambda x: x.sem()
    )

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    months = np.arange(1, 13)

    # Observed
    ax.errorbar(
        months, obs_grouped["mean"], yerr=obs_grouped["sem"],
        fmt="o", color=_OBS_COLOUR, capsize=4, linewidth=1.5,
        label=f"Observed (mean ± SE, n months)",
    )
    ax.fill_between(
        months,
        obs_grouped["mean"] - obs_grouped["sem"],
        obs_grouped["mean"] + obs_grouped["sem"],
        color=_OBS_COLOUR, alpha=_BAND_ALPHA,
    )

    # Modelled
    ax.plot(months, mod_monthly_mean, "-s", color=_MOD_COLOUR, linewidth=1.4,
            markersize=5, label="Model (post-spinup mean)")

    ax.set_xticks(months)
    ax.set_xticklabels(_MONTH_LABELS, fontsize=8)
    ax.set_xlabel("Month")
    ax.set_ylabel("Aboveground biomass (g m⁻²)")
    ax.set_title(f"Seasonal biomass cycle — {site} {location} {treatment}")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    if own_fig:
        plt.tight_layout()
        plt.show()

    return fig, ax


# ---------------------------------------------------------------------------
# 4. Four-panel dashboard
# ---------------------------------------------------------------------------

def plot_site_dashboard(
    nc_path: str,
    site: str,
    location: str,
    treatment: str,
    skip_spinup_years: int = 3,
    outpath: Optional[str] = None,
) -> Tuple:
    """Four-panel summary figure for one NI site.

    Panels
    ------
    Top-left  : Annual ANPP comparison (model vs observed distribution)
    Top-right : Seasonal biomass cycle (model vs observed)
    Bottom-left  : Surface elevation time series
    Bottom-right : Aboveground biomass time series

    Parameters
    ----------
    outpath:
        If provided, save the figure to this path instead of showing it.

    Returns
    -------
    (fig, axes)  — 2×2 array of Axes
    """
    plt, _ = _require_mpl()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"North Inlet — {site} {location} {treatment}",
        fontsize=13, fontweight="bold",
    )

    # Top-left: ANPP comparison
    plot_anpp_comparison(
        nc_path, site, location, treatment,
        skip_spinup_years=skip_spinup_years,
        ax=axes[0, 0],
    )

    # Top-right: seasonal biomass
    plot_seasonal_biomass(
        nc_path, site, location, treatment,
        skip_spinup_years=skip_spinup_years,
        ax=axes[0, 1],
    )

    # Bottom-left: surface elevation time series
    plot_variable_timeseries(
        nc_path,
        variables=["surface_elevation"],
        skip_spinup_years=skip_spinup_years,
        ax=axes[1, 0],
        title="Surface elevation",
        ylabel="Elevation (m)",
    )

    # Bottom-right: aboveground + belowground biomass
    plot_variable_timeseries(
        nc_path,
        variables=["aboveground_biomass_kg_m2", "belowground_biomass_kg_m2"],
        skip_spinup_years=skip_spinup_years,
        ax=axes[1, 1],
        title="Biomass compartments",
        ylabel="Biomass (kg m⁻²)",
    )

    plt.tight_layout()

    if outpath:
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        print(f"Saved: {outpath}")
    else:
        plt.show()

    return fig, axes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot marsh_muddpile output against NI LTER observations."
    )
    parser.add_argument("--nc", required=True, help="Path to NetCDF output file.")
    parser.add_argument(
        "--site", choices=["GI", "OL"], default="GI",
        help="NI LTER site code (default: GI).",
    )
    parser.add_argument(
        "--location", choices=["HM", "LM"], default="HM",
        help="Marsh zone (default: HM).",
    )
    parser.add_argument(
        "--treatment", choices=["C", "NP"], default="C",
        help="Treatment code (default: C).",
    )
    parser.add_argument(
        "--variable", nargs="+", default=None,
        help="Plot these specific variable(s) instead of the full dashboard.",
    )
    parser.add_argument(
        "--skip-spinup", type=int, default=3,
        help="Years to shade as spin-up (default: 3).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Save figure to this file instead of showing interactively.",
    )
    parser.add_argument(
        "--list-vars", action="store_true",
        help="Print available variable names in the NetCDF file and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.list_vars:
        series = read_time_series(args.nc)
        print(f"Variables in {args.nc}:")
        for name in sorted(series):
            print(f"  {name}  shape={series[name].shape}")
        return

    if args.variable:
        plot_variable_timeseries(
            args.nc,
            variables=args.variable,
            skip_spinup_years=args.skip_spinup,
        )
    else:
        plot_site_dashboard(
            args.nc,
            site=args.site,
            location=args.location,
            treatment=args.treatment,
            skip_spinup_years=args.skip_spinup,
            outpath=args.out,
        )


if __name__ == "__main__":
    main()
