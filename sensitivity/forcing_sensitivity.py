#!/usr/bin/env python3
"""
forcing_sensitivity.py

10-year sensitivity runs varying two environmental forcing parameters:
mean annual air temperature and creek salinity.  Everything else is held
at the North Inlet / yaml_writer defaults.

Three values are tested for each:

  Temperature mean (°C):  8   18   28
    (cool temperate → warm temperate → subtropical)

  Creek salinity (ppt): 5   20   35
    (mesohaline → polyhaline → euhaline)

For salinity, the initial root-zone salinity is set to match the creek
salinity so that there is no spurious spin-up transient from mismatched
initial and boundary conditions.

Produces two figures (one per forcing variable), each with two subplots:
  Top:    Aboveground biomass over the last year (seasonal cycle)
  Bottom: Belowground root mortality rate over the full 10-year time series

Total runs: 2 variables × 3 values = 6

Usage
-----
    python sensitivity/forcing_sensitivity.py --binary ./build/marsh_cli
    python sensitivity/forcing_sensitivity.py --skip-runs
    python sensitivity/forcing_sensitivity.py --plot-only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).parent.resolve()
_CALIB_DIR = _THIS_DIR.parent / "calibration"
sys.path.insert(0, str(_CALIB_DIR))

from site_config import PlotConfig, TideRecord, north_inlet_default_met, north_inlet_default_tides
from yaml_writer import write_config
from model_runner import run_model

try:
    import netCDF4 as nc4
except ImportError as exc:
    raise ImportError("netCDF4 is required: pip install netCDF4") from exc

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError("matplotlib is required: pip install matplotlib") from exc

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_YEARS = 10
SURFACE_ELEVATION_M = 0.30
DISTANCE_FROM_CREEK_M = 50.0

# M2-only sinusoidal tide (North Inlet amplitude)
TIDAL_AMPLITUDE_M = 0.766
TIDAL_PERIOD_HOURS = 12.42
MSL_M = 0.0

OUTPUT_DIR = _THIS_DIR / "runs" / "forcing"
FIGURE_DIR = _THIS_DIR / "figures"

_DAYS_IN_YEAR = 365.25

# ---------------------------------------------------------------------------
# Forcing sweep definitions
# ---------------------------------------------------------------------------

# Three ecologically meaningful temperature regimes (°C annual mean).
# North Inlet default is ~18.9 °C; amplitude and peak-day are unchanged.
TEMPERATURE_VALUES: List[float] = [8.0, 18.0, 28.0]
TEMPERATURE_LABELS: List[str] = ["8 °C\n(cool temperate)", "18 °C\n(warm temperate)",
                                  "28 °C\n(subtropical)"]

# Three salinity regimes (ppt creek salinity).
# North Inlet default is 28 ppt.
SALINITY_VALUES: List[float] = [5.0, 20.0, 35.0]
SALINITY_LABELS: List[str] = ["5 ppt\n(mesohaline)", "20 ppt\n(polyhaline)",
                               "35 ppt\n(euhaline)"]

LINE_COLOURS = ["#4575b4", "#d73027", "#1a9641"]   # low=blue, mid=red, high=green
LINE_STYLES = ["--", "-", "-."]

# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def _sine_tidal_record(creek_salinity_ppt: float = 28.0) -> TideRecord:
    """M2-only tidal record with configurable creek salinity."""
    t = TideRecord()
    t.mean_sea_level_m = MSL_M
    t.mean_high_tide_m = TIDAL_AMPLITUDE_M
    t.tidal_amplitude_m = TIDAL_AMPLITUDE_M
    t.tidal_period_hours = TIDAL_PERIOD_HOURS
    t.M2_amplitude_m = TIDAL_AMPLITUDE_M
    t.M2_phase_deg = 0.0
    t.S2_amplitude_m = 0.0
    t.S2_phase_deg = 0.0
    t.N2_amplitude_m = 0.0
    t.N2_phase_deg = 0.0
    t.K1_amplitude_m = 0.0
    t.K1_phase_deg = 0.0
    t.O1_amplitude_m = 0.0
    t.O1_phase_deg = 0.0
    t.creek_salinity_ppt = creek_salinity_ppt
    t.suspended_sediment_concentration_kg_m3 = 0.020
    t.fine_sediment_concentration_kg_m3 = 0.005
    return t


def _config_for_temperature(run_id: str, temperature_mean_c: float) -> PlotConfig:
    """PlotConfig with a modified annual mean temperature; tide unchanged."""
    met = north_inlet_default_met()
    met.temperature_mean_c = temperature_mean_c
    return PlotConfig(
        site_name="sensitivity_sine",
        plot_id=run_id,
        surface_elevation_m=SURFACE_ELEVATION_M,
        distance_from_creek_m=DISTANCE_FROM_CREEK_M,
        tides=_sine_tidal_record(),
        met=met,
        n_years=N_YEARS,
        output_dir=str(OUTPUT_DIR),
    )


def _config_for_salinity(run_id: str, creek_salinity_ppt: float) -> PlotConfig:
    """PlotConfig with modified creek salinity and matching initial salinity."""
    config = PlotConfig(
        site_name="sensitivity_sine",
        plot_id=run_id,
        surface_elevation_m=SURFACE_ELEVATION_M,
        distance_from_creek_m=DISTANCE_FROM_CREEK_M,
        tides=_sine_tidal_record(creek_salinity_ppt=creek_salinity_ppt),
        met=north_inlet_default_met(),
        n_years=N_YEARS,
        output_dir=str(OUTPUT_DIR),
    )
    # Match initial root-zone salinity to the creek so there is no
    # transient driven by a mismatched boundary condition.
    config.initial_salinity_ppt = creek_salinity_ppt
    return config

# ---------------------------------------------------------------------------
# Run management
# ---------------------------------------------------------------------------

def _run_temperature(cli_binary: str, force: bool) -> Dict[float, Path]:
    results: Dict[float, Path] = {}
    for temp in TEMPERATURE_VALUES:
        stem = f"temperature_{temp:.0f}c"
        yaml_path = OUTPUT_DIR / f"{stem}.yaml"
        nc_path = OUTPUT_DIR / f"{stem}.nc"

        if not force and nc_path.exists():
            print(f"  [T = {temp:.0f} °C] exists, skipping.")
        else:
            print(f"  [T = {temp:.0f} °C]")
            config = _config_for_temperature(stem, temp)
            write_config(config, str(yaml_path))
            try:
                run_model(str(yaml_path), cli_binary=cli_binary, silent=True)
            except RuntimeError as exc:
                print(f"    ERROR: {exc}")
                continue

        if nc_path.exists():
            results[temp] = nc_path
    return results


def _run_salinity(cli_binary: str, force: bool) -> Dict[float, Path]:
    results: Dict[float, Path] = {}
    for sal in SALINITY_VALUES:
        stem = f"salinity_{sal:.0f}ppt"
        yaml_path = OUTPUT_DIR / f"{stem}.yaml"
        nc_path = OUTPUT_DIR / f"{stem}.nc"

        if not force and nc_path.exists():
            print(f"  [S = {sal:.0f} ppt] exists, skipping.")
        else:
            print(f"  [S = {sal:.0f} ppt]")
            config = _config_for_salinity(stem, sal)
            write_config(config, str(yaml_path))
            try:
                run_model(str(yaml_path), cli_binary=cli_binary, silent=True)
            except RuntimeError as exc:
                print(f"    ERROR: {exc}")
                continue

        if nc_path.exists():
            results[sal] = nc_path
    return results

# ---------------------------------------------------------------------------
# Output reader
# ---------------------------------------------------------------------------

def _read_ts(nc_path: Path) -> Dict[str, np.ndarray]:
    series = {}
    with nc4.Dataset(str(nc_path), "r") as ds:
        for name, var in ds.variables.items():
            if var.ndim == 1 and "time" in var.dimensions:
                series[name] = np.array(var[:], dtype=np.float64)
    return series

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_sensitivity(
    run_map: Dict[float, Path],
    value_labels: List[str],
    title: str,
    forcing_label: str,
    outpath: Optional[Path] = None,
) -> None:
    """Two stacked subplots: last-year biomass (top) and root mortality (bottom)."""
    fig, (ax_bio, ax_mort) = plt.subplots(2, 1, figsize=(9, 8), sharex=False)

    values = sorted(run_map.keys())
    label_map = dict(zip(sorted(run_map.keys()), value_labels))

    mort_key = None

    for value, colour, ls in zip(values, LINE_COLOURS, LINE_STYLES):
        nc_path = run_map[value]
        if not nc_path.exists():
            continue
        ts = _read_ts(nc_path)
        t_days = ts["model_time_days"]
        t_yr = t_days / _DAYS_IN_YEAR

        # --- top: biomass, last year ---
        mask = t_days >= (t_days[-1] - _DAYS_IN_YEAR)
        doy = t_days[mask] - (t_days[-1] - _DAYS_IN_YEAR)
        bio = ts.get("aboveground_biomass_kg_m2", np.full(mask.sum(), np.nan))
        line_label = label_map[value].replace("\n", " ")
        ax_bio.plot(doy, bio[mask], color=colour, linestyle=ls,
                    linewidth=1.8, label=line_label)

        # --- bottom: root mortality, full timeseries ---
        if mort_key is None:
            mort_key = ("belowground_mortality_kg_m2_d"
                        if "belowground_mortality_kg_m2_d" in ts
                        else "belowground_mortality")
        if mort_key in ts:
            ax_mort.plot(t_yr, ts[mort_key], color=colour, linestyle=ls,
                         linewidth=1.5, label=line_label)

    # --- decorations ---
    ax_bio.set_xlabel("Day of year (last year)")
    ax_bio.set_ylabel("Aboveground biomass (kg m⁻²)")
    ax_bio.set_title(f"Aboveground biomass — last year\n{forcing_label}")
    ax_bio.set_xlim(0, _DAYS_IN_YEAR)
    ax_bio.legend(fontsize=9, loc="upper left")
    ax_bio.grid(True, linewidth=0.4, alpha=0.5)

    mort_ylabel = ("Belowground mortality rate (kg m⁻² d⁻¹)"
                   if mort_key and "kg_m2_d" in mort_key
                   else "Belowground mortality (kg m⁻²)")
    ax_mort.set_xlabel("Model time (years)")
    ax_mort.set_ylabel(mort_ylabel)
    ax_mort.set_title(f"Root mortality time series — {N_YEARS}-year run\n{forcing_label}")
    ax_mort.set_xlim(0, N_YEARS)
    ax_mort.legend(fontsize=9, loc="upper left")
    ax_mort.grid(True, linewidth=0.4, alpha=0.5)

    fig.suptitle(
        f"{title}\nelevation {SURFACE_ELEVATION_M} m, M2 sine tide, {N_YEARS}-year run",
        fontsize=11,
    )
    plt.tight_layout()

    if outpath:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        print(f"Saved: {outpath}")
    else:
        plt.show()
    plt.close(fig)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--binary", default="marsh_cli")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--skip-runs", action="store_true")
    g.add_argument("--plot-only", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-save", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    force = args.force and not args.skip_runs
    run_all = not args.plot_only and (force or not args.skip_runs)

    # --- temperature runs ---
    if run_all:
        print("Running temperature sensitivity (3 runs)...")
        temp_results = _run_temperature(args.binary, force=force)
    else:
        temp_results = {
            t: OUTPUT_DIR / f"temperature_{t:.0f}c.nc"
            for t in TEMPERATURE_VALUES
            if (OUTPUT_DIR / f"temperature_{t:.0f}c.nc").exists()
        }

    # --- salinity runs ---
    if run_all:
        print("Running salinity sensitivity (3 runs)...")
        sal_results = _run_salinity(args.binary, force=force)
    else:
        sal_results = {
            s: OUTPUT_DIR / f"salinity_{s:.0f}ppt.nc"
            for s in SALINITY_VALUES
            if (OUTPUT_DIR / f"salinity_{s:.0f}ppt.nc").exists()
        }

    # --- plots ---
    print("\nPlotting...")

    save = not args.no_save

    if temp_results:
        _plot_sensitivity(
            temp_results,
            value_labels=TEMPERATURE_LABELS,
            title="Mean annual temperature sensitivity",
            forcing_label="Mean annual air temperature (°C)",
            outpath=FIGURE_DIR / "forcing_sensitivity_temperature.png" if save else None,
        )
    else:
        print("  No temperature outputs found.")

    if sal_results:
        _plot_sensitivity(
            sal_results,
            value_labels=SALINITY_LABELS,
            title="Creek salinity sensitivity",
            forcing_label="Creek salinity (ppt)",
            outpath=FIGURE_DIR / "forcing_sensitivity_salinity.png" if save else None,
        )
    else:
        print("  No salinity outputs found.")


if __name__ == "__main__":
    main()
