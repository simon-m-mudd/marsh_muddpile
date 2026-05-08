#!/usr/bin/env python3
"""
gpp_parameter_sensitivity.py

10-year sensitivity runs for the four vegetation parameters used in the North
Inlet GPP calibration.  Each parameter is tested at three values (low, default,
high) with everything else held at the yaml_writer defaults.  Produces one
figure per parameter, each with two subplots:

  Top:    Aboveground biomass over the last year (seasonal cycle, day-of-year)
  Bottom: Belowground (root) mortality rate over the full 10-year time series

The three test values per parameter are placed symmetrically around the
calibration default in log space:
  low  = geometric_mean(calibration_lower_bound, calibration_default)
  mid  = calibration_default  (from calibrate_ni.PARAM_DEFAULTS)
  high = geometric_mean(calibration_default, calibration_upper_bound)

This keeps the sweep centred on the realistic operating point rather than
running out to the hard search boundaries.

Total runs: 4 parameters × 3 values = 12

Usage
-----
    python sensitivity/gpp_parameter_sensitivity.py
    python sensitivity/gpp_parameter_sensitivity.py --skip-runs
    python sensitivity/gpp_parameter_sensitivity.py --plot-only
    python sensitivity/gpp_parameter_sensitivity.py --binary /path/to/marsh_cli
"""

from __future__ import annotations

import argparse
import math
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

from site_config import PlotConfig, TideRecord, north_inlet_default_met
from yaml_writer import write_config
from model_runner import run_model

# Pull calibration parameter metadata directly from calibrate_ni so that
# any changes to bounds or defaults there are automatically reflected here.
from calibrate_ni import PARAM_NAMES, PARAM_BOUNDS, PARAM_DEFAULTS

try:
    import netCDF4 as nc4
except ImportError as exc:
    raise ImportError("netCDF4 is required: pip install netCDF4") from exc

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError("matplotlib is required: pip install matplotlib") from exc

# ---------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------

N_YEARS = 10
SURFACE_ELEVATION_M = 0.30      # m above MSL
DISTANCE_FROM_CREEK_M = 50.0

# M2-only sinusoidal tide (North Inlet amplitude)
TIDAL_AMPLITUDE_M = 0.766
TIDAL_PERIOD_HOURS = 12.42
MSL_M = 0.0

OUTPUT_DIR = _THIS_DIR / "runs" / "gpp_params"
FIGURE_DIR = _THIS_DIR / "figures"

_DAYS_IN_YEAR = 365.25

# Human-readable labels and units for each parameter.
_PARAM_LABELS: Dict[str, Tuple[str, str]] = {
    "vegetation_aboveground_capacity_kg_m2":
        ("Aboveground capacity", "kg m⁻²"),
    "vegetation_aboveground_mortality_base_per_day":
        ("Base aboveground mortality", "d⁻¹"),
    "vegetation_cold_mortality_slope_per_day_per_c":
        ("Cold mortality slope", "d⁻¹ °C⁻¹"),
    "vegetation_lue_gC_per_umol":
        ("Light-use efficiency", "gC μmol⁻¹"),
}

# ---------------------------------------------------------------------------
# Compute the three test values per parameter
# ---------------------------------------------------------------------------

def _test_values(lo: float, hi: float, default: float) -> List[float]:
    """Three log-symmetric values: [gmean(lo,default), default, gmean(default,hi)]."""
    return [
        math.sqrt(lo * default),
        default,
        math.sqrt(default * hi),
    ]


PARAM_TEST_VALUES: List[List[float]] = [
    _test_values(lo, hi, d)
    for (lo, hi), d in zip(PARAM_BOUNDS, PARAM_DEFAULTS)
]

LINE_COLOURS = ["#4575b4", "#d73027", "#1a9641"]   # low=blue, mid=red, high=green
LINE_STYLES = ["--", "-", "-."]


def _fmt(v: float) -> str:
    """Format a parameter value compactly for plot labels."""
    if v == 0.0:
        return "0"
    if abs(v) >= 0.01:
        return f"{v:.3g}"
    return f"{v:.2e}"


# ---------------------------------------------------------------------------
# Tidal record
# ---------------------------------------------------------------------------

def _sine_tidal_record() -> TideRecord:
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
    t.creek_salinity_ppt = 28.0
    t.suspended_sediment_concentration_kg_m3 = 0.020
    t.fine_sediment_concentration_kg_m3 = 0.005
    return t


def _make_config(run_id: str) -> PlotConfig:
    return PlotConfig(
        site_name="sensitivity_sine",
        plot_id=run_id,
        surface_elevation_m=SURFACE_ELEVATION_M,
        distance_from_creek_m=DISTANCE_FROM_CREEK_M,
        tides=_sine_tidal_record(),
        met=north_inlet_default_met(),
        n_years=N_YEARS,
        output_dir=str(OUTPUT_DIR),
    )


# ---------------------------------------------------------------------------
# Run management
# ---------------------------------------------------------------------------

def _run_stem(param_name: str, value: float) -> str:
    """Filesystem-safe stem for one run."""
    short = param_name.replace("vegetation_", "veg_")
    return f"{short}_{_fmt(value).replace('.','p').replace('-','m').replace('+','')}"


def _run_all(cli_binary: str, force: bool) -> Dict[str, Dict[float, Path]]:
    """Run all 12 simulations.

    Returns nested dict: {param_name: {value: nc_path}}.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[float, Path]] = {n: {} for n in PARAM_NAMES}

    for param_name, values in zip(PARAM_NAMES, PARAM_TEST_VALUES):
        label, units = _PARAM_LABELS[param_name]
        for value in values:
            stem = _run_stem(param_name, value)
            yaml_path = OUTPUT_DIR / f"{stem}.yaml"
            nc_path = OUTPUT_DIR / f"{stem}.nc"

            if not force and nc_path.exists():
                print(f"  [{param_name.split('_')[-2]}={_fmt(value)}] exists, skipping.")
                results[param_name][value] = nc_path
                continue

            print(f"  {label} = {_fmt(value)} {units}")
            config = _make_config(stem)
            write_config(config, str(yaml_path),
                         extra_parameters={param_name: value})
            try:
                run_model(str(yaml_path), cli_binary=cli_binary, silent=True)
                results[param_name][value] = nc_path
            except RuntimeError as exc:
                print(f"    ERROR: {exc}")

    return results


# ---------------------------------------------------------------------------
# Output readers
# ---------------------------------------------------------------------------

def _read_ts(nc_path: Path) -> Dict[str, np.ndarray]:
    series = {}
    with nc4.Dataset(str(nc_path), "r") as ds:
        for name, var in ds.variables.items():
            if var.ndim == 1 and "time" in var.dimensions:
                series[name] = np.array(var[:], dtype=np.float64)
    return series


def _last_year_mask(t_days: np.ndarray) -> np.ndarray:
    return t_days >= (t_days[-1] - _DAYS_IN_YEAR)


def _last_year_doy(t_days: np.ndarray) -> np.ndarray:
    """Day-of-year within the last year (0 = start of last year)."""
    last_start = t_days[-1] - _DAYS_IN_YEAR
    return t_days[_last_year_mask(t_days)] - last_start


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_parameter(
    param_name: str,
    run_map: Dict[float, Path],
    outpath: Optional[Path] = None,
) -> None:
    """One figure with two stacked subplots for a single parameter."""
    label, units = _PARAM_LABELS[param_name]
    values = sorted(run_map.keys())

    fig, (ax_bio, ax_mort) = plt.subplots(
        2, 1, figsize=(9, 8), sharex=False
    )

    for value, colour, ls in zip(values, LINE_COLOURS, LINE_STYLES):
        nc_path = run_map[value]
        if not nc_path.exists():
            continue
        ts = _read_ts(nc_path)
        t_days = ts["model_time_days"]

        # --- top panel: aboveground biomass, last year ---
        mask = _last_year_mask(t_days)
        doy = _last_year_doy(t_days)
        bio = ts.get("aboveground_biomass_kg_m2", np.full(mask.sum(), np.nan))
        ax_bio.plot(doy, bio[mask], color=colour, linestyle=ls,
                    linewidth=1.8, label=f"{_fmt(value)} {units}")

        # --- bottom panel: root mortality, full time series ---
        t_yr = t_days / _DAYS_IN_YEAR
        mort_key = "belowground_mortality_kg_m2_d"
        if mort_key not in ts:
            mort_key = "belowground_mortality"   # legacy fallback
        if mort_key in ts:
            ax_mort.plot(t_yr, ts[mort_key], color=colour, linestyle=ls,
                         linewidth=1.5, label=f"{_fmt(value)} {units}")

    # --- top panel decoration ---
    ax_bio.set_xlabel("Day of year (last year)")
    ax_bio.set_ylabel("Aboveground biomass (kg m⁻²)")
    ax_bio.set_title(f"Aboveground biomass — last year\n{label}")
    ax_bio.set_xlim(0, _DAYS_IN_YEAR)
    ax_bio.legend(fontsize=9, title=f"{label}\n({units})", loc="upper left")
    ax_bio.grid(True, linewidth=0.4, alpha=0.5)

    # --- bottom panel decoration ---
    mort_label = ("Belowground mortality rate (kg m⁻² d⁻¹)"
                  if "kg_m2_d" in mort_key else "Belowground mortality (kg m⁻²)")
    ax_mort.set_xlabel("Model time (years)")
    ax_mort.set_ylabel(mort_label)
    ax_mort.set_title(f"Root mortality time series — {N_YEARS}-year run\n{label}")
    ax_mort.set_xlim(0, N_YEARS)
    ax_mort.legend(fontsize=9, title=f"{label}\n({units})", loc="upper left")
    ax_mort.grid(True, linewidth=0.4, alpha=0.5)

    fig.suptitle(
        f"GPP parameter sensitivity — {label}\n"
        f"elevation {SURFACE_ELEVATION_M} m, M2 sine tide, {N_YEARS}-year run",
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
    g.add_argument("--skip-runs", action="store_true",
                   help="Skip runs if .nc files already exist.")
    g.add_argument("--plot-only", action="store_true",
                   help="Skip all runs; only plot existing outputs.")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if outputs exist.")
    p.add_argument("--no-save", action="store_true",
                   help="Show figures interactively instead of saving.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.plot_only:
        print(f"Running {len(PARAM_NAMES) * 3} simulations ({N_YEARS} years each)...")
        force = args.force and not args.skip_runs
        results = _run_all(
            cli_binary=args.binary,
            force=force or (not args.skip_runs),
        )
    else:
        results = {}
        for param_name, values in zip(PARAM_NAMES, PARAM_TEST_VALUES):
            results[param_name] = {}
            for value in values:
                nc_path = OUTPUT_DIR / f"{_run_stem(param_name, value)}.nc"
                if nc_path.exists():
                    results[param_name][value] = nc_path
                else:
                    print(f"  Missing: {nc_path.name}")

    print(f"\nPlotting {len(PARAM_NAMES)} figures...")

    # Map short name to full name for figure filenames
    short_names = {
        "vegetation_aboveground_capacity_kg_m2": "capacity",
        "vegetation_aboveground_mortality_base_per_day": "mort_base",
        "vegetation_cold_mortality_slope_per_day_per_c": "cold_slope",
        "vegetation_lue_gC_per_umol": "lue",
    }

    for param_name in PARAM_NAMES:
        run_map = results.get(param_name, {})
        if not run_map:
            print(f"  No outputs for {param_name}, skipping.")
            continue

        outpath = (FIGURE_DIR / f"gpp_sensitivity_{short_names[param_name]}.png"
                   if not args.no_save else None)
        _plot_parameter(param_name, run_map, outpath=outpath)


if __name__ == "__main__":
    main()
