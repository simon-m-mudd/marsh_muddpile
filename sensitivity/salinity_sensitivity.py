#!/usr/bin/env python3
"""
salinity_sensitivity.py

10-year marsh simulations exploring how creek salinity, marsh elevation,
distance from creek, and mean air temperature interact to drive root-zone
salinity and biomass.

Factorial design
----------------
  Creek salinity      : 5 ppt (low), 28 ppt (high)
  Marsh elevation     : 0.3 m, 0.6 m above MSL
  Distance from creek : 2, 5, 10, 30 m
  Mean temperature    : 12 °C, 20 °C

Total runs: 2 × 2 × 4 × 2 = 32

Plots (8 PNGs)
--------------
  Figures 1-2 : Aboveground biomass, last year        (temp 12 °C, temp 20 °C)
  Figures 3-4 : Belowground biomass, last year        (temp 12 °C, temp 20 °C)
  Figures 5-6 : Root-zone salinity, January last year (temp 12 °C, temp 20 °C)
  Figures 7-8 : Root-zone salinity, August  last year (temp 12 °C, temp 20 °C)

Each figure uses a 2×2 panel grid (rows = elevation, cols = creek salinity)
with one line per distance from creek.

Usage
-----
    python sensitivity/salinity_sensitivity.py
    python sensitivity/salinity_sensitivity.py --plot-only
    python sensitivity/salinity_sensitivity.py --force
    python sensitivity/salinity_sensitivity.py --binary /path/to/marsh_cli
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).parent.resolve()
_CALIB_DIR = _THIS_DIR.parent / "calibration"
sys.path.insert(0, str(_CALIB_DIR))

from site_config import PlotConfig, north_inlet_default_met
from yaml_writer import _DEFAULT_PARAMETERS, _DEFAULT_MATERIALS, _tidal_constituent_parameters
from model_runner import run_model

try:
    import yaml
except ImportError as exc:
    raise ImportError("pyyaml is required: pip install pyyaml") from exc

try:
    import netCDF4 as nc4
except ImportError as exc:
    raise ImportError("netCDF4 is required: pip install netCDF4") from exc

try:
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
except ImportError as exc:
    raise ImportError("matplotlib is required: pip install matplotlib") from exc

# ---------------------------------------------------------------------------
# Factorial design
# ---------------------------------------------------------------------------

CREEK_SALINITIES_PPT: Dict[str, float] = {"low": 5.0, "high": 28.0}
ELEVATIONS_M: List[float] = [0.3, 0.6]
DISTANCES_M: List[int] = [2, 5, 10, 30]
TEMPERATURES_C: List[int] = [12, 20]

N_YEARS = 10
OUTER_DT_DAYS = 1.0

TIDAL_AMPLITUDE_M = 0.766
TIDAL_PERIOD_HOURS = 12.42
MSL_M = 0.0

SSC_KG_M3 = 0.020
FINE_SSC_KG_M3 = 0.005

_DAYS_IN_YEAR = 365.25

# Day-of-year bounds for month extracts (1-indexed, inclusive start)
_JAN_DOY_START = 1
_JAN_DOY_END   = 31
_AUG_DOY_START = 213
_AUG_DOY_END   = 243

OUTPUT_DIR = _THIS_DIR / "runs" / "salinity"
FIGURE_DIR  = _THIS_DIR / "figures"

# RunKey: (salinity_label, elevation_m, distance_m, temperature_c)
RunKey = Tuple[str, float, int, int]

# ---------------------------------------------------------------------------
# Colours and labels
# ---------------------------------------------------------------------------

_DISTANCE_COLOURS: Dict[int, str] = {
    2:  "#d73027",
    5:  "#f46d43",
    10: "#4575b4",
    30: "#313695",
}

# ---------------------------------------------------------------------------
# Tidal record (M2-only sine, varying creek salinity)
# ---------------------------------------------------------------------------

def _sine_tidal_record(creek_salinity_ppt: float):
    from site_config import TideRecord
    t = TideRecord()
    t.mean_sea_level_m      = MSL_M
    t.mean_high_tide_m      = TIDAL_AMPLITUDE_M
    t.tidal_amplitude_m     = TIDAL_AMPLITUDE_M
    t.tidal_period_hours    = TIDAL_PERIOD_HOURS
    t.M2_amplitude_m        = TIDAL_AMPLITUDE_M
    t.M2_phase_deg          = 0.0
    t.S2_amplitude_m        = 0.0
    t.S2_phase_deg          = 0.0
    t.N2_amplitude_m        = 0.0
    t.N2_phase_deg          = 0.0
    t.K1_amplitude_m        = 0.0
    t.K1_phase_deg          = 0.0
    t.O1_amplitude_m        = 0.0
    t.O1_phase_deg          = 0.0
    t.creek_salinity_ppt    = creek_salinity_ppt
    t.suspended_sediment_concentration_kg_m3 = SSC_KG_M3
    t.fine_sediment_concentration_kg_m3      = FINE_SSC_KG_M3
    return t


# ---------------------------------------------------------------------------
# Forcing builder (daily steps)
# ---------------------------------------------------------------------------

def _sinusoid(mean: float, amp: float, peak_day: float, doy: float) -> float:
    return mean + amp * math.cos(2.0 * math.pi * (doy - peak_day) / _DAYS_IN_YEAR)


def _build_forcing(config: PlotConfig) -> List[dict]:
    n_steps = round(config.n_years * _DAYS_IN_YEAR / OUTER_DT_DAYS)
    met   = config.met
    tides = config.tides
    steps = []
    for i in range(n_steps):
        doy              = ((i + 0.5) * OUTER_DT_DAYS) % _DAYS_IN_YEAR
        model_time_days  = (i + 1) * OUTER_DT_DAYS
        temperature = _sinusoid(
            met.temperature_mean_c, met.temperature_amplitude_c,
            met.temperature_peak_day, doy,
        )
        par = max(0.0, _sinusoid(
            met.par_mean_umol_m2_d, met.par_amplitude_umol_m2_d,
            met.par_peak_day, doy,
        ))
        steps.append({
            "model_time_days":                  round(model_time_days, 4),
            "dt_days":                          round(OUTER_DT_DAYS, 4),
            "mean_sea_level":                   tides.mean_sea_level_m,
            "mean_high_tide":                   tides.mean_high_tide_m,
            "tidal_amplitude":                  tides.tidal_amplitude_m,
            "tidal_period_hours":               tides.tidal_period_hours,
            "temperature":                      round(temperature, 3),
            "precipitation_mm_d":               met.precipitation_mean_mm_d,
            "par_umol_m2_d":                    round(par, 1),
            "creek_salinity_ppt":               tides.creek_salinity_ppt,
            "freshwater_input_mm_d":            met.freshwater_input_mm_d,
            "storm_surge_residual_m":           0.0,
            "suspended_sediment_concentration": tides.suspended_sediment_concentration_kg_m3,
            "fine_sediment_concentration":      tides.fine_sediment_concentration_kg_m3,
            "external_pb210_supply":            0.0,
        })
    return steps


# ---------------------------------------------------------------------------
# YAML writer
# ---------------------------------------------------------------------------

def _run_stem(sal_label: str, elev_m: float, dist_m: int, temp_c: int) -> str:
    return f"sal_{sal_label}_e{int(elev_m * 100):03d}_d{dist_m:03d}_t{temp_c:02d}"


def _write_config(config: PlotConfig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parameters = dict(_DEFAULT_PARAMETERS)
    parameters.update(_tidal_constituent_parameters(config))
    parameters.update({
        "deposition_distance_from_edge_m": config.distance_from_creek_m,
        "deposition_basin_length_m":       max(50.0, config.distance_from_creek_m * 2),
        "deposition_length_scale_beta":    3.0,
        # Initialise salinity model at creek value so runs start at equilibrium
        "salinity_initial_root_zone_ppt":  config.tides.creek_salinity_ppt,
    })

    doc = {
        "simulation": {
            "start_year":                    0,
            "end_year":                      config.n_years,
            "dt_days":                       365.0,
            "water_level_model_name":        "composite_water_level",
            "salinity_model_name":           "distance_flushing_salinity",
            "evapotranspiration_model_name": "simple_canopy_et",
            "vegetation_model_name":         "marsh_gpp_biomass",
            "deposition_model_name":         "edge_distance_deposition",
            "root_allocation_model_name":    "exponential_root_allocation",
            "decay_model_name":              "marsh_decay",
            "compaction_model_name":           "mixing_compaction",
        },
        "site": {
            "distance_from_creek_m":  config.distance_from_creek_m,
            "creek_bank_elevation_m": config.tides.mean_sea_level_m,
            "local_tidal_offset_m":   0.0,
        },
        "parameters": parameters,
        "materials":  _DEFAULT_MATERIALS,
        "forcing":    {"steps": _build_forcing(config)},
        "initial_state": {
            "layers": [
                {
                    "top_elevation_m": config.surface_elevation_m,
                    "thickness_m":     config.initial_column_thickness_m,
                    "porosity":        config.initial_porosity,
                    "age_days":        0.0,
                    "fill_material":   config.initial_fill_material,
                }
            ]
        },
        "initial_ecohydrology_state": {
            "root_zone_salinity_ppt":       config.tides.creek_salinity_ppt,
            "aboveground_biomass_kg_m2":    config.initial_aboveground_biomass_kg_m2,
            "belowground_biomass_kg_m2":    config.initial_belowground_biomass_kg_m2,
            "lai":                          config.initial_lai,
            "litter_kg_m2":                 0.0,
        },
        "output": {
            "file":               str(output_path.with_suffix(".nc")),
            "write_time_series":  True,
            "write_column_snapshots": False,
        },
    }

    with open(output_path, "w") as fh:
        yaml.dump(doc, fh, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------

def _make_config(sal_label: str, sal_ppt: float,
                 elev_m: float, dist_m: int, temp_c: int) -> PlotConfig:
    met = north_inlet_default_met()
    met.temperature_mean_c = float(temp_c)     # override mean only; keep amplitude
    tides = _sine_tidal_record(sal_ppt)
    return PlotConfig(
        site_name="salinity_sensitivity",
        plot_id=_run_stem(sal_label, elev_m, dist_m, temp_c),
        surface_elevation_m=elev_m,
        distance_from_creek_m=float(dist_m),
        tides=tides,
        met=met,
        n_years=N_YEARS,
        output_dir=str(OUTPUT_DIR),
    )


# ---------------------------------------------------------------------------
# Run management
# ---------------------------------------------------------------------------

def _run_all(cli_binary: str, force: bool) -> Dict[RunKey, Path]:
    total = (len(CREEK_SALINITIES_PPT) * len(ELEVATIONS_M)
             * len(DISTANCES_M) * len(TEMPERATURES_C))
    results: Dict[RunKey, Path] = {}
    count = 0

    for sal_label, sal_ppt in CREEK_SALINITIES_PPT.items():
        for elev_m in ELEVATIONS_M:
            for dist_m in DISTANCES_M:
                for temp_c in TEMPERATURES_C:
                    count += 1
                    key: RunKey = (sal_label, elev_m, dist_m, temp_c)
                    stem     = _run_stem(sal_label, elev_m, dist_m, temp_c)
                    yaml_path = OUTPUT_DIR / f"{stem}.yaml"
                    nc_path   = OUTPUT_DIR / f"{stem}.nc"

                    print(f"  [{count:2d}/{total}] {stem}")

                    if not force and nc_path.exists():
                        print("             output exists, skipping.")
                    else:
                        config = _make_config(sal_label, sal_ppt, elev_m, dist_m, temp_c)
                        _write_config(config, yaml_path)
                        try:
                            run_model(str(yaml_path), cli_binary=cli_binary, silent=True)
                        except RuntimeError as exc:
                            print(f"    ERROR: {exc}")
                            continue

                    if nc_path.exists():
                        results[key] = nc_path

    return results


def _collect_existing() -> Dict[RunKey, Path]:
    results: Dict[RunKey, Path] = {}
    for sal_label in CREEK_SALINITIES_PPT:
        for elev_m in ELEVATIONS_M:
            for dist_m in DISTANCES_M:
                for temp_c in TEMPERATURES_C:
                    nc_path = OUTPUT_DIR / f"{_run_stem(sal_label, elev_m, dist_m, temp_c)}.nc"
                    if nc_path.exists():
                        results[(sal_label, elev_m, dist_m, temp_c)] = nc_path
    return results


# ---------------------------------------------------------------------------
# Output readers
# ---------------------------------------------------------------------------

def _read_ts(nc_path: Path) -> Dict[str, np.ndarray]:
    series: Dict[str, np.ndarray] = {}
    with nc4.Dataset(str(nc_path), "r") as ds:
        for name, var in ds.variables.items():
            if var.ndim == 1 and "time" in var.dimensions:
                series[name] = np.array(var[:], dtype=np.float64)
    return series


def _last_year_mask(t: np.ndarray) -> np.ndarray:
    return t >= (t[-1] - _DAYS_IN_YEAR)


def _month_mask(t: np.ndarray, doy_start: int, doy_end: int) -> np.ndarray:
    ly_mask    = _last_year_mask(t)
    year_start = t[ly_mask][0]
    doy        = t - year_start          # day within the last year (0-based)
    return ly_mask & (doy >= doy_start - 1) & (doy < doy_end)


# ---------------------------------------------------------------------------
# Extract functions  →  (x_array, y_array)
# ---------------------------------------------------------------------------

def _biomass_last_year(variable: str) -> Callable:
    def _fn(ts: Dict[str, np.ndarray]):
        t    = ts["model_time_days"]
        mask = _last_year_mask(t)
        doy  = t[mask] - t[mask][0]
        return doy, ts[variable][mask]
    return _fn


def _salinity_month(doy_start: int, doy_end: int) -> Callable:
    def _fn(ts: Dict[str, np.ndarray]):
        t    = ts["model_time_days"]
        mask = _month_mask(t, doy_start, doy_end)
        doy  = t[mask] - t[_last_year_mask(t)][0]
        return doy, ts["root_zone_salinity_ppt"][mask]
    return _fn


# ---------------------------------------------------------------------------
# Panel-grid figure
# ---------------------------------------------------------------------------

def _panel_grid(
    results:   Dict[RunKey, Path],
    temp_c:    int,
    extract:   Callable,
    xlabel:    str,
    ylabel:    str,
    suptitle:  str,
    outpath:   Optional[Path],
) -> None:
    """2×2 grid: rows = elevation, cols = creek salinity; lines = distance."""
    sal_labels = list(CREEK_SALINITIES_PPT.keys())
    fig, axes = plt.subplots(
        len(ELEVATIONS_M), len(sal_labels),
        figsize=(12, 8), sharey=True,
    )

    for row, elev_m in enumerate(ELEVATIONS_M):
        for col, sal_label in enumerate(sal_labels):
            ax      = axes[row][col]
            sal_ppt = CREEK_SALINITIES_PPT[sal_label]

            for dist_m in DISTANCES_M:
                key = (sal_label, elev_m, dist_m, temp_c)
                if key not in results:
                    continue
                ts    = _read_ts(results[key])
                x, y  = extract(ts)
                ax.plot(x, y,
                        color=_DISTANCE_COLOURS[dist_m],
                        linewidth=1.3,
                        label=f"{dist_m} m")

            ax.set_title(
                f"elevation {elev_m} m  |  creek sal {sal_ppt:.0f} ppt",
                fontsize=9,
            )
            ax.grid(True, linewidth=0.4, alpha=0.5)
            if row == len(ELEVATIONS_M) - 1:
                ax.set_xlabel(xlabel, fontsize=9)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)

    handles = [
        mlines.Line2D([], [], color=_DISTANCE_COLOURS[d], linewidth=1.5,
                      label=f"{d} m")
        for d in DISTANCES_M
    ]
    fig.legend(handles=handles, title="Distance from creek",
               loc="lower center", ncol=len(DISTANCES_M),
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{suptitle}  —  mean temp {temp_c} °C", fontsize=11)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    _save_or_show(fig, outpath)


def _save_or_show(fig, outpath: Optional[Path]) -> None:
    if outpath is not None:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        print(f"Saved: {outpath}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Make all 8 figures
# ---------------------------------------------------------------------------

def make_plots(results: Dict[RunKey, Path], save: bool) -> None:
    fig_dir = FIGURE_DIR if save else None

    _PLOTS = [
        # (variable or month, extract_fn, xlabel, ylabel, filename_stem, title)
        (
            "aboveground_biomass_kg_m2",
            _biomass_last_year("aboveground_biomass_kg_m2"),
            "Day of final year",
            "Aboveground biomass (kg m⁻²)",
            "sal_aboveground_biomass",
            "Aboveground biomass — last year",
        ),
        (
            "belowground_biomass_kg_m2",
            _biomass_last_year("belowground_biomass_kg_m2"),
            "Day of final year",
            "Belowground biomass (kg m⁻²)",
            "sal_belowground_biomass",
            "Belowground biomass — last year",
        ),
        (
            "salinity_jan",
            _salinity_month(_JAN_DOY_START, _JAN_DOY_END),
            "Day of January (final year)",
            "Root-zone salinity (ppt)",
            "sal_salinity_jan",
            "Root-zone salinity — January",
        ),
        (
            "salinity_aug",
            _salinity_month(_AUG_DOY_START, _AUG_DOY_END),
            "Day of August (final year)",
            "Root-zone salinity (ppt)",
            "sal_salinity_aug",
            "Root-zone salinity — August",
        ),
    ]

    for temp_c in TEMPERATURES_C:
        for _, extract, xlabel, ylabel, stem, title in _PLOTS:
            outpath = fig_dir / f"{stem}_t{temp_c:02d}.png" if save else None
            _panel_grid(
                results, temp_c, extract,
                xlabel=xlabel, ylabel=ylabel,
                suptitle=title, outpath=outpath,
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--binary",    default="marsh_cli",
                   help="Path to marsh_cli executable")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--skip-runs", action="store_true",
                   help="Skip model runs, plot existing outputs")
    g.add_argument("--plot-only", action="store_true",
                   help="Plot existing outputs without running")
    p.add_argument("--force",   action="store_true",
                   help="Re-run even if output exists")
    p.add_argument("--no-save", action="store_true",
                   help="Display figures instead of saving")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n_runs = (len(CREEK_SALINITIES_PPT) * len(ELEVATIONS_M)
              * len(DISTANCES_M) * len(TEMPERATURES_C))

    if args.plot_only or args.skip_runs:
        results = _collect_existing()
    else:
        print(f"Running {n_runs} simulations ({N_YEARS}-year, 1-day outer step)...")
        results = _run_all(cli_binary=args.binary, force=args.force)

    if not results:
        print("No outputs found.")
        return

    print(f"\nPlotting {len(results)} run(s)...")
    make_plots(results, save=not args.no_save)


if __name__ == "__main__":
    main()
