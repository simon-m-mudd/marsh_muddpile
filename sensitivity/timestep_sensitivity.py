#!/usr/bin/env python3
"""
timestep_sensitivity.py

Run 50-year marsh simulations with different forcing timesteps and plot
key output diagnostics to assess numerical sensitivity to timestep choice.

Tidal forcing: pure M2 sinusoid (simple sine wave) with North Inlet-like
amplitude. Meteorological forcing uses North Inlet ERA5-derived seasonality.

Timesteps tested (days): 1, 7, monthly (~30.4), quarterly (~91.3), annual (365.25)

Plots produced
--------------
Figure 1 — 50-year time series (one line per timestep):
    Panel 1: Surface elevation (m)
    Panel 2: Total organic carbon in column (kg C m-2), from refractory +
             labile organic pools × 0.45 carbon fraction

Figure 2 — Last year only (seasonal cycle):
    Panel 1: Aboveground biomass (kg m-2)
    Panel 2: Live belowground biomass (kg m-2)

Usage
-----
    # Run all simulations then plot:
    python sensitivity/timestep_sensitivity.py

    # Skip re-running if .nc files already exist:
    python sensitivity/timestep_sensitivity.py --skip-runs

    # Only plot from existing outputs (no runs):
    python sensitivity/timestep_sensitivity.py --plot-only

    # Custom binary path:
    python sensitivity/timestep_sensitivity.py --binary /path/to/marsh_cli
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
# Path setup: import helpers from calibration/
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).parent.resolve()
_CALIB_DIR = _THIS_DIR.parent / "calibration"
sys.path.insert(0, str(_CALIB_DIR))

from site_config import PlotConfig, TideRecord, north_inlet_default_met
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
    import matplotlib.cm as cm
except ImportError as exc:
    raise ImportError("matplotlib is required: pip install matplotlib") from exc

# ---------------------------------------------------------------------------
# Simulation defaults
# ---------------------------------------------------------------------------

N_YEARS = 50
SURFACE_ELEVATION_M = 0.30      # m above MSL, mid-marsh
DISTANCE_FROM_CREEK_M = 50.0    # m

# Simple M2-only sinusoidal tide (North Inlet amplitude)
TIDAL_AMPLITUDE_M = 0.766
TIDAL_PERIOD_HOURS = 12.42
MSL_M = 0.0

# Sediment concentrations
SSC_KG_M3 = 0.020
FINE_SSC_KG_M3 = 0.005
CREEK_SALINITY_PPT = 28.0

# Organic carbon fraction applied to organic pool masses
OC_FRACTION = 0.45

# Tidal integration: target 20 substeps per tidal cycle.
# This is the minimum recommended for accurate inundation fraction on a
# mid-marsh site (~20% inundation per tide).  See readme.md for details.
# water_level_substeps_per_step is scaled with dt_days so every run
# sees the same sub-cycle resolution regardless of outer timestep.
_TARGET_SUBSTEPS_PER_CYCLE = 20
_CYCLES_PER_DAY = 24.0 / TIDAL_PERIOD_HOURS   # ~1.93 for M2


def _substeps_for_dt(dt_days: float) -> int:
    """Return water_level_substeps_per_step for a given outer dt_days.

    Targets _TARGET_SUBSTEPS_PER_CYCLE substeps per tidal cycle.
    """
    return max(10, round(dt_days * _CYCLES_PER_DAY * _TARGET_SUBSTEPS_PER_CYCLE))

# Timesteps (days) and display labels
TIMESTEPS_DAYS: List[float] = [1.0, 7.0, 365.25 / 12.0, 365.25 / 4.0, 365.25]
TIMESTEP_LABELS: List[str] = ["1 d", "7 d", "monthly", "quarterly", "annual"]

OUTPUT_DIR = _THIS_DIR / "runs" / "timestep"
FIGURE_DIR = _THIS_DIR / "figures"

_DAYS_IN_YEAR = 365.25
_MONTHLY_DT_DAYS = _DAYS_IN_YEAR / 12.0   # ~30.4 d


# ---------------------------------------------------------------------------
# Sine (M2-only) tidal record
# ---------------------------------------------------------------------------

def _sine_tidal_record() -> TideRecord:
    """TideRecord with only the M2 constituent — produces a pure sine tide."""
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
    t.creek_salinity_ppt = CREEK_SALINITY_PPT
    t.suspended_sediment_concentration_kg_m3 = SSC_KG_M3
    t.fine_sediment_concentration_kg_m3 = FINE_SSC_KG_M3
    return t


# ---------------------------------------------------------------------------
# Forcing builder with variable dt
# ---------------------------------------------------------------------------

def _sinusoid(mean: float, amp: float, peak_day: float, doy: float) -> float:
    return mean + amp * math.cos(2.0 * math.pi * (doy - peak_day) / _DAYS_IN_YEAR)


def _build_forcing(config: PlotConfig, dt_days: float) -> List[dict]:
    """Build forcing steps with the given dt_days."""
    n_steps = round(config.n_years * _DAYS_IN_YEAR / dt_days)
    met = config.met
    tides = config.tides
    steps: List[dict] = []

    for i in range(n_steps):
        doy = ((i + 0.5) * dt_days) % _DAYS_IN_YEAR
        model_time_days = (i + 1) * dt_days

        temperature = _sinusoid(
            met.temperature_mean_c, met.temperature_amplitude_c,
            met.temperature_peak_day, doy,
        )
        par = max(0.0, _sinusoid(
            met.par_mean_umol_m2_d, met.par_amplitude_umol_m2_d,
            met.par_peak_day, doy,
        ))

        steps.append({
            "model_time_days": round(model_time_days, 4),
            "dt_days": round(dt_days, 4),
            "mean_sea_level": tides.mean_sea_level_m,
            "mean_high_tide": tides.mean_high_tide_m,
            "tidal_amplitude": tides.tidal_amplitude_m,
            "tidal_period_hours": tides.tidal_period_hours,
            "temperature": round(temperature, 3),
            "precipitation_mm_d": met.precipitation_mean_mm_d,
            "par_umol_m2_d": round(par, 1),
            "creek_salinity_ppt": tides.creek_salinity_ppt,
            "freshwater_input_mm_d": met.freshwater_input_mm_d,
            "storm_surge_residual_m": 0.0,
            "suspended_sediment_concentration": tides.suspended_sediment_concentration_kg_m3,
            "fine_sediment_concentration": tides.fine_sediment_concentration_kg_m3,
            "external_pb210_supply": 0.0,
        })

    return steps


# ---------------------------------------------------------------------------
# YAML config writer (variable dt version)
# ---------------------------------------------------------------------------

def _write_config_dt(
    config: PlotConfig,
    output_path: Path,
    dt_days: float,
) -> Path:
    """Write a YAML config with the given forcing dt_days."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    forcing_steps = _build_forcing(config, dt_days)
    total_days = config.n_years * _DAYS_IN_YEAR

    parameters = dict(_DEFAULT_PARAMETERS)
    parameters.update(_tidal_constituent_parameters(config))
    parameters.update({
        # Scale tidal substeps with dt so each run integrates over the same number
        # of substeps per tidal cycle (~4), matching the monthly default.
        "water_level_substeps_per_step": _substeps_for_dt(dt_days),
        # edge_distance_deposition parameters
        "deposition_distance_from_edge_m": config.distance_from_creek_m,
        "deposition_basin_length_m": 50.0,
        "deposition_length_scale_beta": 3.0,
    })

    base_top = config.surface_elevation_m
    base_bottom = config.surface_elevation_m - config.initial_column_thickness_m

    doc = {
        "simulation": {
            "start_year": 0,
            "end_year": config.n_years,
            "dt_days": 365.0,
            "water_level_model_name": "composite_water_level",
            "salinity_model_name": "distance_flushing_salinity",
            "evapotranspiration_model_name": "simple_canopy_et",
            "vegetation_model_name": "marsh_gpp_biomass",
            "deposition_model_name": "edge_distance_deposition",
            "root_allocation_model_name": "exponential_root_allocation",
            "decay_model_name": "marsh_decay",
            "compaction_model_name":           "mixing_compaction",
        },
        "site": {
            "distance_from_creek_m": config.distance_from_creek_m,
            "creek_bank_elevation_m": config.tides.mean_sea_level_m,
            "local_tidal_offset_m": 0.0,
        },
        "parameters": parameters,
        "materials": _DEFAULT_MATERIALS,
        "forcing": {
            "steps": forcing_steps,
        },
        "initial_state": {
            "layers": [
                {
                    "top_elevation_m": base_top,
                    "thickness_m": config.initial_column_thickness_m,
                    "porosity": config.initial_porosity,
                    "age_days": 0.0,
                    "fill_material": config.initial_fill_material,
                }
            ]
        },
        "initial_ecohydrology_state": {
            "root_zone_salinity_ppt": config.initial_salinity_ppt,
            "aboveground_biomass_kg_m2": config.initial_aboveground_biomass_kg_m2,
            "belowground_biomass_kg_m2": config.initial_belowground_biomass_kg_m2,
            "lai": config.initial_lai,
            "litter_kg_m2": 0.0,
        },
        "output": {
            "file": str(output_path.with_suffix(".nc")),
            "write_time_series": True,
            "write_column_snapshots": False,
        },
    }

    with open(output_path, "w") as fh:
        yaml.dump(doc, fh, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)

    return output_path


# ---------------------------------------------------------------------------
# Output readers
# ---------------------------------------------------------------------------

def _read_ts(nc_path: Path) -> Dict[str, np.ndarray]:
    """Return all 1-D time series from a NetCDF output file."""
    series: Dict[str, np.ndarray] = {}
    with nc4.Dataset(str(nc_path), "r") as ds:
        for name, var in ds.variables.items():
            if var.ndim == 1 and "time" in var.dimensions:
                series[name] = np.array(var[:], dtype=np.float64)
    return series


def _read_material_names(nc_path: Path) -> List[str]:
    with nc4.Dataset(str(nc_path), "r") as ds:
        if "material_name" not in ds.variables:
            return []
        raw = ds.variables["material_name"][:]
        return [b"".join(row.data).decode("ascii", "ignore").rstrip("\x00").strip()
                for row in raw]


def _read_total_mass_by_material(nc_path: Path) -> Optional[np.ndarray]:
    """Return total_mass_by_material array (time, n_materials), or None."""
    with nc4.Dataset(str(nc_path), "r") as ds:
        if "total_mass_by_material" not in ds.variables:
            return None
        return np.array(ds.variables["total_mass_by_material"][:], dtype=np.float64)


def _compute_oc_kg_m2(nc_path: Path) -> Optional[np.ndarray]:
    """Total OC in column (kg C m-2) from refractory + labile organic pools."""
    mat_names = _read_material_names(nc_path)
    total_mass = _read_total_mass_by_material(nc_path)
    if total_mass is None or not mat_names:
        return None

    oc_pools = {"refractory_organic", "labile_organic"}
    indices = [i for i, n in enumerate(mat_names) if n in oc_pools]
    if not indices:
        return None

    return total_mass[:, indices].sum(axis=1) * OC_FRACTION


# ---------------------------------------------------------------------------
# Run management
# ---------------------------------------------------------------------------

def _label_to_stem(label: str) -> str:
    return "dt_" + label.replace(" ", "_")


def _run_all(
    config: PlotConfig,
    cli_binary: str,
    force: bool,
) -> Dict[str, Path]:
    """Run all timestep configurations; return {label: nc_path} dict."""
    results: Dict[str, Path] = {}
    for dt, label in zip(TIMESTEPS_DAYS, TIMESTEP_LABELS):
        stem = _label_to_stem(label)
        yaml_path = OUTPUT_DIR / f"{stem}.yaml"
        nc_path = OUTPUT_DIR / f"{stem}.nc"

        if not force and nc_path.exists():
            print(f"  [{label}] output exists, skipping run.")
        else:
            print(f"  [{label}]  dt={dt:.4g} d  →  {nc_path.name}")
            _write_config_dt(config, yaml_path, dt)
            try:
                run_model(str(yaml_path), cli_binary=cli_binary, silent=True)
            except RuntimeError as exc:
                print(f"    ERROR: {exc}")
                continue

        if nc_path.exists():
            results[label] = nc_path

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_COLOURS = ["#1b7837", "#762a83", "#e08214", "#4393c3", "#d73027"]


def _make_colormap(n: int):
    return [_COLOURS[i % len(_COLOURS)] for i in range(n)]


def plot_timeseries(
    results: Dict[str, Path],
    outpath: Optional[Path] = None,
) -> None:
    """Figure 1: 50-year surface elevation and column OC time series."""
    colours = _make_colormap(len(results))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax_elev, ax_oc = axes
    any_oc = False

    for (label, nc_path), colour in zip(results.items(), colours):
        ts = _read_ts(nc_path)
        t_yr = ts["model_time_days"] / _DAYS_IN_YEAR

        ax_elev.plot(t_yr, ts["surface_elevation"], color=colour,
                     linewidth=1.2, label=label)

        oc = _compute_oc_kg_m2(nc_path)
        if oc is not None:
            ax_oc.plot(t_yr, oc, color=colour, linewidth=1.2, label=label)
            any_oc = True

    ax_elev.set_ylabel("Surface elevation (m)")
    ax_elev.set_title("Surface elevation — timestep sensitivity (50-year run)")
    ax_elev.legend(fontsize=9, loc="best")
    ax_elev.grid(True, linewidth=0.4, alpha=0.5)

    ax_oc.set_xlabel("Model time (years)")
    ax_oc.set_ylabel("Column OC (kg C m⁻²)")
    ax_oc.set_title("Total column OC (refractory + labile organic × 0.45)")
    if any_oc:
        ax_oc.legend(fontsize=9, loc="best")
    else:
        ax_oc.text(0.5, 0.5, "OC data not available\n(recompile with updated simulator)",
                   ha="center", va="center", transform=ax_oc.transAxes,
                   color="grey", fontsize=10)
    ax_oc.grid(True, linewidth=0.4, alpha=0.5)

    fig.suptitle(
        f"Timestep sensitivity — pure M2 sine tide, {N_YEARS}-year run, "
        f"elevation {SURFACE_ELEVATION_M} m",
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


def plot_last_year_biomass(
    results: Dict[str, Path],
    outpath: Optional[Path] = None,
) -> None:
    """Figure 2: aboveground and live belowground biomass over the last year."""
    colours = _make_colormap(len(results))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_ag, ax_bg = axes

    for (label, nc_path), colour in zip(results.items(), colours):
        ts = _read_ts(nc_path)
        t_days = ts["model_time_days"]
        end = t_days[-1]
        start = end - _DAYS_IN_YEAR

        mask = t_days >= start
        doy = (t_days[mask] - start)   # days since start of last year

        above = ts["aboveground_biomass_kg_m2"][mask]
        below = ts["belowground_biomass_kg_m2"][mask]

        kw = dict(color=colour, linewidth=1.4, label=label)
        ax_ag.plot(doy, above, **kw)
        ax_bg.plot(doy, below, **kw)

    for ax, title, ylabel in [
        (ax_ag, "Aboveground biomass — last year", "Aboveground biomass (kg m⁻²)"),
        (ax_bg, "Live belowground biomass — last year", "Belowground biomass (kg m⁻²)"),
    ]:
        ax.set_xlabel("Day of year (year 50)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.set_xlim(0, _DAYS_IN_YEAR)

    fig.suptitle(
        f"Timestep sensitivity — biomass seasonal cycle (last year), "
        f"elevation {SURFACE_ELEVATION_M} m",
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
    p.add_argument("--binary", default="marsh_cli",
                   help="Path to the marsh_cli executable (default: marsh_cli on PATH).")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--skip-runs", action="store_true",
                   help="Do not re-run if output .nc files already exist.")
    g.add_argument("--plot-only", action="store_true",
                   help="Skip all runs; only plot existing outputs.")
    p.add_argument("--force", action="store_true",
                   help="Re-run all simulations even if output exists.")
    p.add_argument("--no-save", action="store_true",
                   help="Show figures interactively instead of saving to files.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    config = PlotConfig(
        site_name="sensitivity_sine",
        plot_id="timestep_test",
        surface_elevation_m=SURFACE_ELEVATION_M,
        distance_from_creek_m=DISTANCE_FROM_CREEK_M,
        tides=_sine_tidal_record(),
        met=north_inlet_default_met(),
        n_years=N_YEARS,
        output_dir=str(OUTPUT_DIR),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- run phase ----
    if not args.plot_only:
        print(f"Running {len(TIMESTEPS_DAYS)} simulations ({N_YEARS} years each)...")
        force_rerun = args.force and not args.skip_runs
        results = _run_all(config, cli_binary=args.binary,
                           force=force_rerun or (not args.skip_runs))
    else:
        results = {}
        for dt, label in zip(TIMESTEPS_DAYS, TIMESTEP_LABELS):
            nc_path = OUTPUT_DIR / f"{_label_to_stem(label)}.nc"
            if nc_path.exists():
                results[label] = nc_path
            else:
                print(f"  [{label}] no output found at {nc_path}, skipping.")

    if not results:
        print("No outputs found. Run without --plot-only first.")
        return

    # ---- plot phase ----
    print(f"\nPlotting {len(results)} run(s)...")

    if args.no_save:
        plot_timeseries(results, outpath=None)
        plot_last_year_biomass(results, outpath=None)
    else:
        plot_timeseries(
            results,
            outpath=FIGURE_DIR / "timestep_sensitivity_timeseries.png",
        )
        plot_last_year_biomass(
            results,
            outpath=FIGURE_DIR / "timestep_sensitivity_last_year.png",
        )


if __name__ == "__main__":
    main()
