#!/usr/bin/env python3
"""
inner_cycle_sensitivity.py

Run 10-year marsh simulations varying the number of tidal substeps per cycle
used by the composite water level model, with a fixed 1-day outer forcing
timestep.

Background
----------
The model has two distinct temporal scales:

  Outer forcing step (dt_days in forcing.steps):
      How often climate and tidal forcing statistics (temperature, PAR,
      suspended sediment concentration, salinity) are updated.  Here fixed
      at 1 day.

  Inner hydrological substeps (water_level_substeps_per_step):
      Within each forcing step, the composite_water_level_model evaluates
      water level at this many evenly-spaced time points and counts what
      fraction are above the marsh surface.  This inundation_fraction is
      then used by the GPP, salinity, and ET models.

The deposition model (edge_distance_deposition) is *not* affected by this
parameter -- it loops over every tidal cycle independently and uses an
analytical arcsine inundation formula that is exact for a sinusoidal tide.

The inundation fraction estimate from the water level model degrades for
high-marsh sites if substeps are sparse.  Consider a marsh surface that is
inundated for only 20% of each 12.42-hour tidal cycle.  With 4 substeps per
cycle, the expected number of inundated substeps is 0.8 -- so a typical
step returns either 0% or 25% rather than the correct 20%, biasing GPP and
salinity.

This script quantifies how many substeps per tidal cycle are needed for
the inundation fraction (and hence GPP, salinity, ET) to converge.

Substeps per tidal cycle tested: 2, 4, 8, 16, 32, 64, 128
(corresponding water_level_substeps_per_step for a 1-day outer step with a
12.42-hour tide: 4, 8, 15, 31, 62, 124, 247)

Plots produced
--------------
Figure 1 -- 10-year time series:
    Panel 1: Surface elevation (m)
    Panel 2: Total column OC (kg C m-2)

Figure 2 -- Last year seasonal cycle:
    Panel 1: Aboveground biomass (kg m-2)
    Panel 2: Live belowground biomass (kg m-2)

Figure 3 -- Inundation fraction time series (directly shows the sampling
    error for high-marsh substep counts)

Usage
-----
    python sensitivity/inner_cycle_sensitivity.py
    python sensitivity/inner_cycle_sensitivity.py --skip-runs
    python sensitivity/inner_cycle_sensitivity.py --plot-only
    python sensitivity/inner_cycle_sensitivity.py --binary /path/to/marsh_cli
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

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
except ImportError as exc:
    raise ImportError("matplotlib is required: pip install matplotlib") from exc

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_YEARS = 10
OUTER_DT_DAYS = 1.0             # fixed outer forcing timestep (days)
SURFACE_ELEVATION_M = 0.30      # m above MSL
DISTANCE_FROM_CREEK_M = 50.0

TIDAL_AMPLITUDE_M = 0.766
TIDAL_PERIOD_HOURS = 12.42
MSL_M = 0.0

SSC_KG_M3 = 0.020
FINE_SSC_KG_M3 = 0.005
CREEK_SALINITY_PPT = 28.0

OC_FRACTION = 0.45

# Substeps per tidal cycle to test.  Corresponding substeps_per_step for a
# 1-day outer step (1 day / 12.42 h * substeps_per_cycle):
SUBSTEPS_PER_CYCLE = [2, 4, 8, 16, 32, 64, 128]

_DAYS_IN_YEAR = 365.25
_CYCLES_PER_DAY = 24.0 / TIDAL_PERIOD_HOURS   # ~1.93 for M2


def _substeps_for_cycle_count(substeps_per_cycle: int) -> int:
    """Convert substeps-per-tidal-cycle to substeps-per-step for OUTER_DT_DAYS."""
    return max(2, round(OUTER_DT_DAYS * _CYCLES_PER_DAY * substeps_per_cycle))


SUBSTEPS_PER_STEP = [_substeps_for_cycle_count(n) for n in SUBSTEPS_PER_CYCLE]
SUBSTEP_LABELS = [f"{n}/cycle" for n in SUBSTEPS_PER_CYCLE]

OUTPUT_DIR = _THIS_DIR / "runs" / "inner_cycle"
FIGURE_DIR = _THIS_DIR / "figures"

# ---------------------------------------------------------------------------
# Tidal record (M2-only sine)
# ---------------------------------------------------------------------------

def _sine_tidal_record():
    from site_config import TideRecord
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
# Forcing builder (daily steps)
# ---------------------------------------------------------------------------

def _sinusoid(mean, amp, peak_day, doy):
    return mean + amp * math.cos(2.0 * math.pi * (doy - peak_day) / _DAYS_IN_YEAR)


def _build_forcing(config: PlotConfig) -> List[dict]:
    n_steps = round(config.n_years * _DAYS_IN_YEAR / OUTER_DT_DAYS)
    met = config.met
    tides = config.tides
    steps = []
    for i in range(n_steps):
        doy = ((i + 0.5) * OUTER_DT_DAYS) % _DAYS_IN_YEAR
        model_time_days = (i + 1) * OUTER_DT_DAYS
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
            "dt_days": round(OUTER_DT_DAYS, 4),
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
# YAML writer
# ---------------------------------------------------------------------------

def _write_config(
    config: PlotConfig,
    output_path: Path,
    substeps_per_step: int,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    forcing_steps = _build_forcing(config)

    parameters = dict(_DEFAULT_PARAMETERS)
    parameters.update(_tidal_constituent_parameters(config))
    parameters.update({
        "water_level_substeps_per_step": substeps_per_step,
        "deposition_distance_from_edge_m": config.distance_from_creek_m,
        "deposition_basin_length_m": 50.0,
        "deposition_length_scale_beta": 3.0,
    })

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
        "forcing": {"steps": forcing_steps},
        "initial_state": {
            "layers": [
                {
                    "top_elevation_m": config.surface_elevation_m,
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
    series = {}
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


def _compute_oc_kg_m2(nc_path: Path) -> Optional[np.ndarray]:
    with nc4.Dataset(str(nc_path), "r") as ds:
        if "total_mass_by_material" not in ds.variables:
            return None
        mat_names = _read_material_names(nc_path)
        total_mass = np.array(ds.variables["total_mass_by_material"][:])
    oc_pools = {"refractory_organic", "labile_organic"}
    indices = [i for i, n in enumerate(mat_names) if n in oc_pools]
    if not indices:
        return None
    return total_mass[:, indices].sum(axis=1) * OC_FRACTION


# ---------------------------------------------------------------------------
# Run management
# ---------------------------------------------------------------------------

def _label_to_stem(label: str) -> str:
    return "substeps_" + label.replace("/", "_per_")


def _run_all(config, cli_binary, force) -> Dict[str, Path]:
    results = {}
    for substeps, label in zip(SUBSTEPS_PER_STEP, SUBSTEP_LABELS):
        stem = _label_to_stem(label)
        yaml_path = OUTPUT_DIR / f"{stem}.yaml"
        nc_path = OUTPUT_DIR / f"{stem}.nc"

        cycles_check = OUTER_DT_DAYS * _CYCLES_PER_DAY
        cycles_per_cycle = substeps / cycles_check
        print(f"  [{label:10s}]  substeps_per_step={substeps:4d}  "
              f"({cycles_per_cycle:.1f} substeps/tidal-cycle)")

        if not force and nc_path.exists():
            print(f"             output exists, skipping run.")
        else:
            _write_config(config, yaml_path, substeps)
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

_COLOURS = ["#d73027", "#f46d43", "#fdae61", "#fee090",
            "#74add1", "#4575b4", "#313695"]


def plot_timeseries(results: Dict[str, Path], outpath=None) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax_elev, ax_oc = axes
    any_oc = False

    for (label, nc_path), colour in zip(results.items(), _COLOURS):
        ts = _read_ts(nc_path)
        t_yr = ts["model_time_days"] / _DAYS_IN_YEAR
        ax_elev.plot(t_yr, ts["surface_elevation"], color=colour,
                     linewidth=1.1, label=label)
        oc = _compute_oc_kg_m2(nc_path)
        if oc is not None:
            ax_oc.plot(t_yr, oc, color=colour, linewidth=1.1, label=label)
            any_oc = True

    ax_elev.set_ylabel("Surface elevation (m)")
    ax_elev.set_title("Surface elevation — inner cycle substep sensitivity")
    ax_elev.legend(fontsize=9, title="substeps/tidal-cycle", loc="best")
    ax_elev.grid(True, linewidth=0.4, alpha=0.5)

    ax_oc.set_xlabel("Model time (years)")
    ax_oc.set_ylabel("Column OC (kg C m⁻²)")
    ax_oc.set_title("Total column OC (refractory + labile organic × 0.45)")
    if any_oc:
        ax_oc.legend(fontsize=9, title="substeps/tidal-cycle", loc="best")
    else:
        ax_oc.text(0.5, 0.5, "OC data not available\n(recompile with updated simulator)",
                   ha="center", va="center", transform=ax_oc.transAxes,
                   color="grey", fontsize=10)
    ax_oc.grid(True, linewidth=0.4, alpha=0.5)

    fig.suptitle(
        f"Inner cycle substep sensitivity — 1-day outer step, {N_YEARS}-year run, "
        f"elevation {SURFACE_ELEVATION_M} m",
        fontsize=11,
    )
    plt.tight_layout()
    _save_or_show(fig, outpath)


def plot_last_year_biomass(results: Dict[str, Path], outpath=None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_ag, ax_bg = axes

    for (label, nc_path), colour in zip(results.items(), _COLOURS):
        ts = _read_ts(nc_path)
        t_days = ts["model_time_days"]
        mask = t_days >= t_days[-1] - _DAYS_IN_YEAR
        doy = t_days[mask] - (t_days[-1] - _DAYS_IN_YEAR)
        kw = dict(color=colour, linewidth=1.1, label=label)
        ax_ag.plot(doy, ts["aboveground_biomass_kg_m2"][mask], **kw)
        ax_bg.plot(doy, ts["belowground_biomass_kg_m2"][mask], **kw)

    for ax, title, ylabel in [
        (ax_ag, "Aboveground biomass — last year", "Aboveground biomass (kg m⁻²)"),
        (ax_bg, "Live belowground biomass — last year", "Belowground biomass (kg m⁻²)"),
    ]:
        ax.set_xlabel("Day of year (year 10)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9, title="substeps/cycle", loc="best")
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.set_xlim(0, _DAYS_IN_YEAR)

    fig.suptitle(
        f"Inner cycle substep sensitivity — biomass last year, "
        f"elevation {SURFACE_ELEVATION_M} m",
        fontsize=11,
    )
    plt.tight_layout()
    _save_or_show(fig, outpath)


def plot_inundation_fraction(results: Dict[str, Path], outpath=None) -> None:
    """Inundation fraction time series — most sensitive diagnostic to substep count."""
    fig, ax = plt.subplots(figsize=(11, 4))

    for (label, nc_path), colour in zip(results.items(), _COLOURS):
        ts = _read_ts(nc_path)
        if "inundation_fraction" not in ts:
            continue
        t_yr = ts["model_time_days"] / _DAYS_IN_YEAR
        # Plot annual mean to reduce visual noise
        year_idx = (t_yr).astype(int)
        n_years = int(year_idx[-1]) + 1
        ann_mean = np.array([
            ts["inundation_fraction"][year_idx == y].mean()
            for y in range(n_years)
        ])
        ax.plot(np.arange(n_years), ann_mean, color=colour,
                linewidth=1.4, marker=".", markersize=3, label=label)

    ax.set_xlabel("Model year")
    ax.set_ylabel("Annual mean inundation fraction")
    ax.set_title(
        "Inundation fraction convergence with substep count\n"
        "(computed by composite_water_level_model via substep counting)"
    )
    ax.legend(fontsize=9, title="substeps/tidal-cycle", loc="best")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.suptitle(
        f"Inner cycle substep sensitivity — 1-day outer step, elevation {SURFACE_ELEVATION_M} m",
        fontsize=11,
    )
    plt.tight_layout()
    _save_or_show(fig, outpath)


def _save_or_show(fig, outpath) -> None:
    if outpath:
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
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

    config = PlotConfig(
        site_name="sensitivity_sine",
        plot_id="inner_cycle_test",
        surface_elevation_m=SURFACE_ELEVATION_M,
        distance_from_creek_m=DISTANCE_FROM_CREEK_M,
        tides=_sine_tidal_record(),
        met=north_inlet_default_met(),
        n_years=N_YEARS,
        output_dir=str(OUTPUT_DIR),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.plot_only:
        print(f"Running {len(SUBSTEPS_PER_CYCLE)} simulations "
              f"({N_YEARS}-year, 1-day outer step)...")
        force = args.force and not args.skip_runs
        results = _run_all(config, cli_binary=args.binary,
                           force=force or (not args.skip_runs))
    else:
        results = {}
        for label in SUBSTEP_LABELS:
            nc_path = OUTPUT_DIR / f"{_label_to_stem(label)}.nc"
            if nc_path.exists():
                results[label] = nc_path

    if not results:
        print("No outputs found.")
        return

    print(f"\nPlotting {len(results)} run(s)...")
    save = not args.no_save

    plot_timeseries(
        results,
        outpath=FIGURE_DIR / "inner_cycle_timeseries.png" if save else None,
    )
    plot_last_year_biomass(
        results,
        outpath=FIGURE_DIR / "inner_cycle_last_year.png" if save else None,
    )
    plot_inundation_fraction(
        results,
        outpath=FIGURE_DIR / "inner_cycle_inundation.png" if save else None,
    )


if __name__ == "__main__":
    main()
