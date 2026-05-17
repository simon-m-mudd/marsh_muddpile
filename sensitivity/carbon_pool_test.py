#!/usr/bin/env python3
"""
carbon_pool_test.py

Diagnostic run to inspect column carbon-pool mass balance after fixing the
double-counting of belowground mortality in the root allocation model and the
incorrect application of decomposition modifiers in the decay model.

A single 20-year simulation is run with North Inlet-like forcing (M2-only
sine tide).  Parameters are adjusted so the root:shoot ratio is ~3-4,
consistent with field observations of Spartina alterniflora:

    vegetation_aboveground_capacity_kg_m2  1.5   (elevated above calibrated NI default 0.95 to produce a root:shoot ratio of ~3-4)
    vegetation_belowground_capacity_kg_m2  5.0   (unchanged)
    vegetation_shoot_allocation_max        0.35  (35 % to shoots, reduced from 0.65)
    vegetation_shoot_allocation_min        0.20  (20 % to shoots, reduced from 0.25)

These give root:shoot ≈ 3-3.5 at equilibrium under moderate stress.

Outputs
-------
Figure 1 — carbon_pool_test.png
    3 panels (full 20-year time series):
        1. Total column refractory organic mass (kg m⁻²)
        2. Total column labile organic mass      (kg m⁻²)
        3. Total column live root mass           (kg m⁻²)

Figure 2 — carbon_pool_test_depth.png
    3 panels (depth profiles at end of simulation):
        1. Refractory organic concentration (kg m⁻³) vs depth (m)
        2. Labile organic concentration     (kg m⁻³) vs depth (m)
        3. Live root mass per layer (kg m⁻²) vs depth (m)

Usage
-----
    python sensitivity/carbon_pool_test.py
    python sensitivity/carbon_pool_test.py --plot-only
    python sensitivity/carbon_pool_test.py --binary /path/to/marsh_cli
    python sensitivity/carbon_pool_test.py --force
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

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
# Run configuration
# ---------------------------------------------------------------------------

N_YEARS         = 20
OUTER_DT_DAYS   = 1.0
_DAYS_IN_YEAR   = 365.25

TIDAL_AMPLITUDE_M    = 0.766
TIDAL_PERIOD_HOURS   = 12.42
MSL_M                = 0.0
SURFACE_ELEVATION_M  = 0.30
DISTANCE_FROM_CREEK_M = 10.0
CREEK_SALINITY_PPT   = 28.0
SSC_KG_M3            = 0.020
FINE_SSC_KG_M3       = 0.005

OUTPUT_DIR = _THIS_DIR / "runs" / "carbon_pool_test"
FIGURE_DIR = _THIS_DIR / "figures"
_STEM      = "carbon_pool_test"

# ---------------------------------------------------------------------------
# Parameter overrides for realistic root : shoot ratio (~3-4)
# ---------------------------------------------------------------------------
_CARBON_TEST_PARAMS = {
    "vegetation_aboveground_capacity_kg_m2":  1.5,   # elevated above NI default (0.95) to give root:shoot ≈ 3-4
    "vegetation_belowground_capacity_kg_m2":  5.0,   # unchanged
    "vegetation_shoot_allocation_max":         0.35,  # 35 % to shoots (was 0.65)
    "vegetation_shoot_allocation_min":         0.20,  # 20 % to shoots (was 0.25)
}

# ---------------------------------------------------------------------------
# Tidal record (M2-only sine)
# ---------------------------------------------------------------------------

def _sine_tidal_record():
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
    t.creek_salinity_ppt    = CREEK_SALINITY_PPT
    t.suspended_sediment_concentration_kg_m3 = SSC_KG_M3
    t.fine_sediment_concentration_kg_m3      = FINE_SSC_KG_M3
    return t


# ---------------------------------------------------------------------------
# Forcing builder
# ---------------------------------------------------------------------------

def _sinusoid(mean: float, amp: float, peak_day: float, doy: float) -> float:
    return mean + amp * math.cos(2.0 * math.pi * (doy - peak_day) / _DAYS_IN_YEAR)


def _build_forcing(config: PlotConfig):
    n_steps = round(config.n_years * _DAYS_IN_YEAR / OUTER_DT_DAYS)
    met   = config.met
    tides = config.tides
    steps = []
    for i in range(n_steps):
        doy             = ((i + 0.5) * OUTER_DT_DAYS) % _DAYS_IN_YEAR
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

def _write_config(config: PlotConfig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parameters = dict(_DEFAULT_PARAMETERS)
    parameters.update(_tidal_constituent_parameters(config))
    parameters.update({
        "deposition_distance_from_edge_m": config.distance_from_creek_m,
        "deposition_basin_length_m":       50.0,
        "deposition_length_scale_beta":    3.0,
        "salinity_initial_root_zone_ppt":  CREEK_SALINITY_PPT,
    })
    parameters.update(_CARBON_TEST_PARAMS)

    total_days = round(config.n_years * _DAYS_IN_YEAR)

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
            "creek_bank_elevation_m": MSL_M,
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
            "root_zone_salinity_ppt":    CREEK_SALINITY_PPT,
            "aboveground_biomass_kg_m2": config.initial_aboveground_biomass_kg_m2,
            "belowground_biomass_kg_m2": config.initial_belowground_biomass_kg_m2,
            "lai":                       config.initial_lai,
            "litter_kg_m2":              0.0,
        },
        "output": {
            "file":                       str(output_path.with_suffix(".nc")),
            "write_time_series":          True,
            "write_column_snapshots":     True,
            "snapshot_times_days":        [total_days],
        },
    }

    with open(output_path, "w") as fh:
        yaml.dump(doc, fh, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)


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


def _read_material_names(nc_path: Path):
    with nc4.Dataset(str(nc_path), "r") as ds:
        if "material_name" not in ds.variables:
            return []
        raw = ds.variables["material_name"][:]
        return [
            b"".join(row.data).decode("ascii", "ignore").rstrip("\x00").strip()
            for row in raw
        ]


def _read_column_pools(nc_path: Path) -> Dict[str, np.ndarray]:
    """Return {material_name: total_mass_time_series} for pools of interest."""
    pools_of_interest = {"refractory_organic", "labile_organic", "roots"}
    with nc4.Dataset(str(nc_path), "r") as ds:
        if "total_mass_by_material" not in ds.variables:
            return {}
        names = _read_material_names(nc_path)
        total_mass = np.array(ds.variables["total_mass_by_material"][:])
    result = {}
    for i, name in enumerate(names):
        if name in pools_of_interest:
            result[name] = total_mass[:, i]
    return result


def _read_depth_profiles(nc_path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Read the final column snapshot from the NetCDF output.

    Returns
    -------
    depth_m : np.ndarray
        Depth below surface at each layer midpoint (m), positive downward.
    profiles : dict
        {material_name: concentration_or_mass_array} for refractory_organic,
        labile_organic, and roots.
        For refractory and labile: concentration in kg m⁻³ (mass / thickness).
        For roots: mass per layer in kg m⁻² (not divided by thickness, since
        roots are expressed as areal density rather than volumetric).
    """
    names = _read_material_names(nc_path)
    name_to_idx = {n: i for i, n in enumerate(names)}

    with nc4.Dataset(str(nc_path), "r") as ds:
        if "layer_mass" not in ds.variables:
            return np.array([]), {}

        # Use the last (and only requested) snapshot
        layer_mass       = np.array(ds.variables["layer_mass"][-1])        # (layer, material)
        layer_thickness  = np.array(ds.variables["layer_thickness"][-1])   # (layer,)
        layer_top_elev   = np.array(ds.variables["layer_top_elevation"][-1]) # (layer,)
        n_layers_var     = ds.variables.get("snapshot_n_layers", None)
        if n_layers_var is not None:
            n_layers = int(np.array(n_layers_var[-1]))
            layer_mass      = layer_mass[:n_layers]
            layer_thickness = layer_thickness[:n_layers]
            layer_top_elev  = layer_top_elev[:n_layers]

    # Surface elevation = top of layer 0
    surface_elevation = layer_top_elev[0]

    # Midpoint depth: positive downward from surface
    midpoint_depth = (surface_elevation - layer_top_elev) + 0.5 * layer_thickness

    profiles: Dict[str, np.ndarray] = {}
    for pool in ("refractory_organic", "labile_organic", "roots"):
        if pool not in name_to_idx:
            profiles[pool] = np.zeros(len(midpoint_depth))
            continue
        idx  = name_to_idx[pool]
        mass = layer_mass[:, idx]
        # Roots reported as kg m⁻² per layer; refractory/labile as concentration
        if pool == "roots":
            profiles[pool] = mass
        else:
            thickness = np.where(layer_thickness > 0, layer_thickness, np.nan)
            profiles[pool] = mass / thickness

    return midpoint_depth, profiles


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_carbon_pools(nc_path: Path, outpath: Optional[Path]) -> None:
    ts    = _read_ts(nc_path)
    pools = _read_column_pools(nc_path)

    t    = ts["model_time_days"]
    t_yr = t / _DAYS_IN_YEAR

    refractory = pools.get("refractory_organic", np.full(len(t), np.nan))
    labile     = pools.get("labile_organic",     np.full(len(t), np.nan))
    roots      = pools.get("roots",              np.full(len(t), np.nan))

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    _plot_panel(axes[0], t_yr, refractory,
                "Refractory organic (kg m⁻²)", "#7b2d8b")
    _plot_panel(axes[1], t_yr, labile,
                "Labile organic (kg m⁻²)", "#d6604d")
    _plot_panel(axes[2], t_yr, roots,
                "Live roots in column (kg m⁻²)", "#4dac26")

    axes[-1].set_xlabel("Model time (years)", fontsize=10)

    # Root:shoot annotation using final year mean
    above = ts.get("aboveground_biomass_kg_m2", None)
    below = ts.get("belowground_biomass_kg_m2", None)
    if above is not None and below is not None:
        final_mask = t >= (t[-1] - _DAYS_IN_YEAR)
        mean_above = above[final_mask].mean()
        mean_below = below[final_mask].mean()
        ratio = mean_below / mean_above if mean_above > 0 else float("nan")
        fig.text(0.98, 0.97,
                 f"Final-year mean root:shoot = {ratio:.2f}\n"
                 f"(above {mean_above:.3f} kg m⁻²,  below {mean_below:.3f} kg m⁻²)",
                 ha="right", va="top", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                           edgecolor="grey", alpha=0.8))

    fig.suptitle(
        f"Column carbon pools — {N_YEARS}-yr run (full time series)\n"
        f"elev {SURFACE_ELEVATION_M} m, dist {DISTANCE_FROM_CREEK_M} m, "
        f"M2-only tide {TIDAL_AMPLITUDE_M} m\n"
        f"shoot_alloc_max = {_CARBON_TEST_PARAMS['vegetation_shoot_allocation_max']}, "
        f"above_cap = {_CARBON_TEST_PARAMS['vegetation_aboveground_capacity_kg_m2']} kg m⁻², "
        f"below_cap = {_CARBON_TEST_PARAMS['vegetation_belowground_capacity_kg_m2']} kg m⁻²",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if outpath is not None:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        print(f"Saved: {outpath}")
    else:
        plt.show()
    plt.close(fig)


def plot_depth_profiles(nc_path: Path, outpath: Optional[Path]) -> None:
    depth, profiles = _read_depth_profiles(nc_path)

    if len(depth) == 0:
        print("No snapshot data found — skipping depth profile plot.")
        return

    refractory = profiles.get("refractory_organic", np.zeros_like(depth))
    labile     = profiles.get("labile_organic",     np.zeros_like(depth))
    roots      = profiles.get("roots",              np.zeros_like(depth))

    fig, axes = plt.subplots(1, 3, figsize=(12, 7), sharey=True)

    def _depth_panel(ax, x, xlabel, colour):
        # Horizontal step plot: each layer as a horizontal bar
        ax.step(x, depth, where="mid", color=colour, linewidth=1.4)
        ax.fill_betweenx(depth, 0, x, step="mid", color=colour, alpha=0.25)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.invert_yaxis()
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.tick_params(labelsize=8)
        ax.set_xlim(left=0)

    _depth_panel(axes[0], refractory,
                 "Refractory organic\nconcentration (kg m⁻³)", "#7b2d8b")
    _depth_panel(axes[1], labile,
                 "Labile organic\nconcentration (kg m⁻³)", "#d6604d")
    _depth_panel(axes[2], roots,
                 "Live root mass per layer\n(kg m⁻²)", "#4dac26")

    axes[0].set_ylabel("Depth below surface (m)", fontsize=9)

    fig.suptitle(
        f"Carbon depth profiles — end of {N_YEARS}-yr run\n"
        f"elev {SURFACE_ELEVATION_M} m, dist {DISTANCE_FROM_CREEK_M} m, "
        f"M2-only tide {TIDAL_AMPLITUDE_M} m",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if outpath is not None:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        print(f"Saved: {outpath}")
    else:
        plt.show()
    plt.close(fig)


def _plot_panel(ax, t_yr, y, ylabel, colour, zero_line=False):
    ax.plot(t_yr, y, color=colour, linewidth=1.1)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    if zero_line:
        ax.axhline(0, color="k", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=8)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--binary",    default="marsh_cli")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--force",     action="store_true")
    p.add_argument("--no-save",   action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    yaml_path = OUTPUT_DIR / f"{_STEM}.yaml"
    nc_path   = OUTPUT_DIR / f"{_STEM}.nc"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.plot_only:
        if args.force or not nc_path.exists():
            tides  = _sine_tidal_record()
            met    = north_inlet_default_met()
            config = PlotConfig(
                site_name="carbon_pool_test",
                plot_id=_STEM,
                surface_elevation_m=SURFACE_ELEVATION_M,
                distance_from_creek_m=DISTANCE_FROM_CREEK_M,
                tides=tides,
                met=met,
                n_years=N_YEARS,
                output_dir=str(OUTPUT_DIR),
            )
            _write_config(config, yaml_path)
            print(f"Running {N_YEARS}-year simulation...")
            try:
                run_model(str(yaml_path), cli_binary=args.binary, silent=False)
            except RuntimeError as exc:
                print(f"ERROR: {exc}")
                return
        else:
            print(f"Output exists ({nc_path.name}), skipping run. Use --force to re-run.")

    if not nc_path.exists():
        print("No output found.")
        return

    ts_outpath    = FIGURE_DIR / f"{_STEM}.png"         if not args.no_save else None
    depth_outpath = FIGURE_DIR / f"{_STEM}_depth.png"   if not args.no_save else None

    print("Plotting time series...")
    plot_carbon_pools(nc_path, ts_outpath)

    print("Plotting depth profiles...")
    plot_depth_profiles(nc_path, depth_outpath)


if __name__ == "__main__":
    main()
