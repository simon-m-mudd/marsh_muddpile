#!/usr/bin/env python3
"""
restoration_experiment.py

50-year colonisation runs starting from bare sand — simulating marsh
restoration or initial platform establishment on bare intertidal substrate.

Initial column (1 m deep):
  - Sand only (no refractory or labile organic, no live roots)
  - Porosity 0.60, bulk density of mineral fraction = 2650 kg m⁻³
  - Vegetation seed biomass: 0.001 kg m⁻² above- and belowground
    (minimal propagule; model must grow biomass from near-zero)

Parameter sweep (full factorial, 3 × 2 × 2 × 2 × 2 = 48 runs):
  SSC (kg m⁻³):              0.005 / 0.020 / 0.050
  Distance from creek:       5 m / 20 m
  Starting elevation:        0.20 m MSL / 0.50 m MSL
  SLR rate:                  2 mm yr⁻¹ / 5 mm yr⁻¹
  Temperature offset (°C):   0 / +4  (added to every ERA5 monthly value)

Meteorological forcing: real ERA5 monthly means from era5_land_ni_1985_2024_daily.csv,
cycled over the 50-year run.  Tides: North Inlet harmonic constituents (NOAA 8661070).
Fine-sediment fraction: 25 % of total SSC.

Figures saved to sensitivity/figures/:
  restoration_biomass.png     — aboveground biomass (kg m⁻²) vs time
  restoration_elevation.png   — surface elevation (m MSL) vs time + MHW reference
  restoration_total_carbon.png — total organic C (kg m⁻²) vs time
  restoration_labile_carbon.png — labile C (kg m⁻²) vs time
  restoration_oc_profiles.png  — % OC depth profiles at year 50

Each figure uses a 4 × 2 panel grid (rows = temperature × starting elevation,
columns = distance from creek).  Colour = SSC, linestyle = SLR rate.

Usage
-----
    python sensitivity/restoration_experiment.py --binary ./build/marsh_cli
    python sensitivity/restoration_experiment.py --skip-runs   # plot only
    python sensitivity/restoration_experiment.py --force       # re-run all
"""

from __future__ import annotations

import argparse
import math
import sys
from itertools import product as itertools_product
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).parent.resolve()
_CALIB_DIR = _THIS_DIR.parent / "calibration"
sys.path.insert(0, str(_CALIB_DIR))

from site_config import PlotConfig, north_inlet_default_met, north_inlet_default_tides
from yaml_writer import _DEFAULT_PARAMETERS, _DEFAULT_MATERIALS
from forcing_builder import build_generated_forcing_block
from model_runner import run_model

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
# Sweep constants
# ---------------------------------------------------------------------------

N_YEARS = 50
_DAYS_IN_YEAR = 365.25

OUTPUT_DIR = _THIS_DIR / "runs" / "restoration"
FIGURE_DIR  = _THIS_DIR / "figures"

SSC_VALUES         = [0.005, 0.020, 0.050]   # kg m⁻³ total; fine = 25 %
DISTANCE_VALUES    = [5.0,  20.0]             # m from creek
ELEVATION_VALUES   = [0.20, 0.50]             # m above MSL
SLR_VALUES         = [0.002, 0.005]           # m yr⁻¹
TEMPERATURE_OFFSETS = [0.0, 4.0]             # °C added to every ERA5 monthly value

# ---------------------------------------------------------------------------
# Initial column — bare sand only
# ---------------------------------------------------------------------------

_N_LAYERS        = 20
_LAYER_THICKNESS = 0.05     # 20 × 0.05 m = 1 m total
_POROSITY        = 0.60
_RHO_SAND        = 2650.0   # kg m⁻³

_SEED_BIOMASS_KG_M2 = 0.001  # minimal propagule; model grows vegetation from this


def _build_sand_layers(surface_elevation_m: float) -> List[dict]:
    """Pure sand column — no organic matter, no roots."""
    solid_vol = (1.0 - _POROSITY) * _LAYER_THICKNESS
    sand_mass = round(_RHO_SAND * solid_vol, 4)

    layers = []
    for i in range(_N_LAYERS):
        top_elev = surface_elevation_m - i * _LAYER_THICKNESS
        layers.append({
            "top_elevation_m": round(top_elev, 4),
            "thickness_m":     _LAYER_THICKNESS,
            "porosity":        _POROSITY,
            "age_days":        0.0,
            "mass_kg_m2": {
                "sand":               sand_mass,
                "refractory_organic": 0.0,
                "labile_organic":     0.0,
                "roots":              0.0,
            },
        })
    # Convention: layer 0 = deepest, layer n-1 = surface
    return list(reversed(layers))


# ---------------------------------------------------------------------------
# Visual encoding
# ---------------------------------------------------------------------------

_SSC_COLOURS       = {0.005: "#4575b4", 0.020: "#d73027", 0.050: "#1a9641"}
_SLR_STYLES        = {0.002: "-",       0.005: "--"}
_TEMP_OFFSET_LABELS = {0.0: "ERA5 baseline", 4.0: "ERA5 +4 °C"}
_LINE_WIDTH        = 1.6

_ROW_COMBOS = [
    (dt, elev)
    for dt   in TEMPERATURE_OFFSETS
    for elev in ELEVATION_VALUES
]


# ---------------------------------------------------------------------------
# Parameter combination type
# ---------------------------------------------------------------------------

class RunParams(NamedTuple):
    ssc:         float
    distance:    float
    elevation:   float
    slr:         float
    temp_offset: float


def _all_params() -> List[RunParams]:
    return [
        RunParams(ssc, dist, elev, slr, dt)
        for ssc  in SSC_VALUES
        for dist in DISTANCE_VALUES
        for elev in ELEVATION_VALUES
        for slr  in SLR_VALUES
        for dt   in TEMPERATURE_OFFSETS
    ]


def _run_id(p: RunParams) -> str:
    sign = "p" if p.temp_offset >= 0 else "m"
    return (
        f"rst_ssc{p.ssc * 1000:.1f}gm3"
        f"_d{p.distance:.0f}m"
        f"_el{p.elevation * 100:.0f}cm"
        f"_slr{p.slr * 1000:.0f}mm"
        f"_dt{sign}{abs(p.temp_offset):.0f}c"
    )


# ---------------------------------------------------------------------------
# Tidal parameter helper
# ---------------------------------------------------------------------------

def _tidal_params(tides) -> dict:
    return {
        "water_level_constituent_M2_amplitude_m": tides.M2_amplitude_m,
        "water_level_constituent_M2_phase_deg":   tides.M2_phase_deg,
        "water_level_constituent_S2_amplitude_m": tides.S2_amplitude_m,
        "water_level_constituent_S2_phase_deg":   tides.S2_phase_deg,
        "water_level_constituent_N2_amplitude_m": tides.N2_amplitude_m,
        "water_level_constituent_N2_phase_deg":   tides.N2_phase_deg,
        "water_level_constituent_K1_amplitude_m": tides.K1_amplitude_m,
        "water_level_constituent_K1_phase_deg":   tides.K1_phase_deg,
        "water_level_constituent_O1_amplitude_m": tides.O1_amplitude_m,
        "water_level_constituent_O1_phase_deg":   tides.O1_phase_deg,
    }


# ---------------------------------------------------------------------------
# YAML config writer
# ---------------------------------------------------------------------------

def _write_yaml(p: RunParams, yaml_path: Path) -> None:
    tides = north_inlet_default_tides()
    tides.suspended_sediment_concentration_kg_m3 = p.ssc
    tides.fine_sediment_concentration_kg_m3      = p.ssc * 0.25

    met = north_inlet_default_met()
    met.temperature_mean_c += p.temp_offset

    config = PlotConfig(
        site_name="restoration_experiment",
        plot_id=_run_id(p),
        surface_elevation_m=p.elevation,
        distance_from_creek_m=p.distance,
        tides=tides,
        met=met,
        n_years=N_YEARS,
        sea_level_rise_m_yr=p.slr,
        output_dir=str(OUTPUT_DIR),
    )
    config.initial_salinity_ppt = tides.creek_salinity_ppt

    total_days = N_YEARS * _DAYS_IN_YEAR

    parameters = dict(_DEFAULT_PARAMETERS)
    parameters.update(_tidal_params(tides))
    parameters["deposition_distance_from_edge_m"] = p.distance
    parameters.setdefault("deposition_basin_length_m", 50.0)
    parameters.setdefault("deposition_length_scale_beta", 3.0)

    initial_layers = _build_sand_layers(p.elevation)

    doc = {
        "simulation": {
            "start_year": 0,
            "end_year":   N_YEARS,
            "dt_days":    365.0,
            "water_level_model_name":        "composite_water_level",
            "salinity_model_name":           "distance_flushing_salinity",
            "evapotranspiration_model_name": "simple_canopy_et",
            "vegetation_model_name":         "marsh_gpp_biomass",
            "deposition_model_name":         "edge_distance_deposition",
            "root_allocation_model_name":    "exponential_root_allocation",
            "decay_model_name":              "marsh_decay",
            "compaction_model_name":           "mixing_compaction",
            "porewater_chemistry_model_name": "none",
            "methane_model_name":            "none",
        },
        "site": {
            "distance_from_creek_m":  p.distance,
            "creek_bank_elevation_m": tides.mean_sea_level_m,
            "local_tidal_offset_m":   0.0,
        },
        "parameters": parameters,
        "materials":  _DEFAULT_MATERIALS,
        "forcing":    {"generated": build_generated_forcing_block(config)},
        "initial_state": {"layers": initial_layers},
        "initial_ecohydrology_state": {
            "root_zone_salinity_ppt":   config.initial_salinity_ppt,
            "aboveground_biomass_kg_m2": _SEED_BIOMASS_KG_M2,
            "belowground_biomass_kg_m2": _SEED_BIOMASS_KG_M2,
            "lai":         round(_SEED_BIOMASS_KG_M2 * parameters["vegetation_lai_per_kg_m2"], 6),
            "litter_kg_m2": 0.0,
        },
        "output": {
            "file":                str(yaml_path).replace(".yaml", ".nc"),
            "write_time_series":   True,
            "write_column_snapshots": True,
            "snapshot_times_days": [
                round(total_days * 0.20, 1),   # yr 10
                round(total_days * 0.50, 1),   # yr 25
                round(total_days * 0.80, 1),   # yr 40
                round(total_days,        1),   # yr 50
            ],
        },
    }

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as fh:
        yaml.dump(doc, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Run management
# ---------------------------------------------------------------------------

def _run_all(cli_binary: str, force: bool) -> Dict[RunParams, Path]:
    all_p = _all_params()
    n = len(all_p)
    print(f"Running {n} configurations ({N_YEARS} yr each) ...")
    results: Dict[RunParams, Path] = {}
    for i, p in enumerate(all_p, 1):
        rid       = _run_id(p)
        yaml_path = OUTPUT_DIR / f"{rid}.yaml"
        nc_path   = OUTPUT_DIR / f"{rid}.nc"

        if not force and nc_path.exists():
            print(f"  [{i:2d}/{n}] {rid}: exists, skipping.")
        else:
            print(f"  [{i:2d}/{n}] {rid}")
            _write_yaml(p, yaml_path)
            try:
                run_model(str(yaml_path), cli_binary=cli_binary, silent=True)
            except RuntimeError as exc:
                print(f"    ERROR: {exc}")
                continue

        if nc_path.exists():
            results[p] = nc_path
    return results


def _collect_existing() -> Dict[RunParams, Path]:
    results: Dict[RunParams, Path] = {}
    for p in _all_params():
        nc_path = OUTPUT_DIR / f"{_run_id(p)}.nc"
        if nc_path.exists():
            results[p] = nc_path
    return results


# ---------------------------------------------------------------------------
# NetCDF reader
# ---------------------------------------------------------------------------

_CARBON_FRACTION = 0.45


def _compute_mhw_offset(tides) -> float:
    deg2rad = np.pi / 180.0
    t_hours = np.arange(0.0, 365.25 * 24.0, 1.0 / 6.0)
    wl = np.zeros(len(t_hours))
    for speed, amp, phase in [
        (28.9841042, tides.M2_amplitude_m, tides.M2_phase_deg),
        (30.0000000, tides.S2_amplitude_m, tides.S2_phase_deg),
        (28.4397295, tides.N2_amplitude_m, tides.N2_phase_deg),
        (15.0410686, tides.K1_amplitude_m, tides.K1_phase_deg),
        (13.9430356, tides.O1_amplitude_m, tides.O1_phase_deg),
    ]:
        wl += amp * np.cos(speed * deg2rad * t_hours - phase * deg2rad)
    is_peak = (wl[1:-1] > wl[:-2]) & (wl[1:-1] > wl[2:])
    return float(np.mean(wl[1:-1][is_peak]))


def _read_run(nc_path: Path) -> dict:
    """Read time series from a restoration run."""
    with nc4.Dataset(str(nc_path), "r") as ds:
        t          = np.array(ds.variables["model_time_days"][:],      dtype=np.float64)
        surf_elev  = np.array(ds.variables["surface_elevation"][:],    dtype=np.float64)
        mass       = np.array(ds.variables["total_mass_by_material"][:], dtype=np.float64)
        mat_names  = [
            "".join(ds.variables["material_name"][j].compressed().astype("U"))
            for j in range(ds.dimensions["material"].size)
        ]

        # Aboveground biomass time series (if present)
        agb = None
        if "aboveground_biomass" in ds.variables:
            agb = np.array(ds.variables["aboveground_biomass"][:], dtype=np.float64)

        # Final column snapshot for OC profiles
        snap = None
        if "snapshot" in ds.dimensions and ds.dimensions["snapshot"].size > 0:
            snap_idx = ds.dimensions["snapshot"].size - 1
            n_lyr = int(ds.variables["snapshot_n_layers"][snap_idx])
            if n_lyr > 0:
                top_elev  = np.array(ds.variables["layer_top_elevation"][snap_idx, :n_lyr], dtype=np.float64)
                thickness = np.array(ds.variables["layer_thickness"][snap_idx, :n_lyr],     dtype=np.float64)
                lyr_mass  = np.array(ds.variables["layer_mass"][snap_idx, :n_lyr, :],       dtype=np.float64)
                snap = (top_elev, thickness, lyr_mass, float(surf_elev[-1]))

    idx_refrac = mat_names.index("refractory_organic")
    idx_labile = mat_names.index("labile_organic")
    idx_roots  = mat_names.index("roots")

    labile_c = mass[:, idx_labile]
    total_c  = mass[:, idx_refrac] + labile_c + mass[:, idx_roots]

    dt_yr       = np.diff(t) / _DAYS_IN_YEAR
    labile_rate = np.diff(labile_c) / np.where(dt_yr > 0, dt_yr, np.nan)
    t_rate_yr   = 0.5 * (t[:-1] + t[1:]) / _DAYS_IN_YEAR

    result = {
        "t_yr":                  t / _DAYS_IN_YEAR,
        "surface_elevation_m":   surf_elev,
        "total_carbon_kg_m2":    total_c,
        "labile_carbon_kg_m2":   labile_c,
        "t_rate_yr":             t_rate_yr,
        "labile_rate_kg_m2_yr":  labile_rate,
        "aboveground_biomass_kg_m2": agb,
        "oc_profile":            None,
    }

    if snap is not None:
        top_elev, thickness, lyr_mass, final_surf = snap
        total_lyr  = lyr_mass.sum(axis=1)
        organic_lyr = (
            lyr_mass[:, mat_names.index("labile_organic")]
            + lyr_mass[:, mat_names.index("refractory_organic")]
            + lyr_mass[:, mat_names.index("roots")]
        )
        pct_oc = np.where(
            total_lyr > 1e-10,
            organic_lyr * _CARBON_FRACTION / total_lyr * 100.0,
            0.0,
        )
        mid_elev  = top_elev - 0.5 * thickness
        depth_cm  = (final_surf - mid_elev) * 100.0
        mask      = (depth_cm >= 0.0) & (depth_cm <= 200.0)
        result["oc_profile"] = {"depth_cm": depth_cm[mask], "pct_oc": pct_oc[mask]}

    return result


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _legend_handles() -> List[mlines.Line2D]:
    handles = []
    for ssc, col in _SSC_COLOURS.items():
        handles.append(mlines.Line2D([], [], color=col, linewidth=_LINE_WIDTH,
                                     label=f"SSC {ssc * 1000:.1f} g m⁻³"))
    for slr, ls in _SLR_STYLES.items():
        handles.append(mlines.Line2D([], [], color="k", linestyle=ls,
                                     linewidth=_LINE_WIDTH,
                                     label=f"SLR {slr * 1000:.0f} mm yr⁻¹"))
    return handles


def _panel_title(dt: float, elev: float, dist: float) -> str:
    return f"{_TEMP_OFFSET_LABELS[dt]},  z₀ = {elev:+.2f} m MSL,  d = {dist:.0f} m"


def _plot_variable(
    data_map: Dict[RunParams, dict],
    x_key:    str,
    y_key:    str,
    xlabel:   str,
    ylabel:   str,
    suptitle: str,
    outpath:  Optional[Path],
) -> None:
    """4 × 2 panel figure; rows = temperature × elevation, cols = distance."""
    nrow = len(_ROW_COMBOS)
    ncol = len(DISTANCE_VALUES)
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 14), sharex=True, sharey=True)

    for ri, (dt, elev) in enumerate(_ROW_COMBOS):
        for ci, dist in enumerate(DISTANCE_VALUES):
            ax = axes[ri, ci]
            for p, data in data_map.items():
                if p.temp_offset != dt or p.elevation != elev or p.distance != dist:
                    continue
                y = data.get(y_key)
                if y is None:
                    continue
                ax.plot(
                    data[x_key], y,
                    color=_SSC_COLOURS[p.ssc],
                    linestyle=_SLR_STYLES[p.slr],
                    linewidth=_LINE_WIDTH, alpha=0.85,
                )
            ax.set_title(_panel_title(dt, elev, dist), fontsize=8)
            ax.grid(True, linewidth=0.3, alpha=0.4)
            if ci == 0:
                ax.set_ylabel(ylabel, fontsize=8)
            if ri == nrow - 1:
                ax.set_xlabel(xlabel, fontsize=8)

    fig.legend(handles=_legend_handles(), loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{suptitle}\n50-year restoration runs, North Inlet tides\n"
                 "Initial column: bare sand, seed biomass 0.001 kg m⁻²",
                 fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    if outpath:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        print(f"  Saved: {outpath}")
    else:
        plt.show()
    plt.close(fig)


def _plot_elevation_with_mhw(
    data_map:     Dict[RunParams, dict],
    mhw_offset_m: float,
    outpath:      Optional[Path],
) -> None:
    nrow = len(_ROW_COMBOS)
    ncol = len(DISTANCE_VALUES)
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 14), sharex=True, sharey=True)
    t_ref = np.linspace(0, N_YEARS, 500)

    for ri, (dt, elev) in enumerate(_ROW_COMBOS):
        for ci, dist in enumerate(DISTANCE_VALUES):
            ax = axes[ri, ci]
            for slr, ls in _SLR_STYLES.items():
                ax.plot(t_ref, mhw_offset_m + slr * t_ref,
                        color="0.55", linestyle=ls, linewidth=1.0, zorder=1)
            for p, data in data_map.items():
                if p.temp_offset != dt or p.elevation != elev or p.distance != dist:
                    continue
                ax.plot(data["t_yr"], data["surface_elevation_m"],
                        color=_SSC_COLOURS[p.ssc], linestyle=_SLR_STYLES[p.slr],
                        linewidth=_LINE_WIDTH, alpha=0.85, zorder=2)
            ax.set_title(_panel_title(dt, elev, dist), fontsize=8)
            ax.grid(True, linewidth=0.3, alpha=0.4)
            if ci == 0:
                ax.set_ylabel("Surface elevation (m MSL)", fontsize=8)
            if ri == nrow - 1:
                ax.set_xlabel("Model time (years)", fontsize=8)

    handles = _legend_handles()
    for slr, ls in _SLR_STYLES.items():
        handles.append(mlines.Line2D([], [], color="0.55", linestyle=ls,
                                     linewidth=1.0,
                                     label=f"MHW — SLR {slr * 1000:.0f} mm yr⁻¹"))
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Surface elevation vs MHW\n50-year restoration runs, North Inlet tides\n"
                 "Initial column: bare sand", fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    if outpath:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        print(f"  Saved: {outpath}")
    else:
        plt.show()
    plt.close(fig)


def _plot_oc_profiles(
    data_map: Dict[RunParams, dict],
    outpath:  Optional[Path],
) -> None:
    nrow = len(_ROW_COMBOS)
    ncol = len(DISTANCE_VALUES)
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 14), sharex=True, sharey=True)

    for ri, (dt, elev) in enumerate(_ROW_COMBOS):
        for ci, dist in enumerate(DISTANCE_VALUES):
            ax = axes[ri, ci]
            for p, data in data_map.items():
                if p.temp_offset != dt or p.elevation != elev or p.distance != dist:
                    continue
                prof = data.get("oc_profile")
                if prof is None:
                    continue
                order = np.argsort(prof["depth_cm"])
                ax.plot(prof["pct_oc"][order], prof["depth_cm"][order],
                        color=_SSC_COLOURS[p.ssc], linestyle=_SLR_STYLES[p.slr],
                        linewidth=_LINE_WIDTH, alpha=0.85)
            ax.set_ylim(200, 0)
            ax.set_title(_panel_title(dt, elev, dist), fontsize=8)
            ax.grid(True, linewidth=0.3, alpha=0.4)
            if ci == 0:
                ax.set_ylabel("Depth below surface (cm)", fontsize=8)
            if ri == nrow - 1:
                ax.set_xlabel("Organic carbon (% dry mass)", fontsize=8)

    fig.legend(handles=_legend_handles(), loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("% OC depth profile at year 50\n50-year restoration runs, North Inlet tides\n"
                 "Initial column: bare sand", fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    if outpath:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        print(f"  Saved: {outpath}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main plot dispatcher
# ---------------------------------------------------------------------------

def _plot_all(results_raw: Dict[RunParams, Path], save: bool) -> None:
    print("Reading outputs ...")
    data_map: Dict[RunParams, dict] = {}
    for p, nc_path in results_raw.items():
        try:
            data_map[p] = _read_run(nc_path)
        except Exception as exc:
            print(f"  Warning: could not read {nc_path.name}: {exc}")

    if not data_map:
        print("No outputs to plot.")
        return

    tides = north_inlet_default_tides()
    mhw_offset = _compute_mhw_offset(tides)
    n = len(data_map)
    print(f"Plotting {n} run(s) ...")

    # Check whether aboveground biomass is available
    has_agb = any(v.get("aboveground_biomass_kg_m2") is not None for v in data_map.values())

    if has_agb:
        _plot_variable(
            data_map,
            x_key="t_yr",
            y_key="aboveground_biomass_kg_m2",
            xlabel="Model time (years)",
            ylabel="Aboveground biomass (kg m⁻²)",
            suptitle="Vegetation establishment — aboveground biomass",
            outpath=FIGURE_DIR / "restoration_biomass.png" if save else None,
        )
    else:
        print("  Note: aboveground_biomass not in output — skipping biomass figure.")

    _plot_elevation_with_mhw(
        data_map, mhw_offset,
        outpath=FIGURE_DIR / "restoration_elevation.png" if save else None,
    )

    _plot_variable(
        data_map,
        x_key="t_yr",
        y_key="total_carbon_kg_m2",
        xlabel="Model time (years)",
        ylabel="Total organic carbon (kg m⁻²)",
        suptitle="Total organic carbon accumulation",
        outpath=FIGURE_DIR / "restoration_total_carbon.png" if save else None,
    )

    _plot_variable(
        data_map,
        x_key="t_yr",
        y_key="labile_carbon_kg_m2",
        xlabel="Model time (years)",
        ylabel="Labile organic carbon (kg m⁻²)",
        suptitle="Labile organic carbon accumulation",
        outpath=FIGURE_DIR / "restoration_labile_carbon.png" if save else None,
    )

    _plot_oc_profiles(
        data_map,
        outpath=FIGURE_DIR / "restoration_oc_profiles.png" if save else None,
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
                   help="Skip model runs; go straight to plotting existing outputs")
    g.add_argument("--plot-only", action="store_true",
                   help="Plot from existing outputs; never invoke the model")
    p.add_argument("--force",     action="store_true",
                   help="Re-run all simulations even if outputs exist")
    p.add_argument("--no-save",   action="store_true",
                   help="Display figures interactively instead of saving to disk")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    force       = args.force and not args.skip_runs
    run_needed  = not args.plot_only and (force or not args.skip_runs)

    if run_needed:
        results_raw = _run_all(args.binary, force=force)
    else:
        results_raw = _collect_existing()
        if not results_raw:
            print("No existing outputs found — run without --plot-only first.")
            return
        print(f"Found {len(results_raw)} existing output(s).")

    _plot_all(results_raw, save=not args.no_save)


if __name__ == "__main__":
    main()
