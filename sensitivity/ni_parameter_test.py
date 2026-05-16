#!/usr/bin/env python3
"""
ni_parameter_test.py

Parameter sensitivity tests calibrated to North Inlet LTER unfertilised plots
(Goat Island, South Carolina).

Observations from unfertilised control plots:
  - Peak aboveground biomass:  ~1 kg m⁻²  (NI LTER edi.135.12)
  - Belowground biomass:       ~1 kg m⁻²
  - Accretion rate:            ~3–4 mm yr⁻¹ (= local SLR; edi.134.12 cm units)
  - Accretion majority from:   organic matter (low SSC at NI)

Run structure (200 years total):
  Years   0–100 spinup  — biomass and sediment column reach equilibrium under
                          constant climate and SLR.  Not analysed.
  Years 100–200 analysis — late-period accretion rate and biomass diagnostics.

Parameter sweep (3 × 3 = 9 runs):
  vegetation_lue_gC_per_umol:         [1.5e-4, 2.5e-4, 3.5e-4]
  (aboveground_capacity, belowground_capacity) kg m⁻²:
                                       [(0.8, 0.8), (1.0, 1.0), (1.5, 1.5)]

All other parameters use the calibrated defaults from yaml_writer.py, including
  vegetation_belowground_mortality_base_per_day = 2.74e-4 d⁻¹  (10 % yr⁻¹,
  Morris & Sundberg 2023) and root_allocation_to_labile_fraction = 0.90.

Figures saved to sensitivity/figures/:
  ni_biomass.png        — aboveground + belowground biomass vs time
  ni_elevation.png      — surface elevation vs time with SLR reference
  ni_accretion.png      — late-period accretion rate summary

Usage
-----
    python sensitivity/ni_parameter_test.py --binary ./build/marsh_cli
    python sensitivity/ni_parameter_test.py --plot-only
    python sensitivity/ni_parameter_test.py --force
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

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
    import matplotlib.patches as mpatches
except ImportError as exc:
    raise ImportError("matplotlib is required: pip install matplotlib") from exc

# ---------------------------------------------------------------------------
# Site and run constants — North Inlet GI low marsh
# ---------------------------------------------------------------------------

N_YEARS      = 400
SPINUP_YEARS = 300         # years discarded from analysis
_DAYS_IN_YEAR = 365.25

# NI long-run equilibrium surface is ~0.50 m above MSL (Miller 2019 parabola peak
# at 35.6 cm above NAVD88 ≈ 50 cm above MSL, assuming MSL ≈ NAVD88 + 0.15 m).
# The two-stage compaction model collapses the initial 1 m column (porosity 0.60)
# by ~0.36 m on the first step, so the pre-compaction elevation must be set to
# 0.50 + 0.36 = 0.86 m to start near the long-run equilibrium and avoid the
# slow transient that occurs when starting at the lower 0.20 m initial elevation.
NI_ELEVATION_M       = 0.86
NI_DISTANCE_M        = 20.0
NI_SLR_M_YR          = 0.003   # 3 mm yr⁻¹ (NOAA Charleston/Springmaid Pier)
NI_SSC_KG_M3         = 0.006   # SSC: NI is a low-turbidity system (5-15 mg/L in literature)

# Observed targets (unfertilised NI plots)
TARGET_ABG_KG_M2 = 1.0   # peak aboveground biomass  (kg m⁻²)
TARGET_BLG_KG_M2 = 1.0   # belowground biomass        (kg m⁻²)
TARGET_ACCRETION_MM_YR_LOW  = 2.0   # lower bound (mm yr⁻¹)
TARGET_ACCRETION_MM_YR_HIGH = 5.0   # upper bound (mm yr⁻¹)

OUTPUT_DIR = _THIS_DIR / "runs" / "ni_parameter_test"
FIGURE_DIR = _THIS_DIR / "figures"

# ---------------------------------------------------------------------------
# Initial column — 1 m deep, uniform mineral base with thin organic surface
# ---------------------------------------------------------------------------

_N_LAYERS         = 20
_LAYER_THICKNESS  = 0.05   # m  (20 × 0.05 = 1 m)
_POROSITY         = 0.60
_ORGANIC_EFOLDING = 0.075  # m

_RHO_SAND   = 2650.0
_RHO_REFRAC = 1400.0
_RHO_LABILE = 1200.0
_RHO_ROOTS  = 1100.0


def _build_initial_layers(surface_elevation_m: float) -> List[dict]:
    solid_vol = (1.0 - _POROSITY) * _LAYER_THICKNESS
    layers = []
    for i in range(_N_LAYERS):
        depth_mid = (i + 0.5) * _LAYER_THICKNESS
        top_elev  = surface_elevation_m - i * _LAYER_THICKNESS

        refrac_frac = 0.05
        exp_fac     = math.exp(-depth_mid / _ORGANIC_EFOLDING)
        labile_frac = 0.05 * exp_fac
        root_frac   = 0.05 * exp_fac
        sand_frac   = max(0.0, 1.0 - refrac_frac - labile_frac - root_frac)

        layers.append({
            "top_elevation_m": round(top_elev, 4),
            "thickness_m": _LAYER_THICKNESS,
            "porosity": _POROSITY,
            "age_days": 0.0,
            "mass_kg_m2": {
                "sand":               round(_RHO_SAND   * sand_frac   * solid_vol, 4),
                "refractory_organic": round(_RHO_REFRAC * refrac_frac * solid_vol, 4),
                "labile_organic":     round(_RHO_LABILE * labile_frac * solid_vol, 6),
                "roots":              round(_RHO_ROOTS  * root_frac   * solid_vol, 6),
            },
        })
    return list(reversed(layers))   # deepest first


# ---------------------------------------------------------------------------
# Parameter combinations
# ---------------------------------------------------------------------------

class RunParams(NamedTuple):
    lue: float           # vegetation_lue_gC_per_umol
    capacity: float      # aboveground AND belowground capacity (kg m⁻²)


LUE_VALUES      = [1.5e-4, 2.5e-4, 3.5e-4]
CAPACITY_VALUES = [0.8, 1.0, 1.5]           # applied to both ab- and belowground


def _all_params() -> List[RunParams]:
    return [RunParams(lue, cap) for lue in LUE_VALUES for cap in CAPACITY_VALUES]


def _run_id(p: RunParams) -> str:
    return f"ni_lue{p.lue:.1e}_cap{p.capacity:.1f}".replace("+", "")


# ---------------------------------------------------------------------------
# Tidal constituent parameter block
# ---------------------------------------------------------------------------

def _tidal_params(tides) -> dict:
    return {
        "water_level_constituent_M2_amplitude_m":  tides.M2_amplitude_m,
        "water_level_constituent_M2_phase_deg":    tides.M2_phase_deg,
        "water_level_constituent_S2_amplitude_m":  tides.S2_amplitude_m,
        "water_level_constituent_S2_phase_deg":    tides.S2_phase_deg,
        "water_level_constituent_N2_amplitude_m":  tides.N2_amplitude_m,
        "water_level_constituent_N2_phase_deg":    tides.N2_phase_deg,
        "water_level_constituent_K1_amplitude_m":  tides.K1_amplitude_m,
        "water_level_constituent_K1_phase_deg":    tides.K1_phase_deg,
        "water_level_constituent_O1_amplitude_m":  tides.O1_amplitude_m,
        "water_level_constituent_O1_phase_deg":    tides.O1_phase_deg,
    }


# ---------------------------------------------------------------------------
# YAML writer
# ---------------------------------------------------------------------------

def _write_yaml(p: RunParams, yaml_path: Path) -> None:
    tides = north_inlet_default_tides()
    tides.suspended_sediment_concentration_kg_m3 = NI_SSC_KG_M3
    tides.fine_sediment_concentration_kg_m3      = NI_SSC_KG_M3 * 0.25

    config = PlotConfig(
        site_name="north_inlet",
        plot_id=_run_id(p),
        surface_elevation_m=NI_ELEVATION_M,
        distance_from_creek_m=NI_DISTANCE_M,
        tides=tides,
        met=north_inlet_default_met(),
        n_years=N_YEARS,
        sea_level_rise_m_yr=NI_SLR_M_YR,
        output_dir=str(OUTPUT_DIR),
    )
    config.initial_salinity_ppt = tides.creek_salinity_ppt

    parameters = dict(_DEFAULT_PARAMETERS)
    parameters.update(_tidal_params(tides))
    parameters["deposition_distance_from_edge_m"] = NI_DISTANCE_M
    parameters.setdefault("deposition_basin_length_m", 50.0)
    parameters.setdefault("deposition_length_scale_beta", 3.0)
    # swept parameters
    parameters["vegetation_lue_gC_per_umol"]             = p.lue
    parameters["vegetation_aboveground_capacity_kg_m2"]  = p.capacity
    parameters["vegetation_belowground_capacity_kg_m2"]  = p.capacity

    initial_layers = _build_initial_layers(NI_ELEVATION_M)
    total_days     = N_YEARS * _DAYS_IN_YEAR

    # Start with modest initial biomass so spinup converges quickly
    initial_abg = min(0.3, p.capacity * 0.3)
    initial_blg = min(0.3, p.capacity * 0.3)

    doc = {
        "simulation": {
            "start_year": 0,
            "end_year": N_YEARS,
            "dt_days": 365.0,
            "water_level_model_name":         "composite_water_level",
            "salinity_model_name":            "distance_flushing_salinity",
            "evapotranspiration_model_name":  "simple_canopy_et",
            "vegetation_model_name":          "marsh_gpp_biomass",
            "deposition_model_name":          "edge_distance_deposition",
            "root_allocation_model_name":     "exponential_root_allocation",
            "decay_model_name":               "marsh_decay",
            "compaction_model_name":          "two_stage_compaction",
            "porewater_chemistry_model_name": "none",
            "methane_model_name":             "none",
        },
        "site": {
            "distance_from_creek_m":  NI_DISTANCE_M,
            "creek_bank_elevation_m": tides.mean_sea_level_m,
            "local_tidal_offset_m":   0.0,
        },
        "parameters": parameters,
        "materials": _DEFAULT_MATERIALS,
        "forcing": {"generated": build_generated_forcing_block(config)},
        "initial_state": {"layers": initial_layers},
        "initial_ecohydrology_state": {
            "root_zone_salinity_ppt":       config.initial_salinity_ppt,
            "aboveground_biomass_kg_m2":    round(initial_abg, 4),
            "belowground_biomass_kg_m2":    round(initial_blg, 4),
            "lai": round(initial_abg * parameters["vegetation_lai_per_kg_m2"], 4),
            "litter_kg_m2": 0.0,
        },
        "output": {
            "file": str(yaml_path).replace(".yaml", ".nc"),
            "write_time_series": True,
            "write_column_snapshots": True,
            "snapshot_times_days": [
                round(total_days * 0.5, 1),
                round(total_days, 1),
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
    print(f"Running {n} configurations ({N_YEARS} yr each, {SPINUP_YEARS} yr spinup) ...")
    results: Dict[RunParams, Path] = {}
    for i, p in enumerate(all_p, 1):
        rid      = _run_id(p)
        yaml_path = OUTPUT_DIR / f"{rid}.yaml"
        nc_path   = OUTPUT_DIR / f"{rid}.nc"

        if not force and nc_path.exists():
            print(f"  [{i:2d}/{n}] {rid}: cached.")
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

def _read_run(nc_path: Path) -> dict:
    with nc4.Dataset(str(nc_path), "r") as ds:
        t    = np.array(ds["model_time_days"][:], dtype=np.float64) / _DAYS_IN_YEAR
        abg  = np.array(ds["aboveground_biomass_kg_m2"][:], dtype=np.float64)
        blg  = np.array(ds["belowground_biomass_kg_m2"][:], dtype=np.float64)
        elev = np.array(ds["surface_elevation"][:], dtype=np.float64)
        gpp  = np.array(ds["gpp_gC_m2_d"][:], dtype=np.float64)

        mat_names = [
            "".join(ds["material_name"][j].compressed().astype("U"))
            for j in range(ds.dimensions["material"].size)
        ]
        mass = np.array(ds["total_mass_by_material"][:], dtype=np.float64)

    analysis = t >= SPINUP_YEARS
    if analysis.sum() < 2:
        accretion_mm_yr = np.nan
        mean_abg = mean_blg = mean_gpp = np.nan
    else:
        slope           = np.polyfit(t[analysis], elev[analysis], 1)[0]
        accretion_mm_yr = slope * 1000.0
        mean_abg        = abg[analysis].mean()
        mean_blg        = blg[analysis].mean()
        mean_gpp        = gpp[analysis].mean()

    # organic vs mineral fraction of total mass at end of run
    idx_refrac = mat_names.index("refractory_organic")
    idx_labile = mat_names.index("labile_organic")
    idx_roots  = mat_names.index("roots")
    organic_total = mass[:, idx_refrac] + mass[:, idx_labile] + mass[:, idx_roots]
    mineral_total = mass.sum(axis=1) - organic_total
    final_organic_frac = (
        organic_total[-1] / mass[-1].sum() if mass[-1].sum() > 0 else np.nan
    )

    return {
        "t_yr":               t,
        "abg_kg_m2":          abg,
        "blg_kg_m2":          blg,
        "surface_elevation_m": elev,
        "gpp_gC_m2_d":        gpp,
        "accretion_mm_yr":    accretion_mm_yr,
        "mean_abg":           mean_abg,
        "mean_blg":           mean_blg,
        "mean_gpp":           mean_gpp,
        "final_organic_frac": final_organic_frac,
    }


# ---------------------------------------------------------------------------
# Visual encoding
# ---------------------------------------------------------------------------

_LUE_COLOURS = {
    1.5e-4: "#4575b4",
    2.5e-4: "#d73027",
    3.5e-4: "#1a9641",
}
_CAP_STYLES = {
    0.8: ":",
    1.0: "-",
    1.5: "--",
}
_LINE_WIDTH = 1.8


def _legend_handles() -> List[mlines.Line2D]:
    handles = []
    for lue, col in _LUE_COLOURS.items():
        handles.append(mlines.Line2D(
            [], [], color=col, linewidth=_LINE_WIDTH,
            label=f"LUE {lue:.1e} gC μmol⁻¹",
        ))
    for cap, ls in _CAP_STYLES.items():
        handles.append(mlines.Line2D(
            [], [], color="k", linestyle=ls, linewidth=_LINE_WIDTH,
            label=f"Capacity {cap:.1f} kg m⁻²",
        ))
    return handles


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_biomass(data_map: Dict[RunParams, dict], save: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    for p, d in data_map.items():
        kw = dict(color=_LUE_COLOURS[p.lue], linestyle=_CAP_STYLES[p.capacity],
                  linewidth=_LINE_WIDTH, alpha=0.85)
        axes[0].plot(d["t_yr"], d["abg_kg_m2"], **kw)
        axes[1].plot(d["t_yr"], d["blg_kg_m2"], **kw)

    for ax, target, label in [
        (axes[0], TARGET_ABG_KG_M2, "Aboveground biomass (kg m⁻²)"),
        (axes[1], TARGET_BLG_KG_M2, "Belowground biomass (kg m⁻²)"),
    ]:
        ax.axhline(target, color="k", linewidth=1.2, linestyle="-.",
                   label=f"NI target ~{target:.0f} kg m⁻²")
        ax.axvline(SPINUP_YEARS, color="0.6", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Model time (years)")
        ax.set_ylabel(label)
        ax.grid(True, linewidth=0.3, alpha=0.4)

    fig.legend(handles=_legend_handles() + [
        mlines.Line2D([], [], color="k", linestyle="-.", linewidth=1.2,
                      label="NI unfertilised target"),
    ], loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(
        "Biomass trajectories — North Inlet parameter test\n"
        f"(elevation {NI_ELEVATION_M:.2f} m MSL, {NI_DISTANCE_M:.0f} m from creek, "
        f"SLR {NI_SLR_M_YR*1000:.0f} mm yr⁻¹)",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0.10, 1, 0.95])
    _save_or_show(fig, FIGURE_DIR / "ni_biomass.png", save)


def _plot_elevation(data_map: Dict[RunParams, dict], save: bool) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    t_ref = np.linspace(0, N_YEARS, 500)
    initial_msl = 0.0
    slr_line = initial_msl + NI_SLR_M_YR * t_ref
    ax.plot(t_ref, slr_line, color="0.4", linewidth=1.2, linestyle="--",
            label=f"MSL rise ({NI_SLR_M_YR*1000:.0f} mm yr⁻¹)")

    for p, d in data_map.items():
        ax.plot(d["t_yr"], d["surface_elevation_m"],
                color=_LUE_COLOURS[p.lue], linestyle=_CAP_STYLES[p.capacity],
                linewidth=_LINE_WIDTH, alpha=0.85)

    ax.axvline(SPINUP_YEARS, color="0.6", linewidth=0.8, linestyle="--",
               label="End of spinup")
    ax.set_xlabel("Model time (years)")
    ax.set_ylabel("Surface elevation (m MSL)")
    ax.grid(True, linewidth=0.3, alpha=0.4)

    fig.legend(handles=_legend_handles() + [
        mlines.Line2D([], [], color="0.4", linestyle="--", linewidth=1.2,
                      label=f"MSL rise ({NI_SLR_M_YR*1000:.0f} mm yr⁻¹)"),
    ], loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(
        "Surface elevation — North Inlet parameter test",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0.10, 1, 0.95])
    _save_or_show(fig, FIGURE_DIR / "ni_elevation.png", save)


def _plot_accretion_summary(data_map: Dict[RunParams, dict], save: bool) -> None:
    """Bar chart of late-period accretion rate and organic fraction per run."""
    params  = list(data_map.keys())
    accretions = [data_map[p]["accretion_mm_yr"] for p in params]
    org_fracs  = [data_map[p]["final_organic_frac"] for p in params]

    x = np.arange(len(params))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    bar_colours = [_LUE_COLOURS[p.lue] for p in params]
    bars = ax1.bar(x, accretions, color=bar_colours, alpha=0.8, edgecolor="k",
                   linewidth=0.5)

    ax1.axhspan(TARGET_ACCRETION_MM_YR_LOW, TARGET_ACCRETION_MM_YR_HIGH,
                color="gold", alpha=0.35, label="NI target range (2–5 mm yr⁻¹)")
    ax1.axhline(NI_SLR_M_YR * 1000, color="k", linestyle="--", linewidth=1.2,
                label=f"SLR rate ({NI_SLR_M_YR*1000:.0f} mm yr⁻¹)")
    ax1.set_ylabel(f"Accretion rate (mm yr⁻¹)\nyears {SPINUP_YEARS}–{N_YEARS}")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", linewidth=0.3, alpha=0.4)

    ax2.bar(x, [f * 100 for f in org_fracs], color=bar_colours, alpha=0.8,
            edgecolor="k", linewidth=0.5)
    ax2.set_ylabel("Organic fraction of total\ncolumn mass (%) at end of run")
    ax2.grid(axis="y", linewidth=0.3, alpha=0.4)

    labels = [
        f"LUE={p.lue:.1e}\ncap={p.capacity:.1f}" for p in params
    ]
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7)

    # Hatch bars by capacity linestyle for a second visual cue
    for bar, p in zip(bars, params):
        if p.capacity == 0.8:
            bar.set_hatch("///")
        elif p.capacity == 1.5:
            bar.set_hatch("...")

    # Legend patches for capacity
    legend_patches = [
        mpatches.Patch(facecolor="white", edgecolor="k", hatch="///",
                       label="Capacity 0.8 kg m⁻²"),
        mpatches.Patch(facecolor="white", edgecolor="k",
                       label="Capacity 1.0 kg m⁻²"),
        mpatches.Patch(facecolor="white", edgecolor="k", hatch="...",
                       label="Capacity 1.5 kg m⁻²"),
    ] + [
        mpatches.Patch(facecolor=col, edgecolor="k",
                       label=f"LUE {lue:.1e}")
        for lue, col in _LUE_COLOURS.items()
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=6, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"Accretion rate and organic fraction — North Inlet parameter test\n"
        f"(years {SPINUP_YEARS}–{N_YEARS} analysis period, "
        f"belowground mortality = 2.74×10⁻⁴ d⁻¹ = 10 % yr⁻¹)",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    _save_or_show(fig, FIGURE_DIR / "ni_accretion.png", save)


def _save_or_show(fig, path: Path, save: bool) -> None:
    if save:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(data_map: Dict[RunParams, dict]) -> None:
    header = f"{'Run':<30} {'Accretion':>12} {'Mean ABG':>10} {'Mean BLG':>10} {'Org frac':>10} {'Mean GPP':>10}"
    print()
    print(header)
    print("-" * len(header))
    for p, d in sorted(data_map.items()):
        print(
            f"{_run_id(p):<30}"
            f"  {d['accretion_mm_yr']:>9.2f} mm/yr"
            f"  {d['mean_abg']:>8.3f} kg/m²"
            f"  {d['mean_blg']:>8.3f} kg/m²"
            f"  {d['final_organic_frac']:>8.1%}"
            f"  {d['mean_gpp']:>8.3f} gC/m²/d"
        )
    print()
    print(f"Target: accretion ~{NI_SLR_M_YR*1000:.0f} mm/yr, "
          f"ABG ~{TARGET_ABG_KG_M2:.0f} kg/m², BLG ~{TARGET_BLG_KG_M2:.0f} kg/m²")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--binary", default="marsh_cli",
                   help="Path to marsh_cli executable")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--plot-only", action="store_true",
                   help="Plot from existing outputs; never run the model")
    p.add_argument("--force", action="store_true",
                   help="Re-run all simulations even if outputs exist")
    p.add_argument("--no-save", action="store_true",
                   help="Display figures interactively instead of saving to file")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        results = _collect_existing()
        if not results:
            print("No existing outputs found — run without --plot-only first.")
            return
        print(f"Found {len(results)} existing run(s).")
    else:
        results = _run_all(args.binary, force=args.force)

    if not results:
        print("No outputs to plot.")
        return

    print("Reading outputs ...")
    data_map: Dict[RunParams, dict] = {}
    for p, nc_path in results.items():
        try:
            data_map[p] = _read_run(nc_path)
        except Exception as exc:
            print(f"  Warning: could not read {nc_path.name}: {exc}")

    _print_summary(data_map)

    save = not args.no_save
    print("Plotting ...")
    _plot_biomass(data_map, save)
    _plot_elevation(data_map, save)
    _plot_accretion_summary(data_map, save)


if __name__ == "__main__":
    main()
