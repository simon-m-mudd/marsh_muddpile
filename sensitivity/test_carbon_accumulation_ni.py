#!/usr/bin/env python3
"""
test_carbon_accumulation_ni.py

Annual labile and refractory carbon accumulation, and peak belowground root
biomass, as functions of surface elevation for S. alterniflora at North Inlet.

Each simulation runs for 2 years (24 monthly steps).  Results are extracted
from year 2 only to avoid transients from the initial conditions.

  labile accumulation    = Δ(total column labile_organic mass) over year 2  [kg m⁻² yr⁻¹]
  refractory accumulation = Δ(total column refractory_organic mass) over year 2  [kg m⁻² yr⁻¹]
  peak BG root biomass   = max(belowground_biomass_kg_m2) in year 2  [kg m⁻²]

Calibration parameters match test_biomass_curve_ni.py.

Usage
-----
    python sensitivity/test_carbon_accumulation_ni.py --binary ./build/marsh_cli
    python sensitivity/test_carbon_accumulation_ni.py --plot-only
    python sensitivity/test_carbon_accumulation_ni.py --force

Output
------
    sensitivity/runs/ni_carbon_accumulation/
    sensitivity/figures/ni_carbon_accumulation.png
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import csv

import numpy as np
import yaml

_THIS_DIR  = Path(__file__).parent.resolve()
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
except ImportError as exc:
    raise ImportError("matplotlib is required: pip install matplotlib") from exc


# ---------------------------------------------------------------------------
# Calibration parameters  (mirror test_biomass_curve_ni.py)
# ---------------------------------------------------------------------------

LUE               = 1.6e-6   # gC per umol PAR
CAPACITY_KG_M2    = 0.95     # kg m⁻² aboveground capacity
NI_SSC_KG_M3      = 0.006    # kg m⁻³
NI_SLR_M_YR       = 0.003    # m yr⁻¹
NI_DISTANCE_M     = 30.0     # m from creek (GIHM distance)
NAVD88_BELOW_MSL_M = 0.108   # fitted growing-season MSL 2021 (m NAVD88)

SIGMA_H_BEST = 0.25   # best-fit vegetation_hydroperiod_sigma_fraction
SIGMA_R_BEST = 0.22   # best-fit vegetation_inundation_sigma_fraction

# ---------------------------------------------------------------------------
# Elevation sweep  (same range as biomass curve)
# ---------------------------------------------------------------------------

ELEV_MIN_M  = -0.50
ELEV_MAX_M  =  1.00
ELEV_STEP_M =  0.05

_COMPACTION_OFFSET_M = 0.0

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUTPUT_DIR = _THIS_DIR / "runs" / "ni_carbon_accumulation_v2"
FIGURE_DIR = _THIS_DIR / "figures"

# ---------------------------------------------------------------------------
# Initial column  (identical to test_biomass_curve_ni.py)
# ---------------------------------------------------------------------------

_N_LAYERS        = 20
_LAYER_THICKNESS = 0.05
_POROSITY        = 0.60
_ORGANIC_EFOLDING = 0.08

_RHO_SAND   = 2650.0
_RHO_REFRAC = 1400.0
_RHO_LABILE = 1200.0
_RHO_ROOTS  = 1100.0


def _build_initial_layers(surface_elevation_m: float) -> List[dict]:
    solid_vol = (1.0 - _POROSITY) * _LAYER_THICKNESS
    layers = []
    for i in range(_N_LAYERS):
        depth_mid  = (i + 0.5) * _LAYER_THICKNESS
        top_elev   = surface_elevation_m - i * _LAYER_THICKNESS
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
    return list(reversed(layers))


# ---------------------------------------------------------------------------
# Tidal helper
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
# YAML writer
# ---------------------------------------------------------------------------

def _run_id(target_elev_m: float) -> str:
    sign = "p" if target_elev_m >= 0 else "m"
    return "ni_ca_%s%s" % (sign, ("%.2f" % abs(target_elev_m)).replace(".", "_"))


def _write_yaml(target_elev_m: float, yaml_path: Path) -> None:
    pre_compaction_elev_m = target_elev_m + _COMPACTION_OFFSET_M

    tides = north_inlet_default_tides()
    tides.suspended_sediment_concentration_kg_m3 = NI_SSC_KG_M3
    tides.fine_sediment_concentration_kg_m3      = NI_SSC_KG_M3 * 0.25

    config = PlotConfig(
        site_name="north_inlet",
        plot_id=_run_id(target_elev_m),
        surface_elevation_m=pre_compaction_elev_m,
        distance_from_creek_m=NI_DISTANCE_M,
        tides=tides,
        met=north_inlet_default_met(),
        n_years=5,
        sea_level_rise_m_yr=NI_SLR_M_YR,
        output_dir=str(OUTPUT_DIR),
    )
    config.initial_salinity_ppt = tides.creek_salinity_ppt

    parameters = dict(_DEFAULT_PARAMETERS)
    parameters.update(_tidal_params(tides))
    parameters["deposition_distance_from_edge_m"]      = NI_DISTANCE_M
    parameters.setdefault("deposition_basin_length_m",    50.0)
    parameters.setdefault("deposition_length_scale_beta", 3.0)
    parameters["vegetation_lue_gC_per_umol"]            = LUE
    parameters["vegetation_aboveground_capacity_kg_m2"] = CAPACITY_KG_M2
    parameters["vegetation_belowground_capacity_kg_m2"] = CAPACITY_KG_M2 * 2.0
    parameters["vegetation_hydroperiod_sigma_fraction"] = SIGMA_H_BEST
    parameters["vegetation_inundation_sigma_fraction"]  = SIGMA_R_BEST

    initial_layers = _build_initial_layers(pre_compaction_elev_m)

    doc = {
        "simulation": {
            "start_year": 0,
            "end_year": 5,
            "dt_days": 365.0,
            "water_level_model_name":         "composite_water_level",
            "salinity_model_name":            "distance_flushing_salinity",
            "evapotranspiration_model_name":  "simple_canopy_et",
            "vegetation_model_name":          "marsh_gpp_biomass",
            "deposition_model_name":          "edge_distance_deposition",
            "root_allocation_model_name":     "exponential_root_allocation",
            "decay_model_name":               "marsh_decay",
            "compaction_model_name":          "mixing_compaction",
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
            "root_zone_salinity_ppt":    config.initial_salinity_ppt,
            "aboveground_biomass_kg_m2": 0.01,
            "belowground_biomass_kg_m2": 0.01,
            "lai": round(0.01 * parameters["vegetation_lai_per_kg_m2"], 4),
            "litter_kg_m2": 0.0,
        },
        "output": {
            "file": str(yaml_path).replace(".yaml", ".nc"),
            "write_time_series": True,
            "write_column_snapshots": False,
        },
    }

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as fh:
        yaml.dump(doc, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Run management
# ---------------------------------------------------------------------------

def _target_elevations() -> List[float]:
    n = round((ELEV_MAX_M - ELEV_MIN_M) / ELEV_STEP_M) + 1
    return [round(ELEV_MIN_M + i * ELEV_STEP_M, 4) for i in range(n)]


def _run_all(cli_binary: str, force: bool) -> Dict[float, Path]:
    elevations = _target_elevations()
    n = len(elevations)
    print("Running %d elevation points ..." % n)
    results: Dict[float, Path] = {}
    for i, elev in enumerate(elevations, 1):
        rid       = _run_id(elev)
        yaml_path = OUTPUT_DIR / ("%s.yaml" % rid)
        nc_path   = OUTPUT_DIR / ("%s.nc"   % rid)
        if not force and nc_path.exists():
            print("  [%2d/%d] %+.2f m: cached." % (i, n, elev))
        else:
            print("  [%2d/%d] %+.2f m" % (i, n, elev))
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            _write_yaml(elev, yaml_path)
            try:
                run_model(str(yaml_path), cli_binary=cli_binary, silent=True)
            except RuntimeError as exc:
                print("    ERROR: %s" % exc)
                continue
        if nc_path.exists():
            results[elev] = nc_path
    return results


def _collect_existing() -> Dict[float, Path]:
    results: Dict[float, Path] = {}
    for elev in _target_elevations():
        nc_path = OUTPUT_DIR / ("%s.nc" % _run_id(elev))
        if nc_path.exists():
            results[elev] = nc_path
    return results


# ---------------------------------------------------------------------------
# NetCDF reader
# ---------------------------------------------------------------------------

def _material_index(mat_names: List[str], candidates: List[str]) -> int:
    for c in candidates:
        if c in mat_names:
            return mat_names.index(c)
    return -1


_DT_DAYS = 365.25 / 12.0   # monthly timestep length

# Fractions of BG mortality that enter each organic pool (must match yaml_writer defaults)
_LABILE_FRACTION    = 0.34
_REFRACTORY_FRACTION = 0.66


def _read_run(nc_path: Path) -> Optional[dict]:
    """
    Return year-2 metrics or None on failure.

    Returns a dict:
      elev_m            : mean surface elevation in year 2 (m MSL)
      bg_mortality_yr   : total BG mortality flux in year 2  (kg m⁻² yr⁻¹)
                          = sum(belowground_mortality_kg_m2_d * dt) over all yr-2 steps
      labile_input_yr   : bg_mortality_yr * _LABILE_FRACTION
      refractory_input_yr: bg_mortality_yr * _REFRACTORY_FRACTION
      peak_bg_roots     : peak belowground biomass in year 2  (kg m⁻²)
      peak_abg          : peak aboveground biomass in year 2  (kg m⁻²)
      inund_mean        : mean inundation fraction in year 2
    """
    with nc4.Dataset(str(nc_path), "r") as ds:
        t       = np.array(ds["model_time_days"][:],               dtype=np.float64)
        elev    = np.array(ds["surface_elevation"][:],             dtype=np.float64)
        abg     = np.array(ds["aboveground_biomass_kg_m2"][:],     dtype=np.float64)
        bg      = np.array(ds["belowground_biomass_kg_m2"][:],     dtype=np.float64)
        inund   = np.array(ds["inundation_fraction"][:],           dtype=np.float64)
        bg_mort = np.array(ds["belowground_mortality_kg_m2_d"][:], dtype=np.float64)

    yr5 = t > 4 * 365.25
    if yr5.sum() < 2:
        return None

    # Annual BG mortality flux: rate (kg m⁻² d⁻¹) × timestep length (d), summed over yr 5
    bg_mortality_yr = float(np.sum(bg_mort[yr5] * _DT_DAYS))

    return {
        "elev_m":              float(np.mean(elev[yr5])),
        "bg_mortality_yr":     bg_mortality_yr,
        "labile_input_yr":     bg_mortality_yr * _LABILE_FRACTION,
        "refractory_input_yr": bg_mortality_yr * _REFRACTORY_FRACTION,
        "peak_bg_roots":       float(np.max(bg[yr5])),
        "peak_abg":            float(np.max(abg[yr5])),
        "inund_mean":          float(np.mean(inund[yr5])),
    }


# ---------------------------------------------------------------------------
# Reference data helpers  (mirrored from test_biomass_curve_ni.py)
# ---------------------------------------------------------------------------

def _load_morris_data() -> tuple:
    """Return (elev_cm_navd88, biomass_kg_m2, se_kg_m2) for North Inlet."""
    csv_path = _THIS_DIR / "morris2013_biomass_data.csv"
    elev_cm, bio_g, se_g = [], [], []
    with open(csv_path, newline="") as fh:
        data_lines = (line for line in fh if not line.startswith("#"))
        for row in csv.DictReader(data_lines):
            if row["site"] == "north_inlet":
                elev_cm.append(float(row["elevation_cm_navd88"]))
                bio_g.append(float(row["biomass_g_m2"]))
                se_g.append(float(row["biomass_se_g_m2"]))
    return (np.array(elev_cm),
            np.array(bio_g) / 1000.0,
            np.array(se_g)  / 1000.0)


def _miller_parabola(elev_cm_navd88: np.ndarray) -> np.ndarray:
    """Miller et al. (2019) parabola: B [kg m⁻²], E in cm NAVD88."""
    B_g = 14.8 * elev_cm_navd88 - 0.157 * elev_cm_navd88 ** 2 + 598.0
    return np.maximum(0.0, B_g) / 1000.0


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(data: List[dict], tides, save: bool) -> None:
    elev      = np.array([d["elev_m"]              for d in data])
    bg_mort   = np.array([d["bg_mortality_yr"]      for d in data])
    labile    = np.array([d["labile_input_yr"]      for d in data])
    refract   = np.array([d["refractory_input_yr"]  for d in data])
    bg_roots  = np.array([d["peak_bg_roots"]        for d in data])
    peak_abg  = np.array([d["peak_abg"]             for d in data])

    elev_navd = elev * 100.0   # cm NAVD88 (model outputs in NAVD88)

    mhw_m  = tides.mean_high_tide_m          # m NAVD88
    mhw_cm = mhw_m * 100.0
    msl_cm = tides.mean_sea_level_m * 100.0

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    def _datum_decor(ax):
        ax.axvspan(msl_cm, mhw_cm, color="lightblue", alpha=0.18, zorder=0)
        ax.axvline(msl_cm, color="#4393c3", lw=1.2, ls="--", zorder=1,
                   label="MSL")
        ax.axvline(mhw_cm, color="#2166ac", lw=1.2, ls="--", zorder=1,
                   label="MHW")
        ax.grid(True, linewidth=0.3, alpha=0.45)
        ax.tick_params(labelsize=9)

    # ── panel 1: BG mortality flux → labile + refractory inputs ──────────────
    ax = axes[0]
    _datum_decor(ax)
    # Stacked fill: refractory at bottom, labile on top
    ax.fill_between(elev_navd, 0, refract,
                    color="#8b4513", alpha=0.45, label="Refractory input (10%%)")
    ax.fill_between(elev_navd, refract, refract + labile,
                    color="#1f77b4", alpha=0.45, label="Labile input (90%%)")
    ax.plot(elev_navd, bg_mort, color="k", lw=2.0, marker="o", ms=4,
            label="Total BG mortality flux")
    ax.set_ylabel("BG mortality → organic\n(kg m⁻² yr⁻¹)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(
        "Belowground organic matter input and biomass vs elevation\n"
        "S. alterniflora, NI fitted harmonics, %d m from creek  —  year 5 of 5-year run" % NI_DISTANCE_M,
        fontsize=10,
    )

    # ── panel 2: peak belowground root biomass ───────────────────────────────
    ax = axes[1]
    _datum_decor(ax)
    ax.plot(elev_navd, bg_roots, color="#2ca02c", lw=2.0, marker="o", ms=4,
            label="Peak BG live roots")
    ax.set_ylabel("Peak belowground\nbiomass (kg m⁻²)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")

    # ── panel 3: peak aboveground biomass + Morris 2013 data ─────────────────
    ax = axes[2]
    _datum_decor(ax)

    ni_elev_cm, ni_bio, ni_se = _load_morris_data()
    z_ref = np.linspace(elev_navd.min() - 5, elev_navd.max() + 5, 400)
    ax.plot(z_ref, _miller_parabola(z_ref),
            color="#d73027", lw=1.6, ls="--", zorder=3,
            label="Miller et al. (2019) parabola")
    ax.errorbar(ni_elev_cm, ni_bio, yerr=ni_se,
                fmt="o", color="#d73027", ms=6, lw=1.2, capsize=3, capthick=1.2,
                zorder=5, label="Morris et al. (2013) NI data")
    ax.plot(elev_navd, peak_abg, color="#1a9641", lw=2.0, marker="s", ms=5,
            markerfacecolor="white", markeredgecolor="#1a9641", markeredgewidth=1.5,
            zorder=4, label="Model peak AG (year 2)")
    ax.set_ylabel("Peak aboveground\nbiomass (kg m⁻²)", fontsize=10)
    ax.set_xlabel("Surface elevation (cm NAVD88)", fontsize=10)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc="upper left")

    # Secondary x-axis (m above MSL) on top panel
    ax2 = axes[0].twiny()
    x0 = (elev_navd.min() - 5) / 100.0 - tides.mean_sea_level_m
    x1 = (elev_navd.max() + 5) / 100.0 - tides.mean_sea_level_m
    ax2.set_xlim(x0, x1)
    ax2.set_xlabel("Elevation (m above MSL)", fontsize=9)

    plt.tight_layout()

    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        out = FIGURE_DIR / "ni_carbon_accumulation.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print("  Saved: %s" % out)
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(data: List[dict]) -> None:
    print()
    hdr = "%8s  %10s  %10s  %10s  %8s" % (
        "elev(cm)", "BG_mort_yr", "labile_in", "refract_in", "peak_BG")
    print(hdr)
    print("-" * len(hdr))
    for d in data:
        print("%8.1f  %10.5f  %10.5f  %10.5f  %8.4f" % (
            d["elev_m"] * 100,
            d["bg_mortality_yr"],
            d["labile_input_yr"],
            d["refractory_input_yr"],
            d["peak_bg_roots"],
        ))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--binary",    default="marsh_cli")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--plot-only", action="store_true")
    g.add_argument("--force",     action="store_true")
    p.add_argument("--no-save",   action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.plot_only:
        results = _collect_existing()
        if not results:
            print("No existing outputs — run without --plot-only first.")
            return
        print("Found %d existing outputs." % len(results))
    else:
        results = _run_all(args.binary, force=args.force)

    if not results:
        print("No outputs to plot.")
        return

    print("Reading outputs ...")
    data = []
    for elev in sorted(results):
        d = _read_run(results[elev])
        if d is not None:
            data.append(d)
        else:
            print("  Warning: could not read %s" % results[elev].name)

    if not data:
        print("No valid outputs.")
        return

    _print_summary(data)

    tides = north_inlet_default_tides()
    print("Plotting ...")
    _plot(data, tides, save=not args.no_save)


if __name__ == "__main__":
    main()
