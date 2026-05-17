#!/usr/bin/env python3
"""
ni_carbon_profile_sensitivity.py

Sensitivity runs to tune the marsh model to reproduce organic matter (OM)
depth profiles at North Inlet, South Carolina.

Two scenarios
-------------
A) Stevens OL High Marsh (target: ~5 % OM near surface, ~2 % OM deep)
   SSC = 28 mg/L (0.028 kg m-3), elevation ~ 0.25 m above MSL, 50-yr run.

B) Morris tube experiment (target: ~20 % OM near surface, ~8 % at 20 cm depth)
   Fresh inorganic substrate, SSC ~ 0 kg m-3, 5-yr run.

Parameters varied
-----------------
  root_allocation_to_refractory_fraction  [REFRAC_GRID]
  vegetation_belowground_mortality_base_per_day  [MORT_GRID]

All OM fractions are LOI-equivalent (fraction organic matter by mass).

Usage
-----
    python sensitivity/ni_carbon_profile_sensitivity.py --binary ./build/marsh_cli
    python sensitivity/ni_carbon_profile_sensitivity.py --plot-only
    python sensitivity/ni_carbon_profile_sensitivity.py --force
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError("matplotlib is required: pip install matplotlib") from exc


# ---------------------------------------------------------------------------
# Fixed physical settings
# ---------------------------------------------------------------------------

LUE              = 1.6e-6     # gC per umol PAR
CAPACITY_KG_M2   = 0.95       # aboveground capacity (kg m-2)
NI_SLR_M_YR      = 0.003      # m yr-1
NI_DISTANCE_M    = 20.0       # m from creek

# Scenario A: OL High Marsh — scan multiple elevations
# CB_H (Stevens 2024, North Inlet) is at 0.8 m in CCN data (datum uncertain;
# ~0.65 m above MSL if NAVD88, or 0.80 m if already MSL-referenced).
# MHW at NI = 0.762 m MSL; high S. alterniflora marsh ~ 0.55–0.80 m MSL.
OL_HM_ELEV_GRID  = [0.65, 0.75]  # m above MSL — CB_H range
OL_HM_SSC        = 0.028      # kg m-3 (Gardner et al. 1989, 28 mg/L)
OL_HM_YEARS      = 50

# Scenario B: Morris tube experiment
# Morris et al. used a mid-intertidal elevation with good biomass.
# Fresh substrate = thin inorganic column (25 cm, 5 layers).
MORRIS_ELEV_M    = 0.35       # m above MSL (peak biomass zone)
MORRIS_SSC       = 0.0        # kg m-3 (fresh substrate, no sediment input)
MORRIS_YEARS     = 5

# Target OM fractions
TARGET_STEVENS_SURFACE = 0.05   # 5 % near surface
TARGET_STEVENS_DEEP    = 0.02   # 2 % deep (refractory floor)
TARGET_MORRIS_SURFACE  = 0.20   # 20 % near top
TARGET_MORRIS_20CM     = 0.08   # 8 % at 20 cm depth

# Compaction offset: pre-compaction elevation = target + offset
_COMPACTION_OFFSET_M = 0.36

# ---------------------------------------------------------------------------
# Fine-silt-only sediment for North Inlet
# ---------------------------------------------------------------------------
# North Inlet carries fine particles.  Gardner et al. (1989) report 28 mg/L
# SSC.  A previous study found ~4 mm/yr settling at 30 g/L with phi=0.50,
# rho=2600 kg/m3 → ~5.2 kg/m2/yr mass flux.  NI has finer particles, so
# we expect a lower effective settling velocity.  Back-calculating from the
# reference gives Vs_eff ≈ 1.5e-5 m/s for fine silt in a turbulent estuary
# (≈13 × below Stokes for 15 µm).
#
# We deposit ALL SSC as fine_silt (suspended_sediment = 0, fine_sediment =
# 0.028 kg/m3).  The fine_silt material settling velocity is overridden from
# its default (1.5e-4 m/s) to the tuned effective value below.

NI_FINE_SILT_VS  = 1.5e-5    # m/s — effective settling, North Inlet fine silt

def _ni_materials() -> list:
    """Return _DEFAULT_MATERIALS with NI-tuned settling and k_0=0 for refractory."""
    mats = copy.deepcopy(_DEFAULT_MATERIALS)
    for mat in mats:
        if mat["name"] == "fine_silt":
            mat["settling"]["settling_velocity"] = NI_FINE_SILT_VS
        if mat["name"] == "refractory_organic":
            mat["decay"]["k_0"] = 0.0   # refractory does not decay
    return mats

# ---------------------------------------------------------------------------
# Parameter grids
# ---------------------------------------------------------------------------

REFRAC_GRID   = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
MORT_GRID     = [1.0e-4, 2.74e-4, 5.0e-4, 8.0e-4, 1.5e-3]   # /day
EFOLDING_GRID = [0.03, 0.05, 0.10]   # m — root e-folding depth

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------

OUT_A    = _THIS_DIR / "runs" / "ni_carbon_ol_hm_v5"
OUT_B    = _THIS_DIR / "runs" / "ni_carbon_morris_tube_v4"
FIG_DIR  = _THIS_DIR / "figures"


# ---------------------------------------------------------------------------
# Initial column
# ---------------------------------------------------------------------------

_N_LAYERS        = 20
_LAYER_THICKNESS = 0.05   # m
_POROSITY        = 0.60
_RHO_SAND        = 2650.0
_RHO_REFRAC      = 1400.0
_RHO_LABILE      = 1200.0
_RHO_ROOTS       = 1100.0

# Morris tube: thin mineral column (5 layers = 25 cm)
_MORRIS_N_LAYERS = 5


def _build_layers_mineral(surface_elevation_m: float,
                           n_layers: int = _N_LAYERS) -> List[dict]:
    """Pure inorganic starting column.

    Use n_layers=_MORRIS_N_LAYERS (5 layers, 25 cm) for the Morris tube
    scenario so root-derived OM is not swamped by a massive mineral column.
    """
    layers = []
    for i in range(n_layers):
        top_elev  = surface_elevation_m - i * _LAYER_THICKNESS
        solid_vol = (1.0 - _POROSITY) * _LAYER_THICKNESS
        layers.append({
            "top_elevation_m": round(top_elev, 4),
            "thickness_m": _LAYER_THICKNESS,
            "porosity": _POROSITY,
            "age_days": 0.0,
            "mass_kg_m2": {
                "sand":               round(_RHO_SAND * solid_vol, 4),
                "refractory_organic": 0.0,
                "labile_organic":     0.0,
                "roots":              0.0,
            },
        })
    return list(reversed(layers))


def _build_layers_organic(surface_elevation_m: float) -> List[dict]:
    """Organic-bearing column (OL HM scenario — typical marsh initial state)."""
    efolding = 0.075   # m
    layers = []
    for i in range(_N_LAYERS):
        depth_mid = (i + 0.5) * _LAYER_THICKNESS
        top_elev  = surface_elevation_m - i * _LAYER_THICKNESS
        solid_vol = (1.0 - _POROSITY) * _LAYER_THICKNESS
        refrac_frac = 0.035
        exp_fac     = math.exp(-depth_mid / efolding)
        labile_frac = 0.05 * exp_fac
        root_frac   = 0.03 * exp_fac
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
# Tidal parameters helper
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
# Run ID helpers
# ---------------------------------------------------------------------------

def _run_id_a(refrac: float, mort: float, elev: float, efolding: float = 0.10) -> str:
    sign = "p" if elev >= 0 else "m"
    return (f"olhm_e{sign}{abs(elev):.2f}_rf{refrac:.2f}_mo{mort:.2e}_ef{efolding:.2f}"
            .replace(".", "_").replace("-", "m"))


def _run_id_b(refrac: float, mort: float) -> str:
    return f"morris_rf{refrac:.2f}_mo{mort:.2e}".replace(".", "_").replace("-", "m")


# ---------------------------------------------------------------------------
# Snapshot times
# ---------------------------------------------------------------------------

def _snap_times(n_years: int) -> List[float]:
    """Snapshot at each year boundary."""
    return [float(y) * 365.25 for y in range(1, n_years + 1)]


# ---------------------------------------------------------------------------
# YAML builder
# ---------------------------------------------------------------------------

def _build_doc(
    elev_m: float,
    ssc: float,
    n_years: int,
    refrac_frac: float,
    mort_per_day: float,
    nc_path: Path,
    mineral_start: bool = False,
    efolding_m: float = 0.10,
) -> dict:
    pre_elev = elev_m + _COMPACTION_OFFSET_M

    tides = north_inlet_default_tides()
    # Route all SSC as fine_silt (NI carries fine particles; silt = 0)
    tides.suspended_sediment_concentration_kg_m3 = 0.0
    tides.fine_sediment_concentration_kg_m3      = ssc

    config = PlotConfig(
        site_name="north_inlet",
        plot_id=nc_path.stem,
        surface_elevation_m=pre_elev,
        distance_from_creek_m=NI_DISTANCE_M,
        tides=tides,
        met=north_inlet_default_met(),
        n_years=n_years,
        sea_level_rise_m_yr=NI_SLR_M_YR,
        output_dir=str(nc_path.parent),
    )
    config.initial_salinity_ppt = tides.creek_salinity_ppt

    parameters = dict(_DEFAULT_PARAMETERS)
    parameters.update(_tidal_params(tides))
    parameters["deposition_distance_from_edge_m"] = NI_DISTANCE_M
    parameters.setdefault("deposition_basin_length_m", 50.0)
    parameters.setdefault("deposition_length_scale_beta", 3.0)

    parameters["vegetation_lue_gC_per_umol"]             = LUE
    parameters["vegetation_aboveground_capacity_kg_m2"]  = CAPACITY_KG_M2
    parameters["vegetation_belowground_capacity_kg_m2"]  = CAPACITY_KG_M2 * 2.0

    parameters["root_allocation_to_refractory_fraction"] = float(refrac_frac)
    parameters["root_allocation_to_labile_fraction"]     = float(1.0 - refrac_frac)
    parameters["vegetation_belowground_mortality_base_per_day"] = float(mort_per_day)
    parameters["root_efolding_depth_m"]                  = float(efolding_m)

    if mineral_start:
        # Morris tube: thin column so roots can dominate OM fraction
        layers = _build_layers_mineral(pre_elev, n_layers=_MORRIS_N_LAYERS)
    else:
        layers = _build_layers_organic(pre_elev)

    # Snapshot at end of each year
    snap_times = _snap_times(n_years)

    return {
        "simulation": {
            "start_year":                      0,
            "end_year":                        n_years,
            "dt_days":                         30.4375,
            "water_level_model_name":          "composite_water_level",
            "salinity_model_name":             "distance_flushing_salinity",
            "evapotranspiration_model_name":   "simple_canopy_et",
            "vegetation_model_name":           "marsh_gpp_biomass",
            "deposition_model_name":           "edge_distance_deposition",
            "root_allocation_model_name":      "exponential_root_allocation",
            "decay_model_name":                "marsh_decay",
            "compaction_model_name":           "mixing_compaction",
            "porewater_chemistry_model_name":  "none",
            "methane_model_name":              "none",
        },
        "site": {
            "distance_from_creek_m":  NI_DISTANCE_M,
            "creek_bank_elevation_m": tides.mean_sea_level_m,
            "local_tidal_offset_m":   0.0,
        },
        "parameters": parameters,
        "materials": _ni_materials(),
        "forcing": {"generated": build_generated_forcing_block(config)},
        "initial_state": {"layers": layers},
        "initial_ecohydrology_state": {
            "root_zone_salinity_ppt":    tides.creek_salinity_ppt,
            "aboveground_biomass_kg_m2": 0.3,
            "belowground_biomass_kg_m2": 0.3,
            "lai": round(0.3 * parameters["vegetation_lai_per_kg_m2"], 4),
            "litter_kg_m2": 0.0,
        },
        "output": {
            "file":                   str(nc_path),
            "write_time_series":      True,
            "write_column_snapshots": True,
            "snapshot_times_days":    snap_times,
        },
    }


# ---------------------------------------------------------------------------
# Run one simulation
# ---------------------------------------------------------------------------

def _run_one(
    run_id: str,
    out_dir: Path,
    elev_m: float,
    ssc: float,
    n_years: int,
    refrac: float,
    mort: float,
    cli_binary: str,
    force: bool,
    mineral_start: bool = False,
    efolding_m: float = 0.10,
) -> Optional[Path]:
    yaml_path = out_dir / f"{run_id}.yaml"
    nc_path   = out_dir / f"{run_id}.nc"

    if not force and nc_path.exists():
        return nc_path

    out_dir.mkdir(parents=True, exist_ok=True)
    doc = _build_doc(elev_m, ssc, n_years, refrac, mort, nc_path, mineral_start, efolding_m)
    with open(yaml_path, "w") as fh:
        yaml.dump(doc, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)

    try:
        run_model(str(yaml_path), cli_binary=cli_binary, silent=True)
    except RuntimeError as exc:
        print(f"    ERROR {run_id}: {exc}")
        return None

    return nc_path if nc_path.exists() else None


# ---------------------------------------------------------------------------
# Read OM profile from final snapshot
# ---------------------------------------------------------------------------

def _read_om_profile(nc_path: Path, snap_idx: int = -1
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (depth_cm, om_fraction, mineral_fraction) from one snapshot.

    depth_cm: depth below surface (midpoint of each layer), 0 = surface.
    om_fraction: (refractory + labile) / total_mass  (fraction OM, LOI proxy).
    """
    with nc4.Dataset(str(nc_path), "r") as ds:
        mat_names = [b"".join(r.compressed()).decode()
                     for r in ds.variables["material_name"][:]]
        snap_n    = int(ds.variables["snapshot_n_layers"][snap_idx])
        mass      = np.array(ds.variables["layer_mass"][snap_idx, :snap_n, :])  # (n, m)
        thick     = np.array(ds.variables["layer_thickness"][snap_idx, :snap_n])
        top_elev  = np.array(ds.variables["layer_top_elevation"][snap_idx, :snap_n])

    # Layers sorted deepest-first in file; sort top-down
    order     = np.argsort(top_elev)[::-1]
    mass      = mass[order]
    thick     = thick[order]
    top_elev  = top_elev[order]

    surface_elev = top_elev[0]
    depth_cm = ((surface_elev - (top_elev - thick / 2.0)) * 100.0)

    i_refrac = mat_names.index("refractory_organic")
    i_labile = mat_names.index("labile_organic")
    i_roots  = mat_names.index("roots")
    total    = mass.sum(axis=1)
    organic  = mass[:, i_refrac] + mass[:, i_labile] + mass[:, i_roots]
    om_frac  = np.where(total > 0, organic / total, 0.0)

    return depth_cm, om_frac, total


def _read_annual_fluxes(nc_path: Path) -> dict:
    """Return dict of annual-mean rates over the whole run."""
    with nc4.Dataset(str(nc_path), "r") as ds:
        t         = np.array(ds.variables["model_time_days"][:])
        bg_growth = np.array(ds.variables["belowground_growth_kg_m2_d"][:])
        bg_mort   = np.array(ds.variables["belowground_mortality_kg_m2_d"][:])
        mat_mass  = np.array(ds.variables["total_mass_by_material"][:])   # (t, m)
        mat_names = [b"".join(r.compressed()).decode()
                     for r in ds.variables["material_name"][:]]

    dt = float(t[1] - t[0])   # days per step
    n  = len(t)
    steps_per_yr = max(1, round(365.25 / dt))

    i_ref = mat_names.index("refractory_organic")
    i_lab = mat_names.index("labile_organic")
    i_slt = mat_names.index("silt")
    i_fsl = mat_names.index("fine_silt")

    years, bg_g_yr, bg_m_yr, ref_yr, lab_yr, min_yr = [], [], [], [], [], []
    for yr in range(n // steps_per_yr):
        i0, i1 = yr * steps_per_yr, (yr + 1) * steps_per_yr
        years.append(yr + 1)
        bg_g_yr.append(float(bg_growth[i0:i1].mean()) * 365.25)
        bg_m_yr.append(float(bg_mort[i0:i1].mean()) * 365.25)

        # Refractory / labile production estimated from mass change + loss
        if yr == 0:
            ref_yr.append(float(mat_mass[i1 - 1, i_ref]) - float(mat_mass[0, i_ref]))
            lab_yr.append(float(mat_mass[i1 - 1, i_lab]) - float(mat_mass[0, i_lab]))
            min_yr.append(float(mat_mass[i1 - 1, i_slt] + mat_mass[i1 - 1, i_fsl])
                          - float(mat_mass[0, i_slt] + mat_mass[0, i_fsl]))
        else:
            prev = (yr - 1) * steps_per_yr
            ref_yr.append(float(mat_mass[i1 - 1, i_ref]) - float(mat_mass[prev, i_ref]))
            lab_yr.append(float(mat_mass[i1 - 1, i_lab]) - float(mat_mass[prev, i_lab]))
            min_yr.append(float(mat_mass[i1 - 1, i_slt] + mat_mass[i1 - 1, i_fsl])
                          - float(mat_mass[prev, i_slt] + mat_mass[prev, i_fsl]))

    return {
        "years":      np.array(years),
        "bg_growth":  np.array(bg_g_yr),
        "bg_mort":    np.array(bg_m_yr),
        "refrac_accum": np.array(ref_yr),
        "labile_accum": np.array(lab_yr),
        "mineral_accum": np.array(min_yr),
    }


# ---------------------------------------------------------------------------
# Score: penalised distance from targets
# ---------------------------------------------------------------------------

def _score_om(depth_cm, om_frac, surface_target, deep_target,
              surface_depth_cm=5.0, deep_depth_cm=30.0) -> float:
    """
    Simple L2 score: sum of squared deviations from targets at two depths.
    Lower = better.
    """
    if len(depth_cm) == 0:
        return np.inf
    # surface OM: mean of layers above surface_depth_cm
    mask_s = depth_cm <= surface_depth_cm
    mask_d = depth_cm >= deep_depth_cm
    om_s = om_frac[mask_s].mean() if mask_s.any() else 0.0
    om_d = om_frac[mask_d].mean() if mask_d.any() else 0.0
    return (om_s - surface_target) ** 2 + (om_d - deep_target) ** 2


# ---------------------------------------------------------------------------
# Run all scans
# ---------------------------------------------------------------------------

def _run_scan_elev(
    label: str,
    out_dir: Path,
    elev_grid: List[float],
    ssc: float,
    n_years: int,
    cli_binary: str,
    force: bool,
    mineral_start: bool = False,
) -> Dict[Tuple[float, float, float, float], Optional[Path]]:
    """Scan (refrac, mort, efolding) at every elevation in elev_grid."""
    combos = [(r, m, e, ef)
              for e in elev_grid
              for r in REFRAC_GRID
              for m in MORT_GRID
              for ef in EFOLDING_GRID]
    n = len(combos)
    print(f"\n{label}: {n} runs "
          f"({len(elev_grid)} elevs × {len(REFRAC_GRID)} refrac × {len(MORT_GRID)} mort "
          f"× {len(EFOLDING_GRID)} efolding) "
          f"× {n_years} yr, SSC={ssc*1000:.1f} mg/L")
    results: Dict[Tuple[float, float, float, float], Optional[Path]] = {}
    for i, (refrac, mort, elev, ef) in enumerate(combos, 1):
        rid = _run_id_a(refrac, mort, elev, ef)
        nc  = out_dir / f"{rid}.nc"
        if not force and nc.exists():
            results[(refrac, mort, elev, ef)] = nc
            continue
        print(f"  [{i:3d}/{n}] elev={elev:+.2f} rf={refrac:.2f} mo={mort:.2e} ef={ef:.2f} …",
              end=" ", flush=True)
        path = _run_one(rid, out_dir, elev, ssc, n_years, refrac, mort,
                        cli_binary, force, mineral_start, ef)
        status = "ok" if path else "FAILED"
        print(status)
        results[(refrac, mort, elev, ef)] = path
    return results


def _run_scan(
    label: str,
    out_dir: Path,
    elev_m: float,
    ssc: float,
    n_years: int,
    run_id_fn,
    cli_binary: str,
    force: bool,
    mineral_start: bool = False,
) -> Dict[Tuple[float, float], Optional[Path]]:
    """Single-elevation scan (used for Morris tube)."""
    combos = [(r, m) for r in REFRAC_GRID for m in MORT_GRID]
    n = len(combos)
    print(f"\n{label}: {n} parameter combinations × {n_years} yrs "
          f"(SSC={ssc*1000:.1f} mg/L, elev={elev_m:+.2f} m MSL)")
    results: Dict[Tuple[float, float], Optional[Path]] = {}
    for i, (refrac, mort) in enumerate(combos, 1):
        rid = run_id_fn(refrac, mort)
        nc  = out_dir / f"{rid}.nc"
        if not force and nc.exists():
            print(f"  [{i:2d}/{n}] refrac={refrac:.2f} mort={mort:.2e}: cached")
            results[(refrac, mort)] = nc
            continue
        print(f"  [{i:2d}/{n}] refrac={refrac:.2f} mort={mort:.2e} …", end=" ", flush=True)
        path = _run_one(rid, out_dir, elev_m, ssc, n_years, refrac, mort,
                        cli_binary, force, mineral_start)
        status = "ok" if path else "FAILED"
        print(status)
        results[(refrac, mort)] = path
    return results


# ---------------------------------------------------------------------------
# Diagnostic plot for a single run
# ---------------------------------------------------------------------------

def _plot_diagnostic(nc_path: Path, label: str, surface_target: float,
                     deep_target: float, out_path: Path) -> None:
    fluxes = _read_annual_fluxes(nc_path)
    depth, om, _ = _read_om_profile(nc_path, snap_idx=-1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel 1: annual fluxes
    ax = axes[0]
    ax.bar(fluxes["years"] - 0.3, fluxes["bg_growth"],  width=0.28, label="BG growth",  color="#2c7bb6", alpha=0.8)
    ax.bar(fluxes["years"],       fluxes["bg_mort"],     width=0.28, label="BG mortality", color="#d7191c", alpha=0.8)
    ax.bar(fluxes["years"] + 0.3, fluxes["mineral_accum"], width=0.28, label="Mineral accum.", color="#756bb1", alpha=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mass flux (kg m⁻² yr⁻¹)")
    ax.set_title("Annual mass fluxes")
    ax.legend(fontsize=7)

    # Panel 2: organic accumulation
    ax = axes[1]
    ax.plot(fluxes["years"], np.cumsum(np.maximum(fluxes["refrac_accum"], 0)),
            "o-", color="#1a9641", label="Cumul. refrac. (net accum.)")
    ax.plot(fluxes["years"], np.cumsum(np.maximum(fluxes["labile_accum"], 0)),
            "s--", color="#fdae61", label="Cumul. labile (net accum.)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative mass (kg m⁻²)")
    ax.set_title("Organic accumulation")
    ax.legend(fontsize=7)

    # Panel 3: final OM depth profile
    ax = axes[2]
    ax.plot(om * 100, depth, "o-", color="#1a9641", lw=1.8, ms=4)
    ax.axvline(surface_target * 100, color="#d7191c", ls="--", lw=1.2,
               label=f"Surface target {surface_target*100:.0f}%")
    ax.axvline(deep_target * 100, color="#2c7bb6", ls=":", lw=1.2,
               label=f"Deep target {deep_target*100:.0f}%")
    ax.invert_yaxis()
    ax.set_xlabel("OM fraction (%)")
    ax.set_ylabel("Depth (cm)")
    ax.set_title("Final OM profile")
    ax.legend(fontsize=7)

    fig.suptitle(label, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Heatmap: score vs (refrac, mort)
# ---------------------------------------------------------------------------

def _plot_heatmap(
    results: Dict[Tuple[float, float], Optional[Path]],
    surface_target: float,
    deep_target: float,
    title: str,
    out_path: Path,
    snap_idx: int = -1,
) -> None:
    refrac_vals = sorted({r for r, _ in results})
    mort_vals   = sorted({m for _, m in results})

    score_grid  = np.full((len(mort_vals), len(refrac_vals)), np.nan)
    om_surf_grid = np.full_like(score_grid, np.nan)
    om_deep_grid = np.full_like(score_grid, np.nan)

    best_score = np.inf
    best_params = None

    for (r, m), nc_path in results.items():
        if nc_path is None or not nc_path.exists():
            continue
        try:
            depth, om, _ = _read_om_profile(nc_path, snap_idx=snap_idx)
        except Exception:
            continue
        ri = refrac_vals.index(r)
        mi = mort_vals.index(m)
        sc = _score_om(depth, om, surface_target, deep_target)
        score_grid[mi, ri]   = sc
        om_surf_grid[mi, ri] = om[depth <= 5].mean() if (depth <= 5).any() else np.nan
        om_deep_grid[mi, ri] = om[depth >= 30].mean() if (depth >= 30).any() else np.nan
        if sc < best_score:
            best_score = sc
            best_params = (r, m)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    xtick = [f"{r:.2f}" for r in refrac_vals]
    ytick = [f"{m:.1e}" for m in mort_vals]

    def _hm(ax, data, cmap, fmt, cbar_label, title_):
        im = ax.imshow(data, aspect="auto", cmap=cmap, origin="lower")
        ax.set_xticks(range(len(refrac_vals))); ax.set_xticklabels(xtick, fontsize=7)
        ax.set_yticks(range(len(mort_vals)));   ax.set_yticklabels(ytick, fontsize=7)
        ax.set_xlabel("Refractory fraction", fontsize=8)
        ax.set_ylabel("BG mortality rate (d⁻¹)", fontsize=8)
        ax.set_title(title_, fontsize=8)
        for mi in range(len(mort_vals)):
            for ri in range(len(refrac_vals)):
                v = data[mi, ri]
                if np.isfinite(v):
                    ax.text(ri, mi, fmt.format(v), ha="center", va="center",
                            fontsize=6, color="black")
        plt.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)

    _hm(axes[0], om_surf_grid * 100, "RdYlGn", "{:.1f}",
        "Surface OM (%)", f"Surface OM (%) — target={surface_target*100:.0f}%")
    _hm(axes[1], om_deep_grid * 100, "RdYlGn", "{:.1f}",
        "Deep OM (%)", f"Deep OM (%) — target={deep_target*100:.0f}%")
    _hm(axes[2], np.sqrt(score_grid) * 100, "RdYlGn_r", "{:.2f}",
        "RMSE OM (%)", "Score (RMSE OM %)")

    if best_params is not None:
        ri = refrac_vals.index(best_params[0])
        mi = mort_vals.index(best_params[1])
        for ax in axes:
            ax.add_patch(plt.Rectangle((ri - 0.5, mi - 0.5), 1, 1,
                                        fc="none", ec="blue", lw=2.5))
        print(f"  Best: refrac={best_params[0]:.2f}, "
              f"mort={best_params[1]:.2e} /d  (score={best_score:.4f})")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return best_params


# ---------------------------------------------------------------------------
# Profile comparison: all parameter combos in one figure
# ---------------------------------------------------------------------------

def _plot_all_profiles(
    results: Dict[Tuple[float, float], Optional[Path]],
    surface_target: float,
    deep_target: float,
    title: str,
    out_path: Path,
    snap_idx: int = -1,
) -> None:
    refrac_vals = sorted({r for r, _ in results})
    mort_vals   = sorted({m for _, m in results})
    ncols = len(refrac_vals)
    nrows = len(mort_vals)

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(3 * ncols, 4 * nrows),
                              sharey=True, sharex=True)
    axes = np.array(axes).reshape(nrows, ncols)

    for mi, mort in enumerate(mort_vals):
        for ri, refrac in enumerate(refrac_vals):
            ax = axes[mi, ri]
            nc_path = results.get((refrac, mort))
            if nc_path is None or not nc_path.exists():
                ax.set_visible(False)
                continue
            try:
                depth, om, _ = _read_om_profile(nc_path, snap_idx=snap_idx)
            except Exception:
                ax.set_visible(False)
                continue
            ax.plot(om * 100, depth, "o-", lw=1.5, ms=3, color="#1a9641")
            ax.axvline(surface_target * 100, color="#d7191c", ls="--", lw=0.8, alpha=0.7)
            ax.axvline(deep_target * 100,    color="#2c7bb6", ls=":",  lw=0.8, alpha=0.7)
            om_surf = om[depth <= 5].mean() * 100 if (depth <= 5).any() else np.nan
            om_deep = om[depth >= 30].mean() * 100 if (depth >= 30).any() else np.nan
            ax.set_title(f"rf={refrac:.2f} mo={mort:.1e}\n"
                         f"surf={om_surf:.1f}% deep={om_deep:.1f}%",
                         fontsize=6)
            ax.invert_yaxis()

    for mi in range(nrows):
        axes[mi, 0].set_ylabel(f"mort={mort_vals[mi]:.1e}\nDepth (cm)", fontsize=7)
    for ri in range(ncols):
        axes[-1, ri].set_xlabel("OM (%)", fontsize=7)

    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--binary",    default="marsh_cli",
                   help="Path to marsh_cli executable")
    p.add_argument("--plot-only", action="store_true",
                   help="Skip model runs; plot from existing outputs")
    p.add_argument("--force",     action="store_true",
                   help="Re-run all simulations even if outputs exist")
    return p.parse_args()


def _plot_elevation_summary(
    results: Dict[Tuple, Optional[Path]],
    surface_target: float,
    deep_target: float,
    title: str,
    out_path: Path,
) -> None:
    """For each elevation, show best surface OM and deep OM across all parameter combos."""
    elev_vals = sorted({k[2] for k in results})
    best_surf = []
    best_deep = []
    min_score = []

    for e in elev_vals:
        surf_list, deep_list, sc_list = [], [], []
        for key, nc_path in results.items():
            ev = key[2]
            if ev != e or nc_path is None or not nc_path.exists():
                continue
            try:
                depth, om, _ = _read_om_profile(nc_path, snap_idx=-1)
            except Exception:
                continue
            sc = _score_om(depth, om, surface_target, deep_target)
            om_s = om[depth <= 5].mean() if (depth <= 5).any() else np.nan
            om_d = om[depth >= 30].mean() if (depth >= 30).any() else np.nan
            surf_list.append(om_s)
            deep_list.append(om_d)
            sc_list.append(sc)
        best_surf.append(np.nanmax(surf_list) * 100 if surf_list else np.nan)
        best_deep.append(np.nanmax(deep_list) * 100 if deep_list else np.nan)
        min_score.append(np.nanmin(sc_list) * 100 if sc_list else np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(elev_vals, best_surf, "o-", color="#1a9641", label="Best surface OM (%)")
    ax.plot(elev_vals, best_deep, "s--", color="#2c7bb6", label="Best deep OM (%)")
    ax.axhline(surface_target * 100, color="#1a9641", ls=":", lw=1.2,
               label=f"Surface target {surface_target*100:.0f}%")
    ax.axhline(deep_target * 100,    color="#2c7bb6", ls=":", lw=1.2,
               label=f"Deep target {deep_target*100:.0f}%")
    ax.set_xlabel("Surface elevation (m above MSL)", fontsize=10)
    ax.set_ylabel("Best OM fraction (%)", fontsize=10)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # annotate accretion rate
    for e, bs, bd in zip(elev_vals, best_surf, best_deep):
        ax.annotate(f"{e:.2f}m", xy=(e, bs), xytext=(0, 6),
                    textcoords="offset points", fontsize=7, ha="center")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _plot_best_heatmap_for_elev(
    results: Dict[Tuple, Optional[Path]],
    elev_m: float,
    surface_target: float,
    deep_target: float,
    title: str,
    out_path: Path,
) -> Optional[Tuple]:
    """Heatmap at a specific elevation (best efolding collapsed); return best key."""
    # Collapse over efolding: keep best score for each (refrac, mort) combo
    sub: Dict[Tuple[float, float], Optional[Path]] = {}
    sub_scores: Dict[Tuple[float, float], float] = {}
    for key, nc_path in results.items():
        ev = key[2]
        if ev != elev_m or nc_path is None or not nc_path.exists():
            continue
        r, m = key[0], key[1]
        try:
            depth, om, _ = _read_om_profile(nc_path, snap_idx=-1)
            sc = _score_om(depth, om, surface_target, deep_target)
        except Exception:
            continue
        if (r, m) not in sub_scores or sc < sub_scores[(r, m)]:
            sub_scores[(r, m)] = sc
            sub[(r, m)] = nc_path
    best = _plot_heatmap(sub, surface_target, deep_target, title, out_path)
    if best:
        return (best[0], best[1], elev_m)
    return None


def main() -> None:
    args = _parse_args()

    # ------------------------------------------------------------------
    # Scenario A: OL High Marsh — multiple elevations
    # ------------------------------------------------------------------
    print("\n=== Scenario A: OL High Marsh (elevation sweep) ===")
    res_a = _run_scan_elev(
        "OL HM Stevens target 5% surf / 2% deep",
        OUT_A, OL_HM_ELEV_GRID, OL_HM_SSC, OL_HM_YEARS,
        args.binary, args.force,
        mineral_start=False,
    )

    print("\nPlotting elevation summary …")
    _plot_elevation_summary(
        res_a,
        TARGET_STEVENS_SURFACE, TARGET_STEVENS_DEEP,
        f"OL HM: best OM vs elevation — SSC={OL_HM_SSC*1000:.0f} mg/L, {OL_HM_YEARS} yr\n"
        f"Targets: surface={TARGET_STEVENS_SURFACE*100:.0f}%, deep={TARGET_STEVENS_DEEP*100:.0f}%",
        FIG_DIR / "ni_olhm_elevation_summary.png",
    )

    # Heatmaps at each elevation
    best_by_elev = {}
    for e in OL_HM_ELEV_GRID:
        print(f"\nHeatmap at elev={e:+.2f} m …")
        b = _plot_best_heatmap_for_elev(
            res_a, e,
            TARGET_STEVENS_SURFACE, TARGET_STEVENS_DEEP,
            f"OL HM — elev={e:+.2f} m MSL\n"
            f"SSC={OL_HM_SSC*1000:.0f} mg/L, {OL_HM_YEARS} yr — "
            f"target: surf={TARGET_STEVENS_SURFACE*100:.0f}% deep={TARGET_STEVENS_DEEP*100:.0f}%",
            FIG_DIR / f"ni_olhm_heatmap_e{e:.2f}.png".replace(".", "p"),
        )
        if b:
            best_by_elev[e] = b

    # Profile grid for the elevation with best overall score
    all_scores = []
    for key, nc_path in res_a.items():
        if nc_path is None or not nc_path.exists():
            continue
        try:
            depth, om, _ = _read_om_profile(nc_path, snap_idx=-1)
            sc = _score_om(depth, om, TARGET_STEVENS_SURFACE, TARGET_STEVENS_DEEP)
            all_scores.append((sc,) + key + (nc_path,))
        except Exception:
            continue
    if all_scores:
        all_scores.sort()
        best_entry = all_scores[0]
        best_sc = best_entry[0]
        best_r, best_m, best_e, best_ef = best_entry[1], best_entry[2], best_entry[3], best_entry[4]
        best_nc = best_entry[5]
        print(f"\nOverall best: elev={best_e:+.2f} m, refrac={best_r:.2f}, "
              f"mort={best_m:.2e}/d, efolding={best_ef:.2f}m (score={best_sc:.4f})")
        _plot_diagnostic(
            best_nc,
            f"OL HM overall best: elev={best_e:+.2f}m, refrac={best_r:.2f}, "
            f"mort={best_m:.2e}/d, efolding={best_ef:.2f}m",
            TARGET_STEVENS_SURFACE, TARGET_STEVENS_DEEP,
            FIG_DIR / "ni_olhm_diagnostic_overall_best.png",
        )
        # All profiles at best elevation and efolding — vary refrac/mort
        sub_best_e = {(r, m): nc for (r, m, e, ef), nc in res_a.items()
                      if e == best_e and ef == best_ef}
        _plot_all_profiles(
            sub_best_e,
            TARGET_STEVENS_SURFACE, TARGET_STEVENS_DEEP,
            f"OL HM profiles — elev={best_e:+.2f} m MSL, efolding={best_ef:.2f}m, "
            f"SSC={OL_HM_SSC*1000:.0f} mg/L",
            FIG_DIR / "ni_olhm_profiles_best_elev.png",
        )

    # ------------------------------------------------------------------
    # Scenario B: Morris tube experiment (fresh substrate, no sediment)
    # ------------------------------------------------------------------
    print("\n=== Scenario B: Morris tube experiment ===")
    res_b = _run_scan(
        "Morris tube (target: 20% surface / 8% at 20 cm, 5 yr)",
        OUT_B, MORRIS_ELEV_M, MORRIS_SSC, MORRIS_YEARS,
        _run_id_b, args.binary, args.force,
        mineral_start=True,
    )

    print("\nPlotting Scenario B heatmap …")
    best_b = _plot_heatmap(
        res_b,
        TARGET_MORRIS_SURFACE, TARGET_MORRIS_20CM,
        f"Morris tube scan — target: surface={TARGET_MORRIS_SURFACE*100:.0f}% "
        f"deep={TARGET_MORRIS_20CM*100:.0f}%\nSSC≈0, {MORRIS_YEARS} yr, "
        f"25 cm fresh mineral substrate, elev={MORRIS_ELEV_M:+.2f} m MSL",
        FIG_DIR / "ni_morris_tube_heatmap_v3.png",
    )

    print("\nPlotting Scenario B all profiles …")
    _plot_all_profiles(
        res_b,
        TARGET_MORRIS_SURFACE, TARGET_MORRIS_20CM,
        f"Morris tube OM profiles — SSC=0, {MORRIS_YEARS} yr, 25 cm fresh substrate",
        FIG_DIR / "ni_morris_tube_profiles_v3.png",
    )

    if best_b:
        best_b_path = res_b.get(best_b)
        if best_b_path and best_b_path.exists():
            print(f"\nDiagnostic for best-fit Scenario B params ({best_b}):")
            _plot_diagnostic(
                best_b_path,
                f"Morris tube — best fit (refrac={best_b[0]:.2f}, mort={best_b[1]:.2e}/d)\n"
                f"25 cm fresh mineral substrate, SSC=0, {MORRIS_YEARS} yr",
                TARGET_MORRIS_SURFACE, TARGET_MORRIS_20CM,
                FIG_DIR / "ni_morris_tube_diagnostic_best_v3.png",
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
