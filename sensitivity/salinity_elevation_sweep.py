#!/usr/bin/env python3
"""
salinity_elevation_sweep.py

1-year simulations of root-zone salinity vs surface elevation at North Inlet.
Two creek distances (near: 2 m, far: 30 m) are compared across the full
intertidal elevation range covered by Morris et al. (2013).

The 80 ppt model cap (salinity_max_ppt) is configurable but not binding at NI
where observed porewater salinity peaks around 45-50 ppt.

Observational overlays
----------------------
  NI (edi.136.11): porewater salinity at GI HM (~0.25 m MSL) and LM (~0 m MSL)
                   Summer (JJA) and winter (DJF) means ± 1 std
  PIE (knb-lter-pie.625.1): surface-layer (0-5 cm) porewater salinity vs NAVD88
                   elevation — note different datum from NI model output

Usage
-----
    python sensitivity/salinity_elevation_sweep.py --binary build/marsh_cli
    python sensitivity/salinity_elevation_sweep.py --plot-only
    python sensitivity/salinity_elevation_sweep.py --force
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

_THIS_DIR  = Path(__file__).parent.resolve()
_REPO_DIR  = _THIS_DIR.parent
_CALIB_DIR = _REPO_DIR / "calibration"
_DATA_DIR  = _REPO_DIR / "lter_data"
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
    import matplotlib.patches as mpatches
except ImportError as exc:
    raise ImportError("matplotlib is required: pip install matplotlib") from exc


# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------

ELEV_MIN_M  = -0.50   # -35 cm NAVD88 (leftmost Morris 2013 NI point)
ELEV_MAX_M  =  0.95   # 110 cm NAVD88 (rightmost Morris 2013 NI point)
ELEV_STEP_M =  0.05

DISTANCES_M = [2, 30]          # near creek, far from creek

N_YEARS        = 1
NI_SSC_KG_M3   = 0.006         # low-turbidity NI conditions
NI_SLR_M_YR    = 0.003
NAVD88_BELOW_MSL_M = 0.15      # MSL ≈ NAVD88 + 0.15 m at North Inlet, SC

_COMPACTION_OFFSET_M = 0.36    # pre-compaction elevation offset for 1-m col at porosity 0.60

OUTPUT_DIR = _THIS_DIR / "runs" / "salinity_elev_sweep"
FIGURE_DIR = _THIS_DIR / "figures"

_DISTANCE_COLOURS = {2: "#d73027", 30: "#4575b4"}
_DISTANCE_LABELS  = {2: "2 m (near creek)", 30: "30 m from creek"}

# ---------------------------------------------------------------------------
# Observational data loaders
# ---------------------------------------------------------------------------

_NI_POREWATER_CSV = _DATA_DIR / "edi.136.11" / "NILTREB_porewater.csv"
_NI_ELEVATION_CSV = _DATA_DIR / "edi.134.12" / "NILTREB_marsh_surface_elevation_init.csv"
_PIE_SOIL_CSV     = _DATA_DIR / "knb-lter-pie.625.1" / "MAR_RO_ST_Soil.csv"


def _load_ni_observations():
    """Return NI porewater salinity stats (mean, std) by location (HM/LM)
    paired with median surface elevation from SET init data.

    Elevation is converted NAVD88 → MSL using NAVD88_BELOW_MSL_M.
    Summer = JJA (months 6-8), Winter = DJF (months 12,1,2).

    Returns dict keyed by location label:
        {'HM': (elev_m_msl, summer_mean, summer_std, winter_mean, winter_std),
         'LM': ...}
    """
    import pandas as pd

    # Surface elevations from SET init data (NAVD88)
    elev_df = pd.read_csv(_NI_ELEVATION_CSV)
    elev_df = elev_df[elev_df['SITE'] == 'GI']
    elev_by_loc = (
        elev_df.groupby('LOCATION')['MARSH_SURFACE_ELEVATION']
        .median()
        - NAVD88_BELOW_MSL_M
    )

    # Porewater salinity
    por_df = pd.read_csv(_NI_POREWATER_CSV)
    por_df['DATE'] = pd.to_datetime(por_df['DATE'])
    por_df['month'] = por_df['DATE'].dt.month
    por_df['season'] = por_df['month'].map(
        lambda m: 'summer' if m in [6, 7, 8] else ('winter' if m in [12, 1, 2] else None)
    )
    por_df = por_df[por_df['TREATMENT'] == 'C']   # control plots only
    por_df = por_df[por_df['SALINITY'].notna()]
    por_df = por_df[por_df['season'].notna()]

    results = {}
    for loc in ['HM', 'LM']:
        if loc not in elev_by_loc.index:
            continue
        elev = elev_by_loc[loc]
        sub = por_df[por_df['LOCATION'] == loc]
        summer = sub[sub['season'] == 'summer']['SALINITY']
        winter = sub[sub['season'] == 'winter']['SALINITY']
        results[loc] = (
            elev,
            summer.mean(), summer.std(),
            winter.mean(), winter.std(),
        )
    return results


def _load_pie_observations():
    """Return PIE surface porewater salinity vs elevation.

    Uses 0-5 cm depth samples only.
    Elevation is in NAVD88 (PIE MSL ≈ 0 m NAVD88, MHW ≈ 1.22 m NAVD88).

    Returns (elev_navd88, salinity_ppt) arrays.
    """
    import pandas as pd
    df = pd.read_csv(_PIE_SOIL_CSV)
    df = df[df['Depth_start'] == 0]
    df = df[df['Salinity'].notna() & df['Elevation'].notna()]
    return df['Elevation'].values, df['Salinity'].values


# ---------------------------------------------------------------------------
# Elevation list and run IDs
# ---------------------------------------------------------------------------

def _target_elevations() -> List[float]:
    n = round((ELEV_MAX_M - ELEV_MIN_M) / ELEV_STEP_M) + 1
    return [round(ELEV_MIN_M + i * ELEV_STEP_M, 4) for i in range(n)]


def _run_id(elev_m: float, dist_m: int) -> str:
    sign = "p" if elev_m >= 0 else "m"
    return f"se_{sign}{abs(elev_m):.2f}_d{dist_m:03d}".replace(".", "_")


# ---------------------------------------------------------------------------
# YAML writer
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


def _write_yaml(elev_m: float, dist_m: int, yaml_path: Path) -> None:
    pre_comp_elev = elev_m + _COMPACTION_OFFSET_M

    tides = north_inlet_default_tides()
    tides.suspended_sediment_concentration_kg_m3 = NI_SSC_KG_M3
    tides.fine_sediment_concentration_kg_m3      = NI_SSC_KG_M3 * 0.25

    config = PlotConfig(
        site_name="north_inlet",
        plot_id=_run_id(elev_m, dist_m),
        surface_elevation_m=pre_comp_elev,
        distance_from_creek_m=float(dist_m),
        tides=tides,
        met=north_inlet_default_met(),
        n_years=N_YEARS,
        sea_level_rise_m_yr=NI_SLR_M_YR,
        output_dir=str(OUTPUT_DIR),
    )
    config.initial_salinity_ppt = tides.creek_salinity_ppt

    parameters = dict(_DEFAULT_PARAMETERS)
    parameters.update(_tidal_params(tides))
    parameters["deposition_distance_from_edge_m"] = float(dist_m)
    parameters.setdefault("deposition_basin_length_m", 50.0)
    parameters.setdefault("deposition_length_scale_beta", 3.0)

    # Use initial salinity equal to creek (runs start near equilibrium)
    parameters["salinity_initial_root_zone_ppt"] = tides.creek_salinity_ppt

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
            "compaction_model_name":           "mixing_compaction",
            "porewater_chemistry_model_name": "none",
            "methane_model_name":             "none",
        },
        "site": {
            "distance_from_creek_m":  float(dist_m),
            "creek_bank_elevation_m": tides.mean_sea_level_m,
            "local_tidal_offset_m":   0.0,
        },
        "parameters": parameters,
        "materials":  _DEFAULT_MATERIALS,
        "forcing":    {"generated": build_generated_forcing_block(config)},
        "initial_state": {
            "layers": [{
                "top_elevation_m": pre_comp_elev,
                "thickness_m":     1.0,
                "porosity":        0.60,
                "age_days":        0.0,
                "fill_material":   "organic",
            }]
        },
        "initial_ecohydrology_state": {
            "root_zone_salinity_ppt":    config.initial_salinity_ppt,
            "aboveground_biomass_kg_m2": 0.3,
            "belowground_biomass_kg_m2": 0.3,
            "lai": round(0.3 * parameters["vegetation_lai_per_kg_m2"], 4),
            "litter_kg_m2": 0.0,
        },
        "output": {
            "file":                   str(yaml_path).replace(".yaml", ".nc"),
            "write_time_series":      True,
            "write_column_snapshots": False,
        },
    }

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as fh:
        yaml.dump(doc, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Run management
# ---------------------------------------------------------------------------

RunKey = Tuple[float, int]   # (target_elev_m, dist_m)


def _run_all(cli_binary: str, force: bool) -> Dict[RunKey, Path]:
    elevations = _target_elevations()
    total = len(elevations) * len(DISTANCES_M)
    results: Dict[RunKey, Path] = {}
    count = 0
    for elev in elevations:
        for dist in DISTANCES_M:
            count += 1
            key  = (elev, dist)
            rid  = _run_id(elev, dist)
            yaml_path = OUTPUT_DIR / f"{rid}.yaml"
            nc_path   = OUTPUT_DIR / f"{rid}.nc"

            if not force and nc_path.exists():
                print(f"  [{count:2d}/{total}] {elev:+.2f} m, {dist:2d} m: cached.")
            else:
                print(f"  [{count:2d}/{total}] {elev:+.2f} m, {dist:2d} m")
                _write_yaml(elev, dist, yaml_path)
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
    for elev in _target_elevations():
        for dist in DISTANCES_M:
            nc_path = OUTPUT_DIR / f"{_run_id(elev, dist)}.nc"
            if nc_path.exists():
                results[(elev, dist)] = nc_path
    return results


# ---------------------------------------------------------------------------
# NetCDF reader — returns (elev_at_end, annual_mean_sal, summer_mean_sal, winter_mean_sal)
# ---------------------------------------------------------------------------

def _read_run(nc_path: Path) -> Tuple[float, float, float, float]:
    with nc4.Dataset(str(nc_path), "r") as ds:
        t   = np.array(ds["model_time_days"][:], dtype=np.float64)
        sal = np.array(ds["root_zone_salinity_ppt"][:], dtype=np.float64)
        elev = np.array(ds["surface_elevation"][:], dtype=np.float64)

    if sal.size == 0:
        return (np.nan, np.nan, np.nan, np.nan)

    # DOY within the 1-year run
    doy = ((t - 0.5 * (t[1] - t[0])) % 365.25)

    annual_mean = float(np.mean(sal))
    summer_mask = (doy >= 152) & (doy < 244)   # Jun–Aug
    winter_mask = (doy < 59) | (doy >= 335)     # Dec–Feb

    summer_mean = float(np.mean(sal[summer_mask])) if summer_mask.any() else np.nan
    winter_mean = float(np.mean(sal[winter_mask])) if winter_mask.any() else np.nan
    elev_end    = float(elev[-1])

    return elev_end, annual_mean, summer_mean, winter_mean


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot(results: Dict[RunKey, Path], tides, save: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    msl_m = 0.0
    mhw_m = tides.mean_high_tide_m

    # ---- collect model results ----
    model: Dict[int, Dict] = {d: {"elev": [], "ann": [], "sum": [], "win": []}
                               for d in DISTANCES_M}
    for (elev_tgt, dist), nc_path in sorted(results.items()):
        elev_end, ann, summer, winter = _read_run(nc_path)
        model[dist]["elev"].append(elev_end)
        model[dist]["ann"].append(ann)
        model[dist]["sum"].append(summer)
        model[dist]["win"].append(winter)

    # ---- load observations ----
    ni_obs = _load_ni_observations() if _NI_POREWATER_CSV.exists() else {}
    pie_elev, pie_sal = _load_pie_observations() if _PIE_SOIL_CSV.exists() else (np.array([]), np.array([]))

    for ax_idx, (ax, season_key, season_label) in enumerate(zip(
        axes,
        ["sum", "win"],
        ["Summer (JJA)", "Winter (DJF)"],
    )):
        # shading and datum lines
        ax.axvspan(msl_m, mhw_m, color="lightblue", alpha=0.18, zorder=0,
                   label=f"Intertidal (MSL – MHW)")
        ax.axvline(msl_m, color="#4393c3", lw=1.3, ls="--", zorder=2, label="MSL")
        ax.axvline(mhw_m, color="#2166ac", lw=1.3, ls="--", zorder=2,
                   label=f"MHW = {mhw_m:.2f} m")
        ax.axhline(tides.creek_salinity_ppt, color="grey", lw=1.0, ls=":",
                   zorder=1, label=f"Creek sal. ({tides.creek_salinity_ppt:.0f} ppt)")

        # model lines
        for dist in DISTANCES_M:
            d = model[dist]
            if not d["elev"]:
                continue
            x   = np.array(d["elev"])
            y   = np.array(d[season_key])
            idx = np.argsort(x)
            ax.plot(x[idx], y[idx],
                    color=_DISTANCE_COLOURS[dist], lw=2.0,
                    label=f"Model {_DISTANCE_LABELS[dist]}", zorder=4)

        # NI observations (only on summer/winter panels)
        ni_season_idx = 1 if season_key == "sum" else 3   # index in ni_obs tuple
        ni_se_idx     = 2 if season_key == "sum" else 4
        for loc, vals in ni_obs.items():
            elev_ni, s_mean, s_std, w_mean, w_std = vals
            mean_val = s_mean if season_key == "sum" else w_mean
            std_val  = s_std  if season_key == "sum" else w_std
            ax.errorbar(
                elev_ni, mean_val, yerr=std_val,
                fmt="D", color="black", markersize=7, capsize=4, zorder=6,
                label=f"NI GI {loc} obs. (edi.136.11)" if ax_idx == 0 else "_",
            )
            ax.text(elev_ni + 0.02, mean_val + 1, f"GI {loc}", fontsize=7.5)

        # PIE observations as scatter (left panel only to avoid clutter)
        if ax_idx == 0 and pie_elev.size > 0:
            # PIE datum: MSL ≈ 0 m NAVD88; NI: MSL = NAVD88 + 0.15 m
            # Plot on secondary x-axis rather than converting — note in caption
            ax.scatter(
                pie_elev - 1.224 + mhw_m,   # shift so PIE MHW aligns with NI MHW
                pie_sal,
                s=18, color="#984ea3", alpha=0.4, zorder=3,
                label="PIE obs. (pie.625.1)\n(shifted so PIE MHW = NI MHW)",
            )

        ax.set_xlabel("Surface elevation (m above MSL, NI)", fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel("Root-zone salinity (ppt)", fontsize=10)
        ax.set_title(season_label, fontsize=11)
        ax.set_xlim(ELEV_MIN_M - 0.03, ELEV_MAX_M + 0.03)
        ax.set_ylim(0, 85)
        ax.axhline(80, color="firebrick", lw=0.8, ls="--", alpha=0.6,
                   label="80 ppt model cap" if ax_idx == 0 else "_")
        ax.grid(True, lw=0.3, alpha=0.5)

        # secondary x axis in cm NAVD88
        ax2 = ax.twiny()
        ax2.set_xlim(
            (ELEV_MIN_M - 0.03 + NAVD88_BELOW_MSL_M) * 100,
            (ELEV_MAX_M + 0.03 + NAVD88_BELOW_MSL_M) * 100,
        )
        ax2.set_xlabel("Elevation (cm above NAVD88, NI)", fontsize=9)

    axes[0].legend(fontsize=7.5, loc="upper left", framealpha=0.9, ncol=1)

    fig.suptitle(
        "Root-zone salinity vs elevation — North Inlet S. alterniflora\n"
        f"SSC = {NI_SSC_KG_M3*1000:.0f} mg L⁻¹,  tidal amp ×0.85,  "
        f"creek sal = {north_inlet_default_tides().creek_salinity_ppt:.0f} ppt",
        fontsize=10,
    )
    plt.tight_layout()

    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        out = FIGURE_DIR / "ni_salinity_elevation_sweep.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--binary", default="marsh_cli")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--plot-only", action="store_true")
    p.add_argument("--force",   action="store_true")
    p.add_argument("--no-save", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    elevations = _target_elevations()
    total = len(elevations) * len(DISTANCES_M)

    if args.plot_only:
        results = _collect_existing()
        if not results:
            print("No existing outputs — run without --plot-only first.")
            return
        print(f"Found {len(results)} existing run(s).")
    else:
        print(f"Running {total} simulations ({N_YEARS}-yr, "
              f"{len(elevations)} elevations × {len(DISTANCES_M)} distances)...")
        results = _run_all(cli_binary=args.binary, force=args.force)

    if not results:
        print("No outputs to plot.")
        return

    tides = north_inlet_default_tides()
    print("Plotting ...")
    _plot(results, tides, save=not args.no_save)


if __name__ == "__main__":
    main()
