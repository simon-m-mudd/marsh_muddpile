#!/usr/bin/env python3
"""
equilibrium_elevation.py

Find the equilibrium surface elevation of an S. alterniflora marsh for
combinations of SSC, SLR rate, and mean annual temperature.

Parameter sweep (3 × 3 × 2 = 18 runs):
  SSC (kg m⁻³):   0.005 / 0.025 / 0.050
  SLR (m yr⁻¹):  0.002 / 0.005 / 0.007
  Temp mean (°C): 15 / 22

Fixed: 20 m from creek, starting elevation 0.20 m MSL.

Each run advances for up to MAX_YEARS years with monthly steps.
Post-processing finds the first year at which the 5-yr rolling mean
annual accretion rate is within EQUIL_TOL (5 %) of the SLR rate.
A run is flagged as drowned if the 5-yr mean inundation fraction
exceeds DROWN_IF for DROWN_CONSEC or more consecutive years.

Outputs
-------
  sensitivity/runs/equilibrium/<run_id>.yaml / .nc
  sensitivity/figures/equilibrium_elevation.png
  sensitivity/figures/equilibrium_year.png

Usage
-----
  python sensitivity/equilibrium_elevation.py --binary ./build/marsh_cli
  python sensitivity/equilibrium_elevation.py --plot-only
  python sensitivity/equilibrium_elevation.py --force
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

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
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
except ImportError as exc:
    raise ImportError("matplotlib is required: pip install matplotlib") from exc

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_YEARS    = 500
_DT_DAYS     = 365.25 / 12.0
_DAYS_IN_YEAR = 365.25

OUTPUT_DIR = _THIS_DIR / "runs" / "equilibrium"
FIGURE_DIR = _THIS_DIR / "figures"

SSC_VALUES  = [0.005, 0.025, 0.050]   # kg m⁻³
SLR_VALUES  = [0.002, 0.005, 0.007]   # m yr⁻¹
TEMP_MEANS  = [15.0, 22.0]            # °C mean annual temperature

DISTANCE_M      = 20.0
START_ELEV_M    = 0.20   # m MSL — solidly in vegetated S. alterniflora range

# Equilibrium criterion: 5-yr rolling mean accretion within 5 % of SLR
EQUIL_TOL    = 0.05
EQUIL_WINDOW = 5    # years of rolling mean
EQUIL_CONSEC = 5    # consecutive years that must satisfy criterion

# Drowning criterion: 5-yr mean inundation fraction above threshold
DROWN_IF     = 0.65
DROWN_CONSEC = 5

# NI tidal amplitude used to draw MHW reference line on plots
_NI_TEMP_AMPLITUDE_C = 8.46
_NI_TEMP_PEAK_DAY    = 202.8

# ---------------------------------------------------------------------------
# Visual encoding
# ---------------------------------------------------------------------------

_SSC_COLOURS = {0.005: "#4575b4", 0.025: "#f46d43", 0.050: "#1a9641"}
_SLR_MARKERS = {0.002: "o", 0.005: "s", 0.007: "^"}
_LINE_WIDTH  = 1.8

# ---------------------------------------------------------------------------
# Run parameter type
# ---------------------------------------------------------------------------

class RunParams(NamedTuple):
    ssc: float
    slr: float
    temp_mean: float


def _all_params() -> List[RunParams]:
    return [RunParams(ssc, slr, t)
            for ssc in SSC_VALUES
            for slr in SLR_VALUES
            for t in TEMP_MEANS]


def _run_id(p: RunParams) -> str:
    return (
        f"eq_ssc{p.ssc * 1000:.1f}gm3"
        f"_slr{p.slr * 1000:.0f}mm"
        f"_t{p.temp_mean:.0f}c"
    )

# ---------------------------------------------------------------------------
# Initial column — 20 layers × 5 cm = 1 m
# ---------------------------------------------------------------------------

_N_LAYERS        = 20
_LAYER_THICKNESS = 0.05
_POROSITY        = 0.60
_RHO_SAND        = 2650.0
_RHO_REFRAC      = 1400.0
_RHO_LABILE      = 1200.0
_RHO_ROOTS       = 1100.0
_EXP_EFOLD       = 0.075


def _build_initial_layers(surface_elevation_m: float) -> List[dict]:
    solid = (1.0 - _POROSITY) * _LAYER_THICKNESS
    layers = []
    for i in range(_N_LAYERS):
        depth_mid = (i + 0.5) * _LAYER_THICKNESS
        top_elev  = surface_elevation_m - i * _LAYER_THICKNESS
        exp_fac   = math.exp(-depth_mid / _EXP_EFOLD)
        refrac    = 0.10
        labile    = 0.10 * exp_fac
        roots     = 0.10 * exp_fac
        sand      = max(0.0, 1.0 - refrac - labile - roots)
        layers.append({
            "top_elevation_m": round(top_elev, 4),
            "thickness_m":     _LAYER_THICKNESS,
            "porosity":        _POROSITY,
            "age_days":        0.0,
            "mass_kg_m2": {
                "sand":               round(_RHO_SAND  * sand   * solid, 4),
                "refractory_organic": round(_RHO_REFRAC * refrac * solid, 4),
                "labile_organic":     round(_RHO_LABILE * labile * solid, 6),
                "roots":              round(_RHO_ROOTS  * roots  * solid, 6),
            },
        })
    return list(reversed(layers))   # deepest first


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


def _write_yaml(p: RunParams, yaml_path: Path) -> None:
    tides = north_inlet_default_tides()
    tides.suspended_sediment_concentration_kg_m3 = p.ssc
    tides.fine_sediment_concentration_kg_m3      = p.ssc * 0.25

    met = north_inlet_default_met()
    met.temperature_mean_c      = p.temp_mean
    met.temperature_amplitude_c = _NI_TEMP_AMPLITUDE_C
    met.temperature_peak_day    = _NI_TEMP_PEAK_DAY

    config = PlotConfig(
        site_name="equilibrium_elevation",
        plot_id=_run_id(p),
        surface_elevation_m=START_ELEV_M,
        distance_from_creek_m=DISTANCE_M,
        tides=tides,
        met=met,
        n_years=MAX_YEARS,
        sea_level_rise_m_yr=p.slr,
        output_dir=str(OUTPUT_DIR),
    )
    config.initial_salinity_ppt = tides.creek_salinity_ppt

    parameters = dict(_DEFAULT_PARAMETERS)
    parameters.update(_tidal_params(tides))
    parameters["deposition_distance_from_edge_m"] = DISTANCE_M
    parameters.setdefault("deposition_basin_length_m",    50.0)
    parameters.setdefault("deposition_length_scale_beta", 3.0)

    initial_layers = _build_initial_layers(START_ELEV_M)
    total_roots    = sum(l["mass_kg_m2"]["roots"] for l in initial_layers)

    total_days = MAX_YEARS * _DAYS_IN_YEAR
    doc = {
        "simulation": {
            "dt_days":                       round(_DT_DAYS, 4),
            "water_level_model_name":        "composite_water_level",
            "salinity_model_name":           "distance_flushing_salinity",
            "evapotranspiration_model_name": "simple_canopy_et",
            "vegetation_model_name":         "marsh_gpp_biomass",
            "deposition_model_name":         "edge_distance_deposition",
            "root_allocation_model_name":    "exponential_root_allocation",
            "decay_model_name":              "marsh_decay",
            "compaction_model_name":         "mixing_compaction",
            "porewater_chemistry_model_name":"none",
            "methane_model_name":            "none",
        },
        "site": {
            "distance_from_creek_m":  DISTANCE_M,
            "creek_bank_elevation_m": tides.mean_sea_level_m,
            "local_tidal_offset_m":   0.0,
        },
        "parameters": parameters,
        "materials":  _DEFAULT_MATERIALS,
        "forcing":    {"generated": build_generated_forcing_block(config)},
        "initial_state":           {"layers": initial_layers},
        "initial_ecohydrology_state": {
            "root_zone_salinity_ppt":    config.initial_salinity_ppt,
            "aboveground_biomass_kg_m2": 0.30,
            "belowground_biomass_kg_m2": round(total_roots, 4),
            "lai":   round(0.30 * parameters["vegetation_lai_per_kg_m2"], 4),
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

def _run_all(cli_binary: str, force: bool) -> Dict[RunParams, Path]:
    all_p = _all_params()
    n = len(all_p)
    print(f"Running {n} configurations (up to {MAX_YEARS} yr each) ...")
    results: Dict[RunParams, Path] = {}
    for i, p in enumerate(all_p, 1):
        rid       = _run_id(p)
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
# NetCDF reader and equilibrium detector
# ---------------------------------------------------------------------------

def _annual_means(t_days: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Bin a monthly time series into calendar years; return (year_centres, annual_means)."""
    t_yr   = t_days / _DAYS_IN_YEAR
    y_bins = np.arange(0, MAX_YEARS + 2, 1.0)
    sums   = np.zeros(MAX_YEARS + 1)
    counts = np.zeros(MAX_YEARS + 1, dtype=int)
    for i, tv in enumerate(t_yr):
        idx = int(tv)
        if 0 <= idx <= MAX_YEARS:
            sums[idx]   += values[i]
            counts[idx] += 1
    valid = counts > 0
    means  = np.where(valid, sums / np.where(valid, counts, 1), np.nan)
    centres = np.arange(MAX_YEARS + 1) + 0.5
    return centres[valid], means[valid]


def _find_equilibrium(
    t_yr: np.ndarray,
    elev: np.ndarray,
    inund: np.ndarray,
    slr: float,
) -> dict:
    """
    Returns a dict with keys:
      status          : 'equilibrium' | 'drowned' | 'not_converged'
      equil_year      : year at which equilibrium first satisfied (or nan)
      equil_elev_m    : mean surface elevation (m MSL) in equilibrium window (or nan)
      equil_elev_navd : same in cm NAVD88 (MSL = NAVD88 + 0.15 m)
      drown_year      : first year of sustained drowning (or nan)
    """
    NAVD88_TO_MSL = 0.15   # m; MSL = NAVD88 datum + 0.15 m at NI

    # Annual accretion and inundation
    yr_c_e, ann_elev   = _annual_means(t_yr * _DAYS_IN_YEAR, elev)
    yr_c_i, ann_inund  = _annual_means(t_yr * _DAYS_IN_YEAR, inund)

    n = len(ann_elev)
    ann_accretion = np.full(n, np.nan)
    ann_accretion[1:] = np.diff(ann_elev)

    # Rolling mean (EQUIL_WINDOW years)
    def rolling_mean(arr, w):
        out = np.full(len(arr), np.nan)
        for i in range(w - 1, len(arr)):
            out[i] = np.nanmean(arr[i - w + 1: i + 1])
        return out

    roll_accretion = rolling_mean(ann_accretion, EQUIL_WINDOW)
    roll_inund     = rolling_mean(ann_inund,     EQUIL_WINDOW)

    slr_per_yr = slr   # already in m yr⁻¹; annual accretion also in m yr⁻¹

    # Check drowning first
    drown_year = np.nan
    consec = 0
    for i, ri in enumerate(roll_inund):
        if not np.isnan(ri) and ri > DROWN_IF:
            consec += 1
            if consec >= DROWN_CONSEC:
                drown_year = float(yr_c_i[i - DROWN_CONSEC + 1])
                break
        else:
            consec = 0

    # Check equilibrium
    equil_year = np.nan
    equil_elev = np.nan
    consec = 0
    for i, ra in enumerate(roll_accretion):
        if (not np.isnan(ra) and slr_per_yr > 0 and
                abs(ra - slr_per_yr) / slr_per_yr <= EQUIL_TOL):
            consec += 1
            if consec >= EQUIL_CONSEC:
                eq_start = i - EQUIL_CONSEC + 1
                equil_year = float(yr_c_e[eq_start])
                equil_elev = float(np.nanmean(ann_elev[eq_start: i + 1]))
                break
        else:
            consec = 0

    # Determine status
    if not np.isnan(drown_year) and (np.isnan(equil_year) or drown_year < equil_year):
        status = "drowned"
    elif not np.isnan(equil_year):
        status = "equilibrium"
    else:
        status = "not_converged"

    equil_elev_navd = (equil_elev + NAVD88_TO_MSL) * 100.0 if not np.isnan(equil_elev) else np.nan

    return {
        "status":          status,
        "equil_year":      equil_year,
        "equil_elev_m":    equil_elev,
        "equil_elev_navd": equil_elev_navd,
        "drown_year":      drown_year,
    }


def _read_run(nc_path: Path, p: RunParams) -> dict:
    with nc4.Dataset(str(nc_path), "r") as ds:
        t     = np.array(ds.variables["model_time_days"][:],         dtype=np.float64)
        elev  = np.array(ds.variables["surface_elevation"][:],       dtype=np.float64)
        inund = np.array(ds.variables["inundation_fraction"][:],     dtype=np.float64)
        ag    = np.array(ds.variables["aboveground_biomass_kg_m2"][:], dtype=np.float64)

    t_yr   = t / _DAYS_IN_YEAR
    result = _find_equilibrium(t_yr, elev, inund, p.slr)
    result["t_yr"]   = t_yr
    result["elev"]   = elev
    result["inund"]  = inund
    result["ag_bio"] = ag
    snap = _read_run_snapshot(nc_path)
    if snap is not None:
        result.update(snap)
    return result


_PROFILE_BIN_M = 0.005   # 5 mm depth bins for organic profile

def _read_run_snapshot(nc_path: Path) -> Optional[dict]:
    """Return final-snapshot organic fraction depth profile, binned at 5 mm.

    Returns a dict with:
      depth_m    : bin mid-depths (m below surface, positive down), shape (n_bins,)
      org_frac   : mass-weighted organic fraction per bin, shape (n_bins,)
      refract_frac, labile_frac, roots_frac : individual organic components
      surface_elev_m : absolute surface elevation at snapshot time
    """
    with nc4.Dataset(str(nc_path), "r") as ds:
        n_snap = len(ds.dimensions["snapshot"])
        if n_snap == 0:
            return None

        n_layers = int(ds.variables["snapshot_n_layers"][-1])
        if n_layers == 0:
            return None

        mat_names = [
            str(nc4.chartostring(ds.variables["material_name"][i])).strip()
            for i in range(len(ds.dimensions["material"]))
        ]

        def _idx(candidates):
            for c in candidates:
                if c in mat_names:
                    return mat_names.index(c)
            return -1

        refract_idx  = _idx(["refractory_organic", "refractory_carbon"])
        labile_idx   = _idx(["labile_organic",     "labile_carbon"])
        roots_idx    = _idx(["roots",               "live_roots",  "root"])

        mass      = np.array(ds.variables["layer_mass"][-1, :n_layers, :], dtype=np.float64)
        top_elev  = np.array(ds.variables["layer_top_elevation"][-1, :n_layers], dtype=np.float64)
        thickness = np.array(ds.variables["layer_thickness"][-1, :n_layers],     dtype=np.float64)

    surface_elev = float(top_elev[-1] + thickness[-1])
    mid_elev     = top_elev + 0.5 * thickness
    depth_m      = surface_elev - mid_elev   # positive downward

    total_mass   = mass.sum(axis=1)
    refract_mass = mass[:, refract_idx] if refract_idx >= 0 else np.zeros(n_layers)
    labile_mass  = mass[:, labile_idx]  if labile_idx  >= 0 else np.zeros(n_layers)
    roots_mass   = mass[:, roots_idx]   if roots_idx   >= 0 else np.zeros(n_layers)
    organic_mass = refract_mass + labile_mass + roots_mass

    # Bin by depth — use mass-weighted mean organic fraction per bin
    max_depth  = max(float(depth_m[depth_m >= 0].max()) if (depth_m >= 0).any() else 0.30,
                     0.30)
    bin_edges  = np.arange(0.0, max_depth + _PROFILE_BIN_M, _PROFILE_BIN_M)
    n_bins     = len(bin_edges) - 1
    bin_depth  = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    def _bin(component_mass):
        num = np.zeros(n_bins)
        den = np.zeros(n_bins)
        for li in range(n_layers):
            d = depth_m[li]
            if d < 0 or total_mass[li] <= 0:
                continue
            b = int(d / _PROFILE_BIN_M)
            if b >= n_bins:
                continue
            num[b] += component_mass[li]
            den[b] += total_mass[li]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)

    return {
        "depth_m":       bin_depth,
        "org_frac":      _bin(organic_mass),
        "refract_frac":  _bin(refract_mass),
        "labile_frac":   _bin(labile_mass),
        "roots_frac":    _bin(roots_mass),
        "surface_elev_m": surface_elev,
    }


# ---------------------------------------------------------------------------
# Organic profile plotting
# ---------------------------------------------------------------------------

_TEMP_STYLES = {15.0: "-", 22.0: "--"}
_TEMP_LABELS = {15.0: "15 °C", 22.0: "22 °C"}
_COMP_COLOURS = {
    "org_frac":     "k",
    "refract_frac": "#8b4513",
    "roots_frac":   "#2ca02c",
    "labile_frac":  "#1f77b4",
}


def _plot_organic_profiles(data_map: Dict[RunParams, dict], save: bool) -> None:
    """3×3 grid (rows=SSC, cols=SLR); lines=temperature; stacked components."""
    n_ssc = len(SSC_VALUES)
    n_slr = len(SLR_VALUES)

    fig, axes = plt.subplots(
        n_ssc, n_slr,
        figsize=(14, 10),
        sharex=False,
        sharey=True,
    )

    x_max_global = 0.0
    for d in data_map.values():
        if "org_frac" in d:
            v = np.nanmax(d["org_frac"]) if np.any(~np.isnan(d["org_frac"])) else 0.0
            x_max_global = max(x_max_global, v)
    x_max_global = max(x_max_global * 1.15, 0.02)

    for row, ssc in enumerate(SSC_VALUES):
        for col, slr in enumerate(SLR_VALUES):
            ax = axes[row, col]

            for temp in TEMP_MEANS:
                p = RunParams(ssc, slr, temp)
                if p not in data_map:
                    continue
                d = data_map[p]
                if "depth_m" not in d:
                    continue

                depth    = d["depth_m"]
                org      = d["org_frac"]
                refract  = d["refract_frac"]
                labile   = d["labile_frac"]
                roots    = d["roots_frac"]
                ls       = _TEMP_STYLES[temp]
                status   = d.get("status", "unknown")
                alpha    = 1.0 if status == "equilibrium" else 0.55

                # Stack area: refract (brown), roots (green), labile (blue)
                # Draw total org_frac as the outer line, then fill components
                left = np.zeros_like(depth)
                for comp, colour in [
                    ("refract_frac", _COMP_COLOURS["refract_frac"]),
                    ("roots_frac",   _COMP_COLOURS["roots_frac"]),
                    ("labile_frac",  _COMP_COLOURS["labile_frac"]),
                ]:
                    vals = np.nan_to_num(d[comp])
                    right = left + vals
                    ax.fill_betweenx(depth, left, right,
                                     color=colour, alpha=0.25 * alpha,
                                     linewidth=0)
                    left = right

                ax.plot(np.nan_to_num(org), depth,
                        color="k", lw=1.2, ls=ls, alpha=alpha,
                        label="%s (%s)" % (_TEMP_LABELS[temp], status))

            ax.set_ylim(bottom=max(ax.get_ylim()[1], 0.35), top=0)
            ax.set_xlim(left=0, right=x_max_global)
            ax.grid(True, linewidth=0.3, alpha=0.4)

            if row == 0:
                ax.set_title("SLR = %g mm yr⁻¹" % (slr * 1000), fontsize=9)
            if col == 0:
                ax.set_ylabel("SSC %g mg L⁻¹\nDepth below surface (m)" % (ssc * 1000),
                              fontsize=8)
            if row == n_ssc - 1:
                ax.set_xlabel("Organic fraction (–)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6, loc="lower right")

    # Colour-patch legend for components
    import matplotlib.patches as mpatch
    comp_legend = [
        mpatch.Patch(color=_COMP_COLOURS["refract_frac"], alpha=0.5, label="Refractory OM"),
        mpatch.Patch(color=_COMP_COLOURS["roots_frac"],   alpha=0.5, label="Live roots"),
        mpatch.Patch(color=_COMP_COLOURS["labile_frac"],  alpha=0.5, label="Labile OM"),
        mlines.Line2D([], [], color="k", lw=1.2, ls="-",  label="Total (15 °C)"),
        mlines.Line2D([], [], color="k", lw=1.2, ls="--", label="Total (22 °C)"),
    ]
    fig.legend(handles=comp_legend, loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        "Organic fraction depth profiles — end of run\n"
        "S. alterniflora, NI tides, 20 m from creek  |  faded = not at equilibrium",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        out = FIGURE_DIR / "equilibrium_organic_profiles.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print("  Saved: %s" % out)
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# MHW helper (reused from longterm_sensitivity)
# ---------------------------------------------------------------------------

def _compute_mhw_msl(tides) -> float:
    deg     = np.pi / 180.0
    t_hours = np.arange(0.0, 365.25 * 24.0, 1.0 / 6.0)
    wl      = np.zeros(len(t_hours))
    for speed, amp, phase in [
        (28.9841042, tides.M2_amplitude_m, tides.M2_phase_deg),
        (30.0000000, tides.S2_amplitude_m, tides.S2_phase_deg),
        (28.4397295, tides.N2_amplitude_m, tides.N2_phase_deg),
        (15.0410686, tides.K1_amplitude_m, tides.K1_phase_deg),
        (13.9430356, tides.O1_amplitude_m, tides.O1_phase_deg),
    ]:
        wl += amp * np.cos(speed * deg * t_hours - phase * deg)
    is_peak = (wl[1:-1] > wl[:-2]) & (wl[1:-1] > wl[2:])
    return float(np.mean(wl[1:-1][is_peak]))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_results(data_map: Dict[RunParams, dict], save: bool) -> None:
    tides   = north_inlet_default_tides()
    mhw_msl = _compute_mhw_msl(tides)

    # ── Figure 1: equilibrium elevation vs SLR ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for col_idx, temp in enumerate(TEMP_MEANS):
        ax = axes[col_idx]
        ax.axhline((mhw_msl + 0.15) * 100, color="0.55", lw=1.0, ls="--",
                   label="MHW (NAVD88)")
        ax.axhline(0, color="0.75", lw=0.7, ls=":")

        for ssc in SSC_VALUES:
            equil_navd = []
            slr_plot   = []
            for slr in SLR_VALUES:
                p = RunParams(ssc, slr, temp)
                if p not in data_map:
                    continue
                d = data_map[p]
                if d["status"] == "equilibrium":
                    equil_navd.append(d["equil_elev_navd"])
                    slr_plot.append(slr * 1000)
                elif d["status"] == "drowned":
                    # Plot X at the bottom
                    ax.scatter([slr * 1000], [-20],
                               marker="x", s=100, color=_SSC_COLOURS[ssc],
                               linewidths=2, zorder=5)
                else:
                    # Not converged — open circle at final elevation
                    final_navd = (d["elev"][-1] + 0.15) * 100
                    ax.scatter([slr * 1000], [final_navd],
                               marker="o", s=60, facecolors="none",
                               edgecolors=_SSC_COLOURS[ssc], linewidths=1.5,
                               zorder=4)

            if equil_navd:
                ax.plot(slr_plot, equil_navd,
                        color=_SSC_COLOURS[ssc], lw=_LINE_WIDTH,
                        marker="o", ms=7, label=f"SSC {ssc*1000:.0f} mg L⁻¹")

        ax.set_title(f"Temperature mean = {temp:.0f} °C", fontsize=10)
        ax.set_xlabel("SLR rate (mm yr⁻¹)", fontsize=9)
        if col_idx == 0:
            ax.set_ylabel("Equilibrium elevation (cm NAVD88)", fontsize=9)
        ax.set_xticks([s * 1000 for s in SLR_VALUES])
        ax.grid(True, linewidth=0.3, alpha=0.4)
        ax.legend(fontsize=8)

    # Legend entries for non-equilibrium symbols
    from matplotlib.lines import Line2D
    extra = [
        Line2D([0], [0], marker="x", color="0.4", lw=0, ms=9,
               markeredgewidth=2, label="drowned"),
        Line2D([0], [0], marker="o", color="0.4", lw=0, ms=7,
               markerfacecolor="none", markeredgewidth=1.5,
               label="not converged (final elev shown)"),
    ]
    axes[1].legend(handles=axes[1].get_legend_handles_labels()[0] + extra,
                   fontsize=8)

    fig.suptitle(
        "S. alterniflora equilibrium elevation — NI tides, 20 m from creek\n"
        f"Equilibrium: 5-yr rolling accretion within {EQUIL_TOL*100:.0f}% of SLR "
        f"for {EQUIL_CONSEC} consecutive years",
        fontsize=10,
    )
    plt.tight_layout()
    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        out = FIGURE_DIR / "equilibrium_elevation.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")
    else:
        plt.show()
    plt.close(fig)

    # ── Figure 2: years to equilibrium ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for col_idx, temp in enumerate(TEMP_MEANS):
        ax = axes[col_idx]
        for ssc in SSC_VALUES:
            yr_list  = []
            slr_plot = []
            for slr in SLR_VALUES:
                p = RunParams(ssc, slr, temp)
                if p not in data_map:
                    continue
                d = data_map[p]
                if d["status"] == "equilibrium":
                    yr_list.append(d["equil_year"])
                    slr_plot.append(slr * 1000)
                elif d["status"] == "drowned":
                    ax.scatter([slr * 1000], [d["drown_year"]],
                               marker="x", s=100, color=_SSC_COLOURS[ssc],
                               linewidths=2, zorder=5)
            if yr_list:
                ax.plot(slr_plot, yr_list,
                        color=_SSC_COLOURS[ssc], lw=_LINE_WIDTH,
                        marker="o", ms=7, label=f"SSC {ssc*1000:.0f} mg L⁻¹")

        ax.set_title(f"Temperature mean = {temp:.0f} °C", fontsize=10)
        ax.set_xlabel("SLR rate (mm yr⁻¹)", fontsize=9)
        if col_idx == 0:
            ax.set_ylabel("Years to equilibrium", fontsize=9)
        ax.set_xticks([s * 1000 for s in SLR_VALUES])
        ax.grid(True, linewidth=0.3, alpha=0.4)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Years to reach equilibrium — S. alterniflora, NI tides, 20 m from creek\n"
        "× = drowned before equilibrium",
        fontsize=10,
    )
    plt.tight_layout()
    if save:
        out = FIGURE_DIR / "equilibrium_year.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")
    else:
        plt.show()
    plt.close(fig)

    # ── Figure 3: organic fraction depth profiles ────────────────────────────
    _plot_organic_profiles(data_map, save=save)

    # ── Summary table ────────────────────────────────────────────────────────
    print()
    header = ("ssc(mg/L)  slr(mm/yr)  temp(C)  status          "
              "equil_yr  equil_elev(cmNAVD)  drown_yr")
    print(header)
    print("-" * len(header))
    for p in _all_params():
        if p not in data_map:
            continue
        d = data_map[p]
        ey  = f"{d['equil_year']:.0f}"  if not np.isnan(d["equil_year"])  else "—"
        ee  = f"{d['equil_elev_navd']:.1f}" if not np.isnan(d["equil_elev_navd"]) else "—"
        dy  = f"{d['drown_year']:.0f}"  if not np.isnan(d["drown_year"])  else "—"
        print(f"{p.ssc*1000:>9.1f}  {p.slr*1000:>10.0f}  {p.temp_mean:>7.0f}  "
              f"{d['status']:<16}{ey:>8}  {ee:>18}  {dy:>8}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--binary", default="marsh_cli")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--plot-only", action="store_true")
    g.add_argument("--force",     action="store_true")
    ap.add_argument("--no-save",  action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        results_raw = _collect_existing()
        if not results_raw:
            print("No existing outputs — run without --plot-only first.")
            return
        print(f"Found {len(results_raw)} existing outputs.")
    else:
        results_raw = _run_all(args.binary, force=args.force)

    print("Analysing outputs ...")
    data_map: Dict[RunParams, dict] = {}
    for p, nc_path in results_raw.items():
        try:
            data_map[p] = _read_run(nc_path, p)
        except Exception as exc:
            print(f"  Warning: could not read {nc_path.name}: {exc}")

    if not data_map:
        print("No outputs to plot.")
        return

    _plot_results(data_map, save=not args.no_save)


if __name__ == "__main__":
    main()
