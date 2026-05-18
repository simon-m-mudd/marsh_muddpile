#!/usr/bin/env python3
"""
test_biomass_curve_ni.py

Peak aboveground biomass vs surface elevation for S. alterniflora at North Inlet.

Each simulation runs for 1 year (12 monthly steps) across a range of surface
elevations.  The model is initialised with a modest starting biomass (0.3 kg m⁻²)
at the beginning of the year; the peak ABG recorded over all 12 steps is the
'peak aboveground biomass' at that elevation.

Parameters (current best-fit North Inlet calibration):
  LUE       = 1.6e-6 gC μmol⁻¹
  Capacity  = 0.95 kg m⁻² aboveground, 1.9 kg m⁻² belowground
  SSC       = 0.006 kg m⁻³  (low-turbidity NI conditions)
  Tidal constituents from NOAA 8662245 (Oyster Landing, directly at North Inlet)
    — no damping applied; MSL = 0.108 m NAVD88 (fitted growing-season mean, 2021)
  Compaction model: mixing_compaction (Morris et al. 2016 depth-dependent LOI mixing)
    — initial column is close to mixing-model equilibrium; no pre-compaction offset
  Refractory organic decay: k_0 = 0.0 (inert on marsh timescales)

Reference overlays:
  MSL = 0.108 m NAVD88 (Apr–Sep 2021 growing-season mean, NOAA 8662245)
  MHW = 0.625 m NAVD88 (NOAA 8662245 datum)
  Miller et al. (2019) parabola fit (coefficients from figure):
      B = 14.8·E − 0.157·E² + 598  [g m⁻²]  E = cm NAVD88
      Peak at E = 47 cm NAVD88  (B_max ≈ 947 g m⁻²)
  NI data points (Morris et al. 2013 Oceanography, 2008 field data)
  with error bars from morris2013_biomass_data.csv

All elevations (model outputs, axis, reference data) are in m NAVD88.

Usage
-----
    python sensitivity/test_biomass_curve_ni.py --binary ./build/marsh_cli
    python sensitivity/test_biomass_curve_ni.py --plot-only
    python sensitivity/test_biomass_curve_ni.py --force

Output
------
    sensitivity/figures/ni_biomass_curve.png
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
    import matplotlib.patches as mpatches
except ImportError as exc:
    raise ImportError("matplotlib is required: pip install matplotlib") from exc


# ---------------------------------------------------------------------------
# Calibration parameters (best-fit from ni_parameter_test)
# ---------------------------------------------------------------------------

LUE            = 1.6e-6   # gC per umol PAR — calibrated to Morris et al. (2013) NI data
CAPACITY_KG_M2      = 0.95   # kg m⁻² aboveground — RMSE-optimal vs Morris 2013 data
CAPACITY_KG_M2_HIGH = 1.20   # kg m⁻² aboveground — for comparison

# Best-fit σ values from ni_biomass_rmse_scan (v2, corrected MSL=0.108):
SIGMA_H_BEST = 0.25   # high-elevation (dry-side) half-Gaussian  — was 0.18
SIGMA_R_BEST = 0.22   # low-elevation  (wet-side) half-Gaussian  — was 0.25
NI_SSC_KG_M3  = 0.006     # kg m⁻³
NI_SLR_M_YR   = 0.003     # m yr⁻¹
NI_DISTANCE_M  = 20.0     # m from creek

# MSL offset above NAVD88 for the 2021 growing season (Apr–Sep) at NOAA 8662245.
# Used to convert Morris NAVD88 data to the same coordinate as the model MSL.
# Updating this keeps the biomass-elevation x-axis in m NAVD88 throughout.
NAVD88_BELOW_MSL_M = 0.108

# Elevations (m NAVD88) of the three low-outlier data points (alternating low/high
# scatter in Morris 2013 field plots at +5, +25, +55 cm NAVD88).
_LOW_OUTLIER_ELEV_M = {0.05, 0.25, 0.55}

# Initial biomass for 1-year runs
INITIAL_BIOMASS_KG_M2 = 0.3   # kg m⁻²

# ---------------------------------------------------------------------------
# Elevation sweep
# ---------------------------------------------------------------------------

# Target surface elevations (m NAVD88).
# Covers the full Morris 2013 NI data range (-35 to 110 cm NAVD88).
ELEV_MIN_M  = -0.50   # m NAVD88 (below subtidal zone)
ELEV_MAX_M  =  0.95   # m NAVD88 (supratidal fringe)
ELEV_STEP_M =  0.05

# The mixing_compaction model produces near-zero net column-height change for
# a φ=0.60 initial column (surface layers expand ~6%, deep mineral layers
# contract ~2%, net ≈ ±0.01 m per 1 m column).  No pre-inflation is needed.
_COMPACTION_OFFSET_M = 0.0

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUTPUT_DIR      = _THIS_DIR / "runs" / "ni_biomass_curve_v2"
OUTPUT_DIR_HIGH = _THIS_DIR / "runs" / "ni_biomass_curve_v2_k120"
FIGURE_DIR = _THIS_DIR / "figures"

# ---------------------------------------------------------------------------
# Initial column builder  (same geometry as ni_parameter_test)
# ---------------------------------------------------------------------------

_N_LAYERS        = 20
_LAYER_THICKNESS = 0.05   # m
_POROSITY        = 0.60
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
# Tidal parameters
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
    return f"ni_bc_{sign}{abs(target_elev_m):.2f}".replace(".", "_")


def _write_yaml(
    target_elev_m: float,
    yaml_path: Path,
    capacity_kg_m2: float = CAPACITY_KG_M2,
    out_dir: Path = OUTPUT_DIR,
) -> None:
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
        n_years=1,
        sea_level_rise_m_yr=NI_SLR_M_YR,
        output_dir=str(out_dir),
    )
    config.initial_salinity_ppt = tides.creek_salinity_ppt

    parameters = dict(_DEFAULT_PARAMETERS)
    parameters.update(_tidal_params(tides))
    parameters["deposition_distance_from_edge_m"] = NI_DISTANCE_M
    parameters.setdefault("deposition_basin_length_m", 50.0)
    parameters.setdefault("deposition_length_scale_beta", 3.0)
    parameters["vegetation_lue_gC_per_umol"]                    = LUE
    parameters["vegetation_aboveground_capacity_kg_m2"]         = capacity_kg_m2
    parameters["vegetation_belowground_capacity_kg_m2"]         = capacity_kg_m2 * 2.0
    parameters["vegetation_hydroperiod_sigma_fraction"]         = SIGMA_H_BEST
    parameters["vegetation_inundation_sigma_fraction"]          = SIGMA_R_BEST

    initial_layers = _build_initial_layers(pre_compaction_elev_m)

    doc = {
        "simulation": {
            "start_year": 0,
            "end_year": 1,
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
            "aboveground_biomass_kg_m2": INITIAL_BIOMASS_KG_M2,
            "belowground_biomass_kg_m2": INITIAL_BIOMASS_KG_M2,
            "lai": round(INITIAL_BIOMASS_KG_M2 * parameters["vegetation_lai_per_kg_m2"], 4),
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


def _run_all(
    cli_binary: str,
    force: bool,
    capacity_kg_m2: float = CAPACITY_KG_M2,
    out_dir: Path = OUTPUT_DIR,
) -> Dict[float, Path]:
    elevations = _target_elevations()
    n = len(elevations)
    print(f"Running {n} elevation points (K={capacity_kg_m2:.2f} kg m⁻², "
          f"init biomass = {INITIAL_BIOMASS_KG_M2:.1f} kg m⁻²) ...")
    results: Dict[float, Path] = {}
    for i, elev in enumerate(elevations, 1):
        rid       = _run_id(elev)
        yaml_path = out_dir / f"{rid}.yaml"
        nc_path   = out_dir / f"{rid}.nc"

        if not force and nc_path.exists():
            print(f"  [{i:2d}/{n}] {elev:+.2f} m: cached.")
        else:
            print(f"  [{i:2d}/{n}] {elev:+.2f} m")
            out_dir.mkdir(parents=True, exist_ok=True)
            _write_yaml(elev, yaml_path, capacity_kg_m2=capacity_kg_m2, out_dir=out_dir)
            try:
                run_model(str(yaml_path), cli_binary=cli_binary, silent=True)
            except RuntimeError as exc:
                print(f"    ERROR: {exc}")
                continue

        if nc_path.exists():
            results[elev] = nc_path
    return results


def _collect_existing(out_dir: Path = OUTPUT_DIR) -> Dict[float, Path]:
    results: Dict[float, Path] = {}
    for elev in _target_elevations():
        nc_path = out_dir / f"{_run_id(elev)}.nc"
        if nc_path.exists():
            results[elev] = nc_path
    return results


# ---------------------------------------------------------------------------
# NetCDF reader — returns (elevation at peak, peak ABG)
# ---------------------------------------------------------------------------

def _read_run(nc_path: Path) -> Tuple[float, float, float]:
    """Return (target_elev, elev_at_peak_m, peak_abg_kg_m2)."""
    with nc4.Dataset(str(nc_path), "r") as ds:
        elev = np.array(ds["surface_elevation"][:], dtype=np.float64)
        abg  = np.array(ds["aboveground_biomass_kg_m2"][:], dtype=np.float64)
        t    = np.array(ds["model_time_days"][:], dtype=np.float64)

    if abg.size == 0:
        return (np.nan, np.nan)

    # Use the second half of the year (months 7-12) to avoid contamination
    # from the arbitrary initial conditions in the first 6 months.
    half = max(1, len(abg) // 2)
    peak_idx       = int(np.argmax(abg[half:])) + half
    peak_abg       = float(abg[peak_idx])
    elev_at_peak   = float(elev[peak_idx])
    return elev_at_peak, peak_abg


# ---------------------------------------------------------------------------
# Miller (2019) reference parabola and digitised data
# ---------------------------------------------------------------------------

def _miller_parabola_kg_m2(z_navd88_m: np.ndarray) -> np.ndarray:
    """
    B = 14.8·E − 0.157·E² + 598  [g m⁻²]  (coefficients from Miller 2019 figure)
    E  = cm NAVD88  →  E = z_navd88_m × 100

    Peak at E ≈ 47.1 cm NAVD88  (B_max ≈ 947 g m⁻²)

    Returns B in kg m⁻² (zero-clipped).
    """
    E_cm = z_navd88_m * 100.0
    B_g  = 14.8 * E_cm - 0.157 * E_cm ** 2 + 598.0
    return np.maximum(0.0, B_g) / 1000.0


def _load_morris_data(site: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load digitised Morris et al. (2013) data for one site from the CSV.

    Returns (elevation_m_navd88, biomass_kg_m2, se_kg_m2) arrays.
    Elevation is in m NAVD88 (same coordinate as the model surface elevation).
    """
    csv_path = _THIS_DIR / "morris2013_biomass_data.csv"
    elev_cm, bio_g, se_g = [], [], []
    with open(csv_path, newline="") as fh:
        data_lines = (line for line in fh if not line.startswith("#"))
        for row in csv.DictReader(data_lines):
            if row["site"] == site:
                elev_cm.append(float(row["elevation_cm_navd88"]))
                bio_g.append(float(row["biomass_g_m2"]))
                se_g.append(float(row["biomass_se_g_m2"]))
    elev_m_navd88 = np.array(elev_cm) / 100.0
    return elev_m_navd88, np.array(bio_g) / 1000.0, np.array(se_g) / 1000.0


# ---------------------------------------------------------------------------
# RMSE helpers
# ---------------------------------------------------------------------------

def _model_bio_at_data_elev(
    target: List[float],
    peaks: List[float],
    data_elev: np.ndarray,
) -> np.ndarray:
    """Interpolate model peak-ABG curve to data point elevations."""
    return np.interp(data_elev, target, peaks)


def _compute_rmse(
    predicted: np.ndarray,
    observed: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """RMSE in kg m⁻², optionally restricted to mask (bool array)."""
    if mask is not None:
        predicted = predicted[mask]
        observed  = observed[mask]
    return float(np.sqrt(np.mean((predicted - observed) ** 2)))


def _outlier_mask(data_elev: np.ndarray) -> np.ndarray:
    """Boolean mask True = NOT a low outlier (i.e. include in robust RMSE)."""
    mask = np.ones(len(data_elev), dtype=bool)
    for elev_m in _LOW_OUTLIER_ELEV_M:
        mask &= ~np.isclose(data_elev, elev_m, atol=0.01)
    return mask


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot(
    elev_at_peak: List[float],
    peak_abg: List[float],
    target_elev: List[float],
    tides,
    save: bool,
    elev_at_peak_high: Optional[List[float]] = None,
    peak_abg_high: Optional[List[float]] = None,
) -> None:
    ni_elev, ni_bio, ni_se = _load_morris_data("north_inlet")
    mask = _outlier_mask(ni_elev)

    # ---- RMSE table ----
    par_bio = _miller_parabola_kg_m2(ni_elev)
    mod_bio = _model_bio_at_data_elev(
        [float(e) for e in target_elev],
        [float(b) for b in peak_abg],
        ni_elev,
    )
    rmse_par     = _compute_rmse(par_bio, ni_bio)
    rmse_par_rob = _compute_rmse(par_bio, ni_bio, mask)
    rmse_mod     = _compute_rmse(mod_bio, ni_bio)
    rmse_mod_rob = _compute_rmse(mod_bio, ni_bio, mask)

    rmse_high = rmse_high_rob = None
    if elev_at_peak_high is not None and peak_abg_high is not None:
        mod_bio_high = _model_bio_at_data_elev(
            [float(e) for e in target_elev],
            [float(b) for b in peak_abg_high],
            ni_elev,
        )
        rmse_high     = _compute_rmse(mod_bio_high, ni_bio)
        rmse_high_rob = _compute_rmse(mod_bio_high, ni_bio, mask)

    fig, ax = plt.subplots(figsize=(11, 6.0))

    # ---- tidal datum shading (m NAVD88) ----
    msl_m = tides.mean_sea_level_m    # 0.108 m NAVD88, Apr-Sep 2021 growing-season MSL
    mhw_m = tides.mean_high_tide_m    # 0.625 m NAVD88
    ax.axvspan(msl_m, mhw_m, color="lightblue", alpha=0.18, zorder=0,
               label=f"Intertidal zone (MSL – MHW)")
    ax.axvline(msl_m, color="#4393c3", linewidth=1.4, linestyle="--", zorder=2,
               label=f"MSL = {msl_m:.3f} m NAVD88")
    ax.axvline(mhw_m, color="#2166ac", linewidth=1.4, linestyle="--", zorder=2,
               label=f"MHW = {mhw_m:.3f} m NAVD88")

    # ---- Miller (2019) parabola ----
    z_ref = np.linspace(ELEV_MIN_M - 0.05, ELEV_MAX_M + 0.05, 500)
    B_ref = _miller_parabola_kg_m2(z_ref)
    ax.plot(
        z_ref, B_ref,
        color="#d73027", linewidth=1.8, linestyle="--", zorder=3,
        label=(
            rf"Miller et al. (2019) parabola  "
            rf"[RMSE={rmse_par*1000:.0f} g m$^{{-2}}$, "
            rf"{rmse_par_rob*1000:.0f} robust]"
        ),
    )

    # ---- digitised NI data ----
    ax.errorbar(
        ni_elev, ni_bio, yerr=ni_se,
        fmt="o", color="#d73027", markerfacecolor="#d73027",
        markersize=6, linewidth=1.2, capsize=3, capthick=1.2,
        zorder=5, label="Morris et al. (2013) NI data (2008)",
    )
    # mark the 3 low outliers
    ax.plot(
        ni_elev[~mask], ni_bio[~mask],
        "v", color="#d73027", markersize=9, markerfacecolor="none",
        markeredgewidth=1.5, zorder=6,
        label="Low-outlier plots (excluded from robust RMSE)",
    )

    # ---- model: default K = 0.95 ----
    ax.plot(
        elev_at_peak, peak_abg,
        color="#1a9641", linewidth=2.2, marker="s", markersize=5,
        markerfacecolor="white", markeredgecolor="#1a9641", markeredgewidth=1.5,
        zorder=4,
        label=(
            rf"Model  $K$={CAPACITY_KG_M2:.2f} kg m$^{{-2}}$  "
            rf"[RMSE={rmse_mod*1000:.0f}, "
            rf"{rmse_mod_rob*1000:.0f} robust]"
        ),
    )

    # ---- model: high K = 1.20 ----
    if elev_at_peak_high is not None and peak_abg_high is not None:
        ax.plot(
            elev_at_peak_high, peak_abg_high,
            color="#f46d43", linewidth=2.0, marker="^", markersize=5,
            markerfacecolor="white", markeredgecolor="#f46d43", markeredgewidth=1.5,
            linestyle="-.", zorder=4,
            label=(
                rf"Model  $K$={CAPACITY_KG_M2_HIGH:.2f} kg m$^{{-2}}$  "
                rf"[RMSE={rmse_high*1000:.0f}, "
                rf"{rmse_high_rob*1000:.0f} robust]"
            ),
        )

    ax.set_xlabel("Surface elevation (m NAVD88)", fontsize=11)
    ax.set_ylabel("Peak aboveground biomass (kg m⁻²)", fontsize=11)
    ax.set_xlim(ELEV_MIN_M - 0.05, ELEV_MAX_M + 0.05)
    ax.set_ylim(bottom=0.0, top=1.65)

    ax2 = ax.twiny()
    ax2.set_xlim(
        (ELEV_MIN_M - 0.05) * 100,
        (ELEV_MAX_M + 0.05) * 100,
    )
    ax2.set_xlabel("Elevation (cm NAVD88)", fontsize=10)

    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.set_title(
        "Peak aboveground biomass vs elevation — North Inlet S. alterniflora\n"
        f"σ_H={SIGMA_H_BEST:.2f}  "
        f"σ_R={SIGMA_R_BEST:.2f}  "
        f"LUE={LUE:.1e} gC μmol⁻¹   "
        f"SSC={NI_SSC_KG_M3*1000:.0f} mg L⁻¹  "
        f"MSL={NAVD88_BELOW_MSL_M:.3f} m NAVD88\n"
        "mixing_compaction | refractory k₀=0 | NOAA 8662245 constituents | "
        "RMSE: all 14 pts / excl. 3 low-outlier plots",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    plt.tight_layout()
    _save_or_show(fig, FIGURE_DIR / "ni_biomass_curve.png", save)


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

def _print_summary(
    target: List[float],
    elev_at_peak: List[float],
    peak_abg: List[float],
) -> None:
    print()
    hdr = f"{'Target elev (m MSL)':>20} {'Elev at peak (m MSL)':>22} {'Peak ABG (kg/m²)':>18}"
    print(hdr)
    print("-" * len(hdr))
    for t, e, b in zip(target, elev_at_peak, peak_abg):
        print(f"  {t:+.2f}{'':14}  {e:+.3f}{'':15}  {b:.3f}")


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
                   help="Display interactively instead of saving")
    return p.parse_args()


def _collect_results(results: Dict[float, Path]) -> Tuple[List[float], List[float], List[float]]:
    targets: List[float] = []
    elevs_at_peak: List[float] = []
    peaks: List[float] = []
    for target_elev in sorted(results.keys()):
        nc_path = results[target_elev]
        try:
            elev_pk, pk_abg = _read_run(nc_path)
            targets.append(target_elev)
            elevs_at_peak.append(elev_pk)
            peaks.append(pk_abg)
        except Exception as exc:
            print(f"  Warning: could not read {nc_path.name}: {exc}")
    return targets, elevs_at_peak, peaks


def main() -> None:
    args = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_HIGH.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        results      = _collect_existing(OUTPUT_DIR)
        results_high = _collect_existing(OUTPUT_DIR_HIGH)
        if not results:
            print("No existing outputs found — run without --plot-only first.")
            return
        print(f"Found {len(results)} low-K run(s), {len(results_high)} high-K run(s).")
    else:
        print(f"=== K = {CAPACITY_KG_M2:.2f} kg m⁻² (default) ===")
        results = _run_all(args.binary, force=args.force,
                           capacity_kg_m2=CAPACITY_KG_M2, out_dir=OUTPUT_DIR)
        print(f"=== K = {CAPACITY_KG_M2_HIGH:.2f} kg m⁻² (high, for comparison) ===")
        results_high = _run_all(args.binary, force=args.force,
                                capacity_kg_m2=CAPACITY_KG_M2_HIGH, out_dir=OUTPUT_DIR_HIGH)

    if not results:
        print("No outputs to plot.")
        return

    print("Reading outputs ...")
    targets, elevs_at_peak, peaks = _collect_results(results)
    targets_h, elevs_high, peaks_high = _collect_results(results_high)

    _print_summary(targets, elevs_at_peak, peaks)

    tides = north_inlet_default_tides()
    save  = not args.no_save
    print("Plotting ...")
    _plot(
        elevs_at_peak, peaks, targets, tides, save,
        elev_at_peak_high=elevs_high if elevs_high else None,
        peak_abg_high=peaks_high if peaks_high else None,
    )


if __name__ == "__main__":
    main()
