#!/usr/bin/env python3
"""
ni_biomass_rmse_scan.py

Grid scan of (K, sigma_H, sigma_R) to minimise RMSE against the 14 Morris et al.
(2013) North Inlet S. alterniflora biomass–elevation data points.

Reports RMSE for the Miller (2019) parabola and for each model combination,
then shows the best-fit parameters.

Usage
-----
    python sensitivity/ni_biomass_rmse_scan.py --binary ./build/marsh_cli
    python sensitivity/ni_biomass_rmse_scan.py --binary ./build/marsh_cli --force
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError as exc:
    raise ImportError("matplotlib is required: pip install matplotlib") from exc

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


# ---------------------------------------------------------------------------
# Fixed scan settings (same as test_biomass_curve_ni.py)
# ---------------------------------------------------------------------------

LUE            = 1.6e-6
NI_SSC_KG_M3  = 0.006
NI_SLR_M_YR   = 0.003
NI_DISTANCE_M  = 20.0
NAVD88_BELOW_MSL_M = 0.108   # growing-season MSL offset above NAVD88 (Apr-Sep 2021)
INITIAL_BIOMASS_KG_M2 = 0.3

OUTPUT_DIR = _THIS_DIR / "runs" / "ni_rmse_scan_v2"

_N_LAYERS        = 20
_LAYER_THICKNESS = 0.05
_POROSITY        = 0.60
_ORGANIC_EFOLDING = 0.075
_COMPACTION_OFFSET_M = 0.0   # mixing_compaction: near-zero net column height change

_RHO_SAND   = 2650.0
_RHO_REFRAC = 1400.0
_RHO_LABILE = 1200.0
_RHO_ROOTS  = 1100.0


# ---------------------------------------------------------------------------
# Parameter grid to scan
# ---------------------------------------------------------------------------

K_GRID      = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20]  # kg m-2 (aboveground)
SIGMA_H_GRID = [0.12, 0.15, 0.18, 0.21, 0.25]   # high-elevation (dry-side) half-Gaussian
SIGMA_R_GRID = [0.18, 0.20, 0.22, 0.25, 0.28, 0.32]  # low-elevation (wet-side) half-Gaussian


# ---------------------------------------------------------------------------
# Morris 2013 data
# ---------------------------------------------------------------------------

def _load_morris_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    csv_path = _THIS_DIR / "morris2013_biomass_data.csv"
    elev_cm, bio_g, se_g = [], [], []
    with open(csv_path, newline="") as fh:
        data_lines = (line for line in fh if not line.startswith("#"))
        for row in csv.DictReader(data_lines):
            if row["site"] == "north_inlet":
                elev_cm.append(float(row["elevation_cm_navd88"]))
                bio_g.append(float(row["biomass_g_m2"]))
                se_g.append(float(row["biomass_se_g_m2"]))
    elev_m_navd88 = np.array(elev_cm) / 100.0
    return elev_m_navd88, np.array(bio_g) / 1000.0, np.array(se_g) / 1000.0


# ---------------------------------------------------------------------------
# Parabola RMSE
# ---------------------------------------------------------------------------

def _miller_parabola_kg_m2(z_navd88_m: np.ndarray) -> np.ndarray:
    E_cm = z_navd88_m * 100.0
    B_g  = 14.8 * E_cm - 0.157 * E_cm ** 2 + 598.0
    return np.maximum(0.0, B_g) / 1000.0


def _parabola_rmse(data_elev: np.ndarray, data_bio: np.ndarray) -> float:
    predicted = _miller_parabola_kg_m2(data_elev)
    return float(np.sqrt(np.mean((predicted - data_bio) ** 2)))


# ---------------------------------------------------------------------------
# Column / YAML helpers
# ---------------------------------------------------------------------------

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
    return list(reversed(layers))


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


def _run_id(k: float, sh: float, sr: float, elev: float) -> str:
    sign = "p" if elev >= 0 else "m"
    return (
        f"scan_k{k:.3f}_sh{sh:.3f}_sr{sr:.3f}_{sign}{abs(elev):.2f}"
        .replace(".", "_")
    )


def _write_and_run(
    k: float, sh: float, sr: float,
    target_elev: float,
    cli_binary: str,
    force: bool,
) -> float:
    """Return peak second-half ABG for one (k, sh, sr, elev) combination."""
    # Ensure Python floats — numpy.float64 is serialised as !!binary by yaml.dump
    target_elev = float(target_elev)
    k = float(k); sh = float(sh); sr = float(sr)

    rid = _run_id(k, sh, sr, target_elev)
    yaml_path = OUTPUT_DIR / f"{rid}.yaml"
    nc_path   = OUTPUT_DIR / f"{rid}.nc"

    if not force and nc_path.exists():
        return _read_peak_abg(nc_path)

    tides = north_inlet_default_tides()
    tides.suspended_sediment_concentration_kg_m3 = NI_SSC_KG_M3
    tides.fine_sediment_concentration_kg_m3      = NI_SSC_KG_M3 * 0.25

    pre_compaction_elev_m = target_elev + _COMPACTION_OFFSET_M

    parameters = dict(_DEFAULT_PARAMETERS)
    parameters.update(_tidal_params(tides))
    parameters["deposition_distance_from_edge_m"] = NI_DISTANCE_M
    parameters.setdefault("deposition_basin_length_m", 50.0)
    parameters.setdefault("deposition_length_scale_beta", 3.0)
    parameters["vegetation_lue_gC_per_umol"]                  = LUE
    parameters["vegetation_aboveground_capacity_kg_m2"]        = k
    parameters["vegetation_belowground_capacity_kg_m2"]        = k * 2.0
    parameters["vegetation_hydroperiod_sigma_fraction"]        = sh
    parameters["vegetation_inundation_sigma_fraction"]         = sr

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
        "forcing": {"generated": build_generated_forcing_block(
            PlotConfig(
                site_name="north_inlet",
                plot_id=rid,
                surface_elevation_m=pre_compaction_elev_m,
                distance_from_creek_m=NI_DISTANCE_M,
                tides=tides,
                met=north_inlet_default_met(),
                n_years=1,
                sea_level_rise_m_yr=NI_SLR_M_YR,
                output_dir=str(OUTPUT_DIR),
            )
        )},
        "initial_state": {"layers": initial_layers},
        "initial_ecohydrology_state": {
            "root_zone_salinity_ppt":    tides.creek_salinity_ppt,
            "aboveground_biomass_kg_m2": INITIAL_BIOMASS_KG_M2,
            "belowground_biomass_kg_m2": INITIAL_BIOMASS_KG_M2,
            "lai": round(INITIAL_BIOMASS_KG_M2 * parameters["vegetation_lai_per_kg_m2"], 4),
            "litter_kg_m2": 0.0,
        },
        "output": {
            "file": str(nc_path),
            "write_time_series": True,
            "write_column_snapshots": False,
        },
    }

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as fh:
        yaml.dump(doc, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)

    try:
        run_model(str(yaml_path), cli_binary=cli_binary, silent=True)
    except RuntimeError as exc:
        print(f"    ERROR {rid}: {exc}")
        return np.nan

    return _read_peak_abg(nc_path)


def _read_peak_abg(nc_path: Path) -> float:
    with nc4.Dataset(str(nc_path), "r") as ds:
        abg = np.array(ds["aboveground_biomass_kg_m2"][:], dtype=np.float64)
    if abg.size == 0:
        return np.nan
    half = max(1, len(abg) // 2)
    return float(np.max(abg[half:]))


# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------

def _scan(cli_binary: str, force: bool) -> tuple:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data_elev, data_bio, data_se = _load_morris_data()
    n_data = len(data_elev)

    parabola_rmse = _parabola_rmse(data_elev, data_bio)
    print(f"Morris 2013 NI data points loaded: {n_data}")
    print(f"Miller (2019) parabola RMSE = {parabola_rmse*1000:.1f} g m⁻²")
    print()

    combos = list(itertools.product(K_GRID, SIGMA_H_GRID, SIGMA_R_GRID))
    n_runs = len(combos) * n_data
    print(f"Scanning {len(combos)} parameter combinations × {n_data} elevations = {n_runs} runs")
    print(f"  K values:      {K_GRID}")
    print(f"  sigma_H values: {SIGMA_H_GRID}")
    print(f"  sigma_R values: {SIGMA_R_GRID}")
    print()

    results = []  # (rmse, k, sh, sr, model_bio_array)

    for i, (k, sh, sr) in enumerate(combos, 1):
        print(f"  [{i:3d}/{len(combos)}] K={k:.2f}  σ_H={sh:.2f}  σ_R={sr:.2f} ...", end="", flush=True)
        model_bio = np.full(n_data, np.nan)
        for j, elev in enumerate(data_elev):
            model_bio[j] = _write_and_run(k, sh, sr, elev, cli_binary, force)

        valid = ~np.isnan(model_bio)
        if valid.sum() < n_data:
            print(f"  WARNING: {n_data - valid.sum()} runs failed")
        rmse = float(np.sqrt(np.mean((model_bio[valid] - data_bio[valid]) ** 2)))
        results.append((rmse, k, sh, sr, model_bio))
        print(f"  RMSE = {rmse*1000:.1f} g m⁻²")

    results.sort(key=lambda r: r[0])

    print()
    print("=" * 72)
    print("TOP 10 PARAMETER COMBINATIONS BY RMSE")
    print("=" * 72)
    print(f"{'Rank':>4}  {'K (kg/m²)':>10}  {'σ_H':>6}  {'σ_R':>6}  {'RMSE (g/m²)':>12}")
    print("-" * 44)
    for rank, (rmse, k, sh, sr, _) in enumerate(results[:10], 1):
        marker = " ← best" if rank == 1 else ""
        print(f"  {rank:2d}    {k:8.2f}    {sh:5.2f}    {sr:5.2f}    {rmse*1000:10.1f}{marker}")

    print()
    print(f"  Miller (2019) parabola RMSE = {parabola_rmse*1000:.1f} g m⁻²")
    print(f"  Best model RMSE             = {results[0][0]*1000:.1f} g m⁻²")
    print(f"  Current params (K=0.95, σ_H=0.18, σ_R=0.25) RMSE = ", end="")
    # find current params in results
    for rmse, k, sh, sr, _ in results:
        if abs(k - 0.95) < 0.01 and abs(sh - 0.18) < 0.01 and abs(sr - 0.25) < 0.01:
            print(f"{rmse*1000:.1f} g m⁻²")
            break
    else:
        print("not in scan grid")

    print()
    print("=" * 72)
    print("BEST-FIT POINT-BY-POINT COMPARISON")
    print("=" * 72)
    best_rmse, best_k, best_sh, best_sr, best_bio = results[0]
    print(f"Parameters: K={best_k:.2f} kg m⁻²,  σ_H={best_sh:.2f},  σ_R={best_sr:.2f}")
    print()
    print(f"{'Elev (m MSL)':>13}  {'Data (kg/m²)':>13}  {'±SE':>7}  {'Model (kg/m²)':>14}  {'SE offset':>10}")
    print("-" * 64)
    for elev, bio, se, mbio in zip(data_elev, data_bio, data_se, best_bio):
        offset = (mbio - bio) / se if se > 0 else float("nan")
        flag = " ✓" if abs(offset) <= 2 else " ✗"
        print(f"  {elev:+.2f}         {bio:8.3f}       {se:6.3f}    {mbio:8.3f}         {offset:+6.1f}σ{flag}")

    return results, parabola_rmse


# ---------------------------------------------------------------------------
# Parameter-space plot
# ---------------------------------------------------------------------------

def _plot_parameter_space(results: list, parabola_rmse: float) -> None:
    """Produce a 3-panel figure showing RMSE across the parameter grid.

    Panel 1: RMSE vs K (line plot) at fixed sigma_H = 0.18, sigma_R = 0.25
    Panel 2: Heatmap K vs sigma_R at fixed sigma_H = 0.18
    Panel 3: Heatmap K vs sigma_H at fixed sigma_R = 0.25
    """
    # ---- organise data ----
    rmse_arr  = np.array([r[0] for r in results]) * 1000.0   # g m-2
    k_arr     = np.array([r[1] for r in results])
    sh_arr    = np.array([r[2] for r in results])
    sr_arr    = np.array([r[3] for r in results])

    FIXED_SH = 0.18
    FIXED_SR = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    cmap = "RdYlGn_r"

    # ─── Panel 1: RMSE vs K (fixed SH=0.18, SR=0.25) ───
    ax = axes[0]
    mask_line = np.isclose(sh_arr, FIXED_SH) & np.isclose(sr_arr, FIXED_SR)
    k_line   = k_arr[mask_line]
    rmse_line = rmse_arr[mask_line]
    order = np.argsort(k_line)
    ax.plot(k_line[order], rmse_line[order], "o-", color="#2166ac",
            linewidth=2.0, markersize=7, markerfacecolor="white",
            markeredgewidth=2, label=f"σ_H={FIXED_SH}, σ_R={FIXED_SR}")
    ax.axhline(parabola_rmse * 1000, color="#d73027", linewidth=1.5,
               linestyle="--", label=f"Parabola RMSE = {parabola_rmse*1000:.0f} g m⁻²")
    ax.set_xlabel("Aboveground capacity K (kg m⁻²)", fontsize=10)
    ax.set_ylabel("RMSE (g m⁻²)", fontsize=10)
    ax.set_title(f"RMSE vs K\n(σ_H={FIXED_SH}, σ_R={FIXED_SR})", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.5)

    # ─── Panel 2: heatmap K vs sigma_R at fixed SH ───
    ax = axes[1]
    mask_sh = np.isclose(sh_arr, FIXED_SH)
    k_vals  = sorted(set(k_arr[mask_sh]))
    sr_vals = sorted(set(sr_arr[mask_sh]))
    grid_ksr = np.full((len(k_vals), len(sr_vals)), np.nan)
    for i, k in enumerate(k_vals):
        for j, sr in enumerate(sr_vals):
            sel = mask_sh & np.isclose(k_arr, k) & np.isclose(sr_arr, sr)
            if sel.any():
                grid_ksr[i, j] = rmse_arr[sel][0]

    im = ax.pcolormesh(sr_vals, k_vals, grid_ksr, cmap=cmap, shading="nearest",
                       vmin=rmse_arr.min(), vmax=min(350, rmse_arr.max()))
    fig.colorbar(im, ax=ax, label="RMSE (g m⁻²)")
    ax.set_xlabel("σ_R (low-elevation stressor width)", fontsize=10)
    ax.set_ylabel("K (kg m⁻²)", fontsize=10)
    ax.set_title(f"RMSE: K × σ_R  (σ_H={FIXED_SH})", fontsize=9)
    # mark best
    best_idx = np.argmin(grid_ksr)
    bi, bj = np.unravel_index(best_idx, grid_ksr.shape)
    ax.plot(sr_vals[bj], k_vals[bi], "*", color="black", markersize=14, zorder=5,
            label=f"Best: K={k_vals[bi]:.2f}, σ_R={sr_vals[bj]:.2f}")
    ax.axhline(0.95, color="white", linewidth=1.5, linestyle=":", alpha=0.7)
    ax.axvline(FIXED_SR, color="white", linewidth=1.5, linestyle=":", alpha=0.7,
               label=f"σ_R = {FIXED_SR}")
    ax.legend(fontsize=7, loc="upper right")

    # ─── Panel 3: heatmap K vs sigma_H at fixed SR ───
    ax = axes[2]
    mask_sr = np.isclose(sr_arr, FIXED_SR)
    k_vals2  = sorted(set(k_arr[mask_sr]))
    sh_vals  = sorted(set(sh_arr[mask_sr]))
    grid_ksh = np.full((len(k_vals2), len(sh_vals)), np.nan)
    for i, k in enumerate(k_vals2):
        for j, sh in enumerate(sh_vals):
            sel = mask_sr & np.isclose(k_arr, k) & np.isclose(sh_arr, sh)
            if sel.any():
                grid_ksh[i, j] = rmse_arr[sel][0]

    im2 = ax.pcolormesh(sh_vals, k_vals2, grid_ksh, cmap=cmap, shading="nearest",
                        vmin=rmse_arr.min(), vmax=min(350, rmse_arr.max()))
    fig.colorbar(im2, ax=ax, label="RMSE (g m⁻²)")
    ax.set_xlabel("σ_H (high-elevation stressor width)", fontsize=10)
    ax.set_ylabel("K (kg m⁻²)", fontsize=10)
    ax.set_title(f"RMSE: K × σ_H  (σ_R={FIXED_SR})", fontsize=9)
    best_idx2 = np.argmin(grid_ksh)
    bi2, bj2 = np.unravel_index(best_idx2, grid_ksh.shape)
    ax.plot(sh_vals[bj2], k_vals2[bi2], "*", color="black", markersize=14, zorder=5,
            label=f"Best: K={k_vals2[bi2]:.2f}, σ_H={sh_vals[bj2]:.2f}")
    ax.axhline(0.95, color="white", linewidth=1.5, linestyle=":", alpha=0.7)
    ax.axvline(FIXED_SH, color="white", linewidth=1.5, linestyle=":", alpha=0.7,
               label=f"σ_H = {FIXED_SH}")
    ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        "RMSE parameter space — S. alterniflora biomass–elevation (Morris et al. 2013 NI)\n"
        f"Parabola RMSE = {parabola_rmse*1000:.0f} g m⁻²  |  "
        "★ = global RMSE minimum  |  dotted lines = default values",
        fontsize=9,
    )
    plt.tight_layout()
    out_path = _THIS_DIR / "figures" / "ni_biomass_rmse_parameter_space.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--binary", default="marsh_cli",
                   help="Path to marsh_cli executable")
    p.add_argument("--force", action="store_true",
                   help="Re-run all simulations even if outputs exist")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    results, parabola_rmse = _scan(args.binary, args.force)
    print("\nGenerating parameter-space plot ...")
    _plot_parameter_space(results, parabola_rmse)


if __name__ == "__main__":
    main()
