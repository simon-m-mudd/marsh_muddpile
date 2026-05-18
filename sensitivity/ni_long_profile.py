#!/usr/bin/env python3
"""
ni_long_profile.py

500-year marsh column simulations starting from a sand pile at MHW elevation.
Four SLR scenarios: 2, 3, 4, 5 mm/yr.  SSC = 10 mg/L.  Seed biomass = 0.1 kg/m².

Marker horizon analysis: the surface elevation at year 470 is the emplacement
elevation.  At year 500 the marker's absolute position is identified from
layer ages in the final column snapshot (layers deposited before year 470 have
age > 30 yr in the final snapshot).  Subsidence rate = (emplacement elevation
− marker elevation at year 500) / 30 yr.

Usage
-----
    python sensitivity/ni_long_profile.py --binary ./build/marsh_cli
    python sensitivity/ni_long_profile.py --plot-only
    python sensitivity/ni_long_profile.py --force

Output
------
    sensitivity/runs/ni_long_profile/
    sensitivity/figures/ni_long_profile.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

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
    raise ImportError("netCDF4 required: pip install netCDF4") from exc

try:
    import matplotlib.pyplot as plt
    import matplotlib
except ImportError as exc:
    raise ImportError("matplotlib required: pip install matplotlib") from exc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SLR_RATES_M_YR  = [0.002, 0.003, 0.004, 0.005]
RUN_YEARS       = 300
MARKER_YEAR     = 270              # year marker is emplaced
INITIAL_ELEV_M  = 0.625            # MHW at North Inlet (m NAVD88)
SSC_KG_M3       = 0.010            # 10 mg/L
SEED_BIOMASS    = 0.1              # kg/m²
NI_DISTANCE_M   = 30.0

SIGMA_H_BEST    = 0.25
SIGMA_R_BEST    = 0.22
CAPACITY_KG_M2  = 0.95
LUE             = 1.6e-6

# Age of marker layers in the final snapshot (days)
_MARKER_AGE_DAYS = (RUN_YEARS - MARKER_YEAR) * 365.25

# Initial column uses build_initial_column_layers (20 layers × 0.05 m, organic/silt
# profile calibrated for mixing_compaction).  This is the standard starting
# condition for North Inlet; the 500-year run builds the full profile above it.

OUTPUT_DIR = _THIS_DIR / "runs" / "ni_long_profile"
FIGURE_DIR = _THIS_DIR / "figures"

COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

ANIM_FPS        = 15
ANIM_BIN_M      = 0.01     # 1 cm elevation bins for animation
ANIM_STEP_YR    = 3        # one frame every N years


# ---------------------------------------------------------------------------
# YAML construction
# ---------------------------------------------------------------------------

def _run_id(slr_m_yr: float) -> str:
    return "ni_lp_slr%d" % round(slr_m_yr * 1000)


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


def _write_yaml(slr_m_yr: float, yaml_path: Path) -> None:
    tides = north_inlet_default_tides()
    tides.suspended_sediment_concentration_kg_m3 = SSC_KG_M3
    tides.fine_sediment_concentration_kg_m3      = SSC_KG_M3 * 0.25

    config = PlotConfig(
        site_name="north_inlet",
        plot_id=_run_id(slr_m_yr),
        surface_elevation_m=INITIAL_ELEV_M,
        distance_from_creek_m=NI_DISTANCE_M,
        tides=tides,
        met=north_inlet_default_met(),
        n_years=RUN_YEARS,
        sea_level_rise_m_yr=slr_m_yr,
        output_dir=str(OUTPUT_DIR),
    )
    config.initial_salinity_ppt = tides.creek_salinity_ppt

    parameters = dict(_DEFAULT_PARAMETERS)
    parameters["layer_merging_enable"] = 0.0   # disabled: merging causes sawtooth artifacts
    parameters.update(_tidal_params(tides))
    parameters["deposition_distance_from_edge_m"]      = NI_DISTANCE_M
    parameters.setdefault("deposition_basin_length_m",    50.0)
    parameters.setdefault("deposition_length_scale_beta", 3.0)
    parameters["vegetation_lue_gC_per_umol"]            = LUE
    parameters["vegetation_aboveground_capacity_kg_m2"] = CAPACITY_KG_M2
    parameters["vegetation_belowground_capacity_kg_m2"] = CAPACITY_KG_M2 * 2.0
    parameters["vegetation_hydroperiod_sigma_fraction"] = SIGMA_H_BEST
    parameters["vegetation_inundation_sigma_fraction"]  = SIGMA_R_BEST

    doc = {
        "simulation": {
            "start_year": 0,
            "end_year":   RUN_YEARS,
            "dt_days":    365.0,
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
        "materials":  _DEFAULT_MATERIALS,
        "forcing":    {"generated": build_generated_forcing_block(config)},
        "initial_state": {
            "equilibrate_surface_m": INITIAL_ELEV_M,
            "uniform_column": {
                "n_layers":          100,
                "total_thickness_m": 1.0,
                "fill_material":     "sand",
                "porosity":          0.40,
            },
        },
        "initial_ecohydrology_state": {
            "root_zone_salinity_ppt":    config.initial_salinity_ppt,
            "aboveground_biomass_kg_m2": SEED_BIOMASS,
            "belowground_biomass_kg_m2": SEED_BIOMASS,
            "lai": round(SEED_BIOMASS * parameters["vegetation_lai_per_kg_m2"], 4),
            "litter_kg_m2": 0.0,
        },
        "output": {
            "file":                    str(yaml_path).replace(".yaml", ".nc"),
            "write_time_series":       True,
            "write_column_snapshots":  True,
            "snapshot_every_n_steps":  36,   # every 3 years
        },
    }

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as fh:
        yaml.dump(doc, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Run management
# ---------------------------------------------------------------------------

def _run_all(cli_binary: str, force: bool) -> Dict[float, Path]:
    results: Dict[float, Path] = {}
    n = len(SLR_RATES_M_YR)
    for i, slr in enumerate(SLR_RATES_M_YR, 1):
        rid       = _run_id(slr)
        yaml_path = OUTPUT_DIR / ("%s.yaml" % rid)
        nc_path   = OUTPUT_DIR / ("%s.nc"   % rid)
        mm        = round(slr * 1000)
        if not force and nc_path.exists():
            print("  [%d/%d] SLR=%d mm/yr: cached." % (i, n, mm))
        else:
            print("  [%d/%d] SLR=%d mm/yr ..." % (i, n, mm))
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            _write_yaml(slr, yaml_path)
            try:
                run_model(str(yaml_path), cli_binary=cli_binary, silent=False)
            except RuntimeError as exc:
                print("    ERROR: %s" % exc)
                continue
        if nc_path.exists():
            results[slr] = nc_path
    return results


def _collect_existing() -> Dict[float, Path]:
    results: Dict[float, Path] = {}
    for slr in SLR_RATES_M_YR:
        nc_path = OUTPUT_DIR / ("%s.nc" % _run_id(slr))
        if nc_path.exists():
            results[slr] = nc_path
    return results


# ---------------------------------------------------------------------------
# Profile extraction
# ---------------------------------------------------------------------------

def _decode_mat_names(raw) -> List[str]:
    names = []
    for row in raw:
        chars = []
        for c in row:
            if hasattr(c, "decode"):
                ch = c.decode()
            elif isinstance(c, str):
                ch = c
            else:
                break   # masked or None
            if ch == "\x00":
                break
            chars.append(ch)
        names.append("".join(chars).strip())
    return names


def _read_profile(nc_path: Path, slr_m_yr: float) -> Optional[dict]:
    with nc4.Dataset(str(nc_path)) as ds:
        t       = np.array(ds["model_time_days"][:],              dtype=np.float64)
        surf_ts = np.array(ds["surface_elevation"][:],            dtype=np.float64)
        abg_ts  = np.array(ds["aboveground_biomass_kg_m2"][:],    dtype=np.float64)
        blg_ts  = np.array(ds["belowground_biomass_kg_m2"][:],    dtype=np.float64)
        if_ts   = np.array(ds["inundation_fraction"][:],          dtype=np.float64)
        mid_ts  = np.array(ds["mean_inundation_depth_m"][:],      dtype=np.float64)
        mass_ts = np.array(ds["total_mass_by_material"][:],       dtype=np.float64)

        mat_names = _decode_mat_names(ds["material_name"][:])
        sand_i  = mat_names.index("sand")
        lab_i   = mat_names.index("labile_organic")
        ref_i   = mat_names.index("refractory_organic")
        root_i  = mat_names.index("roots")
        silt_indices = [i for i, n in enumerate(mat_names)
                        if "silt" in n.lower() or n.lower() == "fine_silt"]

        # surface elevation at marker emplacement (year 470)
        idx_470       = int(np.argmin(np.abs(t - MARKER_YEAR * 365.25)))
        elev_at_marker = float(surf_ts[idx_470])

        # final column snapshot
        snap_t  = np.array(ds["snapshot_time_days"][:], dtype=np.float64)
        fi      = int(np.argmax(snap_t))
        n_lay   = int(ds["snapshot_n_layers"][fi])

        thick    = np.array(ds["layer_thickness"][fi,    :n_lay],    dtype=np.float64)
        top_elev = np.array(ds["layer_top_elevation"][fi, :n_lay],   dtype=np.float64)
        age      = np.array(ds["layer_age"][fi,           :n_lay],   dtype=np.float64)
        mass     = np.array(ds["layer_mass"][fi,          :n_lay, :], dtype=np.float64)

    # sort layers top → bottom
    order    = np.argsort(top_elev)[::-1]
    thick    = thick[order]
    top_elev = top_elev[order]
    age      = age[order]
    mass     = mass[order, :]

    surf_final = float(top_elev[0])   # top of highest layer = column surface

    total_mass   = mass.sum(axis=1)
    organic_mass = mass[:, lab_i] + mass[:, ref_i] + mass[:, root_i]

    # marker horizon: first layer (top→bottom) whose age exceeds _MARKER_AGE_DAYS
    marker_elev_final = None
    for k in range(n_lay):
        if age[k] > _MARKER_AGE_DAYS:
            marker_elev_final = float(top_elev[k])
            break
    if marker_elev_final is None:
        marker_elev_final = surf_final

    yrs_since_marker = RUN_YEARS - MARKER_YEAR   # 30
    subsidence_m_yr  = (elev_at_marker - marker_elev_final) / yrs_since_marker
    accretion_m_yr   = (surf_final     - elev_at_marker)    / yrs_since_marker

    # Aggregate into 1 cm depth bins (avoids noise from thousands of thin layers)
    bin_cm   = 1.0
    max_cm   = 250.0
    bins     = np.arange(0.0, max_cm + bin_cm, bin_cm)
    n_bins   = len(bins) - 1
    bin_mass_total   = np.zeros(n_bins)
    bin_mass_organic = np.zeros(n_bins)
    bin_thick        = np.zeros(n_bins)

    for k in range(n_lay):
        if thick[k] <= 0:
            continue
        depth_top = surf_final - top_elev[k]          # depth to top of layer
        depth_bot = depth_top  + thick[k]             # depth to bottom of layer
        depth_top_cm = depth_top * 100.0
        depth_bot_cm = depth_bot * 100.0

        for b in range(n_bins):
            lo = bins[b]
            hi = bins[b + 1]
            # overlap between [depth_top_cm, depth_bot_cm] and [lo, hi]
            overlap = max(0.0, min(depth_bot_cm, hi) - max(depth_top_cm, lo))
            if overlap <= 0:
                continue
            frac = overlap / (depth_bot_cm - depth_top_cm)
            bin_mass_total[b]   += total_mass[k]   * frac
            bin_mass_organic[b] += organic_mass[k] * frac
            bin_thick[b]        += thick[k] * frac

    bin_centres = (bins[:-1] + bins[1:]) / 2.0
    valid = bin_thick > 1e-6
    bin_loi = np.where(valid, bin_mass_organic / np.where(bin_mass_total > 1e-9,
                                                           bin_mass_total, 1.0) * 100.0, 0.0)
    bin_loi[~valid] = np.nan

    # Root, labile, and refractory weight fractions per depth bin
    bin_mass_root   = np.zeros(n_bins)
    bin_mass_labile = np.zeros(n_bins)
    bin_mass_ref    = np.zeros(n_bins)
    for k in range(n_lay):
        if thick[k] <= 0:
            continue
        depth_top_cm = (surf_final - top_elev[k]) * 100.0
        depth_bot_cm = depth_top_cm + thick[k] * 100.0
        for b in range(n_bins):
            lo = bins[b]; hi = bins[b + 1]
            overlap = max(0.0, min(depth_bot_cm, hi) - max(depth_top_cm, lo))
            if overlap <= 0:
                continue
            frac = overlap / (depth_bot_cm - depth_top_cm)
            bin_mass_root[b]   += mass[k, root_i] * frac
            bin_mass_labile[b] += mass[k, lab_i]  * frac
            bin_mass_ref[b]    += mass[k, ref_i]  * frac

    # weight percent (mass of component / total mass in bin × 100)
    denom = np.where(bin_mass_total > 1e-9, bin_mass_total, np.nan)
    bin_root_wt_pct   = np.where(valid, bin_mass_root   / denom * 100.0, np.nan)
    bin_labile_wt_pct = np.where(valid, bin_mass_labile / denom * 100.0, np.nan)
    bin_ref_wt_pct    = np.where(valid, bin_mass_ref    / denom * 100.0, np.nan)

    # --- Annual smoothed rate time series ---
    # Compute annual totals by binning into 1-yr windows
    years_edge = np.arange(0.0, RUN_YEARS + 1.0, 1.0)
    t_yr       = t / 365.25
    n_yr       = len(years_edge) - 1

    mineral_ts = mass_ts[:, silt_indices].sum(axis=1)          # kg/m²
    organic_ts = mass_ts[:, lab_i] + mass_ts[:, ref_i]         # kg/m²  (excl. roots)

    rate_yr      = np.full(n_yr, np.nan)
    mineral_yr   = np.full(n_yr, np.nan)
    organic_yr   = np.full(n_yr, np.nan)
    abg_yr       = np.full(n_yr, np.nan)
    blg_yr       = np.full(n_yr, np.nan)
    if_yr        = np.full(n_yr, np.nan)
    mid_yr       = np.full(n_yr, np.nan)

    for yi in range(n_yr):
        lo, hi = years_edge[yi], years_edge[yi + 1]
        mask_lo = np.searchsorted(t_yr, lo)
        mask_hi = np.searchsorted(t_yr, hi)
        if mask_hi <= mask_lo:
            continue
        sl = slice(mask_lo, mask_hi)
        # elevation change over the year (m → mm)
        rate_yr[yi]    = (surf_ts[mask_hi - 1] - surf_ts[mask_lo]) * 1000.0
        # mass change over the year (kg/m²/yr)
        mineral_yr[yi] = mineral_ts[mask_hi - 1] - mineral_ts[mask_lo]
        organic_yr[yi] = organic_ts[mask_hi - 1] - organic_ts[mask_lo]
        # annual means
        abg_yr[yi] = abg_ts[sl].mean()
        blg_yr[yi] = blg_ts[sl].mean()
        if_yr[yi]  = if_ts[sl].mean()
        mid_yr[yi] = mid_ts[sl].mean()

    rate_years = (years_edge[:-1] + years_edge[1:]) / 2.0

    return {
        "slr_m_yr":           slr_m_yr,
        "depth_cm":           bin_centres,
        "loi":                bin_loi,
        "roots_wt_pct":       bin_root_wt_pct,
        "labile_wt_pct":      bin_labile_wt_pct,
        "ref_wt_pct":         bin_ref_wt_pct,
        "valid":              valid,
        "surf_final":         surf_final,
        "elev_at_marker":     elev_at_marker,
        "marker_elev_final":  marker_elev_final,
        "subsidence_m_yr":    subsidence_m_yr,
        "accretion_m_yr":     accretion_m_yr,
        "final_abg":          float(abg_ts[-1]),
        # time series (unsmoothed)
        "rate_years":         rate_years,
        "rate_mm_yr":         rate_yr,
        "mineral_kg_m2_yr":   mineral_yr,
        "organic_kg_m2_yr":   organic_yr,
        "abg_kg_m2":          abg_yr,
        "blg_kg_m2":          blg_yr,
        "inundation_fraction": if_yr,
        "mean_inundation_depth_m": mid_yr,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(profiles: List[dict], save: bool) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 8))
    ax_loi, ax_root, ax_lab, ax_ref = axes

    max_depth_cm = 200.0

    for prof, col in zip(profiles, COLORS):
        mm    = round(prof["slr_m_yr"] * 1000)
        label = "SLR %d mm/yr" % mm
        depth_cm = prof["depth_cm"]
        mask  = prof["valid"] & (depth_cm <= max_depth_cm)

        # depth at which initial sand surface lies
        init_depth_cm = (prof["surf_final"] - INITIAL_ELEV_M) * 100.0

        ax_loi.plot(prof["loi"][mask],           depth_cm[mask], color=col, lw=1.8, label=label)
        ax_root.plot(prof["roots_wt_pct"][mask],  depth_cm[mask], color=col, lw=1.8, label=label)
        ax_lab.plot(prof["labile_wt_pct"][mask],  depth_cm[mask], color=col, lw=1.8, label=label)
        ax_ref.plot(prof["ref_wt_pct"][mask],     depth_cm[mask], color=col, lw=1.8, label=label)

        for ax in (ax_loi, ax_root, ax_lab, ax_ref):
            ax.axhline(init_depth_cm, color=col, lw=0.8, ls="--", alpha=0.6)

    for ax in (ax_loi, ax_root, ax_lab, ax_ref):
        ax.invert_yaxis()
        ax.set_ylim(max_depth_cm, 0)
        ax.set_xlim(left=0)
        ax.grid(True, lw=0.3, alpha=0.4)
        ax.tick_params(labelsize=9)
        ax.set_ylabel("Depth (cm)", fontsize=10)

    ax_loi.set_xlabel("LOI (%)", fontsize=10)
    ax_loi.legend(fontsize=8)
    ax_loi.set_title("Loss on ignition vs depth", fontsize=10)

    ax_root.set_xlabel("Weight %", fontsize=10)
    ax_root.legend(fontsize=8)
    ax_root.set_title("Live root wt% vs depth", fontsize=10)

    ax_lab.set_xlabel("Weight %", fontsize=10)
    ax_lab.legend(fontsize=8)
    ax_lab.set_title("Labile organic wt% vs depth", fontsize=10)

    ax_ref.set_xlabel("Weight %", fontsize=10)
    ax_ref.legend(fontsize=8)
    ax_ref.set_title("Refractory organic wt% vs depth", fontsize=10)

    sub_str = "  ".join(
        "SLR%d → %.1f mm/yr" % (round(p["slr_m_yr"]*1000), p["subsidence_m_yr"]*1000)
        for p in profiles
    )
    fig.suptitle(
        "%d-yr NI  |  start MHW %.2f m NAVD88  |  SSC=10 mg/L  |  30 m from creek\n"
        "Marker subsidence (yr %d–%d):  %s" % (
            RUN_YEARS, INITIAL_ELEV_M, MARKER_YEAR, RUN_YEARS, sub_str),
        fontsize=9,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        out = FIGURE_DIR / "ni_long_profile.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print("Saved: %s" % out)
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Accretion rate through time
# ---------------------------------------------------------------------------

def _plot_rates(profiles: List[dict], save: bool) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    (ax_acc, ax_min, ax_org, ax_mid), (ax_if, ax_abg, ax_blg, ax_surf) = axes

    PLOT_START_YR = 5
    for prof, col in zip(profiles, COLORS):
        mm    = round(prof["slr_m_yr"] * 1000)
        label = "SLR %d mm/yr" % mm
        yrs   = prof["rate_years"]
        m     = yrs >= PLOT_START_YR

        ax_acc.plot(yrs[m], prof["rate_mm_yr"][m],              color=col, lw=1.6, label=label)
        ax_min.plot(yrs[m], prof["mineral_kg_m2_yr"][m],        color=col, lw=1.6, label=label)
        ax_org.plot(yrs[m], prof["organic_kg_m2_yr"][m],        color=col, lw=1.6, label=label)
        ax_mid.plot(yrs[m], prof["mean_inundation_depth_m"][m], color=col, lw=1.6, label=label)
        ax_if.plot( yrs[m], prof["inundation_fraction"][m],     color=col, lw=1.6, label=label)
        ax_abg.plot(yrs[m], prof["abg_kg_m2"][m],               color=col, lw=1.6, label=label)
        ax_blg.plot(yrs[m], prof["blg_kg_m2"][m],               color=col, lw=1.6, label=label)
        # surface elevation (raw annual mean, unsmoothed)
        with nc4.Dataset(str(OUTPUT_DIR / ("%s.nc" % _run_id(prof["slr_m_yr"])))) as ds:
            t_raw   = np.array(ds["model_time_days"][:], dtype=np.float64)
            surf_raw = np.array(ds["surface_elevation"][:], dtype=np.float64)
        t_yr_raw = t_raw / 365.25
        ann_surf = np.full(RUN_YEARS, np.nan)
        for yi in range(RUN_YEARS):
            lo = np.searchsorted(t_yr_raw, float(yi))
            hi = np.searchsorted(t_yr_raw, float(yi + 1))
            if hi > lo:
                ann_surf[yi] = surf_raw[lo:hi].mean()
        surf_yrs = np.arange(1, RUN_YEARS + 1)
        sm = surf_yrs >= PLOT_START_YR
        ax_surf.plot(surf_yrs[sm], ann_surf[sm], color=col, lw=1.0, label=label)

    for slr, col in zip(SLR_RATES_M_YR, COLORS):
        ax_acc.axhline(slr * 1000, color=col, lw=0.8, ls=":", alpha=0.5)

    for ax in axes.flat:
        ax.set_xlabel("Year", fontsize=10)
        ax.set_xlim(PLOT_START_YR, RUN_YEARS)
        ax.grid(True, lw=0.3, alpha=0.4)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=8)

    for ax in (ax_acc, ax_min, ax_org, ax_mid, ax_if, ax_abg, ax_blg):
        ax.set_ylim(bottom=0)

    ax_acc.set_ylabel("mm yr⁻¹",     fontsize=10)
    ax_acc.set_title("Surface accretion rate",               fontsize=10)
    ax_min.set_ylabel("kg m⁻² yr⁻¹", fontsize=10)
    ax_min.set_title("Mineral accumulation rate",            fontsize=10)
    ax_org.set_ylabel("kg m⁻² yr⁻¹", fontsize=10)
    ax_org.set_title("Organic (labile+ref) accumulation",   fontsize=10)
    ax_mid.set_ylabel("m",            fontsize=10)
    ax_mid.set_title("Mean inundation depth",                fontsize=10)
    ax_if.set_ylabel( "fraction",     fontsize=10)
    ax_if.set_title( "Inundation fraction",                  fontsize=10)
    ax_abg.set_ylabel("kg m⁻²",      fontsize=10)
    ax_abg.set_title("Aboveground biomass",                  fontsize=10)
    ax_blg.set_ylabel("kg m⁻²",      fontsize=10)
    ax_blg.set_title("Belowground biomass",                  fontsize=10)
    ax_surf.set_ylabel("m NAVD88",    fontsize=10)
    ax_surf.set_title("Surface elevation (annual mean)",     fontsize=10)

    fig.suptitle(
        "%d-yr NI  |  start MHW %.2f m NAVD88  |  SSC=10 mg/L  |  30 m from creek" % (
            RUN_YEARS, INITIAL_ELEV_M),
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        out = FIGURE_DIR / "ni_long_profile_rates.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print("Saved: %s" % out)
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(profiles: List[dict]) -> None:
    print()
    hdr = "%-10s  %-10s  %-12s  %-14s  %-14s  %-14s" % (
        "SLR(mm/yr)", "surf(m)", "elev@470(m)", "marker@500(m)",
        "subsid(mm/yr)", "accret(mm/yr)")
    print(hdr)
    print("-" * len(hdr))
    for p in profiles:
        print("%-10d  %-10.3f  %-12.3f  %-14.3f  %-14.2f  %-14.2f" % (
            round(p["slr_m_yr"] * 1000),
            p["surf_final"],
            p["elev_at_marker"],
            p["marker_elev_final"],
            p["subsidence_m_yr"] * 1000,
            p["accretion_m_yr"]  * 1000,
        ))


# ---------------------------------------------------------------------------
# Animation helpers
# ---------------------------------------------------------------------------

def _precompute_anim_data(nc_path: Path, bin_edges: np.ndarray) -> dict:
    """
    Read every column snapshot from *nc_path* and return per-snapshot
    profiles binned in absolute-elevation space (m NAVD88).

    Returns dict with keys:
        snap_years      (n_snaps,)
        loi             (n_snaps, n_bins)  LOI %
        root_wt         (n_snaps, n_bins)  root wt%
        labile_wt       (n_snaps, n_bins)  labile organic wt%
        ref_wt          (n_snaps, n_bins)  refractory organic wt%
        surf_elev       (n_snaps,)         surface elevation m NAVD88
    """
    bin_lo = bin_edges[:-1]
    bin_hi = bin_edges[1:]
    n_bins = len(bin_lo)

    with nc4.Dataset(str(nc_path)) as ds:
        mat_names = _decode_mat_names(ds["material_name"][:])
        lab_i  = mat_names.index("labile_organic")
        ref_i  = mat_names.index("refractory_organic")
        root_i = mat_names.index("roots")

        snap_t  = np.array(ds["snapshot_time_days"][:], dtype=np.float64)
        n_snaps = len(snap_t)

        loi_all    = np.full((n_snaps, n_bins), np.nan)
        root_all   = np.full((n_snaps, n_bins), np.nan)
        labile_all = np.full((n_snaps, n_bins), np.nan)
        ref_all    = np.full((n_snaps, n_bins), np.nan)
        surf_all   = np.full(n_snaps, np.nan)

        for si in range(n_snaps):
            n_lay = int(ds["snapshot_n_layers"][si])
            if n_lay == 0:
                continue
            thick  = np.array(ds["layer_thickness"][si,    :n_lay], dtype=np.float64)
            top_el = np.array(ds["layer_top_elevation"][si, :n_lay], dtype=np.float64)
            mass   = np.array(ds["layer_mass"][si,          :n_lay, :], dtype=np.float64)

            bot_el = top_el - thick
            surf_all[si] = top_el.max()

            # overlap[b, k] = fraction of layer k that falls in elevation bin b
            overlap = np.maximum(0.0,
                np.minimum(top_el[np.newaxis, :], bin_hi[:, np.newaxis]) -
                np.maximum(bot_el[np.newaxis, :], bin_lo[:, np.newaxis])
            )  # (n_bins, n_lay)
            safe_thick = np.where(thick > 0, thick, np.inf)
            frac = overlap / safe_thick[np.newaxis, :]  # (n_bins, n_lay)

            bin_mass  = frac @ mass                                  # (n_bins, n_mats)
            bin_total = bin_mass.sum(axis=1)                         # (n_bins,)
            bin_thick = (frac * thick[np.newaxis, :]).sum(axis=1)    # (n_bins,)

            valid = bin_thick > 1e-6
            denom = np.where(bin_total > 1e-9, bin_total, np.nan)

            organic = bin_mass[:, lab_i] + bin_mass[:, ref_i] + bin_mass[:, root_i]
            loi_all[si]    = np.where(valid, organic              / denom * 100.0, np.nan)
            root_all[si]   = np.where(valid, bin_mass[:, root_i]  / denom * 100.0, np.nan)
            labile_all[si] = np.where(valid, bin_mass[:, lab_i]   / denom * 100.0, np.nan)
            ref_all[si]    = np.where(valid, bin_mass[:, ref_i]   / denom * 100.0, np.nan)

    return {
        "snap_years": snap_t / 365.25,
        "loi":        loi_all,
        "root_wt":    root_all,
        "labile_wt":  labile_all,
        "ref_wt":     ref_all,
        "surf_elev":  surf_all,
    }


def _make_animation(profiles: List[dict], results: Dict[float, Path], save: bool) -> None:
    """
    Animate the growing column for all SLR scenarios simultaneously.
    Profiles are plotted in absolute elevation (m NAVD88) so the column
    grows upward.  The y-axis is fixed to match the final-frame extent.
    """
    try:
        import imageio.v3 as iio
    except ImportError:
        print("imageio required for animation: pip install imageio[pyav]")
        return

    base_elev = INITIAL_ELEV_M - 1.0    # bottom of initial sand column
    max_surf  = max(p["surf_final"] for p in profiles)

    # elevation bins: fixed across all frames
    bin_edges  = np.arange(base_elev, max_surf + ANIM_BIN_M * 2, ANIM_BIN_M)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    print("Pre-computing snapshot profiles ...")
    anim_data = []
    for prof, col in zip(profiles, COLORS):
        slr = prof["slr_m_yr"]
        nc_path = OUTPUT_DIR / ("%s.nc" % _run_id(slr))
        print("  SLR=%d mm/yr" % round(slr * 1000))
        anim_data.append(_precompute_anim_data(nc_path, bin_edges))

    # x-axis limits: max value across all scenarios and all snapshots
    xlim_loi    = (0.0, max(np.nanmax(d["loi"])    for d in anim_data))
    xlim_root   = (0.0, max(np.nanmax(d["root_wt"]) for d in anim_data))
    xlim_labile = (0.0, max(np.nanmax(d["labile_wt"]) for d in anim_data))
    xlim_ref    = (0.0, max(np.nanmax(d["ref_wt"])   for d in anim_data))
    ylim = (base_elev - 0.02, max_surf + 0.05)

    # synthetic year-0 data: initial sand column, all organics = 0
    yr0_valid = (bin_centres >= base_elev) & (bin_centres < INITIAL_ELEV_M)
    yr0_zeros = np.where(yr0_valid, 0.0, np.nan)

    # frame target years
    frame_years = np.arange(0, RUN_YEARS + ANIM_STEP_YR, ANIM_STEP_YR)

    # --- set up figure (created once, updated in place) ---
    matplotlib.use("Agg")
    fig, axes = plt.subplots(1, 4, figsize=(18, 8))
    ax_loi, ax_root, ax_lab, ax_ref = axes

    panel_info = [
        (ax_loi,  "LOI (%)",           xlim_loi,    "loi"),
        (ax_root, "Live root wt%",     xlim_root,   "root_wt"),
        (ax_lab,  "Labile organic wt%", xlim_labile, "labile_wt"),
        (ax_ref,  "Refractory org wt%", xlim_ref,   "ref_wt"),
    ]

    # MHW reference line on every panel
    for ax, xlabel, xlim, key in panel_info:
        ax.axhline(INITIAL_ELEV_M, color="grey", lw=0.7, ls="--", alpha=0.5,
                   label="Initial surface (MHW)")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Elevation (m NAVD88)", fontsize=10)
        ax.grid(True, lw=0.3, alpha=0.4)
        ax.tick_params(labelsize=9)

    ax_loi.set_title("Loss on ignition",        fontsize=10)
    ax_root.set_title("Live root wt%",           fontsize=10)
    ax_lab.set_title("Labile organic wt%",       fontsize=10)
    ax_ref.set_title("Refractory organic wt%",   fontsize=10)

    # create line objects for each scenario × panel
    lines = {}   # (scenario_idx, key) -> Line2D
    for si, (prof, col) in enumerate(zip(profiles, COLORS)):
        mm    = round(prof["slr_m_yr"] * 1000)
        label = "SLR %d mm/yr" % mm
        for ax, xlabel, xlim, key in panel_info:
            ln, = ax.plot([], [], color=col, lw=1.6, label=label)
            lines[(si, key)] = ln

    ax_loi.legend(fontsize=8, loc="upper right")

    title = fig.suptitle("Year 0", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    fig.canvas.draw()

    def _get_frame_arrays(target_yr: float):
        """Return {key: (x_arr, y_arr)} for each scenario at target_yr."""
        result = []
        for si, d in enumerate(anim_data):
            if target_yr <= 0:
                row = {k: yr0_zeros for k in ("loi", "root_wt", "labile_wt", "ref_wt")}
            else:
                snap_idx = int(np.argmin(np.abs(d["snap_years"] - target_yr)))
                row = {k: d[k][snap_idx] for k in ("loi", "root_wt", "labile_wt", "ref_wt")}
            result.append(row)
        return result

    print("Rendering %d frames at %d fps ..." % (len(frame_years), ANIM_FPS))
    frames_rgba = []
    for target_yr in frame_years:
        frame_data = _get_frame_arrays(target_yr)
        for si in range(len(anim_data)):
            for ax, xlabel, xlim, key in panel_info:
                vals = frame_data[si][key]
                mask = ~np.isnan(vals)
                ln   = lines[(si, key)]
                ln.set_data(vals[mask], bin_centres[mask])

        title.set_text("Year %d" % int(round(target_yr)))
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        rgba = np.frombuffer(buf, dtype=np.uint8).reshape(
            fig.canvas.get_width_height()[::-1] + (4,)
        )
        frames_rgba.append(rgba[:, :, :3].copy())   # RGB only

    plt.close(fig)

    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        out_mp4 = FIGURE_DIR / "ni_long_profile_anim.mp4"
        iio.imwrite(str(out_mp4), frames_rgba, fps=ANIM_FPS, codec="libx264")
        print("Saved: %s  (%d frames, %d fps)" % (out_mp4, len(frames_rgba), ANIM_FPS))
    else:
        # show as interactive HTML in fallback
        print("(--no-save: animation not written to disk)")


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
    p.add_argument("--animate",   action="store_true",
                   help="Generate growing-column animation (ni_long_profile_anim.mp4)")
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
        print("Running %d long-profile simulations (%d yr each) ..." % (
            len(SLR_RATES_M_YR), RUN_YEARS))
        results = _run_all(args.binary, force=args.force)

    if not results:
        print("No outputs available.")
        return

    print("Reading profiles ...")
    profiles = []
    for slr in sorted(results):
        p = _read_profile(results[slr], slr)
        if p:
            profiles.append(p)
        else:
            print("  Warning: could not read %s" % results[slr].name)

    if not profiles:
        print("No valid profiles.")
        return

    _print_summary(profiles)

    if args.animate:
        _make_animation(profiles, results, save=not args.no_save)
    else:
        print("Plotting ...")
        _plot(profiles, save=not args.no_save)
        _plot_rates(profiles, save=not args.no_save)


if __name__ == "__main__":
    main()
