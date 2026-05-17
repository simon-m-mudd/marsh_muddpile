#!/usr/bin/env python3
"""
animate_core_growth.py

Runs a single restoration (bare-sand colonisation) simulation and produces
an animation showing:

  LEFT — growing core rectangle
    • Y axis: elevation (m relative to initial MSL)
    • X axis: fractional composition at each depth, shown as filled wedges:
        sediment (yellow) | refractory C (black) | labile C (brown) | roots (green)
      stacked left → right at every depth layer
    • Horizontal dashed lines: MSL (blue) and MHW (cyan)
    • Rectangle base is fixed; top surface rises with marsh accretion

  RIGHT — three evolving time-series panels (full run as faint background;
           animated window as solid growing line; red cursor)
    • Total organic carbon (refractory + labile + roots, g m⁻²)
    • Live roots (g m⁻²)
    • Labile C and refractory C separately (g m⁻²)

  The animation renders the FIRST --first-yr years and the LAST --last-yr years
  of the simulation, concatenated.  A subtitle on the core panel indicates which
  window is active.

Usage
-----
    # Full 200-year run, animate first 10 + last 10 years (default)
    python3 viz/animate_core_growth.py

    # Custom windows
    python3 viz/animate_core_growth.py --years 200 --first-yr 20 --last-yr 20

    # Use an existing NC (skip model run)
    python3 viz/animate_core_growth.py --nc viz/core_anim_200yr.nc

Options
-------
    --years     INT    Simulation length in years (default 200)
    --first-yr  INT    Animate first N years (default 10)
    --last-yr   INT    Animate last N years (default 10)
    --fps       INT    Frames per second in output (default 12)
    --binary    PATH   Path to marsh_cli (default ./build/marsh_cli)
    --nc        PATH   Existing NC file; skip model run
    --out       PATH   Output mp4 path
    --dpi       INT    Output DPI (default 120)
"""

from __future__ import annotations
import argparse
import subprocess
from pathlib import Path

import numpy as np
import yaml

# ── paths ─────────────────────────────────────────────────────────────────────

_VIZ_DIR  = Path(__file__).parent
_ROOT_DIR = _VIZ_DIR.parent

# ── model config ──────────────────────────────────────────────────────────────

_DT_DAYS = 365.25 / 12.0   # one nominal month

_TIDAL_CONSTITUENTS = {
    "M2": {"amplitude": 0.6205, "phase_deg": 357.9},
    "S2": {"amplitude": 0.1045, "phase_deg": 20.3},
    "N2": {"amplitude": 0.1454, "phase_deg": 340.6},
    "K1": {"amplitude": 0.0867, "phase_deg": 187.6},
    "O1": {"amplitude": 0.0646, "phase_deg": 191.4},
}
_TIDAL_AMPLITUDE_M = 0.6511


def _initial_layers(start_elev_m: float = -0.75, top_elev_m: float = 0.20,
                    step_m: float = 0.05):
    layers = []
    z = start_elev_m
    while z < top_elev_m - 1e-6:
        layers.append({
            "top_elevation_m":   round(z, 4),
            "thickness_m":       step_m,
            "porosity":          0.6,
            "age_days":          0.0,
            "mass_kg_m2": {"sand": 53.0, "refractory_organic": 0.0,
                           "labile_organic": 0.0, "roots": 0.0},
        })
        z = round(z + step_m, 6)
    return layers


def build_yaml(nc_path: Path, n_years: int) -> Path:
    """Write a single-run YAML with monthly snapshots and return its path."""
    yaml_path = nc_path.with_suffix(".yaml")
    n_steps   = n_years * 12

    doc = {
        "simulation": {
            "start_year": 0,
            "end_year":   n_years,
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
            "distance_from_creek_m":  20.0,
            "creek_bank_elevation_m": 0.0,
            "local_tidal_offset_m":   0.0,
        },
        "parameters": {
            # water level
            "water_level_substeps_per_step": 240,
            "water_level_reference_time_hours": 0.0,
            # layers
            "layer_merging_enable": 1.0,
            "layer_merging_max_layers": 200.0,
            "layer_merging_minimum_unmerged_depth_m": 0.3,
            "layer_merging_target_thickness_m": 0.05,
            # surface properties
            "surface_property_active_depth_m": 0.1,
            "surface_property_fine_grain_size_threshold_m": 6.3e-5,
            "surface_property_default_grain_size_m": 2.0e-5,
            "surface_property_default_organic_carbon_fraction": 0.45,
            # salinity
            "salinity_min_ppt": 0.0, "salinity_max_ppt": 80.0,
            "salinity_initial_root_zone_ppt": 20.0,
            "salinity_default_creek_ppt": 30.0,
            "salinity_et_concentration_rate_ppt_per_mm": 0.02,
            "salinity_precipitation_dilution_rate_ppt_per_mm": 0.03,
            "salinity_freshwater_dilution_rate_ppt_per_mm": 0.03,
            "salinity_base_flushing_rate_per_day": 0.5,
            "salinity_flushing_efolding_distance_m": 50.0,
            "salinity_exchange_coarse_sensitivity": 0.75,
            "salinity_exchange_organic_sensitivity": 0.5,
            "salinity_storage_fine_sensitivity": 1.0,
            "salinity_storage_organic_sensitivity": 1.0,
            "salinity_storage_porosity_sensitivity": 0.5,
            "salinity_min_exchange_modifier": 0.1,
            "salinity_min_storage_modifier": 0.25,
            "salinity_subsurface_flushing_rate_per_day": 0.003,
            # ET
            "et_background_mm_d": 0.5,
            "et_temperature_base_c": 0.0,
            "et_temperature_coefficient_mm_d_per_c": 0.12,
            "et_par_coefficient_mm_d_per_umol_m2_d": 3.0e-8,
            "et_cover_extinction_coefficient": 0.7,
            "et_kc_min": 0.2, "et_kc_max": 1.1,
            "et_salinity_threshold_ppt": 20.0,
            "et_salinity_stress_per_ppt": 0.03,
            "et_inundation_threshold_fraction": 0.25,
            "et_inundation_stress_per_fraction": 1.2,
            "et_min_transpiration_stress": 0.05,
            "et_transpiration_fraction_min": 0.15,
            "et_transpiration_fraction_max": 0.85,
            "et_evaporation_fine_sensitivity": 0.25,
            "et_evaporation_organic_sensitivity": 0.2,
            "et_evaporation_porosity_sensitivity": 0.15,
            "et_evaporation_coarse_sensitivity": 0.15,
            "et_min_evaporation_modifier": 0.2,
            "et_biomass_to_lai_coefficient": 3.0,
            # vegetation
            "vegetation_days_per_year": 365.0,
            "vegetation_lue_gC_per_umol": 1.6e-6,
            "vegetation_carbon_use_efficiency": 0.5,
            "vegetation_biomass_carbon_fraction": 0.45,
            "vegetation_lai_light_extinction": 0.8,
            "vegetation_max_fpar": 0.95,
            "vegetation_lai_per_kg_m2": 3.0,
            "vegetation_max_lai": 6.0,
            "vegetation_temperature_optimum_c": 25.0,
            "vegetation_temperature_sigma_c": 10.0,
            "vegetation_hydroperiod_optimum_fraction": 0.25,
            "vegetation_hydroperiod_sigma_fraction": 0.18,
            "vegetation_salinity_threshold_ppt": 20.0,
            "vegetation_salinity_stress_per_ppt": 0.03,
            "vegetation_shoot_allocation_min": 0.25,
            "vegetation_shoot_allocation_max": 0.65,
            "vegetation_aboveground_capacity_kg_m2": 0.95,
            "vegetation_root_shoot_ratio": 2.0,
            "vegetation_aboveground_mortality_base_per_day": 0.001,
            "vegetation_aboveground_mortality_seasonal_amp_per_day": 0.0,
            "vegetation_mortality_peak_day": 300.0,
            "vegetation_cold_mortality_threshold_c": 20.0,
            "vegetation_cold_mortality_slope_per_day_per_c": 0.002,
            "vegetation_aboveground_hydroperiod_mortality_threshold": 0.5,
            "vegetation_aboveground_hydroperiod_mortality_slope_per_day": 0.01,
            "vegetation_aboveground_salinity_mortality_threshold_ppt": 30.0,
            "vegetation_aboveground_salinity_mortality_slope_per_day_per_ppt": 0.0005,
            "vegetation_belowground_mortality_base_per_day": 0.000274,
            "vegetation_belowground_hydroperiod_mortality_threshold": 0.6,
            "vegetation_belowground_hydroperiod_mortality_slope_per_day": 0.005,
            "vegetation_belowground_salinity_mortality_threshold_ppt": 35.0,
            "vegetation_belowground_salinity_mortality_slope_per_day_per_ppt": 0.00025,
            "vegetation_inundation_optimum_fraction": 0.25,
            "vegetation_inundation_sigma_fraction": 0.25,
            "vegetation_low_inundation_mortality_threshold": 0.05,
            "vegetation_low_inundation_mortality_slope_per_day": 0.1,
            # roots
            "root_efolding_depth_m": 0.1,
            "root_profile_normalize_to_column": 1.0,
            "root_allocation_to_labile_fraction": 0.9,
            "root_allocation_to_refractory_fraction": 0.1,
            # deposition
            "deposition_reference_biomass_g_m2": 500.0,
            "deposition_reference_flow_velocity_m_s": 0.05,
            "deposition_limiting_flow_velocity_m_s": 0.15,
            "deposition_time_fractions_per_tide": 720.0,
            "deposition_min_water_depth_m": 0.002,
            "deposition_von_karman": 0.4,
            "deposition_tke_coefficient": 0.2,
            "deposition_water_density_kg_m3": 1000.0,
            "deposition_aa": 0.46, "deposition_bb": 3.8,
            "deposition_alpha_turb": 0.9, "deposition_alpha_zero": 11.0,
            "deposition_alpha": 0.55, "deposition_beta": 0.4,
            "deposition_mu": 0.00066, "deposition_phi": 0.29,
            "deposition_kappa": 0.224, "deposition_nu_m2_s": 1.0e-5,
            "deposition_gamma_tr": 0.718, "deposition_epsilon": 2.08,
            "deposition_distance_from_edge_m": 20.0,
            "deposition_basin_length_m": 50.0,
            "deposition_length_scale_beta": 3.0,
            # decay
            "decay_reference_temperature_c": 20.0,
            "decay_temperature_factor_per_degree": 0.0,
            "decay_use_exact_solution": 1.0,
            "decomposition_modifier_global_min_multiplier": 0.05,
            "decomposition_modifier_global_max_multiplier": 2.0,
            "decomposition_labile_hydro_sensitivity": 2.5,
            "decomposition_labile_fine_sensitivity": 0.8,
            "decomposition_labile_organic_sensitivity": 0.6,
            "decomposition_labile_porosity_sensitivity": 0.4,
            "decomposition_labile_salinity_threshold_ppt": 18.0,
            "decomposition_labile_salinity_sensitivity": 0.035,
            "decomposition_refractory_hydro_sensitivity": 1.2,
            "decomposition_refractory_fine_sensitivity": 0.5,
            "decomposition_refractory_organic_sensitivity": 0.4,
            "decomposition_refractory_porosity_sensitivity": 0.2,
            "decomposition_refractory_salinity_threshold_ppt": 25.0,
            "decomposition_refractory_salinity_sensitivity": 0.015,
            "decomposition_root_hydro_sensitivity": 1.8,
            "decomposition_root_fine_sensitivity": 0.7,
            "decomposition_root_organic_sensitivity": 0.5,
            "decomposition_root_porosity_sensitivity": 0.3,
            "decomposition_root_salinity_threshold_ppt": 22.0,
            "decomposition_root_salinity_sensitivity": 0.025,
            # tidal harmonic constituents
            "water_level_constituent_M2_amplitude_m": _TIDAL_CONSTITUENTS["M2"]["amplitude"],
            "water_level_constituent_M2_phase_deg":   _TIDAL_CONSTITUENTS["M2"]["phase_deg"],
            "water_level_constituent_S2_amplitude_m": _TIDAL_CONSTITUENTS["S2"]["amplitude"],
            "water_level_constituent_S2_phase_deg":   _TIDAL_CONSTITUENTS["S2"]["phase_deg"],
            "water_level_constituent_N2_amplitude_m": _TIDAL_CONSTITUENTS["N2"]["amplitude"],
            "water_level_constituent_N2_phase_deg":   _TIDAL_CONSTITUENTS["N2"]["phase_deg"],
            "water_level_constituent_K1_amplitude_m": _TIDAL_CONSTITUENTS["K1"]["amplitude"],
            "water_level_constituent_K1_phase_deg":   _TIDAL_CONSTITUENTS["K1"]["phase_deg"],
            "water_level_constituent_O1_amplitude_m": _TIDAL_CONSTITUENTS["O1"]["amplitude"],
            "water_level_constituent_O1_phase_deg":   _TIDAL_CONSTITUENTS["O1"]["phase_deg"],
            # compaction
            "compaction_loi_intercept_percent": 0.0,
            "compaction_loi_slope_percent_per_organic_fraction": 100.0,
            "compaction_default_grain_size_m": 2.0e-5,
            "compaction_e_1_intercept": 0.5, "compaction_e_1_loi_slope": 0.04,
            "compaction_e_1_grain_size_slope": 0.0,
            "compaction_c_r_intercept": 0.005, "compaction_c_r_loi_slope": 0.0015,
            "compaction_c_r_grain_size_slope": 0.0,
            "compaction_c_c_intercept": 0.1, "compaction_c_c_loi_slope": 0.02,
            "compaction_c_c_grain_size_slope": 0.0,
            "compaction_sigma_y_intercept_kpa": 12.0,
            "compaction_sigma_y_loi_slope": -0.15,
            "compaction_sigma_y_grain_size_slope": 0.0,
            "compaction_min_e_1": 0.1, "compaction_min_c_r": 0.0001,
            "compaction_min_c_c": 0.001, "compaction_min_sigma_y_kpa": 3.0,
            "compaction_min_effective_stress_kpa": 0.01,
            "compaction_reference_stress_kpa": 1.0,
            "compaction_min_void_ratio": 0.01,
            # NH4 / CH4 (off but parameters required)
            "nh4_carbon_fraction": 0.45, "nh4_c_to_n_molar_ratio": 25.0,
            "nh4_creek_umol_L": 5.0, "nh4_tidal_flushing_rate_per_d": 0.5,
            "nh4_flushing_depth_scale_m": 0.05,
            "nh4_diffusion_coeff_m2_per_d": 1.0e-5,
            "nh4_uptake_vmax_umol_L_d_per_kg_m3": 10.0,
            "nh4_km_umol_L": 50.0, "nh4_initial_umol_L": 100.0,
            "nh4_min_umol_L": 0.0, "nh4_max_umol_L": 2000.0,
            "ch4_carbon_fraction": 0.45,
            "ch4_initial_so4_umol_L": 19600.0, "ch4_initial_ch4_umol_L": 0.0,
            "ch4_creek_so4_umol_L": 19600.0,
            "ch4_tidal_flushing_rate_per_d": 0.3,
            "ch4_flushing_depth_scale_m": 0.1,
            "ch4_km_so4_umol_L": 1000.0, "ch4_c_to_ch4": 2.0,
            "ch4_diffusion_coeff_m2_per_d": 1.0e-5,
            "ch4_so4_diffusion_coeff_m2_per_d": 1.0e-5,
            "ch4_oxidation_rate_per_d": 0.05,
            "ch4_oxidation_depth_scale_m": 0.05,
            "ch4_ebullition_threshold_umol_L": 500.0,
            "ch4_plant_transport_factor": 2.0,
            "ch4_min_umol_L": 0.0, "ch4_max_umol_L": 2000.0,
        },
        "materials": [
            {"name": "sand",  "category": "mineral", "density": 2650.0,
             "allow_surface_deposition": False,
             "settling": {"diameter": 0.0002, "settling_velocity": 0.02}},
            {"name": "silt",  "category": "mineral", "density": 2600.0,
             "allow_surface_deposition": True,
             "settling": {"diameter": 2.0e-5, "settling_velocity": 3.7e-5}},
            {"name": "fine_silt", "category": "mineral", "density": 2600.0,
             "allow_surface_deposition": True,
             "settling": {"diameter": 1.5e-5, "settling_velocity": 1.5e-4}},
            {"name": "refractory_organic", "category": "organic_refractory",
             "density": 1400.0, "allow_root_input": True,
             "decay": {"k_0": 0.0, "gamma": 2.0, "temperature_sensitive": True}},
            {"name": "labile_organic", "category": "organic_labile",
             "density": 1200.0, "allow_root_input": True,
             "decay": {"k_0": 0.01, "gamma": 0.1, "temperature_sensitive": True}},
            {"name": "roots", "category": "live_root", "density": 1100.0,
             "allow_root_input": True},
        ],
        "forcing": {
            "generated": {
                "n_steps":     n_steps,
                "dt_days":     round(_DT_DAYS, 4),
                "start_time_days": 0.0,
                "initial_mean_sea_level": 0.0,
                "sea_level_rise_rate_m_per_yr": 0.002,
                "temperature_mean_c":      18.92,
                "temperature_amplitude_c": 8.46,
                "temperature_peak_day":    202.8,
                "temperature_trend_c_per_yr": 0.0,
                "par_mean_umol_m2_d":      34172288.0,
                "par_amplitude_umol_m2_d": 14160082.0,
                "par_peak_day":            164.8,
                "par_trend_umol_m2_d_per_yr": 0.0,
                "tidal_amplitude":    _TIDAL_AMPLITUDE_M,
                "tidal_period_hours": 12.42,
                "creek_salinity_ppt": 28.0,
                "suspended_sediment_concentration": 0.02,
                "fine_sediment_concentration":      0.005,
                "precipitation_mm_d": 3.0,
                "freshwater_input_mm_d": 0.0,
                "storm_surge_residual_m": 0.0,
                "external_pb210_supply": 0.0,
            }
        },
        "initial_state": {"layers": _initial_layers()},
        "initial_ecohydrology_state": {
            "root_zone_salinity_ppt":    28.0,
            "aboveground_biomass_kg_m2": 0.001,
            "belowground_biomass_kg_m2": 0.001,
            "lai": 0.003,
            "litter_kg_m2": 0.0,
        },
        "output": {
            "file": str(nc_path),
            "write_time_series":       True,
            "write_column_snapshots":  True,
            "snapshot_every_n_steps":  1,
        },
    }

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as fh:
        yaml.dump(doc, fh, default_flow_style=False, sort_keys=False)
    print(f"  Wrote {yaml_path}")
    return yaml_path


# ── model runner ──────────────────────────────────────────────────────────────

def run_model(yaml_path: Path, binary: str) -> None:
    print(f"  Running {binary} {yaml_path.name} …")
    result = subprocess.run([binary, str(yaml_path)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-2000:])
        raise RuntimeError(f"marsh_cli failed (exit {result.returncode})")
    print("  Model run complete.")


# ── data loading ──────────────────────────────────────────────────────────────

def load_nc(nc_path: Path):
    import netCDF4 as nc4
    ds = nc4.Dataset(nc_path)

    t_days  = ds.variables["model_time_days"][:]
    surf    = ds.variables["surface_elevation"][:]
    msl     = ds.variables["mean_water_level_m"][:]
    mhw     = ds.variables["max_water_level_m"][:]
    total_m = ds.variables["total_mass_by_material"][:]

    snap_t   = ds.variables["snapshot_time_days"][:]
    n_layers = ds.variables["snapshot_n_layers"][:]
    top_elev = ds.variables["layer_top_elevation"][:]
    thick    = ds.variables["layer_thickness"][:]
    lmass    = ds.variables["layer_mass"][:]

    ds.close()
    return dict(
        t_days=np.array(t_days),
        surf=np.array(surf),
        msl=np.array(msl),
        mhw=np.array(mhw),
        total_m=np.array(total_m),
        snap_t=np.array(snap_t),
        n_layers=np.array(n_layers, dtype=int),
        top_elev=np.array(top_elev),
        thick=np.array(thick),
        lmass=np.array(lmass),
    )


# ── animation ─────────────────────────────────────────────────────────────────

_MAT_SEDIMENT    = [0, 1, 2]
_MAT_REFRACTORY  = 3
_MAT_LABILE      = 4
_MAT_ROOTS       = 5

_C_SEDIMENT   = "#E8C84A"
_C_REFRACTORY = "#1a1a1a"
_C_LABILE     = "#7B4B2A"
_C_ROOTS      = "#2E7D32"


def _core_fracs(lmass_rows, n_lay=None):
    m = lmass_rows if n_lay is None else lmass_rows[:n_lay, :]
    sed   = m[:, _MAT_SEDIMENT].sum(axis=1)
    refr  = m[:, _MAT_REFRACTORY]
    lab   = m[:, _MAT_LABILE]
    roots = m[:, _MAT_ROOTS]
    total = sed + refr + lab + roots
    total = np.where(total < 1e-12, 1.0, total)
    return np.column_stack([sed, refr, lab, roots]) / total[:, None]


def _build_frame_indices(n_total_snaps: int, n_years_total: int,
                         first_yr: int, last_yr: int) -> np.ndarray:
    """Return snapshot indices for first_yr years and last_yr years."""
    steps_per_year = n_total_snaps / n_years_total
    n_first = int(round(first_yr * steps_per_year))
    n_last  = int(round(last_yr  * steps_per_year))
    first_indices = np.arange(0,             min(n_first, n_total_snaps))
    last_indices  = np.arange(max(0, n_total_snaps - n_last), n_total_snaps)
    # Avoid double-counting if windows overlap
    combined = np.unique(np.concatenate([first_indices, last_indices]))
    return combined


def make_animation(data: dict, out_path: Path, fps: int, dpi: int,
                   frame_indices: np.ndarray, n_years_total: int,
                   first_yr: int, last_yr: int) -> None:
    import imageio_ffmpeg
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    t_days = data["t_days"]
    surf   = data["surf"]
    msl    = data["msl"]
    mhw    = data["mhw"]
    total  = data["total_m"] * 1000.0   # → g/m²
    n_total_snaps = len(data["snap_t"])

    total_c   = total[:, _MAT_REFRACTORY] + total[:, _MAT_LABILE] + total[:, _MAT_ROOTS]
    roots_ts  = total[:, _MAT_ROOTS]
    labile_ts = total[:, _MAT_LABILE]
    refr_ts   = total[:, _MAT_REFRACTORY]
    t_yr      = t_days / 365.25

    # Work out which segment boundary (first-yr / last-yr) each rendered frame is in
    steps_per_year  = n_total_snaps / n_years_total
    first_boundary  = int(round(first_yr * steps_per_year))   # last snap index of first window

    # Y-axis window for the core panel: 0.30 m below initial surface to 0.10 m above max MHW
    init_surf    = surf[0]
    elev_bot     = init_surf - 0.30
    elev_top_max = mhw.max() + 0.10

    # --- figure ---
    fig = plt.figure(figsize=(13, 7))
    gs  = fig.add_gridspec(3, 2, width_ratios=[1, 1.8],
                           hspace=0.45, wspace=0.35,
                           left=0.07, right=0.97, top=0.93, bottom=0.08)

    ax_core = fig.add_subplot(gs[:, 0])
    ax_tc   = fig.add_subplot(gs[0, 1])
    ax_rt   = fig.add_subplot(gs[1, 1])
    ax_lr   = fig.add_subplot(gs[2, 1])

    ax_core.set_ylim(elev_bot, elev_top_max)
    ax_core.set_xlim(0, 1)
    ax_core.set_xlabel("Fraction", fontsize=9)
    ax_core.set_ylabel("Elevation (m relative to initial MSL)", fontsize=9)
    ax_core.set_title("Core stratigraphy", fontsize=10, fontweight="bold")

    legend_patches = [
        mpatches.Patch(color=_C_SEDIMENT,   label="Sediment"),
        mpatches.Patch(color=_C_REFRACTORY, label="Refractory C"),
        mpatches.Patch(color=_C_LABILE,     label="Labile C"),
        mpatches.Patch(color=_C_ROOTS,      label="Live roots"),
    ]
    ax_core.legend(handles=legend_patches + [
        plt.Line2D([0], [0], ls="--", color="royalblue",   label="MSL"),
        plt.Line2D([0], [0], ls="--", color="deepskyblue", label="MHW"),
    ], loc="lower right", fontsize=7, framealpha=0.8)

    ax_tc.set_ylabel("g m⁻²", fontsize=8)
    ax_tc.set_title("Total organic C", fontsize=9)
    ax_rt.set_ylabel("g m⁻²", fontsize=8)
    ax_rt.set_title("Live roots", fontsize=9)
    ax_lr.set_ylabel("g m⁻²", fontsize=8)
    ax_lr.set_title("Labile & refractory C", fontsize=9)
    ax_lr.set_xlabel("Year", fontsize=8)

    for ax in [ax_tc, ax_rt, ax_lr]:
        ax.set_xlim(0, t_yr[-1])
        ax.grid(True, alpha=0.25)

    ax_tc.set_ylim(0, total_c.max() * 1.1 + 1)
    ax_rt.set_ylim(0, roots_ts.max() * 1.1 + 1)
    ax_lr.set_ylim(0, max(labile_ts.max(), refr_ts.max()) * 1.1 + 1)

    # Full-run ghost traces
    ax_tc.plot(t_yr, total_c,   color="black",       lw=0.6, alpha=0.15)
    ax_rt.plot(t_yr, roots_ts,  color=_C_ROOTS,      lw=0.6, alpha=0.15)
    ax_lr.plot(t_yr, labile_ts, color=_C_LABILE,     lw=0.6, alpha=0.15)
    ax_lr.plot(t_yr, refr_ts,   color=_C_REFRACTORY, lw=0.6, alpha=0.15)

    ax_lr.plot([], [], color=_C_LABILE,     lw=1.5, label="Labile")
    ax_lr.plot([], [], color=_C_REFRACTORY, lw=1.5, label="Refractory")
    ax_lr.legend(fontsize=7, loc="upper left")

    # Animated artists
    fill_collections = []
    line_msl, = ax_core.plot([], [], "--", color="royalblue",   lw=1.2)
    line_mhw, = ax_core.plot([], [], "--", color="deepskyblue", lw=1.2)

    line_tc,  = ax_tc.plot([], [], color="black",       lw=1.4)
    line_rt,  = ax_rt.plot([], [], color=_C_ROOTS,      lw=1.4)
    line_lab, = ax_lr.plot([], [], color=_C_LABILE,     lw=1.4)
    line_ref, = ax_lr.plot([], [], color=_C_REFRACTORY, lw=1.4)

    vcursors = []
    for ax in [ax_tc, ax_rt, ax_lr]:
        vc, = ax.plot([], [], color="red", lw=0.8, alpha=0.7)
        vcursors.append(vc)

    year_text = ax_core.text(
        0.03, 0.97, "", transform=ax_core.transAxes,
        fontsize=10, va="top", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    seg_text = ax_core.text(
        0.03, 0.90, "", transform=ax_core.transAxes,
        fontsize=8, va="top", color="dimgray",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    fig.suptitle("Marsh restoration — core growth", fontsize=12, fontweight="bold")

    n_frames = len(frame_indices)

    def _update(anim_frame):
        for col in fill_collections:
            col.remove()
        fill_collections.clear()

        snap_i = int(frame_indices[anim_frame])
        n_lay  = data["n_layers"][snap_i]
        tops_all  = data["top_elev"][snap_i, :n_lay]
        thicks_all = data["thick"][snap_i, :n_lay]
        bots_all  = tops_all - thicks_all

        visible = (tops_all >= elev_bot) & (bots_all <= elev_top_max)
        if not visible.any():
            visible[tops_all.argmax()] = True
        idx_vis = np.where(visible)[0]
        tops   = tops_all[idx_vis]
        bots   = bots_all[idx_vis]
        n_vis  = len(idx_vis)

        fracs = _core_fracs(data["lmass"][snap_i][idx_vis], n_vis)

        cum = np.zeros((n_vis, 5))
        for k in range(4):
            cum[:, k+1] = cum[:, k] + fracs[:, k]

        colors = [_C_SEDIMENT, _C_REFRACTORY, _C_LABILE, _C_ROOTS]
        for k, col in enumerate(colors):
            y_steps   = np.empty(2 * n_vis)
            y_steps[0::2] = bots
            y_steps[1::2] = tops
            x_l = np.repeat(cum[:, k],   2)
            x_r = np.repeat(cum[:, k+1], 2)
            coll = ax_core.fill_betweenx(y_steps, x_l, x_r,
                                         color=col, alpha=0.88, linewidth=0)
            fill_collections.append(coll)

        line_msl.set_data([0, 1], [msl[snap_i], msl[snap_i]])
        line_mhw.set_data([0, 1], [mhw[snap_i], mhw[snap_i]])

        yr = t_yr[snap_i]
        year_text.set_text(f"Year {yr:.1f}")

        # Label which segment we are in
        if snap_i < first_boundary:
            seg_text.set_text(f"Early colonisation\n(yr 0 – {first_yr})")
        else:
            seg_text.set_text(f"Late-stage\n(yr {n_years_total - last_yr} – {n_years_total})")

        # Growing time-series line up to current snapshot time
        ts_idx = snap_i + 1   # time series has same length as snapshots
        line_tc.set_data(t_yr[:ts_idx],  total_c[:ts_idx])
        line_rt.set_data(t_yr[:ts_idx],  roots_ts[:ts_idx])
        line_lab.set_data(t_yr[:ts_idx], labile_ts[:ts_idx])
        line_ref.set_data(t_yr[:ts_idx], refr_ts[:ts_idx])

        yr_now = t_yr[snap_i]
        for ax, vc in zip([ax_tc, ax_rt, ax_lr], vcursors):
            ylim = ax.get_ylim()
            vc.set_data([yr_now, yr_now], ylim)

        return (fill_collections + [line_msl, line_mhw, year_text, seg_text,
                                     line_tc, line_rt, line_lab, line_ref]
                + vcursors)

    print(f"  Rendering {n_frames} frames at {fps} fps → {out_path.name}")
    anim = FuncAnimation(fig, _update, frames=n_frames,
                         blit=False, interval=1000 / fps)

    writer = FFMpegWriter(fps=fps, metadata={"title": "Marsh core growth"},
                          extra_args=["-vcodec", "libx264", "-crf", "22",
                                      "-pix_fmt", "yuv420p"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=writer, dpi=dpi,
              progress_callback=lambda i, n: print(f"\r  Frame {i+1}/{n}",
                                                    end="", flush=True))
    print(f"\n  Saved {out_path}")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years",    type=int,  default=200,
                   help="Simulation length in years (default 200)")
    p.add_argument("--first-yr", type=int,  default=10,
                   help="Animate first N years (default 10)")
    p.add_argument("--last-yr",  type=int,  default=10,
                   help="Animate last N years (default 10)")
    p.add_argument("--fps",      type=int,  default=12)
    p.add_argument("--binary",   default=str(_ROOT_DIR / "build" / "marsh_cli"))
    p.add_argument("--nc",       default=None)
    p.add_argument("--out",      default=None)
    p.add_argument("--dpi",      type=int,  default=120)
    args = p.parse_args()

    nc_path = (Path(args.nc) if args.nc
               else _VIZ_DIR / f"core_anim_{args.years}yr.nc")
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = (_VIZ_DIR /
                    f"core_growth_{args.years}yr"
                    f"_first{args.first_yr}_last{args.last_yr}.mp4")

    print(f"\n=== Marsh core growth animation  "
          f"({args.years} yr run, first {args.first_yr} + last {args.last_yr} yr) ===\n")

    if not nc_path.exists():
        print("[1] Building YAML …")
        yaml_path = build_yaml(nc_path, args.years)
        print("[2] Running model …")
        run_model(yaml_path, args.binary)
    else:
        print(f"[1-2] Using existing NC: {nc_path}")

    print("[3] Loading NC …")
    data = load_nc(nc_path)
    n_snaps = len(data["snap_t"])
    print(f"  {n_snaps} snapshots, {len(data['t_days'])} time-series steps")
    print(f"  Surface elevation: {data['surf'][0]:.3f} → {data['surf'][-1]:.3f} m")

    frame_indices = _build_frame_indices(
        n_snaps, args.years, args.first_yr, args.last_yr)
    print(f"  Animating {len(frame_indices)} frames "
          f"(first {args.first_yr} yr + last {args.last_yr} yr)")

    print("[4] Animating …")
    make_animation(data, out_path, args.fps, args.dpi,
                   frame_indices, args.years, args.first_yr, args.last_yr)

    print("\nDone.")


if __name__ == "__main__":
    main()
