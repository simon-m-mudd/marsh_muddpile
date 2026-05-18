# yaml_writer.py
#
# Generates a complete marsh_muddpile YAML configuration file from a
# PlotConfig and a list of forcing steps produced by forcing_builder.
#
# The initial_state uses the 'layers' form (explicit top_elevation_m) rather
# than 'uniform_column' so that the sediment surface is placed at the
# measured plot elevation.  The model datum is MSL = 0 m.
#
# All parameters are taken from the reference run
# (example_runs/run_new_GPP_instance.yaml) and can be overridden by passing
# a dict to write_config(extra_parameters=...).

from __future__ import annotations
import math
import os
from typing import Any, Dict, List, Optional

import yaml

from site_config import PlotConfig
from forcing_builder import build_generated_forcing_block
from species_presets import get_species_parameters


# ---------------------------------------------------------------------------
# Default parameter block - mirrors run_new_GPP_instance.yaml exactly.
# Only values that a user is likely to vary during calibration are broken
# out into PlotConfig; everything else lives here.
# ---------------------------------------------------------------------------
_DEFAULT_PARAMETERS: Dict[str, Any] = {
    # physical constants
    "water_density_kg_m3": 1000.0,
    "gravity_m_s2": 9.81,

    # layer merging
    "layer_merging_enable": 1.0,
    "layer_merging_max_layers": 200.0,
    "layer_merging_minimum_unmerged_depth_m": 0.30,
    "layer_merging_target_thickness_m": 0.05,

    # water level / tidal substeps
    "water_level_substeps_per_step": 240,
    "water_level_reference_time_hours": 0.0,

    # surface property summariser
    "surface_property_active_depth_m": 0.10,
    "surface_property_fine_grain_size_threshold_m": 6.3e-5,
    "surface_property_default_grain_size_m": 2.0e-5,
    "surface_property_default_organic_carbon_fraction": 0.45,

    # salinity model
    "salinity_min_ppt": 0.0,
    "salinity_max_ppt": 80.0,
    "salinity_initial_root_zone_ppt": 20.0,
    "salinity_default_creek_ppt": 30.0,
    "salinity_et_concentration_rate_ppt_per_mm": 0.020,
    "salinity_precipitation_dilution_rate_ppt_per_mm": 0.030,
    "salinity_freshwater_dilution_rate_ppt_per_mm": 0.030,
    "salinity_base_flushing_rate_per_day": 0.50,
    "salinity_flushing_efolding_distance_m": 50.0,
    "salinity_exchange_coarse_sensitivity": 0.75,
    "salinity_exchange_organic_sensitivity": 0.50,
    "salinity_storage_fine_sensitivity": 1.00,
    "salinity_storage_organic_sensitivity": 1.00,
    "salinity_storage_porosity_sensitivity": 0.50,
    "salinity_min_exchange_modifier": 0.10,
    "salinity_min_storage_modifier": 0.25,
    "salinity_subsurface_flushing_rate_per_day": 0.003,

    # ET model
    "et_background_mm_d": 0.5,
    "et_temperature_base_c": 0.0,
    "et_temperature_coefficient_mm_d_per_c": 0.12,
    "et_par_coefficient_mm_d_per_umol_m2_d": 3.0e-8,
    "et_cover_extinction_coefficient": 0.7,
    "et_kc_min": 0.2,
    "et_kc_max": 1.1,
    "et_salinity_threshold_ppt": 20.0,
    "et_salinity_stress_per_ppt": 0.03,
    "et_inundation_threshold_fraction": 0.25,
    "et_inundation_stress_per_fraction": 1.2,
    "et_min_transpiration_stress": 0.05,
    "et_transpiration_fraction_min": 0.15,
    "et_transpiration_fraction_max": 0.85,
    "et_evaporation_fine_sensitivity": 0.25,
    "et_evaporation_organic_sensitivity": 0.20,
    "et_evaporation_porosity_sensitivity": 0.15,
    "et_evaporation_coarse_sensitivity": 0.15,
    "et_min_evaporation_modifier": 0.2,
    "et_biomass_to_lai_coefficient": 3.0,

    # vegetation / GPP
    "vegetation_days_per_year": 365.0,
    "vegetation_lue_gC_per_umol": 1.6e-6,   # reduced from 3.5e-4 so m/g ~ 0.05 at peak elevation
    "vegetation_carbon_use_efficiency": 0.5,
    "vegetation_biomass_carbon_fraction": 0.45,
    "vegetation_lai_light_extinction": 0.8,
    "vegetation_max_fpar": 0.95,
    "vegetation_lai_per_kg_m2": 3.0,
    "vegetation_max_lai": 6.0,
    "vegetation_temperature_optimum_c": 25.0,
    "vegetation_temperature_sigma_c": 10.0,
    "vegetation_hydroperiod_optimum_fraction": 0.25,   # peak at ≈ 47 cm NAVD88 (Morris et al. 2013 NI)
    "vegetation_hydroperiod_sigma_fraction": 0.18,   # half-Gaussian: controls right-limb (high-elevation) decline
    "vegetation_salinity_threshold_ppt": 20.0,
    "vegetation_salinity_stress_per_ppt": 0.03,
    "vegetation_shoot_allocation_min": 0.25,
    "vegetation_shoot_allocation_max": 0.65,
    "vegetation_aboveground_capacity_kg_m2": 0.95,   # RMSE-optimal vs Morris et al. (2013) NI data
    "vegetation_belowground_capacity_kg_m2": 1.9,    # 2× aboveground
    "vegetation_aboveground_mortality_base_per_day": 0.001,
    "vegetation_aboveground_mortality_seasonal_amp_per_day": 0.0,
    "vegetation_mortality_peak_day": 300.0,
    "vegetation_cold_mortality_threshold_c": 20.0,
    "vegetation_cold_mortality_slope_per_day_per_c": 0.002,
    "vegetation_aboveground_hydroperiod_mortality_threshold": 0.50,
    "vegetation_aboveground_hydroperiod_mortality_slope_per_day": 0.01,
    "vegetation_aboveground_salinity_mortality_threshold_ppt": 30.0,
    "vegetation_aboveground_salinity_mortality_slope_per_day_per_ppt": 0.0005,
    "vegetation_belowground_mortality_base_per_day": 2.74e-4,   # 10 % yr⁻¹ (Morris & Sundberg 2023)
    "vegetation_belowground_hydroperiod_mortality_threshold": 0.60,
    "vegetation_belowground_hydroperiod_mortality_slope_per_day": 0.005,
    "vegetation_belowground_salinity_mortality_threshold_ppt": 35.0,
    "vegetation_belowground_salinity_mortality_slope_per_day_per_ppt": 0.00025,
    # inundation range stressor (Gaussian) — empirical bell-curve peak at optimum IF
    # Replaces the earlier tent function (Morris et al. 2013; NI S. alterniflora defaults)
    "vegetation_inundation_optimum_fraction": 0.25,  # IF at peak biomass elevation (~47 cm NAVD88)
    "vegetation_inundation_sigma_fraction": 0.25,    # half-Gaussian: controls left-limb (low-elevation) decline
    # low-inundation mortality — extra shoot+root loss when platform rises above tidal frame
    "vegetation_low_inundation_mortality_threshold": 0.05,
    "vegetation_low_inundation_mortality_slope_per_day": 0.005,
    # desiccation mortality — lethal linear ramp from IF=threshold to IF=0
    "vegetation_desiccation_threshold": 0.01,
    "vegetation_desiccation_mortality_rate_per_day": 0.10,
    # low-inundation GPP stressor — linear ramp from 0 (IF=0) to 1 (IF=threshold)
    # suppresses production at the upper intertidal margin; threshold matches
    # low_inundation_mortality_threshold so both responses share the same reference
    "vegetation_low_inundation_gpp_threshold": 0.05,

    # root allocation
    "root_efolding_depth_m": 0.08,
    "root_profile_normalize_to_column": 1.0,
    "root_allocation_to_labile_fraction": 0.34,
    "root_allocation_to_refractory_fraction": 0.66,

    # TKE deposition
    "deposition_reference_biomass_g_m2": 500.0,
    "deposition_reference_flow_velocity_m_s": 0.05,
    "deposition_limiting_flow_velocity_m_s": 0.15,
    "deposition_time_fractions_per_tide": 720.0,
    "deposition_min_water_depth_m": 0.002,
    "deposition_von_karman": 0.40,
    "deposition_tke_coefficient": 0.20,
    "deposition_water_density_kg_m3": 1000.0,
    "deposition_aa": 0.46,
    "deposition_bb": 3.8,
    "deposition_alpha_turb": 0.9,
    "deposition_alpha_zero": 11.0,
    "deposition_alpha": 0.55,
    "deposition_beta": 0.40,
    "deposition_mu": 0.00066,
    "deposition_phi": 0.29,
    "deposition_kappa": 0.224,
    "deposition_nu_m2_s": 1.0e-5,
    "deposition_gamma_tr": 0.718,
    "deposition_epsilon": 2.08,

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

    # porewater NH4
    "nh4_carbon_fraction": 0.45,
    "nh4_c_to_n_molar_ratio": 25.0,
    "nh4_creek_umol_L": 5.0,
    "nh4_tidal_flushing_rate_per_d": 0.5,
    "nh4_flushing_depth_scale_m": 0.05,
    "nh4_diffusion_coeff_m2_per_d": 1.0e-5,
    "nh4_uptake_vmax_umol_L_d_per_kg_m3": 10.0,
    "nh4_km_umol_L": 50.0,
    "nh4_initial_umol_L": 100.0,
    "nh4_min_umol_L": 0.0,
    "nh4_max_umol_L": 2000.0,

    # sulfate-methane model
    # SO4 boundary: North Inlet mean salinity ~24.5 ppt → ~19.6 mM SO4
    "ch4_carbon_fraction": 0.45,
    "ch4_initial_so4_umol_L": 19600.0,
    "ch4_initial_ch4_umol_L": 0.0,
    "ch4_creek_so4_umol_L": 19600.0,
    "ch4_tidal_flushing_rate_per_d": 0.3,
    "ch4_flushing_depth_scale_m": 0.10,
    "ch4_km_so4_umol_L": 1000.0,
    "ch4_c_to_ch4": 2.0,         # molar C : CH4 ratio for methanogenesis
    "ch4_diffusion_coeff_m2_per_d": 1.0e-5,
    "ch4_so4_diffusion_coeff_m2_per_d": 1.0e-5,
    "ch4_oxidation_rate_per_d": 0.05,
    "ch4_oxidation_depth_scale_m": 0.05,
    "ch4_ebullition_threshold_umol_L": 500.0,
    # plant_transport_factor=2 → total flux = diffusive × 3, calibrated
    # from edi.1828.3 (root+shoot / root-only chambers ≈ 3)
    "ch4_plant_transport_factor": 2.0,
    "ch4_min_umol_L": 0.0,
    "ch4_max_umol_L": 2000.0,

    # compaction
    "compaction_loi_intercept_percent": 0.0,
    "compaction_loi_slope_percent_per_organic_fraction": 100.0,
    "compaction_default_grain_size_m": 2.0e-5,
    "compaction_e_1_intercept": 0.50,
    "compaction_e_1_loi_slope": 0.040,
    "compaction_e_1_grain_size_slope": 0.0,
    "compaction_c_r_intercept": 0.005,
    "compaction_c_r_loi_slope": 0.0015,
    "compaction_c_r_grain_size_slope": 0.0,
    "compaction_c_c_intercept": 0.10,
    "compaction_c_c_loi_slope": 0.020,
    "compaction_c_c_grain_size_slope": 0.0,
    "compaction_sigma_y_intercept_kpa": 12.0,
    "compaction_sigma_y_loi_slope": -0.15,
    "compaction_sigma_y_grain_size_slope": 0.0,
    "compaction_min_e_1": 0.10,
    "compaction_min_c_r": 1.0e-4,
    "compaction_min_c_c": 1.0e-3,
    "compaction_min_sigma_y_kpa": 3.0,
    "compaction_min_effective_stress_kpa": 0.01,
    "compaction_reference_stress_kpa": 1.0,
    "compaction_min_void_ratio": 0.01,
}

_DEFAULT_MATERIALS = [
    {
        "name": "sand",
        "category": "mineral",
        "density": 2650.0,
        "allow_surface_deposition": False,
        "settling": {"diameter": 2.0e-4, "settling_velocity": 2.0e-2},
    },
    {
        "name": "silt",
        "category": "mineral",
        "density": 2600.0,
        "allow_surface_deposition": True,
        "settling": {"diameter": 2.0e-5, "settling_velocity": 3.7e-5},
    },
    {
        "name": "fine_silt",
        "category": "mineral",
        "density": 2600.0,
        "allow_surface_deposition": True,
        "settling": {"diameter": 1.5e-5, "settling_velocity": 1.5e-4},
    },
    {
        "name": "refractory_organic",
        "category": "organic_refractory",
        "density": 1400.0,
        "allow_root_input": True,
        # k_0 = 0.0: refractory carbon does not decay on marsh timescales.
        # Calibrated at North Inlet; non-zero values produce unrealistically low
        # deep OM fractions (half-life 1.9 yr at k_0=0.001/d).
        "decay": {"k_0": 0.0, "gamma": 2.0, "temperature_sensitive": True},
    },
    {
        "name": "labile_organic",
        "category": "organic_labile",
        "density": 1200.0,
        "allow_root_input": True,
        "decay": {"k_0": 0.01, "gamma": 0.10, "temperature_sensitive": True},
    },
    {
        "name": "roots",
        "category": "live_root",
        "density": 1100.0,
        "allow_root_input": True,
    },
    {
        "name": "marker",
        "category": "mineral",
        "density": 2600.0,
        "allow_surface_deposition": False,
    },
]


def _insert_marker_layer(
    layers: List[dict],
    elevation_m: float,
    thickness_m: float,
) -> List[dict]:
    """Insert a thin marker layer into an existing column layer list.

    The marker top is placed at `elevation_m`.  If that elevation falls
    inside an existing layer, that layer is split into an upper and lower
    sub-layer with masses scaled proportionally to their new thicknesses,
    and the marker is inserted between them.  If the elevation aligns with
    a layer boundary no splitting is needed.

    The marker layer contains only 'marker' material (density 2600 kg/m³,
    inert mineral).  Its mass equals rho * (1 - porosity) * thickness.
    """
    _RHO_MARKER = 2600.0
    result: List[dict] = []
    inserted = False

    for layer in layers:
        if inserted:
            result.append(layer)
            continue

        top    = layer["top_elevation_m"]
        h      = layer["thickness_m"]
        phi    = layer["porosity"]
        bottom = round(top - h, 6)
        marker_bottom = round(elevation_m - thickness_m, 6)

        # Marker is above this layer — skip (surface-first ordering).
        if elevation_m > top + 1e-9:
            result.append(layer)
            continue

        # Marker top is at or below this layer's bottom — defer to the next
        # layer (handles the exact-boundary case without creating overlaps).
        if elevation_m <= bottom + 1e-9:
            result.append(layer)
            continue

        # Marker top is strictly inside this layer: split around it.
        upper_thick = round(top - elevation_m, 6)
        lower_thick = round(marker_bottom - bottom, 6)

        if upper_thick > 1e-6:
            frac = upper_thick / h
            result.append({
                "top_elevation_m": round(top, 4),
                "thickness_m":     round(upper_thick, 4),
                "porosity":        phi,
                "age_days":        layer["age_days"],
                "mass_kg_m2":      {k: round(v * frac, 6)
                                    for k, v in layer["mass_kg_m2"].items()},
            })

        marker_mass = round(_RHO_MARKER * (1.0 - phi) * thickness_m, 4)
        result.append({
            "top_elevation_m": round(elevation_m, 4),
            "thickness_m":     round(thickness_m, 4),
            "porosity":        phi,
            "age_days":        0.0,
            "mass_kg_m2":      {"marker": marker_mass},
        })

        if lower_thick > 1e-6:
            frac = lower_thick / h
            result.append({
                "top_elevation_m": round(marker_bottom, 4),
                "thickness_m":     round(lower_thick, 4),
                "porosity":        phi,
                "age_days":        layer["age_days"],
                "mass_kg_m2":      {k: round(v * frac, 6)
                                    for k, v in layer["mass_kg_m2"].items()},
            })

        inserted = True

    if not inserted:
        # Elevation is below the whole column — append at the bottom.
        phi = layers[-1]["porosity"] if layers else 0.60
        marker_mass = round(_RHO_MARKER * (1.0 - phi) * thickness_m, 4)
        result.append({
            "top_elevation_m": round(elevation_m, 4),
            "thickness_m":     round(thickness_m, 4),
            "porosity":        phi,
            "age_days":        0.0,
            "mass_kg_m2":      {"marker": marker_mass},
        })

    return result


def build_initial_column_layers(config: PlotConfig) -> List[dict]:
    """Build a realistic 1-m organic sediment column as a list of layer dicts.

    Profile (each 5 cm layer, depth measured from surface downward):
      - Refractory organic : 10 % of solid volume throughout
      - Labile organic     : exponential decay, 10 % at surface,
                             e-folding depth 10 cm (≈0.5 % at 30 cm)
      - Silt               : remainder of solid volume
      - Roots              : total = mean_abg_biomass × root_to_shoot_ratio,
                             distributed with exponential profile (e-fold 10 cm)
    """
    n = config.initial_column_n_layers
    h = config.initial_column_layer_thickness_m
    phi = config.initial_column_porosity
    solid_vol = (1.0 - phi) * h          # m³ solid per m² per layer

    rho_silt    = 2600.0
    rho_refrac  = 1400.0
    rho_labile  = 1200.0
    rho_roots   = 1100.0

    root_efolding_m = 0.08
    total_roots = (config.mean_aboveground_biomass_kg_m2
                   * config.initial_root_to_shoot_ratio)

    # Unnormalized root weights per layer (midpoint depth)
    root_weights = [
        math.exp(-(i + 0.5) * h / root_efolding_m)
        for i in range(n)
    ]
    total_weight = sum(root_weights)

    layers = []
    for i in range(n):
        depth_mid = (i + 0.5) * h
        top_elev  = config.surface_elevation_m - i * h

        refrac_frac  = 0.10
        labile_frac  = 0.10 * math.exp(-depth_mid / root_efolding_m)
        mineral_frac = max(0.0, 1.0 - refrac_frac - labile_frac)

        root_mass = total_roots * root_weights[i] / total_weight

        layers.append({
            "top_elevation_m": round(top_elev, 4),
            "thickness_m": h,
            "porosity": phi,
            "age_days": 0.0,
            "mass_kg_m2": {
                "silt":               round(rho_silt   * mineral_frac * solid_vol, 4),
                "refractory_organic": round(rho_refrac * refrac_frac  * solid_vol, 4),
                "labile_organic":     round(rho_labile * labile_frac  * solid_vol, 6),
                "roots":              round(root_mass, 6),
            },
        })

    if config.add_marker_horizon:
        layers = _insert_marker_layer(
            layers,
            elevation_m=config.surface_elevation_m,
            thickness_m=config.marker_horizon_thickness_m,
        )

    return layers


def _tidal_constituent_parameters(config: PlotConfig) -> Dict[str, Any]:
    """Return the harmonic constituent entries for the parameters block."""
    t = config.tides
    return {
        "water_level_constituent_M2_amplitude_m": t.M2_amplitude_m,
        "water_level_constituent_M2_phase_deg": t.M2_phase_deg,
        "water_level_constituent_S2_amplitude_m": t.S2_amplitude_m,
        "water_level_constituent_S2_phase_deg": t.S2_phase_deg,
        "water_level_constituent_N2_amplitude_m": t.N2_amplitude_m,
        "water_level_constituent_N2_phase_deg": t.N2_phase_deg,
        "water_level_constituent_K1_amplitude_m": t.K1_amplitude_m,
        "water_level_constituent_K1_phase_deg": t.K1_phase_deg,
        "water_level_constituent_O1_amplitude_m": t.O1_amplitude_m,
        "water_level_constituent_O1_phase_deg": t.O1_phase_deg,
    }


def write_config(
    config: PlotConfig,
    output_path: str,
    extra_parameters: Optional[Dict[str, Any]] = None,
) -> str:
    """Write a complete YAML config file for one plot run.

    Parameters
    ----------
    config:
        PlotConfig describing the plot and site.
    output_path:
        Path where the YAML file will be written.
    extra_parameters:
        Optional dict of parameter overrides (e.g. calibration candidates).
        These are merged on top of _DEFAULT_PARAMETERS before writing.

    Returns
    -------
    The path that was written.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    total_days = config.n_years * 365.25

    parameters = dict(_DEFAULT_PARAMETERS)
    parameters.update(_tidal_constituent_parameters(config))
    # Apply species-specific parameter overrides before any caller overrides.
    # This allows calibration scripts to further refine individual parameters
    # without having to repeat the full species preset.
    species_params = get_species_parameters(config.species)
    parameters.update(species_params)
    # edge_distance_deposition parameters — distance from edge matches the
    # distance_from_creek_m site property used by the salinity model.
    parameters["deposition_distance_from_edge_m"] = config.distance_from_creek_m
    parameters.setdefault("deposition_basin_length_m", 50.0)
    parameters.setdefault("deposition_length_scale_beta", 3.0)
    if extra_parameters:
        parameters.update(extra_parameters)

    initial_layers = build_initial_column_layers(config)

    # Use the merged lai_per_kg (may differ from the default for non-S.alt species)
    lai_per_kg = parameters["vegetation_lai_per_kg_m2"]
    abg_kg = config.mean_aboveground_biomass_kg_m2
    blg_kg = abg_kg * config.initial_root_to_shoot_ratio

    doc = {
        "simulation": {
            "start_year": 0,
            "end_year": config.n_years,
            "dt_days": 365.0,
            "water_level_model_name": "composite_water_level",
            "salinity_model_name": "distance_flushing_salinity",
            "evapotranspiration_model_name": "simple_canopy_et",
            "vegetation_model_name": "marsh_gpp_biomass",
            "deposition_model_name": "edge_distance_deposition",
            "root_allocation_model_name": "exponential_root_allocation",
            "decay_model_name": "marsh_decay",
            "compaction_model_name": "mixing_compaction",
            "porewater_chemistry_model_name": "nh4_porewater",
            "methane_model_name": "sulfate_methane",
        },
        "site": {
            "distance_from_creek_m": config.distance_from_creek_m,
            "creek_bank_elevation_m": config.tides.mean_sea_level_m,
            "local_tidal_offset_m": 0.0,
        },
        "parameters": parameters,
        "materials": _DEFAULT_MATERIALS,
        "forcing": {
            "generated": build_generated_forcing_block(config),
        },
        "initial_state": {
            "layers": initial_layers,
        },
        "initial_ecohydrology_state": {
            "root_zone_salinity_ppt": config.initial_salinity_ppt,
            "aboveground_biomass_kg_m2": round(abg_kg, 4),
            "belowground_biomass_kg_m2": round(blg_kg, 4),
            "lai": round(abg_kg * lai_per_kg, 4),
            "litter_kg_m2": 0.0,
        },
        "output": {
            "file": output_path.replace(".yaml", ".nc"),
            "write_time_series": True,
            "write_column_snapshots": True,
            "snapshot_times_days": [
                round(total_days * 0.25, 1),
                round(total_days * 0.50, 1),
                round(total_days * 0.75, 1),
                round(total_days, 1),
            ],
        },
    }

    with open(output_path, "w") as fh:
        yaml.dump(doc, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return output_path
