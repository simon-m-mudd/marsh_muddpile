# species_presets.py
#
# Parameter overrides for each marsh plant species supported by the
# marsh_gpp_biomass model.  Values here represent prior estimates derived
# from the literature; they should be treated as initial guesses for
# site-specific calibration, not universal constants.
#
# Each preset is a dict of (parameter_name → value) that is merged on top of
# the generic defaults in yaml_writer._DEFAULT_PARAMETERS.  Only parameters
# that differ meaningfully from the Spartina alterniflora defaults are listed;
# everything else inherits the default value.
#
# Usage
# -----
#   from species_presets import get_species_parameters
#   extra = get_species_parameters("spartina_patens")
#   yaml_writer.write_config(config, path, extra_parameters=extra)
#
# References (selected)
# ---------------------
#   Bertness & Ellison (1987) Ecology 68:1042-1052     — S. patens hydroperiod
#   Warren & Niering (1993) Ecology 74:96-103          — PIE biomass
#   Pennings et al. (2005) Ecology 86:2940-2951        — latitudinal variation
#   Daoust & Childers (1999) Oecologia 119:285-293     — root allocation
#   Mancera et al. (2003) Estuaries 26:673-684         — J. roemerianus GPP
#   Darby & Turner (2008) Mar. Ecol. Prog. Ser.        — J. roemerianus roots
#   Kirwan & Guntenspergen (2012) J. Ecology            — salinity tolerance

from __future__ import annotations
from typing import Dict


# ---------------------------------------------------------------------------
# Spartina alterniflora — low marsh, tidal creek banks
# ---------------------------------------------------------------------------
# These values are the baseline defaults in yaml_writer._DEFAULT_PARAMETERS.
# Listed here for documentation and to enable explicit selection.
#
# Ecology: C4 grass, dominant from Newfoundland to Florida on intertidal creek
# banks and low marsh; forms tall (>1 m) and short (<0.5 m) ecotypes.
# Salinity-tolerant; optimum inundation at ~15-25 % of tidal cycle.
SPARTINA_ALTERNIFLORA: Dict[str, float] = {}   # no overrides — inherits all defaults


# ---------------------------------------------------------------------------
# Spartina patens — high marsh, peat-mat forming
# ---------------------------------------------------------------------------
# Ecology: C4 grass, mat-forming with horizontal rhizomes.  Dominant in high-
# marsh zones at PIE (MA), VCR (VA), and NI (SC); tolerates waterlogged soils
# but prefers the high marsh where flooding is infrequent (<5-10 % of tidal
# cycle).  Less salt-tolerant than S. alterniflora.  Shoots are thinner and
# shorter (30-80 cm); LAI per unit biomass is higher.  Seasonal die-back more
# pronounced at northern sites.
SPARTINA_PATENS: Dict[str, float] = {
    # GPP / LUE — similar C4 pathway but lower canopy-level productivity
    # due to shorter stature; Pennings et al. ANPP ~ 500-900 g m-2 yr-1
    "vegetation_lue_gC_per_umol": 2.2e-4,

    # Canopy structure — fine dense culms give higher leaf area per biomass
    "vegetation_lai_per_kg_m2":   4.5,
    "vegetation_max_lai":         4.0,
    "vegetation_lai_light_extinction": 0.85,   # slightly denser canopy packing

    # Capacity — shorter plant, dense root mat
    "vegetation_aboveground_capacity_kg_m2":  1.2,
    "vegetation_belowground_capacity_kg_m2":  4.5,

    # Hydroperiod — prefers drier high-marsh conditions (Bertness & Ellison 1987)
    "vegetation_hydroperiod_optimum_fraction": 0.05,
    "vegetation_hydroperiod_sigma_fraction":   0.12,

    # Temperature — wider geographic range; slightly cooler optimum captures
    # the New England populations (Warren & Niering 1993)
    "vegetation_temperature_optimum_c": 23.0,
    "vegetation_temperature_sigma_c":   12.0,

    # Salinity stress — less tolerant than S. alterniflora (threshold ~15 ppt)
    "vegetation_salinity_threshold_ppt":   15.0,
    "vegetation_salinity_stress_per_ppt":  0.05,

    # Carbon allocation — shallower, denser root mats (Daoust & Childers 1999)
    "vegetation_shoot_allocation_min": 0.30,
    "vegetation_shoot_allocation_max": 0.60,

    # Aboveground mortality — seasonal die-back similar to S. alterniflora but
    # earlier onset at northern sites; cold threshold 15 °C (PIE autumn)
    "vegetation_aboveground_mortality_base_per_day": 8.0e-4,
    "vegetation_cold_mortality_threshold_c":          15.0,
    "vegetation_cold_mortality_slope_per_day_per_c":  3.0e-3,

    # Flooding mortality — dies more readily under sustained flooding
    "vegetation_aboveground_hydroperiod_mortality_threshold": 0.20,
    "vegetation_aboveground_hydroperiod_mortality_slope_per_day": 0.015,

    # Belowground — slow root turnover in dense mat
    "vegetation_belowground_mortality_base_per_day":               3.0e-4,
    "vegetation_belowground_hydroperiod_mortality_threshold":      0.30,
    "vegetation_belowground_salinity_mortality_threshold_ppt":     28.0,

    # Root profile — shallower allocation than S. alterniflora
    "root_efolding_depth_m":                    0.07,
    "root_allocation_to_labile_fraction":       0.50,
    "root_allocation_to_refractory_fraction":   0.50,
}


# ---------------------------------------------------------------------------
# Juncus roemerianus — black needlerush, brackish-transitional upper marsh
# ---------------------------------------------------------------------------
# Ecology: C3 rush; dominant in upper-marsh and brackish zones from NC to TX
# and at GCE (GA) and VCR (VA) upper-marsh transects.  Round, stiff culms
# give a lower light-interception efficiency per unit LAI than flat grass
# blades.  Semi-evergreen in the SE US; minimal cold die-back.  Deep, slow-
# turnover root system; litter is highly refractory (tough culm walls).
# Prefers low-salinity (<20 ppt) and low-flooding conditions.
JUNCUS_ROEMERIANUS: Dict[str, float] = {
    # GPP / LUE — C3 photosynthesis; round culms reduce effective absorptance
    # per unit LAI (Mancera et al. 2003 ANPP ~ 800-1500 g m-2 yr-1 at GCE)
    "vegetation_lue_gC_per_umol": 2.0e-4,
    "vegetation_carbon_use_efficiency": 0.45,   # slightly lower CUE (C3)

    # Canopy structure — round culms: lower area per biomass, lower extinction
    "vegetation_lai_per_kg_m2":        2.5,
    "vegetation_max_lai":              5.0,
    "vegetation_lai_light_extinction": 0.55,   # round culms scatter light

    # Capacity — moderate above-ground stature (0.5-1.5 m); large root system
    "vegetation_aboveground_capacity_kg_m2":  2.5,
    "vegetation_belowground_capacity_kg_m2":  6.0,

    # Hydroperiod — upper high-marsh, tolerates periodic flooding only
    "vegetation_hydroperiod_optimum_fraction": 0.08,
    "vegetation_hydroperiod_sigma_fraction":   0.15,

    # Temperature — subtropical optimum (dominant at GCE, mid-Atlantic)
    "vegetation_temperature_optimum_c": 28.0,
    "vegetation_temperature_sigma_c":   10.0,

    # Salinity stress — intermediate tolerance; declines above ~15-20 ppt
    # (Kirwan & Guntenspergen 2012)
    "vegetation_salinity_threshold_ppt":   15.0,
    "vegetation_salinity_stress_per_ppt":  0.04,

    # Carbon allocation — more aboveground retention (semi-evergreen)
    "vegetation_shoot_allocation_min": 0.35,
    "vegetation_shoot_allocation_max": 0.65,

    # Aboveground mortality — semi-evergreen: very low cold die-back in SE US;
    # harder frost (< 5 °C) triggers some senescence
    "vegetation_aboveground_mortality_base_per_day": 4.0e-4,
    "vegetation_cold_mortality_threshold_c":          5.0,
    "vegetation_cold_mortality_slope_per_day_per_c":  1.0e-3,

    # Flooding mortality — dies quickly under sustained inundation
    "vegetation_aboveground_hydroperiod_mortality_threshold": 0.25,
    "vegetation_aboveground_hydroperiod_mortality_slope_per_day": 0.015,

    # Salinity mortality — more sensitive than S. alterniflora
    "vegetation_aboveground_salinity_mortality_threshold_ppt": 25.0,
    "vegetation_aboveground_salinity_mortality_slope_per_day_per_ppt": 7.0e-4,

    # Belowground — deep, slow-turnover roots (Darby & Turner 2008)
    "vegetation_belowground_mortality_base_per_day":               2.0e-4,
    "vegetation_belowground_hydroperiod_mortality_threshold":      0.35,
    "vegetation_belowground_salinity_mortality_threshold_ppt":     30.0,

    # Root profile — deeper root system than Spartina spp.
    "root_efolding_depth_m":                    0.12,
    "root_allocation_to_labile_fraction":       0.40,
    "root_allocation_to_refractory_fraction":   0.60,   # tough culm litter
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Dict[str, float]] = {
    "spartina_alterniflora": SPARTINA_ALTERNIFLORA,
    "spartina_patens":       SPARTINA_PATENS,
    "juncus_roemerianus":    JUNCUS_ROEMERIANUS,
}

KNOWN_SPECIES = list(_REGISTRY.keys())


def get_species_parameters(species: str) -> Dict[str, float]:
    """Return the parameter-override dict for a named species.

    Parameters
    ----------
    species : one of "spartina_alterniflora", "spartina_patens",
              "juncus_roemerianus"

    Raises
    ------
    ValueError if the species name is not recognised.
    """
    key = species.lower().strip()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown species '{species}'. "
            f"Known species: {KNOWN_SPECIES}"
        )
    return dict(_REGISTRY[key])
