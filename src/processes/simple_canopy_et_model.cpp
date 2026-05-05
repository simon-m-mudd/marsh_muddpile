// simple_canopy_et_model.cpp
//
// Canopy-based evapotranspiration model that partitions potential ET into
// transpiration and soil evaporation components.
//
// Potential ET is a simple empirical function of air temperature and incoming
// PAR. The canopy cover fraction (derived from LAI via Beer-Lambert extinction)
// drives the split between transpiration and evaporation. Transpiration is then
// reduced by salinity and inundation stress. Evaporation is scaled by the
// exposed surface fraction and a soil-texture modifier that accounts for fine
// and coarse fractions, organic content, and porosity.
//
// References:
//   Allen, R.G., Pereira, L.S., Raes, D., Smith, M., 1998. Crop
//     evapotranspiration: guidelines for computing crop water requirements.
//     FAO Irrigation and Drainage Paper 56. FAO, Rome.
//     https://www.fao.org/3/x0490e/x0490e00.htm
//
//   Monsi, M., Saeki, T., 1953. Uber den Lichtfaktor in den Pflanzengesellschaften
//     und seine Bedeutung fur die Stoffproduktion. Japanese Journal of Botany 14,
//     22-52. (English translation: Annals of Botany 95, 549-567, 2005.
//     https://doi.org/10.1093/aob/mci052)

#include "marsh_model/processes/simple_canopy_et_model.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace marsh_model
{
namespace
{
double get_parameter_or_default(
    const parameter_set& parameters,
    const std::string& name,
    double default_value)
{
    if (parameters.has(name))
    {
        return parameters.get(name);
    }

    return default_value;
}

double clamp(double value, double lower, double upper)
{
    return std::max(lower, std::min(value, upper));
}
}

// Computes transpiration and evaporation fluxes for one time step.
// 1. Potential ET is computed from temperature and PAR.
// 2. Canopy cover (from LAI) sets the transpiration/evaporation split.
// 3. Transpiration is multiplied by a combined salinity-and-inundation stress.
// 4. Evaporation is scaled by exposed surface fraction and a soil-texture modifier.
et_fluxes simple_canopy_et_model::compute_et(
    const ecohydrology_state& eco_state,
    const hydrology_diagnostics& hydro,
    const sediment_surface_properties& surface,
    const forcing_step& forcing,
    const site_properties&,
    const parameter_set& parameters) const
{
    et_fluxes fluxes;

    if (forcing.dt_days <= 0.0)
    {
        return fluxes;
    }

    const double effective_lai =
        compute_effective_lai(eco_state, parameters);

    const double potential_et_mm_d =
        compute_potential_et_mm_d(forcing, parameters);

    const double cover_extinction_coefficient =
        get_parameter_or_default(
            parameters,
            "et_cover_extinction_coefficient",
            0.7);

    const double kc_min =
        get_parameter_or_default(
            parameters,
            "et_kc_min",
            0.2);

    const double kc_max =
        get_parameter_or_default(
            parameters,
            "et_kc_max",
            1.1);

    const double salinity_threshold_ppt =
        get_parameter_or_default(
            parameters,
            "et_salinity_threshold_ppt",
            20.0);

    const double salinity_stress_per_ppt =
        get_parameter_or_default(
            parameters,
            "et_salinity_stress_per_ppt",
            0.03);

    const double inundation_threshold_fraction =
        get_parameter_or_default(
            parameters,
            "et_inundation_threshold_fraction",
            0.25);

    const double inundation_stress_per_fraction =
        get_parameter_or_default(
            parameters,
            "et_inundation_stress_per_fraction",
            1.2);

    const double transpiration_fraction_min =
        get_parameter_or_default(
            parameters,
            "et_transpiration_fraction_min",
            0.15);

    const double transpiration_fraction_max =
        get_parameter_or_default(
            parameters,
            "et_transpiration_fraction_max",
            0.85);

    const double evaporation_fine_sensitivity =
        get_parameter_or_default(
            parameters,
            "et_evaporation_fine_sensitivity",
            0.25);

    const double evaporation_organic_sensitivity =
        get_parameter_or_default(
            parameters,
            "et_evaporation_organic_sensitivity",
            0.20);

    const double evaporation_porosity_sensitivity =
        get_parameter_or_default(
            parameters,
            "et_evaporation_porosity_sensitivity",
            0.15);

    const double evaporation_coarse_sensitivity =
        get_parameter_or_default(
            parameters,
            "et_evaporation_coarse_sensitivity",
            0.15);

    const double min_evaporation_modifier =
        get_parameter_or_default(
            parameters,
            "et_min_evaporation_modifier",
            0.2);

    const double min_transpiration_stress =
        get_parameter_or_default(
            parameters,
            "et_min_transpiration_stress",
            0.05);

    const double canopy_cover =
        1.0 - std::exp(-cover_extinction_coefficient * std::max(0.0, effective_lai));

    const double canopy_coefficient =
        kc_min + (kc_max - kc_min) * canopy_cover;

    const double salinity_excess_ppt =
        std::max(0.0, eco_state.root_zone_salinity_ppt - salinity_threshold_ppt);

    const double salinity_stress =
        std::max(
            min_transpiration_stress,
            std::exp(-salinity_stress_per_ppt * salinity_excess_ppt));

    const double inundation_excess_fraction =
        std::max(0.0, hydro.inundation_fraction - inundation_threshold_fraction);

    const double inundation_stress =
        std::max(
            min_transpiration_stress,
            std::exp(-inundation_stress_per_fraction * inundation_excess_fraction));

    const double transpiration_stress =
        salinity_stress * inundation_stress;

    const double transpiration_fraction =
        clamp(
            transpiration_fraction_min +
            (transpiration_fraction_max - transpiration_fraction_min) * canopy_cover,
            0.0,
            1.0);

    const double evaporation_fraction =
        1.0 - transpiration_fraction;

    const double evaporation_modifier =
        std::max(
            min_evaporation_modifier,
            1.0 +
            evaporation_fine_sensitivity * surface.fine_fraction +
            evaporation_organic_sensitivity * surface.organic_mass_fraction +
            evaporation_porosity_sensitivity * surface.porosity -
            evaporation_coarse_sensitivity * surface.coarse_fraction);

    const double exposed_surface_fraction =
        clamp(1.0 - hydro.inundation_fraction, 0.0, 1.0);

    fluxes.transpiration_mm_d =
        potential_et_mm_d *
        canopy_coefficient *
        transpiration_fraction *
        transpiration_stress;

    fluxes.evaporation_mm_d =
        potential_et_mm_d *
        evaporation_fraction *
        exposed_surface_fraction *
        evaporation_modifier;

    fluxes.total_et_mm_d =
        std::max(0.0, fluxes.transpiration_mm_d + fluxes.evaporation_mm_d);

    return fluxes;
}

// Returns the LAI stored in eco_state if it has been set (> 0); otherwise
// estimates it from aboveground biomass using a fixed coefficient.
double simple_canopy_et_model::compute_effective_lai(
    const ecohydrology_state& eco_state,
    const parameter_set& parameters) const
{
    if (eco_state.lai > 0.0)
    {
        return eco_state.lai;
    }

    const double biomass_to_lai_coefficient =
        get_parameter_or_default(
            parameters,
            "et_biomass_to_lai_coefficient",
            3.0);

    return std::max(
        0.0,
        biomass_to_lai_coefficient * eco_state.aboveground_biomass_kg_m2);
}

// Empirical potential ET (mm d^-1) as a linear combination of a background
// rate, a temperature-driven term (above a base temperature), and a PAR-driven
// term. This is intentionally simple - no Penman-Monteith aerodynamic terms.
double simple_canopy_et_model::compute_potential_et_mm_d(
    const forcing_step& forcing,
    const parameter_set& parameters) const
{
    const double temperature_base_c =
        get_parameter_or_default(
            parameters,
            "et_temperature_base_c",
            0.0);

    const double temperature_coefficient_mm_d_per_c =
        get_parameter_or_default(
            parameters,
            "et_temperature_coefficient_mm_d_per_c",
            0.12);

    const double par_coefficient_mm_d_per_umol_m2_d =
        get_parameter_or_default(
            parameters,
            "et_par_coefficient_mm_d_per_umol_m2_d",
            1.0e-4);

    const double et_background_mm_d =
        get_parameter_or_default(
            parameters,
            "et_background_mm_d",
            0.5);

    const double temperature_term =
        temperature_coefficient_mm_d_per_c *
        std::max(0.0, forcing.temperature - temperature_base_c);

    const double par_term =
        par_coefficient_mm_d_per_umol_m2_d *
        std::max(0.0, forcing.par_umol_m2_d);

    return std::max(0.0, et_background_mm_d + temperature_term + par_term);
}
}
