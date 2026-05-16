#include "marsh_model/processes/distance_flushing_salinity_model.hpp"

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

void distance_flushing_salinity_model::update_salinity(
    ecohydrology_state& eco_state,
    const hydrology_diagnostics& hydro,
    const et_fluxes& et,
    const sediment_surface_properties& surface,
    const forcing_step& forcing,
    const site_properties& site,
    const parameter_set& parameters) const
{
    if (forcing.dt_days <= 0.0)
    {
        return;
    }

    const double min_salinity_ppt =
        get_parameter_or_default(parameters, "salinity_min_ppt", 0.0);

    const double max_salinity_ppt =
        get_parameter_or_default(parameters, "salinity_max_ppt", 80.0);

    const double initial_root_zone_salinity_ppt =
        get_parameter_or_default(parameters, "salinity_initial_root_zone_ppt", -1.0);

    const double default_creek_salinity_ppt =
        get_parameter_or_default(parameters, "salinity_default_creek_ppt", 30.0);

    const double et_concentration_rate_ppt_per_mm =
        get_parameter_or_default(
            parameters,
            "salinity_et_concentration_rate_ppt_per_mm",
            0.020);

    const double precipitation_dilution_rate_ppt_per_mm =
        get_parameter_or_default(
            parameters,
            "salinity_precipitation_dilution_rate_ppt_per_mm",
            0.030);

    const double freshwater_dilution_rate_ppt_per_mm =
        get_parameter_or_default(
            parameters,
            "salinity_freshwater_dilution_rate_ppt_per_mm",
            0.030);

    const double base_flushing_rate_per_day =
        get_parameter_or_default(
            parameters,
            "salinity_base_flushing_rate_per_day",
            0.50);

    const double flushing_efolding_distance_m =
        get_parameter_or_default(
            parameters,
            "salinity_flushing_efolding_distance_m",
            50.0);

    const double exchange_coarse_sensitivity =
        get_parameter_or_default(
            parameters,
            "salinity_exchange_coarse_sensitivity",
            0.75);

    const double exchange_organic_sensitivity =
        get_parameter_or_default(
            parameters,
            "salinity_exchange_organic_sensitivity",
            0.50);

    const double storage_fine_sensitivity =
        get_parameter_or_default(
            parameters,
            "salinity_storage_fine_sensitivity",
            1.00);

    const double storage_organic_sensitivity =
        get_parameter_or_default(
            parameters,
            "salinity_storage_organic_sensitivity",
            1.00);

    const double storage_porosity_sensitivity =
        get_parameter_or_default(
            parameters,
            "salinity_storage_porosity_sensitivity",
            0.50);

    const double min_exchange_modifier =
        get_parameter_or_default(
            parameters,
            "salinity_min_exchange_modifier",
            0.10);

    const double min_storage_modifier =
        get_parameter_or_default(
            parameters,
            "salinity_min_storage_modifier",
            0.25);

    const double creek_salinity_ppt =
        (forcing.creek_salinity_ppt > 0.0)
            ? forcing.creek_salinity_ppt
            : default_creek_salinity_ppt;

    double current_salinity_ppt = eco_state.root_zone_salinity_ppt;

    if (current_salinity_ppt <= 0.0)
    {
        current_salinity_ppt =
            (initial_root_zone_salinity_ppt >= 0.0)
                ? initial_root_zone_salinity_ppt
                : creek_salinity_ppt;
    }

    const double distance_from_creek_m =
        std::max(0.0, site.distance_from_creek_m);

    double distance_modifier = 1.0;
    if (flushing_efolding_distance_m > 0.0)
    {
        distance_modifier =
            std::exp(-distance_from_creek_m / flushing_efolding_distance_m);
    }

    const double exchange_modifier =
        std::max(
            min_exchange_modifier,
            1.0 +
            exchange_coarse_sensitivity * surface.coarse_fraction -
            exchange_organic_sensitivity * surface.organic_mass_fraction);

    const double storage_modifier =
        std::max(
            min_storage_modifier,
            1.0 +
            storage_fine_sensitivity * surface.fine_fraction +
            storage_organic_sensitivity * surface.organic_mass_fraction +
            storage_porosity_sensitivity * surface.porosity);

    const double local_concentration_tendency_ppt_per_day =
        (et_concentration_rate_ppt_per_mm *
             std::max(0.0, et.total_et_mm_d) -
         precipitation_dilution_rate_ppt_per_mm *
             std::max(0.0, forcing.precipitation_mm_d) -
         freshwater_dilution_rate_ppt_per_mm *
             std::max(0.0, forcing.freshwater_input_mm_d)) /
        storage_modifier;

    const double salinity_relaxation_tendency_ppt_per_day =
        base_flushing_rate_per_day *
        distance_modifier *
        std::max(0.0, hydro.inundation_fraction) *
        exchange_modifier *
        (creek_salinity_ppt - current_salinity_ppt) /
        storage_modifier;

    // Subsurface lateral exchange: slow background relaxation toward creek
    // salinity driven by porewater diffusion and lateral advection through
    // the saturated column.  This is independent of tidal inundation and
    // prevents runaway concentration on sites that are rarely overtopped.
    // Rate is attenuated by the same distance modifier as tidal flushing.
    // Default 0.0 preserves the original behaviour; typical calibrated
    // values for a site 30–60 m from a creek are 0.002–0.005 d⁻¹.
    const double subsurface_flushing_rate_per_day =
        get_parameter_or_default(
            parameters,
            "salinity_subsurface_flushing_rate_per_day",
            0.0);

    const double subsurface_relaxation_tendency_ppt_per_day =
        subsurface_flushing_rate_per_day *
        distance_modifier *
        (creek_salinity_ppt - current_salinity_ppt) /
        storage_modifier;

    // Total relaxation rate toward creek salinity (d^-1).
    // Both tidal exchange and subsurface diffusion relax S toward creek_salinity_ppt.
    const double total_relaxation_rate_per_day =
        (base_flushing_rate_per_day *
             distance_modifier *
             std::max(0.0, hydro.inundation_fraction) *
             exchange_modifier +
         subsurface_flushing_rate_per_day * distance_modifier) /
        storage_modifier;

    // Exact solution to the linear ODE:
    //   dS/dt = source - rate * (S - creek)
    // where source = local_concentration_tendency (ET minus precip)
    //       rate   = total_relaxation_rate_per_day
    //
    // Equilibrium: S_eq = creek + source / rate  (when rate > 0)
    // S(t+dt) = S_eq + (S0 - S_eq) * exp(-rate * dt)
    //
    // Falls back to forward-Euler only when rate == 0 (no flushing at all),
    // where the exact and Euler solutions are identical.
    double updated_salinity_ppt;
    if (total_relaxation_rate_per_day > 0.0)
    {
        const double source_ppt_per_day =
            local_concentration_tendency_ppt_per_day;   // already / storage_modifier
        const double s_eq =
            creek_salinity_ppt + source_ppt_per_day / total_relaxation_rate_per_day;
        updated_salinity_ppt =
            s_eq +
            (current_salinity_ppt - s_eq) *
                std::exp(-total_relaxation_rate_per_day * forcing.dt_days);
    }
    else
    {
        updated_salinity_ppt =
            current_salinity_ppt +
            forcing.dt_days * local_concentration_tendency_ppt_per_day;
    }

    eco_state.root_zone_salinity_ppt =
        clamp(updated_salinity_ppt, min_salinity_ppt, max_salinity_ppt);
}
}
