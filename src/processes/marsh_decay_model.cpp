#include "marsh_model/processes/marsh_decay_model.hpp"

#include "marsh_model/engine/surface_property_summarizer.hpp"
#include "marsh_model/processes/hydro_salinity_decomposition_modifier_model.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#ifdef marsh_muddpile_use_openmp
#include <omp.h>
#endif

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

hydrology_diagnostics build_fallback_hydrology(
    const column_state& state,
    const forcing_step& forcing,
    const site_properties& site)
{
    hydrology_diagnostics hydro;

    const double surface_elevation_m =
        state.get_surface_elevation();

    const double dt_hours =
        std::max(0.0, 24.0 * forcing.dt_days);

    if (forcing.has_observed_water_level)
    {
        const double water_level_m =
            forcing.observed_water_level_m + site.local_tidal_offset_m;

        const double inundation_depth_m =
            std::max(0.0, water_level_m - surface_elevation_m);

        hydro.mean_water_level_m = water_level_m;
        hydro.max_water_level_m = water_level_m;
        hydro.inundation_fraction = (inundation_depth_m > 0.0) ? 1.0 : 0.0;
        hydro.mean_inundation_depth_m = inundation_depth_m;
        hydro.inundation_duration_hours = hydro.inundation_fraction * dt_hours;
        hydro.storm_surge_residual_m = 0.0;

        return hydro;
    }

    const double mean_water_level_m =
        forcing.mean_sea_level +
        site.local_tidal_offset_m +
        forcing.storm_surge_residual_m;

    hydro.mean_water_level_m = mean_water_level_m;
    hydro.max_water_level_m =
        mean_water_level_m + std::abs(forcing.tidal_amplitude);
    hydro.storm_surge_residual_m = forcing.storm_surge_residual_m;

    if (forcing.tidal_amplitude <= 0.0)
    {
        const double inundation_depth_m =
            std::max(0.0, mean_water_level_m - surface_elevation_m);

        hydro.inundation_fraction = (inundation_depth_m > 0.0) ? 1.0 : 0.0;
        hydro.mean_inundation_depth_m = inundation_depth_m;
        hydro.inundation_duration_hours = hydro.inundation_fraction * dt_hours;
        return hydro;
    }

    const double argument =
        (surface_elevation_m - mean_water_level_m) / forcing.tidal_amplitude;

    if (argument <= -1.0)
    {
        hydro.inundation_fraction = 1.0;
    }
    else if (argument >= 1.0)
    {
        hydro.inundation_fraction = 0.0;
    }
    else
    {
        hydro.inundation_fraction =
            0.5 - std::asin(argument) / 3.14159265358979323846;
    }

    hydro.inundation_fraction =
        std::max(0.0, std::min(1.0, hydro.inundation_fraction));

    hydro.mean_inundation_depth_m =
        std::max(0.0, mean_water_level_m - surface_elevation_m);

    hydro.inundation_duration_hours =
        hydro.inundation_fraction * dt_hours;

    return hydro;
}

double get_default_root_zone_salinity_ppt(
    const forcing_step& forcing,
    const parameter_set& parameters)
{
    if (forcing.creek_salinity_ppt > 0.0)
    {
        return forcing.creek_salinity_ppt;
    }

    return get_parameter_or_default(
        parameters,
        "salinity_default_creek_ppt",
        30.0);
}

double compute_pool_modifier(
    const material_properties& material,
    const decomposition_modifiers& modifiers)
{
    switch (material.category)
    {
        case material_category::organic_labile:
            return modifiers.labile_multiplier;

        case material_category::organic_refractory:
            return modifiers.refractory_multiplier;

        case material_category::live_root:
            return modifiers.root_multiplier;

        default:
            return 1.0;
    }
}
}

void marsh_decay_model::apply_decay(
    column_state& state,
    const forcing_step& forcing,
    const material_catalog& catalog,
    const parameter_set& parameters,
    const decay_context& context) const
{
    const int n_layers = state.n_layers();
    const int n_materials = state.n_materials();

    if (n_layers == 0 || n_materials == 0)
    {
        return;
    }

    // -------------------------------------------------------------------------
    // Build the best available ecohydrologic context.
    //
    // Priority:
    //   1. provided context from simulator
    //   2. fallback estimates from current state + forcing
    // -------------------------------------------------------------------------
    ecohydrology_state fallback_eco_state;
    if (context.eco_state)
    {
        fallback_eco_state = *context.eco_state;
    }
    else
    {
        fallback_eco_state.root_zone_salinity_ppt =
            get_default_root_zone_salinity_ppt(forcing, parameters);
    }

    site_properties fallback_site;
    if (context.site)
    {
        fallback_site = *context.site;
    }

    sediment_surface_properties fallback_surface =
        context.surface
            ? *context.surface
            : surface_property_summarizer::summarize(
                  state,
                  catalog,
                  parameters);

    hydrology_diagnostics fallback_hydro =
        context.hydrology
            ? *context.hydrology
            : build_fallback_hydrology(
                  state,
                  forcing,
                  fallback_site);

    hydro_salinity_decomposition_modifier_model modifier_model;

    const decomposition_modifiers modifiers =
        modifier_model.compute_modifiers(
            fallback_eco_state,
            fallback_hydro,
            fallback_surface,
            forcing,
            fallback_site,
            parameters);

    auto& mass = state.mass();

#ifdef marsh_muddpile_use_openmp
#pragma omp parallel for schedule(static) if(n_layers > 32)
#endif
    for (int layer_index = 0; layer_index < n_layers; ++layer_index)
    {
        const double midpoint_depth_m =
            compute_layer_midpoint_depth_m(state, layer_index);

        for (int material_index = 0; material_index < n_materials; ++material_index)
        {
            const material_properties& material =
                catalog.get_material(material_index);

            if (!material.decay.has_value())
            {
                continue;
            }

            const double current_mass =
                mass(layer_index, material_index);

            if (current_mass <= 0.0)
            {
                continue;
            }

            const double decay_rate_per_day =
                compute_decay_rate_per_day(
                    material,
                    midpoint_depth_m,
                    forcing.temperature,
                    parameters);

            if (decay_rate_per_day <= 0.0)
            {
                continue;
            }

            const double base_multiplier =
                compute_decay_multiplier(
                    decay_rate_per_day,
                    forcing.dt_days,
                    parameters);

            const double pool_modifier =
                compute_pool_modifier(material, modifiers);

            const double combined_multiplier =
                std::max(0.0, base_multiplier * pool_modifier);

            mass(layer_index, material_index) =
                std::max(0.0, current_mass * combined_multiplier);
        }
    }
}

double marsh_decay_model::compute_layer_midpoint_depth_m(
    const column_state& state,
    int layer_index) const
{
    const double surface_elevation =
        state.get_surface_elevation();

    const double layer_top_elevation =
        state.layer_top_elevation()(layer_index);

    const double layer_thickness =
        state.layer_thickness()(layer_index);

    return surface_elevation - layer_top_elevation + 0.5 * layer_thickness;
}

double marsh_decay_model::compute_decay_rate_per_day(
    const material_properties& material,
    double midpoint_depth_m,
    double temperature,
    const parameter_set& parameters) const
{
    const decay_properties& decay = material.decay.value();

    double decay_rate_per_day = decay.k_0;

    if (decay_rate_per_day <= 0.0)
    {
        return 0.0;
    }

    if (decay.gamma > 0.0)
    {
        decay_rate_per_day *= std::exp(-midpoint_depth_m / decay.gamma);
    }

    if (decay.temperature_sensitive)
    {
        const double reference_temperature_c =
            get_parameter_or_default(
                parameters,
                "decay_reference_temperature_c",
                20.0);

        const double temperature_factor_per_degree =
            get_parameter_or_default(
                parameters,
                "decay_temperature_factor_per_degree",
                0.0);

        const double delta_temperature =
            temperature - reference_temperature_c;

        decay_rate_per_day *=
            std::max(
                0.0,
                1.0 + temperature_factor_per_degree * delta_temperature);
    }

    return std::max(0.0, decay_rate_per_day);
}

double marsh_decay_model::compute_decay_multiplier(
    double decay_rate_per_day,
    double dt_days,
    const parameter_set& parameters) const
{
    const double use_exact_solution =
        get_parameter_or_default(
            parameters,
            "decay_use_exact_solution",
            1.0);

    const double k_dt = decay_rate_per_day * dt_days;

    double multiplier = 1.0;

    if (use_exact_solution > 0.5)
    {
        multiplier = std::exp(-k_dt);
    }
    else
    {
        multiplier = 1.0 - k_dt + 0.5 * k_dt * k_dt;
    }

    return std::max(0.0, multiplier);
}
}
