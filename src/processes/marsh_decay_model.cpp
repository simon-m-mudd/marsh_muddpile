// marsh_decay_model.cpp
//
// Part of marsh_muddpile -- https://github.com/simon-m-mudd/marsh_muddpile
//
// Copyright (C) 2026 Simon M. Mudd
// Released under the GNU General Public Licence v3 (GPL-3.0)
// See LICENSE file or https://www.gnu.org/licenses/gpl-3.0.html
//
// -----------------------------------------------------------------------------
// Organic matter decay for the marsh sediment column.
//
// Each time step, every organic layer loses mass according to a first-order
// decay law.  The per-material decay rate is:
//
//   k_eff = k_0 * f_depth(z) * f_temperature(T) * modifier
//
// where:
//   k_0            - surface decay rate from material catalogue (d-1)
//   f_depth(z)     - exponential attenuation with depth below surface:
//                    exp(-z / gamma).  gamma = 0 disables depth attenuation.
//   f_temperature  - linear temperature sensitivity above a reference value:
//                    1 + factor * (T - T_ref).  Disabled when factor = 0.
//   modifier       - dimensionless multiplier from
//                    hydro_salinity_decomposition_modifier_model, which
//                    accounts for anoxia (inundation) and salinity inhibition
//                    separately for labile, refractory, and root-derived pools.
//
// The mass remaining at the end of the step is computed either exactly:
//   M(t+dt) = M(t) * exp(-k_eff * dt)
// or with a second-order Taylor approximation when
// decay_use_exact_solution = 0:
//   M(t+dt) = M(t) * (1 - k_eff*dt + 0.5*(k_eff*dt)^2)
//
// Only materials whose catalogue entry carries decay properties are touched;
// mineral fractions and live roots are skipped.
//
// References:
//   Berner, R.A., 1980. Early Diagenesis: A Theoretical Approach.
//     Princeton University Press, Princeton NJ.
//     (depth-dependent decay: chapter 2)
//
//   Olson, J.S., 1963. Energy storage and the balance of producers and
//     decomposers in ecological systems. Ecology 44, 322-331.
//     https://doi.org/10.2307/1932179
//     (first-order decay model for organic matter)
//
//   Morris, J.T., Bowden, W.B., 1986. A mechanistic, numerical model of
//     sedimentation, mineralisation, and decomposition for marsh sediments.
//     Soil Science Society of America Journal 50, 96-105.
//     https://doi.org/10.2136/sssaj1986.03615995005000010019x
//
//   Mudd, S.M., Howell, S.M., Morris, J.T., 2009. Impact of dynamic feedbacks
//     between sedimentation, sea-level rise, and biomass production on near-surface
//     marsh stratigraphy and carbon accumulation. Estuarine, Coastal and Shelf
//     Science 82, 377-389.
//     https://doi.org/10.1016/j.ecss.2009.01.028
//
//   Kirwan, M.L., Mudd, S.M., 2012. Response of salt-marsh carbon accumulation
//     to climate change. Nature 489, 550-553.
//     https://doi.org/10.1038/nature11440
// -----------------------------------------------------------------------------

#include "marsh_model/processes/marsh_decay_model.hpp"

#include "marsh_model/engine/surface_property_summariser.hpp"
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

// Constructs a minimal hydrology_diagnostics when one has not been supplied by
// the simulator (e.g. when decay is called standalone in tests).  If the
// forcing carries an observed water level it is used directly; otherwise the
// inundation fraction is estimated analytically from the tidal amplitude and
// surface elevation using the arcsine relationship for a sinusoidal tide:
//   inundation_fraction = 0.5 - arcsin((z - MSL) / amplitude) / pi
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

// Returns the best available root-zone salinity when none has been set by the
// salinity model.  Uses the creek salinity from the forcing step if positive,
// otherwise falls back to the salinity_default_creek_ppt parameter.
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

// Maps a material's category to the correct pre-computed modifier from
// hydro_salinity_decomposition_modifier_model.  Mineral materials and any
// unrecognised categories return 1.0 (no modification).
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

// Main entry point called once per time step by the simulator.
// Iterates over every layer-material pair, computes the effective decay rate,
// and multiplies the stored mass by the survival fraction for the step.
// Layers with no decayable material and mineral materials are skipped.
// The layer loop is optionally parallelised with OpenMP for columns with more
// than 32 layers.
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
            : surface_property_summariser::summarize(
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

    // Initialise output flux array if requested.
    if (context.out_fluxes != nullptr)
    {
        context.out_fluxes->mass_decomposed_kg_m2 =
            Eigen::ArrayXXd::Zero(n_layers, n_materials);
    }

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

            const double pool_modifier =
                compute_pool_modifier(material, modifiers);

            // pool_modifier scales the effective decay rate (modifier = 1 is
            // baseline aerobic rate; < 1 slows decay under anoxia or salinity).
            // It must be applied to the rate before computing the survival
            // fraction, not multiplied with the survival fraction afterward.
            // Multiplying survival fractions is mathematically wrong:
            //   exp(-k*dt) * m  !=  exp(-k*m*dt)
            // and turns a modest rate reduction into catastrophic mass loss
            // for slow-decaying pools (e.g. refractory with k=0.001 d-1 and
            // m=0.59 would give 41 % decay per day instead of 0.06 %).
            const double effective_decay_rate_per_day =
                decay_rate_per_day * pool_modifier;

            const double combined_multiplier =
                compute_decay_multiplier(
                    effective_decay_rate_per_day,
                    forcing.dt_days,
                    parameters);

            const double new_mass =
                std::max(0.0, current_mass * combined_multiplier);

            mass(layer_index, material_index) = new_mass;

            if (context.out_fluxes != nullptr)
            {
                context.out_fluxes->mass_decomposed_kg_m2(
                    layer_index, material_index) =
                    current_mass - new_mass;
            }
        }
    }
}

// Returns the depth of the layer midpoint below the sediment surface (m).
// Used to evaluate the depth-attenuation term in compute_decay_rate_per_day.
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

// Computes the effective first-order decay rate (d-1) for one material in one
// layer, applying depth and temperature corrections to the catalogue rate k_0:
//
//   k_eff = k_0 * exp(-z / gamma) * max(0, 1 + factor * (T - T_ref))
//
// gamma = 0 skips depth attenuation; temperature_factor_per_degree = 0 (the
// default) skips temperature sensitivity, giving a purely depth-dependent rate.
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

// Returns the fraction of mass surviving one time step (dimensionless, 0-1).
// The exact exponential solution exp(-k*dt) is used by default.  Setting
// decay_use_exact_solution = 0 switches to a second-order Taylor expansion,
// which is cheaper but becomes inaccurate when k*dt is large.
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
