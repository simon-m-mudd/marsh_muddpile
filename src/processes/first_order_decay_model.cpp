// first_order_decay_model.cpp
//
// Part of marsh_muddpile-- https://github.com/simon-m-mudd/marsh_muddpile
//
// Copyright (C) 2026 Simon M. Mudd
// Released under the GNU General Public Licence v3 (GPL-3.0)
// See LICENSE file or https://www.gnu.org/licenses/gpl-3.0.html
//
// -----------------------------------------------------------------------------
// this component implements a column-scale first-order decay model for marsh
// sediment materials.
//
// this is a modern replacement for the older decay logic that was spread across
// depo_particle, sediment_layer, and sediment_stack.
//
// the model applies decay to any material that has decay_properties in the
// material catalog. This avoids hard-coded material indices and supports new
// pools and tracers more naturally.
//
// the rate law is:
//
//   k(z, T) = k_0 * depth_modifier(z) * temperature_modifier(T)
//
// where depth attenuation follows an exponential form when gamma > 0, similar
// to the older marsh model code, and the decay update can use either an exact
// exponential solution or a second-order approximation.
//
// this version includes a safe OpenMP parallelisation over layers, because
// each layer-material mass update is independent within a timestep.
//
// -----------------------------------------------------------------------------

#include "marsh_model/processes/first_order_decay_model.hpp"

#include "marsh_model/core/material_properties.hpp"

#include <algorithm>
#include <cmath>

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
}

void first_order_decay_model::apply_decay(
    column_state& state,
    const forcing_step& forcing,
    const material_catalog& catalog,
    const parameter_set& parameters) const
{
    const int n_layers = state.n_layers();
    const int n_materials = state.n_materials();

    if (n_layers == 0 || n_materials == 0)
    {
        return;
    }

    auto& mass = state.mass();
    const auto& layer_top_elevation = state.layer_top_elevation();
    const auto& layer_thickness = state.layer_thickness();
    const double surface_elevation = state.get_surface_elevation();

#ifdef marsh_muddpile_use_openmp
#pragma omp parallel for schedule(static) if(n_layers > 32)
#endif
    for (int layer_index = 0; layer_index < n_layers; ++layer_index)
    {
        const double midpoint_depth_m =
            surface_elevation -
            layer_top_elevation(layer_index) +
            0.5 * layer_thickness(layer_index);

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

            const double multiplier =
                compute_decay_multiplier(
                    decay_rate_per_day,
                    forcing.dt_days,
                    parameters);

            mass(layer_index, material_index) =
                std::max(0.0, current_mass * multiplier);
        }
    }
}

// Compute midpoint depth below the current marsh surface.
//
// Layer ordering assumption:
//   - layer 0 is deepest
//   - layer n_layers - 1 is the surface layer
//
// Surface elevation is taken as the top elevation of the uppermost layer.
double first_order_decay_model::compute_layer_midpoint_depth_m(
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

// Compute the first-order decay rate in units of day^-1.
//
// This follows the structure used in the older marsh model:
//
//   if gamma == 0:
//        k = k_0
//    else:
//        k = k_0 * exp( -z / gamma )
//
// A temperature modifier can also be applied when the material is marked as
// temperature_sensitive.
//
// For now the temperature response is linear in temperature anomaly:
//
//   k(T) = k * ( 1 + alpha * delta_T )
//
// where:
//   delta_T = temperature - reference_temperature
//
// This is simple, transparent, and similar in spirit to the older code, but
// can later be replaced by a Q10 or Arrhenius formulation without changing
// the decay_model interface.
double first_order_decay_model::compute_decay_rate_per_day(
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
            get_parameter_or_default(parameters, "decay_reference_temperature_c", 20.0);

        const double temperature_factor_per_degree =
            get_parameter_or_default(parameters, "decay_temperature_factor_per_degree", 0.0);

        const double delta_temperature =
            temperature - reference_temperature_c;

        decay_rate_per_day *=
            std::max(0.0, 1.0 + temperature_factor_per_degree * delta_temperature);
    }

    return std::max(0.0, decay_rate_per_day);
}

// Compute the decay multiplier over one timestep.
//
// Two update forms are supported:
//
//   exact:
//       m_new = m_old * exp( -k dt )
//
//   second-order approximation:
//       m_new = m_old * ( 1 - k dt + 0.5 (k dt)^2 )
//
// The second-order form is included because the old code used it in places
// for efficiency. The exact solution should normally be preferred unless
// profiling shows a real need for approximation.
//
// The switch is controlled by:
//   decay_use_exact_solution
//
// with:
//   > 0.5  -> use exact exponential
//   <= 0.5 -> use second-order approximation
double first_order_decay_model::compute_decay_multiplier(
    double decay_rate_per_day,
    double dt_days,
    const parameter_set& parameters) const
{
    const double use_exact_solution =
        get_parameter_or_default(parameters, "decay_use_exact_solution", 1.0);

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

    // Numerical safety: prevent a negative multiplier if someone uses a large
    // timestep with the second-order approximation.
    return std::max(0.0, multiplier);
}
}
