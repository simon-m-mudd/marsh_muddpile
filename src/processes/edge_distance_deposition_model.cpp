// edge_distance_deposition_model.cpp
//
// Part of marsh_muddpile-- https://github.com/simon-m-mudd/marsh_muddpile
//
// Copyright (C) 2026 Simon M. Mudd
// Released under the GNU General Public Licence v3 (GPL-3.0)
// See LICENSE file or https://www.gnu.org/licenses/gpl-3.0.html
//
// -----------------------------------------------------------------------------
// this component implements a vegetation-independent marsh surface deposition
// model based on edge concentration, marsh-edge distance, hydroperiod, and
// settling velocity.
//
// the formulation is motivated by the non-peer-reviewed preprint:
//
//   Lester et al., "Vegetation Does Not Control Suspended Sediment Deposition
//   in Salt Marshes"
//
// in that framework, tidal-marsh suspended sediment concentrations and
// deposition rates decay inland approximately exponentially away from the
// marsh edge. This implementation translates that idea into a simple 1D
// column-scale deposition model.
//
// this module intentionally ignores vegetation biomass.
//
// -----------------------------------------------------------------------------

#include "marsh_model/processes/edge_distance_deposition_model.hpp"

#include "marsh_model/core/tidal_utilities.hpp"

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
}

Eigen::ArrayXd edge_distance_deposition_model::compute_surface_flux(
    const column_state& state,
    const forcing_step& forcing,
    const biomass_fluxes&,
    const material_catalog& catalog,
    const parameter_set& parameters) const
{
    Eigen::ArrayXd surface_flux = Eigen::ArrayXd::Zero(catalog.size());

    if (catalog.size() == 0 || forcing.dt_days <= 0.0 || forcing.tidal_period_hours <= 0.0)
    {
        return surface_flux;
    }

    const double distance_from_edge_m =
        get_parameter_or_default(parameters, "deposition_distance_from_edge_m", 0.0);

    const double surface_elevation_m = state.get_surface_elevation();

    // Precompute per-material quantities that don't vary with tidal cycle.
    struct material_deposit_props
    {
        bool active = false;
        double local_concentration_kg_m3 = 0.0;
        double settling_velocity_m_s = 0.0;
        double decay_length_m = 0.0;
    };

    const int n_materials = catalog.size();
    std::vector<material_deposit_props> mat_props(n_materials);

    bool any_active = false;
    for (int i = 0; i < n_materials; ++i)
    {
        const material_properties& material = catalog.get_material(i);

        if (!material.allow_surface_deposition || !material.settling.has_value())
            continue;

        const double ws = material.settling->settling_velocity;
        if (ws <= 0.0)
            continue;

        const double edge_c = compute_material_concentration_kg_m3(
            material, forcing, parameters);
        if (edge_c <= 0.0)
            continue;

        const double L = compute_decay_length_m(ws, forcing, parameters);

        double local_c = edge_c;
        if (L > 0.0)
            local_c *= std::exp(-distance_from_edge_m / L);

        mat_props[i].active = true;
        mat_props[i].local_concentration_kg_m3 = local_c;
        mat_props[i].settling_velocity_m_s = ws;
        mat_props[i].decay_length_m = L;
        any_active = true;
    }

    if (any_active)
    {
        // Pre-read harmonic constituent data once — avoids repeated map lookups
        // inside the cycle loop.
        const tidal_utils::tidal_data tidal =
            tidal_utils::prepare_tidal_data(forcing, parameters);

        // Integrate over each tidal cycle within the timestep.  For each cycle,
        // compute the harmonic envelope (spring-neap varying amplitude and mean),
        // then derive the inundation fraction analytically from the sinusoidal
        // approximation with those cycle-specific parameters.
        const int n_cycles = std::max(
            1,
            static_cast<int>(std::round(24.0 * forcing.dt_days / forcing.tidal_period_hours)));

        const double step_start_hours = 24.0 * forcing.model_time_days;

        for (int cycle = 0; cycle < n_cycles; ++cycle)
        {
            const double cycle_start_hours =
                step_start_hours + cycle * forcing.tidal_period_hours;

            const tidal_utils::cycle_envelope env =
                tidal_utils::compute_cycle_envelope(
                    cycle_start_hours,
                    forcing.tidal_period_hours,
                    tidal);

            const double inundation_fraction =
                compute_inundation_fraction(
                    surface_elevation_m,
                    env.amplitude_m,
                    env.mean_m);

            const double inundation_seconds =
                inundation_fraction * forcing.tidal_period_hours * 3600.0;

            if (inundation_seconds <= 0.0)
                continue;

            for (int i = 0; i < n_materials; ++i)
            {
                if (!mat_props[i].active)
                    continue;

                surface_flux(i) +=
                    mat_props[i].settling_velocity_m_s *
                    mat_props[i].local_concentration_kg_m3 *
                    inundation_seconds;
            }
        }
    }

    if (catalog.has_material("pb210"))
    {
        const int pb210_index = catalog.get_material_index("pb210");
        surface_flux(pb210_index) += forcing.external_pb210_supply * forcing.dt_days;
    }

    return surface_flux;
}

double edge_distance_deposition_model::compute_inundation_fraction(
    double surface_elevation_m,
    double cycle_amplitude_m,
    double cycle_mean_m) const
{
    if (cycle_amplitude_m <= 0.0)
    {
        return (cycle_mean_m > surface_elevation_m) ? 1.0 : 0.0;
    }

    const double relative_elevation =
        (surface_elevation_m - cycle_mean_m) / cycle_amplitude_m;

    if (relative_elevation <= -1.0)
        return 1.0;
    if (relative_elevation >= 1.0)
        return 0.0;

    constexpr double pi = 3.14159265358979323846;
    return std::clamp(0.5 - std::asin(relative_elevation) / pi, 0.0, 1.0);
}

double edge_distance_deposition_model::compute_decay_length_m(
    double settling_velocity_m_s,
    const forcing_step& forcing,
    const parameter_set& parameters) const
{
    const double basin_length_m =
        get_parameter_or_default(parameters, "deposition_basin_length_m", 50.0);

    const double beta =
        get_parameter_or_default(parameters, "deposition_length_scale_beta", 3.0);

    const double tidal_amplitude_m =
        std::max(0.0, forcing.tidal_amplitude);

    const double tidal_period_s =
        std::max(0.0, forcing.tidal_period_hours) * 3600.0;

    if (settling_velocity_m_s <= 0.0 || tidal_period_s <= 0.0 || basin_length_m <= 0.0)
    {
        return 0.0;
    }

    // Following the scaling argued in the Lester et al. preprint:
    //
    //   L_D ~ beta * L * (delta / T) / w_s
    //
    // where:
    //   L_D = deposition decay length
    //   L   = marsh basin length scale
    //   delta = tidal amplitude
    //   T   = tidal period
    //   w_s = settling velocity
    return beta * basin_length_m * (tidal_amplitude_m / tidal_period_s) / settling_velocity_m_s;
}

double edge_distance_deposition_model::compute_material_concentration_kg_m3(
    const material_properties& material,
    const forcing_step& forcing,
    const parameter_set& parameters) const
{
    const std::string specific_parameter_name =
        "deposition_concentration_" + material.name + "_kg_m3";

    if (parameters.has(specific_parameter_name))
    {
        return std::max(0.0, parameters.get(specific_parameter_name));
    }

    if (material.name == "silt")
    {
        return std::max(0.0, forcing.suspended_sediment_concentration);
    }

    if (material.name == "fine_silt" || material.name == "fine_sediment")
    {
        return std::max(0.0, forcing.fine_sediment_concentration);
    }

    return 0.0;
}
}
