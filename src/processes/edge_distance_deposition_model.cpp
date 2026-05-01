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

    const double cycles_per_timestep =
        compute_cycles_per_timestep(forcing);

    const double inundation_fraction_per_cycle =
        compute_inundation_fraction_per_cycle(state, forcing);

    const double inundation_seconds_per_cycle =
        inundation_fraction_per_cycle * forcing.tidal_period_hours * 3600.0;

    if (inundation_seconds_per_cycle <= 0.0 || cycles_per_timestep <= 0.0)
    {
        if (catalog.has_material("pb210"))
        {
            const int pb210_index = catalog.get_material_index("pb210");
            surface_flux(pb210_index) += forcing.external_pb210_supply * forcing.dt_days;
        }

        return surface_flux;
    }

    for (int material_index = 0; material_index < catalog.size(); ++material_index)
    {
        const material_properties& material =
            catalog.get_material(material_index);

        if (!material.allow_surface_deposition)
        {
            continue;
        }

        if (!material.settling.has_value())
        {
            continue;
        }

        const double settling_velocity_m_s =
            material.settling->settling_velocity;

        if (settling_velocity_m_s <= 0.0)
        {
            continue;
        }

        const double edge_concentration_kg_m3 =
            compute_material_concentration_kg_m3(
                material,
                forcing,
                parameters);

        if (edge_concentration_kg_m3 <= 0.0)
        {
            continue;
        }

        const double decay_length_m =
            compute_decay_length_m(
                settling_velocity_m_s,
                forcing,
                parameters);

        double local_concentration_kg_m3 = edge_concentration_kg_m3;

        if (decay_length_m > 0.0)
        {
            local_concentration_kg_m3 *=
                std::exp(-distance_from_edge_m / decay_length_m);
        }

        const double deposited_mass_per_cycle_kg_m2 =
            settling_velocity_m_s *
            local_concentration_kg_m3 *
            inundation_seconds_per_cycle;

        surface_flux(material_index) =
            cycles_per_timestep * deposited_mass_per_cycle_kg_m2;
    }

    // Optional direct external supply term for pb210.
    //
    // Convention:
    // forcing.external_pb210_supply is interpreted as kg m^-2 day^-1.
    if (catalog.has_material("pb210"))
    {
        const int pb210_index = catalog.get_material_index("pb210");
        surface_flux(pb210_index) += forcing.external_pb210_supply * forcing.dt_days;
    }

    return surface_flux;
}

double edge_distance_deposition_model::compute_cycles_per_timestep(
    const forcing_step& forcing) const
{
    if (forcing.tidal_period_hours <= 0.0)
    {
        return 0.0;
    }

    return 24.0 * forcing.dt_days / forcing.tidal_period_hours;
}

double edge_distance_deposition_model::compute_inundation_fraction_per_cycle(
    const column_state& state,
    const forcing_step& forcing) const
{
    const double tidal_amplitude_m = forcing.tidal_amplitude;
    const double mean_sea_level_m = forcing.mean_sea_level;
    const double surface_elevation_m = state.get_surface_elevation();

    if (tidal_amplitude_m <= 0.0)
    {
        return (mean_sea_level_m > surface_elevation_m) ? 1.0 : 0.0;
    }

    const double relative_elevation =
        (surface_elevation_m - mean_sea_level_m) / tidal_amplitude_m;

    if (relative_elevation <= -1.0)
    {
        return 1.0;
    }

    if (relative_elevation >= 1.0)
    {
        return 0.0;
    }

    const double pi = 3.14159265358979323846;

    const double inundation_fraction =
        0.5 - std::asin(relative_elevation) / pi;

    return std::clamp(inundation_fraction, 0.0, 1.0);
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
