#pragma once

// edge_distance_deposition_model.hpp
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
// the structure follows the non-peer-reviewed preprint:
//
//   Lester et al., "Vegetation Does Not Control Suspended Sediment Deposition
//   in Salt Marshes"
//
// the core idea used here is that suspended sediment concentration on the marsh
// platform decays approximately exponentially away from the marsh edge, and
// that deposited mass depends on:
//
//   - sediment concentration at the marsh edge
//   - distance from the marsh edge
//   - inundation duration during the tidal cycle
//   - settling velocity
//
// this model intentionally ignores vegetation biomass in the deposition
// calculation.
//
// -----------------------------------------------------------------------------

#include "marsh_model/core/material_properties.hpp"
#include "marsh_model/processes/deposition_model.hpp"

#include <Eigen/Core>

namespace marsh_model
{
class edge_distance_deposition_model : public deposition_model
{
public:
    edge_distance_deposition_model() = default;

    Eigen::ArrayXd compute_surface_flux(
        const column_state& state,
        const forcing_step& forcing,
        const biomass_fluxes& biomass,
        const material_catalog& catalog,
        const parameter_set& parameters) const override;

private:
    double compute_inundation_fraction(
        double surface_elevation_m,
        double cycle_amplitude_m,
        double cycle_mean_m) const;

    double compute_decay_length_m(
        double settling_velocity_m_s,
        const forcing_step& forcing,
        const parameter_set& parameters) const;

    double compute_material_concentration_kg_m3(
        const material_properties& material,
        const forcing_step& forcing,
        const parameter_set& parameters) const;
};
}
