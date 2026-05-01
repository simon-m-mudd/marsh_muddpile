#pragma once

// tke_deposition_model.hpp
//
// Part of marsh_muddpile-- https://github.com/simon-m-mudd/marsh_muddpile
//
// Copyright (C) 2026 Simon M. Mudd
// Released under the GNU General Public Licence v3 (GPL-3.0)
// See LICENSE file or https://www.gnu.org/licenses/gpl-3.0.html
//
// -----------------------------------------------------------------------------
// this component implements a tidal deposition model with settling, trapping,
// and turbulence-reduced effective settling velocity.
//
// the implementation is based on the structure of the older marsh model, in
// particular the TKE-based formulation for:
//   - tidal water depth through the tidal cycle
//   - flow velocity scaled by tidal stage
//   - turbulent kinetic energy
//   - upward turbulent velocity reducing effective settling
//   - vegetative trapping
//
// this module returns surface mass input by material over the current timestep.
//
// this version receives current vegetation biomass through biomass_fluxes,
// rather than reading a temporary value from parameter_set.
//
// -----------------------------------------------------------------------------

#include "marsh_model/core/material_properties.hpp"
#include "marsh_model/processes/deposition_model.hpp"

#include <Eigen/Core>

namespace marsh_model
{
class tke_deposition_model : public deposition_model
{
public:
    tke_deposition_model() = default;

    Eigen::ArrayXd compute_surface_flux(
        const column_state& state,
        const forcing_step& forcing,
        const biomass_fluxes& biomass,
        const material_catalog& catalog,
        const parameter_set& parameters) const override;

private:
    double compute_cycles_per_timestep(
        const forcing_step& forcing) const;

    double compute_reference_slope(
        const parameter_set& parameters) const;

    double compute_peak_flow_velocity_m_s(
        double vegetation_biomass_g_m2,
        double reference_slope,
        const parameter_set& parameters) const;

    Eigen::ArrayXd compute_flux_per_tidal_cycle(
        const column_state& state,
        const forcing_step& forcing,
        const material_catalog& catalog,
        const parameter_set& parameters,
        double vegetation_biomass_g_m2,
        double peak_flow_velocity_m_s) const;

    double compute_material_concentration_kg_m3(
        const material_properties& material,
        const forcing_step& forcing,
        const parameter_set& parameters) const;

    double compute_water_depth_m(
        double tidal_time_hours,
        const forcing_step& forcing,
        double surface_elevation_m) const;

    double compute_water_depth_rate_m_per_hour(
        double tidal_time_hours,
        const forcing_step& forcing) const;

    double compute_turbulent_kinetic_energy(
        double vegetation_biomass_g_m2,
        double flow_velocity_m_s,
        const parameter_set& parameters) const;

    double compute_slope_for_velocity(
        double vegetation_biomass_g_m2,
        double flow_velocity_m_s,
        const parameter_set& parameters) const;

    double compute_velocity_for_slope(
        double vegetation_biomass_g_m2,
        double slope,
        const parameter_set& parameters) const;
};
}
