// tke_deposition_model.cpp
//
// Part of marsh_muddpile-- https://github.com/simon-m-mudd/marsh_muddpile
//
// Copyright (C) 2026 Simon M. Mudd
// Released under the GNU General Public Licence v3 (GPL-3.0)
// See LICENSE file or https://www.gnu.org/licenses/gpl-3.0.html
//
// -----------------------------------------------------------------------------
// this component implements a TKE-based marsh surface deposition model.
//
// this is a modular replacement for the older settling and trapping logic that
// previously lived in the monolithic marsh driver code. The implementation
// follows the same broad ideas:
//
//   - tidal inundation is calculated through a sinusoidal tidal cycle
//   - flow velocity varies through the cycle with water-level change
//   - turbulent kinetic energy reduces effective settling
//   - vegetation enhances trapping
//   - material concentrations may be depleted during ebb tide
//
// the returned quantity is deposited surface mass by material over one model
// timestep.
//
// parameter conventions:
//   - sediment concentrations are in kg m^-3
//   - settling velocities are in m s^-1
//   - biomass is in g m^-2
//   - tidal period is in hours
//   - timestep is in days
//
// this version receives aboveground biomass from the biomass process through
// biomass_fluxes.
//
// this version also includes a safe optimisation:
// material concentrations and settling properties are precomputed once per
// forcing step rather than recomputed inside the tidal substep loop.
//
// -----------------------------------------------------------------------------

#include "marsh_model/processes/tke_deposition_model.hpp"

#include "marsh_model/core/material_properties.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

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

Eigen::ArrayXd tke_deposition_model::compute_surface_flux(
    const column_state& state,
    const forcing_step& forcing,
    const biomass_fluxes& biomass,
    const material_catalog& catalog,
    const parameter_set& parameters) const
{
    Eigen::ArrayXd surface_flux = Eigen::ArrayXd::Zero(catalog.size());

    if (catalog.size() == 0 || forcing.dt_days <= 0.0 || forcing.tidal_period_hours <= 0.0)
    {
        return surface_flux;
    }

    const double vegetation_biomass_g_m2 =
        std::max(1.0, biomass.aboveground_biomass);

    const double reference_slope =
        compute_reference_slope(parameters);

    const double peak_flow_velocity_m_s =
        compute_peak_flow_velocity_m_s(
            vegetation_biomass_g_m2,
            reference_slope,
            parameters);

    const double cycles_per_timestep =
        compute_cycles_per_timestep(forcing);

    const Eigen::ArrayXd flux_per_tidal_cycle =
        compute_flux_per_tidal_cycle(
            state,
            forcing,
            catalog,
            parameters,
            vegetation_biomass_g_m2,
            peak_flow_velocity_m_s);

    surface_flux = cycles_per_timestep * flux_per_tidal_cycle;

    if (catalog.has_material("pb210"))
    {
        const int pb210_index = catalog.get_material_index("pb210");
        surface_flux(pb210_index) += forcing.external_pb210_supply * forcing.dt_days;
    }

    return surface_flux;
}

double tke_deposition_model::compute_cycles_per_timestep(
    const forcing_step& forcing) const
{
    if (forcing.tidal_period_hours <= 0.0)
    {
        return 0.0;
    }

    return 24.0 * forcing.dt_days / forcing.tidal_period_hours;
}

double tke_deposition_model::compute_reference_slope(
    const parameter_set& parameters) const
{
    const double reference_biomass_g_m2 =
        get_parameter_or_default(parameters, "deposition_reference_biomass_g_m2", 200.0);

    const double reference_flow_velocity_m_s =
        get_parameter_or_default(parameters, "deposition_reference_flow_velocity_m_s", 0.05);

    return compute_slope_for_velocity(
        std::max(1.0, reference_biomass_g_m2),
        reference_flow_velocity_m_s,
        parameters);
}

double tke_deposition_model::compute_peak_flow_velocity_m_s(
    double vegetation_biomass_g_m2,
    double reference_slope,
    const parameter_set& parameters) const
{
    const double limiting_flow_velocity_m_s =
        get_parameter_or_default(parameters, "deposition_limiting_flow_velocity_m_s", 0.10);

    double velocity =
        compute_velocity_for_slope(
            std::max(1.0, vegetation_biomass_g_m2),
            reference_slope,
            parameters);

    if (!std::isfinite(velocity) || velocity < 0.0)
    {
        velocity = 0.0;
    }

    return std::min(velocity, limiting_flow_velocity_m_s);
}

Eigen::ArrayXd tke_deposition_model::compute_flux_per_tidal_cycle(
    const column_state& state,
    const forcing_step& forcing,
    const material_catalog& catalog,
    const parameter_set& parameters,
    double vegetation_biomass_g_m2,
    double peak_flow_velocity_m_s) const
{
    const int n_materials = catalog.size();

    Eigen::ArrayXd total_flux = Eigen::ArrayXd::Zero(n_materials);
    Eigen::ArrayXd particle_concentration = Eigen::ArrayXd::Zero(n_materials);
    Eigen::ArrayXd incoming_concentration = Eigen::ArrayXd::Zero(n_materials);
    Eigen::ArrayXd diameter_m = Eigen::ArrayXd::Zero(n_materials);
    Eigen::ArrayXd settling_velocity_m_s = Eigen::ArrayXd::Zero(n_materials);
    Eigen::ArrayXd trapping_flux = Eigen::ArrayXd::Zero(n_materials);
    Eigen::ArrayXd settling_flux = Eigen::ArrayXd::Zero(n_materials);

    std::vector<bool> can_deposit(n_materials, false);

    const double pi = 3.14159265358979323846;

    const double time_fractions =
        get_parameter_or_default(parameters, "deposition_time_fractions_per_tide", 15000.0);

    const double min_water_depth_m =
        get_parameter_or_default(parameters, "deposition_min_water_depth_m", 0.002);

    const double von_karman =
        get_parameter_or_default(parameters, "deposition_von_karman", 0.40);

    const double tke_coefficient =
        get_parameter_or_default(parameters, "deposition_tke_coefficient", 0.20);

    const double water_density_kg_m3 =
        get_parameter_or_default(parameters, "deposition_water_density_kg_m3", 1000.0);

    const double alpha =
        get_parameter_or_default(parameters, "deposition_alpha", 0.55);

    const double beta =
        get_parameter_or_default(parameters, "deposition_beta", 0.40);

    const double mu =
        get_parameter_or_default(parameters, "deposition_mu", 0.00066);

    const double phi =
        get_parameter_or_default(parameters, "deposition_phi", 0.29);

    const double kappa =
        get_parameter_or_default(parameters, "deposition_kappa", 0.224);

    const double nu_m2_s =
        get_parameter_or_default(parameters, "deposition_nu_m2_s", 1.0e-5);

    const double gamma_tr =
        get_parameter_or_default(parameters, "deposition_gamma_tr", 0.718);

    const double epsilon =
        get_parameter_or_default(parameters, "deposition_epsilon", 2.08);

    const double dt_hours =
        forcing.tidal_period_hours / std::max(1.0, time_fractions);

    const double water_depth_amplitude_rate_m_per_hour =
        2.0 * forcing.tidal_amplitude * pi / forcing.tidal_period_hours;

    const double alpha_t =
        alpha * kappa * std::pow(mu, gamma_tr - epsilon) / std::pow(nu_m2_s, gamma_tr);

    const double beta_t =
        phi * (gamma_tr - epsilon) + beta;

    const double surface_elevation_m =
        state.get_surface_elevation();

    // -------------------------------------------------------------------------
    // precompute material-specific constants for the whole forcing step
    // -------------------------------------------------------------------------
    for (int material_index = 0; material_index < n_materials; ++material_index)
    {
        const material_properties& material =
            catalog.get_material(material_index);

        incoming_concentration(material_index) =
            compute_material_concentration_kg_m3(
                material,
                forcing,
                parameters);

        particle_concentration(material_index) =
            incoming_concentration(material_index);

        if (!material.allow_surface_deposition)
        {
            continue;
        }

        if (!material.settling.has_value())
        {
            continue;
        }

        diameter_m(material_index) = material.settling->diameter;
        settling_velocity_m_s(material_index) = material.settling->settling_velocity;

        if (diameter_m(material_index) > 0.0 &&
            settling_velocity_m_s(material_index) > 0.0)
        {
            can_deposit[material_index] = true;
        }
    }

    double tidal_time_hours = 0.0;

    while (tidal_time_hours < forcing.tidal_period_hours)
    {
        tidal_time_hours += dt_hours;

        const double water_depth_m =
            compute_water_depth_m(
                tidal_time_hours,
                forcing,
                surface_elevation_m);

        const double dwater_depth_dt_m_per_hour =
            compute_water_depth_rate_m_per_hour(
                tidal_time_hours,
                forcing);

        if (water_depth_m <= min_water_depth_m)
        {
            continue;
        }

        double flow_velocity_m_s = 0.0;
        if (std::abs(water_depth_amplitude_rate_m_per_hour) > 0.0)
        {
            flow_velocity_m_s =
                peak_flow_velocity_m_s *
                std::abs(dwater_depth_dt_m_per_hour / water_depth_amplitude_rate_m_per_hour);
        }

        const double tke =
            compute_turbulent_kinetic_energy(
                vegetation_biomass_g_m2,
                flow_velocity_m_s,
                parameters);

        const double w_up_m_s =
            von_karman * std::sqrt(std::max(0.0, tke_coefficient * tke / water_density_kg_m3));

        for (int material_index = 0; material_index < n_materials; ++material_index)
        {
            if (!can_deposit[material_index])
            {
                continue;
            }

            double active_concentration_kg_m3 =
                incoming_concentration(material_index);

            // flood tide: refresh concentration from the external source
            // ebb tide: allow concentration depletion within the water column
            if (dwater_depth_dt_m_per_hour <= 0.0)
            {
                active_concentration_kg_m3 =
                    std::max(0.0, particle_concentration(material_index));
            }

            const double starting_mass_kg_m2 =
                active_concentration_kg_m3 * water_depth_m;

            double settling_mass_kg_m2 =
                3600.0 * active_concentration_kg_m3 * dt_hours *
                std::max(0.0, settling_velocity_m_s(material_index) - w_up_m_s);

            double trapping_mass_kg_m2 =
                3600.0 * alpha_t *
                std::pow(diameter_m(material_index), epsilon) *
                std::pow(vegetation_biomass_g_m2, beta_t) *
                active_concentration_kg_m3 *
                std::pow(std::max(0.0, flow_velocity_m_s), gamma_tr + 1.0) *
                water_depth_m *
                dt_hours;

            if (starting_mass_kg_m2 < trapping_mass_kg_m2)
            {
                trapping_mass_kg_m2 = starting_mass_kg_m2;
                settling_mass_kg_m2 = 0.0;
            }
            else if (starting_mass_kg_m2 - trapping_mass_kg_m2 < settling_mass_kg_m2)
            {
                settling_mass_kg_m2 = starting_mass_kg_m2 - trapping_mass_kg_m2;
            }

            trapping_flux(material_index) += trapping_mass_kg_m2;
            settling_flux(material_index) += settling_mass_kg_m2;

            if (dwater_depth_dt_m_per_hour <= 0.0)
            {
                double updated_concentration_kg_m3 =
                    active_concentration_kg_m3
                    - dt_hours * active_concentration_kg_m3 * dwater_depth_dt_m_per_hour / water_depth_m
                    - (settling_mass_kg_m2 + trapping_mass_kg_m2) / water_depth_m;

                particle_concentration(material_index) =
                    std::max(0.0, updated_concentration_kg_m3);
            }
            else
            {
                particle_concentration(material_index) =
                    incoming_concentration(material_index);
            }
        }
    }

    total_flux = trapping_flux + settling_flux;
    return total_flux;
}

double tke_deposition_model::compute_material_concentration_kg_m3(
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

double tke_deposition_model::compute_water_depth_m(
    double tidal_time_hours,
    const forcing_step& forcing,
    double surface_elevation_m) const
{
    const double pi = 3.14159265358979323846;

    const double water_depth_m =
        forcing.tidal_amplitude *
            std::sin(2.0 * pi * (tidal_time_hours / forcing.tidal_period_hours - 0.25)) +
        forcing.mean_sea_level -
        surface_elevation_m;

    return std::max(0.0, water_depth_m);
}

double tke_deposition_model::compute_water_depth_rate_m_per_hour(
    double tidal_time_hours,
    const forcing_step& forcing) const
{
    const double pi = 3.14159265358979323846;

    return
        2.0 * pi * forcing.tidal_amplitude *
        std::cos(2.0 * pi * (tidal_time_hours / forcing.tidal_period_hours - 0.25)) /
        forcing.tidal_period_hours;
}

double tke_deposition_model::compute_turbulent_kinetic_energy(
    double vegetation_biomass_g_m2,
    double flow_velocity_m_s,
    const parameter_set& parameters) const
{
    const double aa =
        get_parameter_or_default(parameters, "deposition_aa", 0.46);

    const double bb =
        get_parameter_or_default(parameters, "deposition_bb", 3.8);

    const double alpha_turb =
        get_parameter_or_default(parameters, "deposition_alpha_turb", 0.9);

    const double alpha_zero =
        get_parameter_or_default(parameters, "deposition_alpha_zero", 11.0);

    const double alpha =
        get_parameter_or_default(parameters, "deposition_alpha", 0.55);

    const double beta =
        get_parameter_or_default(parameters, "deposition_beta", 0.40);

    const double mu =
        get_parameter_or_default(parameters, "deposition_mu", 0.00066);

    const double phi =
        get_parameter_or_default(parameters, "deposition_phi", 0.29);

    const double nu_m2_s =
        get_parameter_or_default(parameters, "deposition_nu_m2_s", 1.0e-5);

    const double term_1 =
        alpha_zero * nu_m2_s /
        (std::max(flow_velocity_m_s, 1.0e-12) * mu * std::pow(vegetation_biomass_g_m2, phi));

    const double term_2 =
        aa + bb * alpha * mu * 3.14159265358979323846 * 0.25 *
        std::pow(vegetation_biomass_g_m2, beta + phi);

    const double term_3 =
        alpha * mu * std::pow(vegetation_biomass_g_m2, beta + phi);

    return
        alpha_turb * alpha_turb * flow_velocity_m_s * flow_velocity_m_s *
        std::pow(2.0 * (term_1 + term_2) * term_3, 2.0 / 3.0);
}

double tke_deposition_model::compute_slope_for_velocity(
    double vegetation_biomass_g_m2,
    double flow_velocity_m_s,
    const parameter_set& parameters) const
{
    const double aa =
        get_parameter_or_default(parameters, "deposition_aa", 0.46);

    const double bb =
        get_parameter_or_default(parameters, "deposition_bb", 3.8);

    const double alpha =
        get_parameter_or_default(parameters, "deposition_alpha", 0.55);

    const double beta =
        get_parameter_or_default(parameters, "deposition_beta", 0.40);

    const double mu =
        get_parameter_or_default(parameters, "deposition_mu", 0.00066);

    const double phi =
        get_parameter_or_default(parameters, "deposition_phi", 0.29);

    const double nu_m2_s =
        get_parameter_or_default(parameters, "deposition_nu_m2_s", 1.0e-5);

    const double alpha_zero =
        get_parameter_or_default(parameters, "deposition_alpha_zero", 11.0);

    const double gravity_m_s2 =
        get_parameter_or_default(parameters, "gravity_m_s2", 9.81);

    const double pi = 3.14159265358979323846;

    const double term_1 =
        aa * std::pow(vegetation_biomass_g_m2, beta) *
        flow_velocity_m_s * flow_velocity_m_s * alpha;

    const double term_2 =
        0.25 *
        std::pow(vegetation_biomass_g_m2, 2.0 * beta + phi) *
        bb * pi * flow_velocity_m_s * flow_velocity_m_s *
        alpha * alpha * mu;

    const double term_3 =
        std::pow(vegetation_biomass_g_m2, beta - phi) *
        flow_velocity_m_s * alpha * alpha_zero * nu_m2_s / mu;

    const double term_4 =
        (1.0 - 0.25 * alpha * mu * pi *
         std::pow(vegetation_biomass_g_m2, beta + phi)) *
        gravity_m_s2;

    if (std::abs(term_4) < 1.0e-12)
    {
        return 0.0;
    }

    return -(term_1 + term_2 + term_3) / term_4;
}

double tke_deposition_model::compute_velocity_for_slope(
    double vegetation_biomass_g_m2,
    double slope,
    const parameter_set& parameters) const
{
    const double aa =
        get_parameter_or_default(parameters, "deposition_aa", 0.46);

    const double bb =
        get_parameter_or_default(parameters, "deposition_bb", 3.8);

    const double alpha =
        get_parameter_or_default(parameters, "deposition_alpha", 0.55);

    const double beta =
        get_parameter_or_default(parameters, "deposition_beta", 0.40);

    const double mu =
        get_parameter_or_default(parameters, "deposition_mu", 0.00066);

    const double phi =
        get_parameter_or_default(parameters, "deposition_phi", 0.29);

    const double nu_m2_s =
        get_parameter_or_default(parameters, "deposition_nu_m2_s", 1.0e-5);

    const double alpha_zero =
        get_parameter_or_default(parameters, "deposition_alpha_zero", 11.0);

    const double gravity_m_s2 =
        get_parameter_or_default(parameters, "gravity_m_s2", 9.81);

    const double pi = 3.14159265358979323846;

    const double term_5 =
        -4.0 * std::pow(vegetation_biomass_g_m2, beta) *
        alpha * alpha_zero * nu_m2_s;

    const double term_6 =
        -4.0 * std::pow(vegetation_biomass_g_m2, phi) * gravity_m_s2 * slope * mu +
        std::pow(vegetation_biomass_g_m2, beta + 2.0 * phi) *
        gravity_m_s2 * pi * slope * alpha * mu * mu;

    const double term_7 =
        -4.0 * aa * std::pow(vegetation_biomass_g_m2, beta + phi) * alpha * mu -
        std::pow(vegetation_biomass_g_m2, 2.0 * beta + 2.0 * phi) *
        bb * pi * alpha * alpha * mu * mu;

    const double term_8 =
        16.0 * std::pow(vegetation_biomass_g_m2, 2.0 * beta) *
        alpha * alpha * alpha_zero * alpha_zero *
        nu_m2_s * nu_m2_s;

    const double term_9 =
        2.0 *
        (4.0 * aa * std::pow(vegetation_biomass_g_m2, beta + phi) * alpha * mu +
         std::pow(vegetation_biomass_g_m2, 2.0 * beta + 2.0 * phi) *
         bb * pi * alpha * alpha * mu * mu);

    const double discriminant =
        -4.0 * term_6 * term_7 + term_8;

    if (term_9 == 0.0 || discriminant < 0.0)
    {
        return 0.0;
    }

    const double velocity =
        (term_5 + std::sqrt(discriminant)) / term_9;

    return std::max(0.0, velocity);
}
}
