#include "marsh_model/processes/composite_water_level_model.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>

namespace marsh_model
{
namespace
{
struct tidal_constituent_definition
{
    const char* name;
    double speed_deg_per_hour;
};

constexpr std::array<tidal_constituent_definition, 11> standard_constituents = {{
    {"M2", 28.9841042},
    {"S2", 30.0000000},
    {"N2", 28.4397295},
    {"K1", 15.0410686},
    {"O1", 13.9430356},
    {"P1", 14.9589314},
    {"K2", 30.0821373},
    {"Q1", 13.3986609},
    {"2N2", 27.8953548},
    {"M4", 57.9682084},
    {"MS4", 58.9841042}
}};

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

double clamp_nonnegative(double value)
{
    return std::max(0.0, value);
}

double degrees_to_radians(double degrees)
{
    return degrees * 3.14159265358979323846 / 180.0;
}

std::string constituent_parameter_name(
    const std::string& constituent_name,
    const std::string& suffix)
{
    return "water_level_constituent_" + constituent_name + "_" + suffix;
}
}

hydrology_diagnostics composite_water_level_model::compute_hydrology(
    const column_state& state,
    const forcing_step& forcing,
    const site_properties& site,
    const parameter_set& parameters) const
{
    hydrology_diagnostics diagnostics;

    const double surface_elevation_m =
        state.get_surface_elevation();

    const double dt_days =
        std::max(0.0, forcing.dt_days);

    const double dt_hours =
        24.0 * dt_days;

    // -------------------------------------------------------------------------
    // Option 1: observed/local water level supplied directly.
    // This is the top-priority mode and is useful for gage forcing or external
    // hydrodynamic model output.
    // -------------------------------------------------------------------------
    if (forcing.has_observed_water_level)
    {
        const double water_level_m =
            forcing.observed_water_level_m + site.local_tidal_offset_m;

        const double inundation_depth_m =
            clamp_nonnegative(water_level_m - surface_elevation_m);

        diagnostics.mean_water_level_m = water_level_m;
        diagnostics.max_water_level_m = water_level_m;
        diagnostics.inundation_fraction = (inundation_depth_m > 0.0) ? 1.0 : 0.0;
        diagnostics.mean_inundation_depth_m = inundation_depth_m;
        diagnostics.inundation_duration_hours =
            diagnostics.inundation_fraction * dt_hours;

        diagnostics.storm_surge_residual_m = 0.0;
        return diagnostics;
    }

    // -------------------------------------------------------------------------
    // Option 2: constituent-based astronomical tide plus optional additive
    // residual (storm surge or other non-tidal residual).
    // -------------------------------------------------------------------------
    const int n_substeps =
        std::max(
            1,
            static_cast<int>(
                std::round(
                    get_parameter_or_default(
                        parameters,
                        "water_level_substeps_per_step",
                        240.0))));

    double water_level_sum_m = 0.0;
    double max_water_level_m = -1.0e300;

    int inundated_substeps = 0;
    double inundated_depth_sum_m = 0.0;

    for (int substep = 0; substep < n_substeps; ++substep)
    {
        const double substep_fraction =
            (static_cast<double>(substep) + 0.5) /
            static_cast<double>(n_substeps);

        const double substep_time_hours =
            24.0 * forcing.model_time_days +
            substep_fraction * dt_hours;

        const double water_level_m =
            compute_astronomical_water_level_m(
                substep_time_hours,
                forcing,
                site,
                parameters);

        water_level_sum_m += water_level_m;
        max_water_level_m = std::max(max_water_level_m, water_level_m);

        const double inundation_depth_m =
            clamp_nonnegative(water_level_m - surface_elevation_m);

        if (inundation_depth_m > 0.0)
        {
            ++inundated_substeps;
            inundated_depth_sum_m += inundation_depth_m;
        }
    }

    diagnostics.mean_water_level_m =
        water_level_sum_m / static_cast<double>(n_substeps);

    diagnostics.max_water_level_m = max_water_level_m;

    diagnostics.inundation_fraction =
        static_cast<double>(inundated_substeps) /
        static_cast<double>(n_substeps);

    if (inundated_substeps > 0)
    {
        diagnostics.mean_inundation_depth_m =
            inundated_depth_sum_m / static_cast<double>(inundated_substeps);
    }

    diagnostics.inundation_duration_hours =
        diagnostics.inundation_fraction * dt_hours;

    diagnostics.storm_surge_residual_m =
        forcing.storm_surge_residual_m;

    return diagnostics;
}

double composite_water_level_model::compute_astronomical_water_level_m(
    double time_hours,
    const forcing_step& forcing,
    const site_properties& site,
    const parameter_set& parameters) const
{
    const double mean_water_level_m =
        forcing.mean_sea_level + site.local_tidal_offset_m;

    const double additive_residual_m =
        forcing.storm_surge_residual_m;

    const double reference_time_hours =
        get_parameter_or_default(
            parameters,
            "water_level_reference_time_hours",
            0.0);

    double astronomical_tide_m = 0.0;
    bool has_any_constituent = false;

    // -------------------------------------------------------------------------
    // Constituent-based harmonic reconstruction:
    //
    // eta(t) = eta_mean + sum_j [ f_j A_j cos( omega_j (t - t_ref) + V_j - phi_j ) ]
    //
    // where:
    //   A_j   = local constituent amplitude
    //   phi_j = local phase lag
    //   f_j   = optional nodal factor
    //   V_j   = optional equilibrium argument
    //
    // Standard constituent speeds are built in below. The user only needs to
    // supply amplitudes and phase lags for the constituents they want to use.
    // -------------------------------------------------------------------------
    for (const auto& constituent : standard_constituents)
    {
        const std::string amplitude_name =
            constituent_parameter_name(constituent.name, "amplitude_m");

        if (!parameters.has(amplitude_name))
        {
            continue;
        }

        const double amplitude_m =
            parameters.get(amplitude_name);

        if (std::abs(amplitude_m) <= 0.0)
        {
            continue;
        }

        const double phase_deg =
            get_parameter_or_default(
                parameters,
                constituent_parameter_name(constituent.name, "phase_deg"),
                0.0);

        const double nodal_factor =
            get_parameter_or_default(
                parameters,
                constituent_parameter_name(constituent.name, "nodal_factor"),
                1.0);

        const double equilibrium_argument_deg =
            get_parameter_or_default(
                parameters,
                constituent_parameter_name(constituent.name, "equilibrium_argument_deg"),
                0.0);

        const double argument_deg =
            constituent.speed_deg_per_hour * (time_hours - reference_time_hours) +
            equilibrium_argument_deg -
            phase_deg;

        astronomical_tide_m +=
            nodal_factor * amplitude_m * std::cos(degrees_to_radians(argument_deg));

        has_any_constituent = true;
    }

    // -------------------------------------------------------------------------
    // Backward-compatible fallback: if no constituents are configured, use the
    // older single-sine tide proxy from mean level + amplitude + period.
    // -------------------------------------------------------------------------
    if (!has_any_constituent)
    {
        const double phase_offset_cycles =
            get_parameter_or_default(
                parameters,
                "water_level_phase_offset_cycles",
                -0.25);

        if (forcing.tidal_period_hours > 0.0 &&
            std::abs(forcing.tidal_amplitude) > 0.0)
        {
            const double phase_rad =
                2.0 * 3.14159265358979323846 *
                (time_hours / forcing.tidal_period_hours + phase_offset_cycles);

            astronomical_tide_m =
                forcing.tidal_amplitude * std::sin(phase_rad);
        }
    }

    return mean_water_level_m + astronomical_tide_m + additive_residual_m;
}
}
