// seasonal_biomass_model.cpp
//
// Part of marsh_muddpile-- https://github.com/simon-m-mudd/marsh_muddpile
//
// Copyright (C) 2026 Simon M. Mudd
// Released under the GNU General Public Licence v3 (GPL-3.0)
// See LICENSE file or https://www.gnu.org/licenses/gpl-3.0.html
//
// -----------------------------------------------------------------------------
// this component implements a seasonal marsh biomass model.
//
// this is a modular replacement for the older biomass functions used in the
// original marsh model. The old model computed:
//
//   - peak biomass as a function of water depth relative to mean high tide
//   - seasonal aboveground biomass
//   - seasonal mortality
//   - belowground biomass using a belowground-to-aboveground ratio
//
// this implementation preserves that structure and returns biomass_fluxes that
// can be consumed by the root allocation process.
//
// unit conventions in this draft are:
//
//   - peak_biomass and aboveground_biomass are stored in g m^-2
//   - belowground_biomass and belowground_mortality are stored in kg m^-2
//
// the reason for this mixed convention is compatibility with the older marsh
// model, where empirical peak biomass relationships were expressed in g m^-2
// but sediment-column mass pools were tracked in kg m^-2.
//
// -----------------------------------------------------------------------------

#include "marsh_model/processes/seasonal_biomass_model.hpp"

#include <algorithm>
#include <cmath>

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

biomass_fluxes seasonal_biomass_model::compute_biomass_fluxes(
    const column_state& state,
    const forcing_step& forcing,
    const material_catalog&,
    const parameter_set& parameters) const
{
    biomass_fluxes fluxes;

    const double day_of_year_days =
        compute_day_of_year_days(forcing, parameters);

    const double peak_biomass_g_m2 =
        compute_peak_aboveground_biomass_g_m2(state, forcing, parameters);

    const double aboveground_biomass_g_m2 =
        compute_aboveground_biomass_g_m2(
            day_of_year_days,
            peak_biomass_g_m2,
            parameters);

    const double aboveground_mortality_g_m2_over_timestep =
        compute_aboveground_mortality_g_m2_over_timestep(
            day_of_year_days,
            peak_biomass_g_m2,
            forcing.dt_days,
            parameters);

    const double belowground_to_aboveground_ratio =
        compute_belowground_to_aboveground_ratio(
            state,
            forcing,
            parameters);

    fluxes.peak_biomass = peak_biomass_g_m2;
    fluxes.aboveground_biomass = aboveground_biomass_g_m2;

    // convert belowground standing biomass from g m^-2 to kg m^-2
    fluxes.belowground_biomass =
        aboveground_biomass_g_m2 * belowground_to_aboveground_ratio / 1000.0;

    // convert belowground mortality from g m^-2 over timestep to kg m^-2
    fluxes.belowground_mortality =
        aboveground_mortality_g_m2_over_timestep * belowground_to_aboveground_ratio / 1000.0;

    return fluxes;
}

// compute day of year from model_time_days
//
// by default the model year is 365 days long and day-of-year is wrapped into
// the interval [0, 365). An optional phase offset can be added later if needed.
double seasonal_biomass_model::compute_day_of_year_days(
    const forcing_step& forcing,
    const parameter_set& parameters) const
{
    const double days_per_year =
        get_parameter_or_default(parameters, "biomass_days_per_year", 365.0);

    if (days_per_year <= 0.0)
    {
        return 0.0;
    }

    double day_of_year =
        std::fmod(forcing.model_time_days, days_per_year);

    if (day_of_year < 0.0)
    {
        day_of_year += days_per_year;
    }

    return day_of_year;
}

// compute peak aboveground biomass as a function of water depth relative to
// mean high tide.
//
// this follows the triangular form used in the older marsh model:
//
//   if water_depth > max_depth or water_depth < min_depth:
//       biomass = 0
//   else:
//       biomass = max_biomass * (water_depth - min_depth) / (max_depth - min_depth)
//
// an optional linear temperature effect is included:
//   biomass = biomass * (1 + biomass_temperature_factor_per_degree * delta_t)
double seasonal_biomass_model::compute_peak_aboveground_biomass_g_m2(
    const column_state& state,
    const forcing_step& forcing,
    const parameter_set& parameters) const
{
    const double min_depth_m =
        get_parameter_or_default(parameters, "biomass_min_depth_m", -0.10);

    const double max_depth_m =
        get_parameter_or_default(parameters, "biomass_max_depth_m", 0.30);

    const double max_biomass_g_m2 =
        get_parameter_or_default(parameters, "biomass_max_biomass_g_m2", 1500.0);

    const double biomass_temperature_factor_per_degree =
        get_parameter_or_default(parameters, "biomass_temperature_factor_per_degree", 0.0);

    const double biomass_reference_temperature_c =
        get_parameter_or_default(parameters, "biomass_reference_temperature_c", 20.0);

    const double surface_elevation_m =
        state.get_surface_elevation();

    const double water_depth_m =
        forcing.mean_high_tide - surface_elevation_m;

    double peak_biomass_g_m2 = 0.0;

    if (water_depth_m > max_depth_m || water_depth_m < min_depth_m)
    {
        peak_biomass_g_m2 = 0.0;
    }
    else
    {
        const double depth_range_m = max_depth_m - min_depth_m;

        if (depth_range_m > 0.0)
        {
            peak_biomass_g_m2 =
                max_biomass_g_m2 *
                (water_depth_m - min_depth_m) /
                depth_range_m;
        }
    }

    const double delta_temperature_c =
        forcing.temperature - biomass_reference_temperature_c;

    peak_biomass_g_m2 *=
        std::max(0.0, 1.0 + biomass_temperature_factor_per_degree * delta_temperature_c);

    return std::max(0.0, peak_biomass_g_m2);
}

// compute seasonal aboveground biomass.
//
// this follows the cosine seasonal cycle used in the old biomass code:
//
//   b(t) = 0.5 * [ b_min + b_peak + (b_peak - b_min) * cos( 2*pi*(t - t_peak)/365 ) ]
//
// parameters:
//   biomass_min_fraction_of_peak
//   biomass_day_peak_season_days
double seasonal_biomass_model::compute_aboveground_biomass_g_m2(
    double day_of_year_days,
    double peak_biomass_g_m2,
    const parameter_set& parameters) const
{
    const double days_per_year =
        get_parameter_or_default(parameters, "biomass_days_per_year", 365.0);

    const double min_fraction_of_peak =
        get_parameter_or_default(parameters, "biomass_min_fraction_of_peak", 0.1);

    const double day_peak_season_days =
        get_parameter_or_default(parameters, "biomass_day_peak_season_days", 200.0);

    const double min_biomass_g_m2 =
        peak_biomass_g_m2 * min_fraction_of_peak;

    const double argument =
        2.0 * M_PI * (day_of_year_days - day_peak_season_days) / days_per_year;

    const double aboveground_biomass_g_m2 =
        0.5 *
        (min_biomass_g_m2 +
         peak_biomass_g_m2 +
         (peak_biomass_g_m2 - min_biomass_g_m2) * std::cos(argument));

    return std::max(0.0, aboveground_biomass_g_m2);
}

// compute seasonal aboveground mortality over the timestep.
//
// this follows the structure of the old biomass_and_mortality2 formulation.
// The mortality expression is written as an integrated seasonal flux over dt.
//
// parameters:
//   biomass_min_fraction_of_peak
//   biomass_peak_growth_rate_fraction_per_day
//   biomass_min_growth_rate_fraction_per_day
//   biomass_day_peak_season_days
//   biomass_phase_shift_days
double seasonal_biomass_model::compute_aboveground_mortality_g_m2_over_timestep(
    double day_of_year_days,
    double peak_biomass_g_m2,
    double dt_days,
    const parameter_set& parameters) const
{
    const double days_per_year =
        get_parameter_or_default(parameters, "biomass_days_per_year", 365.0);

    const double min_fraction_of_peak =
        get_parameter_or_default(parameters, "biomass_min_fraction_of_peak", 0.1);

    const double peak_growth_rate_fraction_per_day =
        get_parameter_or_default(parameters, "biomass_peak_growth_rate_fraction_per_day", 0.02);

    const double min_growth_rate_fraction_per_day =
        get_parameter_or_default(parameters, "biomass_min_growth_rate_fraction_per_day", 0.002);

    const double day_peak_season_days =
        get_parameter_or_default(parameters, "biomass_day_peak_season_days", 200.0);

    const double phase_shift_days =
        get_parameter_or_default(parameters, "biomass_phase_shift_days", 30.0);

    const double min_biomass_g_m2 =
        peak_biomass_g_m2 * min_fraction_of_peak;

    const double peak_growth_g_m2_per_day =
        peak_growth_rate_fraction_per_day * peak_biomass_g_m2;

    const double min_growth_g_m2_per_day =
        min_growth_rate_fraction_per_day * peak_biomass_g_m2;

    const double mortality_g_m2 =
        (1.0 / (4.0 * M_PI)) *
        (2.0 * dt_days * (min_growth_g_m2_per_day + peak_growth_g_m2_per_day) * M_PI +
         2.0 * (min_biomass_g_m2 - peak_biomass_g_m2) * M_PI *
             (std::cos(2.0 * M_PI * (day_of_year_days - day_peak_season_days) / days_per_year) -
              std::cos(2.0 * M_PI * (dt_days - day_of_year_days + day_peak_season_days) / days_per_year)) -
         days_per_year * (min_growth_g_m2_per_day - peak_growth_g_m2_per_day) *
             (std::sin(2.0 * M_PI * (day_of_year_days - day_peak_season_days + phase_shift_days) / days_per_year) +
              std::sin(2.0 * M_PI * (dt_days - day_of_year_days + day_peak_season_days - phase_shift_days) / days_per_year)));

    return std::max(0.0, mortality_g_m2);
}

// compute the belowground-to-aboveground biomass ratio.
//
// this follows the older marsh model logic:
//
//   bg_to_ag_ratio = theta_bm * depth_below_mht + d_mbm
//
// with a minimum value clamp.
double seasonal_biomass_model::compute_belowground_to_aboveground_ratio(
    const column_state& state,
    const forcing_step& forcing,
    const parameter_set& parameters) const
{
    const double theta_bm =
        get_parameter_or_default(parameters, "biomass_bg_to_ag_slope", 1.0);

    const double d_mbm =
        get_parameter_or_default(parameters, "biomass_bg_to_ag_intercept", 0.5);

    const double minimum_ratio =
        get_parameter_or_default(parameters, "biomass_bg_to_ag_minimum", 0.1);

    const double depth_below_mht_m =
        forcing.mean_high_tide - state.get_surface_elevation();

    const double ratio =
        theta_bm * depth_below_mht_m + d_mbm;

    return std::max(minimum_ratio, ratio);
}
}
