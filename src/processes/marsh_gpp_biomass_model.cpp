#include "marsh_model/processes/marsh_gpp_biomass_model.hpp"

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

double clamp(double value, double lower, double upper)
{
    return std::max(lower, std::min(value, upper));
}
}

vegetation_diagnostics marsh_gpp_biomass_model::update_vegetation(
    ecohydrology_state& eco_state,
    const hydrology_diagnostics& hydro,
    const sediment_surface_properties&,
    const forcing_step& forcing,
    const site_properties&,
    const parameter_set& parameters) const
{
    vegetation_diagnostics diagnostics;

    if (forcing.dt_days <= 0.0)
    {
        diagnostics.lai = eco_state.lai;
        return diagnostics;
    }

    const double day_of_year_days =
        compute_day_of_year_days(forcing, parameters);

    const double lai =
        compute_effective_lai(eco_state, parameters);

    const double fpar =
        compute_fpar(lai, parameters);

    const double temperature_modifier =
        compute_temperature_modifier(forcing, parameters);

    const double hydroperiod_stress =
        compute_hydroperiod_stress(hydro, parameters);

    const double salinity_stress =
        compute_salinity_stress(eco_state, parameters);

    const double lue_gC_per_umol =
        get_parameter_or_default(
            parameters,
            "vegetation_lue_gC_per_umol",
            2.5e-4);

    const double carbon_use_efficiency =
        clamp(
            get_parameter_or_default(
                parameters,
                "vegetation_carbon_use_efficiency",
                0.5),
            0.0,
            1.0);

    const double biomass_carbon_fraction =
        clamp(
            get_parameter_or_default(
                parameters,
                "vegetation_biomass_carbon_fraction",
                0.45),
            1.0e-6,
            1.0);

    const double aboveground_capacity_kg_m2 =
        std::max(
            1.0e-6,
            get_parameter_or_default(
                parameters,
                "vegetation_aboveground_capacity_kg_m2",
                3.0));

    const double belowground_capacity_kg_m2 =
        std::max(
            1.0e-6,
            get_parameter_or_default(
                parameters,
                "vegetation_belowground_capacity_kg_m2",
                5.0));

    const double apar_umol_m2_d =
        fpar * std::max(0.0, forcing.par_umol_m2_d);

    diagnostics.gpp_gC_m2_d =
        lue_gC_per_umol *
        apar_umol_m2_d *
        temperature_modifier *
        hydroperiod_stress *
        salinity_stress;

    diagnostics.gpp_gC_m2_d =
        std::max(0.0, diagnostics.gpp_gC_m2_d);

    diagnostics.npp_gC_m2_d =
        carbon_use_efficiency * diagnostics.gpp_gC_m2_d;

    const double shoot_allocation_fraction =
        compute_shoot_allocation_fraction(
            hydroperiod_stress,
            salinity_stress,
            parameters);

    const double root_allocation_fraction =
        1.0 - shoot_allocation_fraction;

    const double aboveground_capacity_modifier =
        clamp(
            1.0 - eco_state.aboveground_biomass_kg_m2 / aboveground_capacity_kg_m2,
            0.0,
            1.0);

    const double belowground_capacity_modifier =
        clamp(
            1.0 - eco_state.belowground_biomass_kg_m2 / belowground_capacity_kg_m2,
            0.0,
            1.0);

    const double aboveground_growth_gC_m2_d =
        diagnostics.npp_gC_m2_d *
        shoot_allocation_fraction *
        aboveground_capacity_modifier;

    const double belowground_growth_gC_m2_d =
        diagnostics.npp_gC_m2_d *
        root_allocation_fraction *
        belowground_capacity_modifier;

    diagnostics.aboveground_growth_kg_m2_d =
        aboveground_growth_gC_m2_d / (1000.0 * biomass_carbon_fraction);

    diagnostics.belowground_growth_kg_m2_d =
        belowground_growth_gC_m2_d / (1000.0 * biomass_carbon_fraction);

    const double aboveground_mortality_rate_per_day =
        compute_aboveground_mortality_rate_per_day(
            eco_state,
            hydro,
            day_of_year_days,
            parameters);

    const double belowground_mortality_rate_per_day =
        compute_belowground_mortality_rate_per_day(
            eco_state,
            hydro,
            parameters);

    diagnostics.aboveground_mortality_kg_m2_d =
        aboveground_mortality_rate_per_day *
        std::max(0.0, eco_state.aboveground_biomass_kg_m2);

    diagnostics.belowground_mortality_kg_m2_d =
        belowground_mortality_rate_per_day *
        std::max(0.0, eco_state.belowground_biomass_kg_m2);

    eco_state.aboveground_biomass_kg_m2 =
        std::max(
            0.0,
            eco_state.aboveground_biomass_kg_m2 +
            forcing.dt_days *
                (diagnostics.aboveground_growth_kg_m2_d -
                 diagnostics.aboveground_mortality_kg_m2_d));

    eco_state.belowground_biomass_kg_m2 =
        std::max(
            0.0,
            eco_state.belowground_biomass_kg_m2 +
            forcing.dt_days *
                (diagnostics.belowground_growth_kg_m2_d -
                 diagnostics.belowground_mortality_kg_m2_d));

    eco_state.litter_kg_m2 +=
        forcing.dt_days *
        (diagnostics.aboveground_mortality_kg_m2_d +
         diagnostics.belowground_mortality_kg_m2_d);

    eco_state.lai =
        compute_lai_from_aboveground_biomass(
            eco_state.aboveground_biomass_kg_m2,
            parameters);

    diagnostics.lai = eco_state.lai;

    return diagnostics;
}

double marsh_gpp_biomass_model::compute_day_of_year_days(
    const forcing_step& forcing,
    const parameter_set& parameters) const
{
    const double days_per_year =
        std::max(
            1.0,
            get_parameter_or_default(
                parameters,
                "vegetation_days_per_year",
                365.0));

    double day_of_year =
        std::fmod(forcing.model_time_days, days_per_year);

    if (day_of_year < 0.0)
    {
        day_of_year += days_per_year;
    }

    return day_of_year;
}

double marsh_gpp_biomass_model::compute_effective_lai(
    const ecohydrology_state& eco_state,
    const parameter_set& parameters) const
{
    if (eco_state.lai > 0.0)
    {
        return eco_state.lai;
    }

    return compute_lai_from_aboveground_biomass(
        eco_state.aboveground_biomass_kg_m2,
        parameters);
}

double marsh_gpp_biomass_model::compute_fpar(
    double lai,
    const parameter_set& parameters) const
{
    const double max_fpar =
        clamp(
            get_parameter_or_default(
                parameters,
                "vegetation_max_fpar",
                0.95),
            0.0,
            1.0);

    const double light_extinction =
        std::max(
            1.0e-6,
            get_parameter_or_default(
                parameters,
                "vegetation_lai_light_extinction",
                0.8));

    return max_fpar * (1.0 - std::exp(-light_extinction * std::max(0.0, lai)));
}

double marsh_gpp_biomass_model::compute_temperature_modifier(
    const forcing_step& forcing,
    const parameter_set& parameters) const
{
    const double optimum_c =
        get_parameter_or_default(
            parameters,
            "vegetation_temperature_optimum_c",
            25.0);

    const double sigma_c =
        std::max(
            1.0e-6,
            get_parameter_or_default(
                parameters,
                "vegetation_temperature_sigma_c",
                10.0));

    const double temperature_c = forcing.temperature;
    const double normalized =
        (temperature_c - optimum_c) / sigma_c;

    return std::exp(-0.5 * normalized * normalized);
}

double marsh_gpp_biomass_model::compute_hydroperiod_stress(
    const hydrology_diagnostics& hydro,
    const parameter_set& parameters) const
{
    const double optimum_fraction =
        clamp(
            get_parameter_or_default(
                parameters,
                "vegetation_hydroperiod_optimum_fraction",
                0.20),
            0.0,
            1.0);

    const double sigma_fraction =
        std::max(
            1.0e-6,
            get_parameter_or_default(
                parameters,
                "vegetation_hydroperiod_sigma_fraction",
                0.20));

    const double normalized =
        (hydro.inundation_fraction - optimum_fraction) / sigma_fraction;

    return std::exp(-0.5 * normalized * normalized);
}

double marsh_gpp_biomass_model::compute_salinity_stress(
    const ecohydrology_state& eco_state,
    const parameter_set& parameters) const
{
    const double salinity_threshold_ppt =
        get_parameter_or_default(
            parameters,
            "vegetation_salinity_threshold_ppt",
            20.0);

    const double salinity_stress_per_ppt =
        std::max(
            0.0,
            get_parameter_or_default(
                parameters,
                "vegetation_salinity_stress_per_ppt",
                0.03));

    const double salinity_excess_ppt =
        std::max(
            0.0,
            eco_state.root_zone_salinity_ppt - salinity_threshold_ppt);

    return std::exp(-salinity_stress_per_ppt * salinity_excess_ppt);
}

double marsh_gpp_biomass_model::compute_shoot_allocation_fraction(
    double hydroperiod_stress,
    double salinity_stress,
    const parameter_set& parameters) const
{
    const double shoot_allocation_min =
        clamp(
            get_parameter_or_default(
                parameters,
                "vegetation_shoot_allocation_min",
                0.25),
            0.0,
            1.0);

    const double shoot_allocation_max =
        clamp(
            get_parameter_or_default(
                parameters,
                "vegetation_shoot_allocation_max",
                0.65),
            0.0,
            1.0);

    const double stress_index =
        clamp(hydroperiod_stress * salinity_stress, 0.0, 1.0);

    return
        shoot_allocation_min +
        (shoot_allocation_max - shoot_allocation_min) * stress_index;
}

double marsh_gpp_biomass_model::compute_lai_from_aboveground_biomass(
    double aboveground_biomass_kg_m2,
    const parameter_set& parameters) const
{
    const double lai_per_kg_m2 =
        std::max(
            1.0e-6,
            get_parameter_or_default(
                parameters,
                "vegetation_lai_per_kg_m2",
                3.0));

    const double max_lai =
        std::max(
            1.0e-6,
            get_parameter_or_default(
                parameters,
                "vegetation_max_lai",
                6.0));

    const double raw_lai =
        lai_per_kg_m2 * std::max(0.0, aboveground_biomass_kg_m2);

    return std::min(raw_lai, max_lai);
}

double marsh_gpp_biomass_model::compute_aboveground_mortality_rate_per_day(
    const ecohydrology_state& eco_state,
    const hydrology_diagnostics& hydro,
    double day_of_year_days,
    const parameter_set& parameters) const
{
    const double days_per_year =
        std::max(
            1.0,
            get_parameter_or_default(
                parameters,
                "vegetation_days_per_year",
                365.0));

    const double baseline_rate =
        std::max(
            0.0,
            get_parameter_or_default(
                parameters,
                "vegetation_aboveground_mortality_base_per_day",
                0.001));

    const double seasonal_amplitude =
        std::max(
            0.0,
            get_parameter_or_default(
                parameters,
                "vegetation_aboveground_mortality_seasonal_amp_per_day",
                0.002));

    const double seasonal_peak_day =
        get_parameter_or_default(
            parameters,
            "vegetation_mortality_peak_day",
            300.0);

    const double hydro_threshold =
        clamp(
            get_parameter_or_default(
                parameters,
                "vegetation_aboveground_hydroperiod_mortality_threshold",
                0.50),
            0.0,
            1.0);

    const double hydro_slope =
        std::max(
            0.0,
            get_parameter_or_default(
                parameters,
                "vegetation_aboveground_hydroperiod_mortality_slope_per_day",
                0.01));

    const double salinity_threshold =
        get_parameter_or_default(
            parameters,
            "vegetation_aboveground_salinity_mortality_threshold_ppt",
            30.0);

    const double salinity_slope =
        std::max(
            0.0,
            get_parameter_or_default(
                parameters,
                "vegetation_aboveground_salinity_mortality_slope_per_day_per_ppt",
                0.0005));

    const double seasonal_term =
        0.5 *
        (1.0 -
         std::cos(
             2.0 * 3.14159265358979323846 *
             (day_of_year_days - seasonal_peak_day) /
             days_per_year));

    const double hydro_excess =
        std::max(0.0, hydro.inundation_fraction - hydro_threshold);

    const double salinity_excess =
        std::max(0.0, eco_state.root_zone_salinity_ppt - salinity_threshold);

    return
        baseline_rate +
        seasonal_amplitude * seasonal_term +
        hydro_slope * hydro_excess +
        salinity_slope * salinity_excess;
}

double marsh_gpp_biomass_model::compute_belowground_mortality_rate_per_day(
    const ecohydrology_state& eco_state,
    const hydrology_diagnostics& hydro,
    const parameter_set& parameters) const
{
    const double baseline_rate =
        std::max(
            0.0,
            get_parameter_or_default(
                parameters,
                "vegetation_belowground_mortality_base_per_day",
                0.0005));

    const double hydro_threshold =
        clamp(
            get_parameter_or_default(
                parameters,
                "vegetation_belowground_hydroperiod_mortality_threshold",
                0.60),
            0.0,
            1.0);

    const double hydro_slope =
        std::max(
            0.0,
            get_parameter_or_default(
                parameters,
                "vegetation_belowground_hydroperiod_mortality_slope_per_day",
                0.005));

    const double salinity_threshold =
        get_parameter_or_default(
            parameters,
            "vegetation_belowground_salinity_mortality_threshold_ppt",
            35.0);

    const double salinity_slope =
        std::max(
            0.0,
            get_parameter_or_default(
                parameters,
                "vegetation_belowground_salinity_mortality_slope_per_day_per_ppt",
                0.00025));

    const double hydro_excess =
        std::max(0.0, hydro.inundation_fraction - hydro_threshold);

    const double salinity_excess =
        std::max(0.0, eco_state.root_zone_salinity_ppt - salinity_threshold);

    return
        baseline_rate +
        hydro_slope * hydro_excess +
        salinity_slope * salinity_excess;
}
}
