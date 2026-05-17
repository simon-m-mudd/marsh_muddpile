// marsh_gpp_biomass_model.cpp
//
// Light-use efficiency (LUE) vegetation model for tidal marsh plants.
//
// GPP is driven by absorbed PAR (via Beer-Lambert canopy interception) multiplied
// by stress scalars for temperature, hydroperiod, and root-zone salinity. NPP is
// split between aboveground shoots and belowground roots depending on stress
// conditions, with logistic capacity limits on each pool. Separate mortality rates
// handle aboveground seasonal senescence (with a cosine annual cycle) and
// belowground root turnover. The ecohydrology state - biomass pools, litter, and
// LAI - is updated in-place each time step.
//
// References:
//   Monteith, J.L., 1972. Solar radiation and productivity in tropical
//     ecosystems. Journal of Applied Ecology 9, 747-766.
//     https://doi.org/10.2307/2401901
//
//   Oikawa, P.Y., Jenerette, G.D., Knox, S.H., Sturtevant, C., Verfaillie, J.,
//     Dronova, I., Poindexter, C.M., Eichelmann, E., Baldocchi, D.D., 2017.
//     Evaluation of a hierarchy of models reveals importance of substrate
//     limitation for predicting carbon dioxide and methane exchange in restored
//     wetlands. Journal of Geophysical Research: Biogeosciences 122, 145-167.
//     https://doi.org/10.1002/2016JG003438
//
//   Oikawa, P.Y. et al., PEPRMT-Tidal v1.0 (tidal wetland extension of PEPRMT).
//     https://github.com/pattyoikawa/PEPRMT-Tidal/tree/v1.0

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

// Advances vegetation one time step. Computes GPP from LUE * APAR * stress terms,
// derives NPP via carbon-use efficiency, partitions NPP to shoots and roots with
// capacity modifiers, applies mortality to both pools, accumulates litter, and
// updates LAI. Returns per-day flux diagnostics for the current step.
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

    const double lai =
        compute_effective_lai(eco_state, parameters);

    const double fpar =
        compute_fpar(lai, parameters);

    const double temperature_modifier =
        compute_temperature_modifier(forcing, parameters);

    const double hydroperiod_stress =
        compute_hydroperiod_stress(hydro, parameters);

    const double inundation_range_stressor =
        compute_inundation_range_stressor(hydro, parameters);

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

    const double root_shoot_ratio =
        std::max(
            1.0e-6,
            get_parameter_or_default(
                parameters,
                "vegetation_root_shoot_ratio",
                2.0));

    // Dynamic belowground capacity tracks the root:shoot target.
    // When aboveground is zero the cap collapses to near-zero so roots decay
    // at their turnover rate rather than accumulating against a fixed ceiling.
    const double belowground_capacity_kg_m2 =
        std::max(1.0e-6,
                 eco_state.aboveground_biomass_kg_m2 * root_shoot_ratio);

    const double apar_umol_m2_d =
        fpar * std::max(0.0, forcing.par_umol_m2_d);

    diagnostics.gpp_gC_m2_d =
        lue_gC_per_umol *
        apar_umol_m2_d *
        temperature_modifier *
        hydroperiod_stress *
        salinity_stress *
        inundation_range_stressor;

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
            forcing.temperature,
            parameters);

    const double belowground_mortality_rate_per_day =
        compute_belowground_mortality_rate_per_day(
            eco_state,
            hydro,
            parameters);

    // Aboveground ODE: exact solution to
    //   dbio/dt = g*(1 - bio/cap) - m*bio  =  g - B_eff*bio
    // where B_eff = g/cap + m  and  bio_eq = g/B_eff.
    // This avoids forward-Euler overshoot with monthly timesteps.
    const double g_above =
        diagnostics.npp_gC_m2_d * shoot_allocation_fraction /
        (1000.0 * biomass_carbon_fraction);
    const double B_eff_above =
        g_above / aboveground_capacity_kg_m2 + aboveground_mortality_rate_per_day;

    const double g_below =
        diagnostics.npp_gC_m2_d * root_allocation_fraction /
        (1000.0 * biomass_carbon_fraction);

    const double bio0_above = eco_state.aboveground_biomass_kg_m2;
    const double bio0_below = eco_state.belowground_biomass_kg_m2;

    if (B_eff_above > 0.0)
    {
        const double bio_eq = g_above / B_eff_above;
        eco_state.aboveground_biomass_kg_m2 = std::max(
            0.0,
            bio_eq + (bio0_above - bio_eq) * std::exp(-B_eff_above * forcing.dt_days));
    }

    // Belowground ODE — two regimes to prevent winter cap-collapse from killing
    // roots/rhizomes. The root:shoot target only constrains growth; it never
    // accelerates mortality when aboveground biomass is seasonally low.
    //
    // Growth regime (bio < cap):
    //   dbio/dt = g*(1 - bio/cap) - m*bio  →  logistic approach to target
    //
    // Decay regime (bio >= cap, i.e. shoots died back, cap has shrunk):
    //   dbio/dt = -m*bio  →  pure turnover at the natural belowground rate;
    //   rhizomes persist through winter and are not driven to zero by shoot death.
    double litter_below_kg_m2 = 0.0;
    if (bio0_below < belowground_capacity_kg_m2)
    {
        const double B_eff_below =
            g_below / belowground_capacity_kg_m2 + belowground_mortality_rate_per_day;
        if (B_eff_below > 0.0)
        {
            const double bio_eq = g_below / B_eff_below;
            eco_state.belowground_biomass_kg_m2 = std::max(
                0.0,
                bio_eq + (bio0_below - bio_eq) * std::exp(-B_eff_below * forcing.dt_days));
            litter_below_kg_m2 =
                (belowground_mortality_rate_per_day / B_eff_below) *
                std::max(0.0, g_below * forcing.dt_days + bio0_below -
                              eco_state.belowground_biomass_kg_m2);
        }
    }
    else
    {
        // Roots/rhizomes above the current root:shoot target: no new growth,
        // purely decay at the natural turnover rate.
        const double m = belowground_mortality_rate_per_day;
        eco_state.belowground_biomass_kg_m2 = std::max(
            0.0, bio0_below * std::exp(-m * forcing.dt_days));
        litter_below_kg_m2 = bio0_below - eco_state.belowground_biomass_kg_m2;
    }

    // Aboveground litter from the exact ODE mass-balance identity:
    //   integral_0^dt m*B(t) dt = (m / B_eff) * (g*dt + B0 - B_final)
    const double litter_above_kg_m2 =
        (B_eff_above > 0.0)
        ? (aboveground_mortality_rate_per_day / B_eff_above) *
          std::max(0.0, g_above * forcing.dt_days + bio0_above -
                        eco_state.aboveground_biomass_kg_m2)
        : 0.0;

    eco_state.litter_kg_m2 += litter_above_kg_m2 + litter_below_kg_m2;

    diagnostics.aboveground_mortality_kg_m2_d = litter_above_kg_m2 / forcing.dt_days;
    diagnostics.belowground_mortality_kg_m2_d = litter_below_kg_m2 / forcing.dt_days;

    eco_state.lai =
        compute_lai_from_aboveground_biomass(
            eco_state.aboveground_biomass_kg_m2,
            parameters);

    diagnostics.lai = eco_state.lai;

    return diagnostics;
}

// Returns the LAI stored in eco_state if it has been set (> 0); otherwise
// bootstraps an estimate from aboveground biomass for the first step.
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

// Fraction of incoming PAR absorbed by the canopy using Beer-Lambert extinction.
// FPAR = max_fpar * (1 - exp(-k * LAI)), capped at max_fpar.
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

// Gaussian response centred on the optimum temperature. Returns 1 at the
// optimum and declines symmetrically above and below it.
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

// Gaussian stress factor centred on the optimum inundation fraction. Productivity
// peaks at moderate flooding; both too-dry and too-wet conditions reduce GPP.
double marsh_gpp_biomass_model::compute_hydroperiod_stress(
    const hydrology_diagnostics& hydro,
    const parameter_set& parameters) const
{
    const double optimum_fraction =
        clamp(
            get_parameter_or_default(
                parameters,
                "vegetation_hydroperiod_optimum_fraction",
                0.25),
            0.0,
            1.0);

    const double sigma_fraction =
        std::max(
            1.0e-6,
            get_parameter_or_default(
                parameters,
                "vegetation_hydroperiod_sigma_fraction",
                0.18));

    // Half-Gaussian (asymmetric): no penalty on the wet side (IF >= optimum,
    // i.e. deeper/lower platform) — flooding itself does not limit GPP there.
    // Stress is applied only when IF < optimum (drier/higher platform), where
    // reduced tidal inundation means less nutrient delivery and less freshwater
    // supply, producing the high-elevation biomass decline observed in the field
    // (Morris et al. 2013, Oceanography).  Together with compute_inundation_range_
    // stressor (which handles the wet-side decline), the two complementary
    // half-Gaussians produce the full asymmetric biomass-elevation curve.
    if (hydro.inundation_fraction >= optimum_fraction)
        return 1.0;

    const double normalized =
        (hydro.inundation_fraction - optimum_fraction) / sigma_fraction;

    return std::exp(-0.5 * normalized * normalized);
}

// Gaussian inundation-range stressor centred on the empirical optimum inundation
// fraction for the species.  Produces a smooth bell-shaped biomass-elevation
// response consistent with field observations (Morris et al. 2013 Oceanography).
//
// Replaces the earlier tent function, which imposed hard zero-growth cutoffs at
// F_min and F_max and produced a flat-topped biomass plateau across the mid-
// intertidal zone.  The Gaussian has non-zero values across the full [0,1]
// inundation range, so no hard elevation boundaries are needed.
//
// Default parameters calibrated to North Inlet, SC S. alterniflora
// (tidal range ≈ 0.65 m, MHW ≈ 0.65 m above MSL):
//   optimum_fraction = 0.25  — peak at ≈ 47 cm NAVD88 (Morris et al. 2013)
//   sigma_fraction   = 0.15  — gives near-zero values at IF ≈ 0 (above MHW)
//                              and IF ≈ 0.75 (permanently flooded mudflat)
double marsh_gpp_biomass_model::compute_inundation_range_stressor(
    const hydrology_diagnostics& hydro,
    const parameter_set& parameters) const
{
    const double optimum_fraction =
        clamp(
            get_parameter_or_default(
                parameters,
                "vegetation_inundation_optimum_fraction",
                0.25),
            0.0,
            1.0);

    const double sigma_fraction =
        std::max(
            1.0e-6,
            get_parameter_or_default(
                parameters,
                "vegetation_inundation_sigma_fraction",
                0.25));   // half-Gaussian left-limb sigma (wet-side decay)

    // Half-Gaussian (asymmetric): no penalty on the dry side (IF <= optimum,
    // i.e. drier/higher platform) — species such as S. alterniflora are highly
    // flood-tolerant and productivity is not limited by *too little* flooding
    // within the normal tidal frame. Flooding/anaerobic stress is only applied
    // when IF > optimum (wetter/lower platform), producing a one-sided decay
    // that correctly gives near-zero GPP in the permanently flooded mudflat
    // while leaving the high-intertidal to mid-intertidal stressor = 1.
    if (hydro.inundation_fraction <= optimum_fraction)
        return 1.0;

    const double normalized =
        (hydro.inundation_fraction - optimum_fraction) / sigma_fraction;

    return std::exp(-0.5 * normalized * normalized);
}

// Exponential GPP reduction for root-zone salinity above a threshold.
// Represents osmotic and ionic stress limiting carbon assimilation.
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

// NPP fraction directed to shoots. Under low combined stress more carbon goes
// aboveground; under high stress allocation shifts toward roots.
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

// Converts aboveground biomass to LAI with a fixed specific leaf area coefficient,
// capped at a maximum LAI to prevent unbounded canopy growth.
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

// Aboveground mortality rate (d^-1) composed of three additive terms:
//   baseline rate  - background senescence throughout the year
//   cold term      - linear increase below a temperature threshold, representing
//                    cold-driven senescence (replaces the former cosine cycle)
//   stress terms   - excess mortality when inundation or salinity exceeds thresholds
double marsh_gpp_biomass_model::compute_aboveground_mortality_rate_per_day(
    const ecohydrology_state& eco_state,
    const hydrology_diagnostics& hydro,
    double temperature_c,
    const parameter_set& parameters) const
{
    const double baseline_rate =
        std::max(
            0.0,
            get_parameter_or_default(
                parameters,
                "vegetation_aboveground_mortality_base_per_day",
                0.001));

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

    const double hydro_excess =
        std::max(0.0, hydro.inundation_fraction - hydro_threshold);

    const double salinity_excess =
        std::max(0.0, eco_state.root_zone_salinity_ppt - salinity_threshold);

    const double cold_mortality_threshold_c =
        get_parameter_or_default(
            parameters,
            "vegetation_cold_mortality_threshold_c",
            20.0);

    const double cold_mortality_slope =
        std::max(
            0.0,
            get_parameter_or_default(
                parameters,
                "vegetation_cold_mortality_slope_per_day_per_c",
                0.002));

    const double cold_excess_c =
        std::max(0.0, cold_mortality_threshold_c - temperature_c);

    // Extra mortality when inundation is below the minimum viable fraction
    // (surface above MHHW: plants become too dry to survive).
    const double low_inundation_threshold =
        clamp(
            get_parameter_or_default(
                parameters,
                "vegetation_low_inundation_mortality_threshold",
                0.05),
            0.0,
            1.0);

    const double low_inundation_slope =
        std::max(
            0.0,
            get_parameter_or_default(
                parameters,
                "vegetation_low_inundation_mortality_slope_per_day",
                0.10));

    const double low_inundation_deficit =
        std::max(0.0, low_inundation_threshold - hydro.inundation_fraction);

    return
        baseline_rate +
        cold_mortality_slope * cold_excess_c +
        hydro_slope * hydro_excess +
        salinity_slope * salinity_excess +
        low_inundation_slope * low_inundation_deficit;
}

// Belowground mortality rate (d^-1): a baseline root-turnover rate plus linear
// penalty terms for inundation and salinity above their respective thresholds.
// No seasonal cycle - root turnover is treated as approximately year-round.
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

    const double low_inundation_threshold =
        clamp(
            get_parameter_or_default(
                parameters,
                "vegetation_low_inundation_mortality_threshold",
                0.05),
            0.0,
            1.0);

    const double low_inundation_slope =
        std::max(
            0.0,
            get_parameter_or_default(
                parameters,
                "vegetation_low_inundation_mortality_slope_per_day",
                0.10));

    const double low_inundation_deficit =
        std::max(0.0, low_inundation_threshold - hydro.inundation_fraction);

    return
        baseline_rate +
        hydro_slope * hydro_excess +
        salinity_slope * salinity_excess +
        low_inundation_slope * low_inundation_deficit;
}
}
