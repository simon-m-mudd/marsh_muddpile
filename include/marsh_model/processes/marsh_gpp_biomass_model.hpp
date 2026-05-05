#pragma once

// marsh_gpp_biomass_model.hpp
//
// Light-use efficiency (LUE) vegetation model for tidal marsh plants.
//
// Gross primary production is computed each time step as:
//   GPP = LUE * APAR * f(temperature) * f(hydroperiod) * f(salinity)
//
// NPP is partitioned between shoots and roots based on plant stress, with
// capacity limits on each pool. Separate mortality rates (with a seasonal
// cycle for aboveground tissue) are applied and the ecohydrology state is
// updated in-place.
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

#include "marsh_model/core/ecohydrology_state.hpp"
#include "marsh_model/core/forcing_step.hpp"
#include "marsh_model/core/hydrology_diagnostics.hpp"
#include "marsh_model/core/parameter_set.hpp"
#include "marsh_model/core/sediment_surface_properties.hpp"
#include "marsh_model/core/site_properties.hpp"
#include "marsh_model/core/vegetation_diagnostics.hpp"
#include "marsh_model/processes/vegetation_model.hpp"

namespace marsh_model
{
class marsh_gpp_biomass_model : public vegetation_model
{
public:
    // Full vegetation time step: computes GPP/NPP, allocates carbon to shoots
    // and roots, applies mortality, and updates eco_state in-place.
    vegetation_diagnostics update_vegetation(
        ecohydrology_state& eco_state,
        const hydrology_diagnostics& hydro,
        const sediment_surface_properties& surface,
        const forcing_step& forcing,
        const site_properties& site,
        const parameter_set& parameters) const override;

private:
    // Wraps absolute model time to a day-of-year value for seasonal mortality.
    double compute_day_of_year_days(
        const forcing_step& forcing,
        const parameter_set& parameters) const;

    // Returns current LAI, falling back to a biomass-derived estimate if unset.
    double compute_effective_lai(
        const ecohydrology_state& eco_state,
        const parameter_set& parameters) const;

    // Beer-Lambert fraction of PAR absorbed by the canopy (FPAR).
    double compute_fpar(
        double lai,
        const parameter_set& parameters) const;

    // Gaussian GPP multiplier centred on the optimum temperature.
    double compute_temperature_modifier(
        const forcing_step& forcing,
        const parameter_set& parameters) const;

    // Gaussian GPP multiplier centred on the optimum inundation fraction.
    double compute_hydroperiod_stress(
        const hydrology_diagnostics& hydro,
        const parameter_set& parameters) const;

    // Exponential GPP decline above a root-zone salinity threshold.
    double compute_salinity_stress(
        const ecohydrology_state& eco_state,
        const parameter_set& parameters) const;

    // Shoot allocation fraction: shifts toward roots under combined stress.
    double compute_shoot_allocation_fraction(
        double hydroperiod_stress,
        double salinity_stress,
        const parameter_set& parameters) const;

    // Linear biomass-to-LAI conversion, capped at a maximum value.
    double compute_lai_from_aboveground_biomass(
        double aboveground_biomass_kg_m2,
        const parameter_set& parameters) const;

    // Baseline senescence plus a seasonal cosine cycle and excess-stress terms
    // for flooding and salinity.
    double compute_aboveground_mortality_rate_per_day(
        const ecohydrology_state& eco_state,
        const hydrology_diagnostics& hydro,
        double day_of_year_days,
        const parameter_set& parameters) const;

    // Baseline root turnover amplified by excess flooding or salinity stress.
    double compute_belowground_mortality_rate_per_day(
        const ecohydrology_state& eco_state,
        const hydrology_diagnostics& hydro,
        const parameter_set& parameters) const;
};
}
