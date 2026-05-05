#pragma once

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
    vegetation_diagnostics update_vegetation(
        ecohydrology_state& eco_state,
        const hydrology_diagnostics& hydro,
        const sediment_surface_properties& surface,
        const forcing_step& forcing,
        const site_properties& site,
        const parameter_set& parameters) const override;

private:
    double compute_day_of_year_days(
        const forcing_step& forcing,
        const parameter_set& parameters) const;

    double compute_effective_lai(
        const ecohydrology_state& eco_state,
        const parameter_set& parameters) const;

    double compute_fpar(
        double lai,
        const parameter_set& parameters) const;

    double compute_temperature_modifier(
        const forcing_step& forcing,
        const parameter_set& parameters) const;

    double compute_hydroperiod_stress(
        const hydrology_diagnostics& hydro,
        const parameter_set& parameters) const;

    double compute_salinity_stress(
        const ecohydrology_state& eco_state,
        const parameter_set& parameters) const;

    double compute_shoot_allocation_fraction(
        double hydroperiod_stress,
        double salinity_stress,
        const parameter_set& parameters) const;

    double compute_lai_from_aboveground_biomass(
        double aboveground_biomass_kg_m2,
        const parameter_set& parameters) const;

    double compute_aboveground_mortality_rate_per_day(
        const ecohydrology_state& eco_state,
        const hydrology_diagnostics& hydro,
        double day_of_year_days,
        const parameter_set& parameters) const;

    double compute_belowground_mortality_rate_per_day(
        const ecohydrology_state& eco_state,
        const hydrology_diagnostics& hydro,
        const parameter_set& parameters) const;
};
}
