#pragma once

#include "marsh_model/core/ecohydrology_state.hpp"
#include "marsh_model/core/et_fluxes.hpp"
#include "marsh_model/core/forcing_step.hpp"
#include "marsh_model/core/hydrology_diagnostics.hpp"
#include "marsh_model/core/parameter_set.hpp"
#include "marsh_model/core/sediment_surface_properties.hpp"
#include "marsh_model/core/site_properties.hpp"
#include "marsh_model/processes/evapotranspiration_model.hpp"

namespace marsh_model
{
class simple_canopy_et_model : public evapotranspiration_model
{
public:
    et_fluxes compute_et(
        const ecohydrology_state& eco_state,
        const hydrology_diagnostics& hydro,
        const sediment_surface_properties& surface,
        const forcing_step& forcing,
        const site_properties& site,
        const parameter_set& parameters) const override;

private:
    double compute_effective_lai(
        const ecohydrology_state& eco_state,
        const parameter_set& parameters) const;

    double compute_potential_et_mm_d(
        const forcing_step& forcing,
        const parameter_set& parameters) const;
};
}
