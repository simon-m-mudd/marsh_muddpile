#pragma once

#include "marsh_model/core/ecohydrology_state.hpp"
#include "marsh_model/core/et_fluxes.hpp"
#include "marsh_model/core/forcing_step.hpp"
#include "marsh_model/core/hydrology_diagnostics.hpp"
#include "marsh_model/core/parameter_set.hpp"
#include "marsh_model/core/sediment_surface_properties.hpp"
#include "marsh_model/core/site_properties.hpp"

namespace marsh_model
{
class salinity_model
{
public:
    virtual ~salinity_model() = default;

    virtual void update_salinity(
        ecohydrology_state& eco_state,
        const hydrology_diagnostics& hydro,
        const et_fluxes& et,
        const sediment_surface_properties& surface,
        const forcing_step& forcing,
        const site_properties& site,
        const parameter_set& parameters) const = 0;
};
}
