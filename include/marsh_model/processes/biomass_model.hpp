#pragma once

#include "marsh_model/core/column_state.hpp"
#include "marsh_model/core/forcing_step.hpp"
#include "marsh_model/core/material_catalog.hpp"
#include "marsh_model/core/parameter_set.hpp"

namespace marsh_model
{
struct biomass_fluxes
{
    double aboveground_biomass = 0.0;
    double belowground_biomass = 0.0;
    double belowground_mortality = 0.0;
    double peak_biomass = 0.0;
};

class biomass_model
{
public:
    virtual ~biomass_model() = default;

    virtual biomass_fluxes compute_biomass_fluxes(
        const column_state& state,
        const forcing_step& forcing,
        const material_catalog& catalog,
        const parameter_set& parameters) const = 0;
};
}
