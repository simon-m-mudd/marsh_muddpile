#pragma once

#include "marsh_model/processes/biomass_model.hpp"

namespace marsh_model
{
class null_biomass_model : public biomass_model
{
public:
    biomass_fluxes compute_biomass_fluxes(
        const column_state& state,
        const forcing_step& forcing,
        const material_catalog& catalog,
        const parameter_set& parameters) const override;
};
}
