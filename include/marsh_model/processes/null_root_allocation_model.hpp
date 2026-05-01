#pragma once

#include "marsh_model/processes/root_allocation_model.hpp"

namespace marsh_model
{
class null_root_allocation_model : public root_allocation_model
{
public:
    Eigen::ArrayXXd compute_root_mass_change(
        const column_state& state,
        const biomass_fluxes& biomass,
        const material_catalog& catalog,
        const parameter_set& parameters) const override;
};
}
