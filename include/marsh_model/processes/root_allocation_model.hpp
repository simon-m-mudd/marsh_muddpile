#pragma once

#include "marsh_model/core/column_state.hpp"
#include "marsh_model/core/material_catalog.hpp"
#include "marsh_model/core/parameter_set.hpp"
#include "marsh_model/processes/biomass_model.hpp"
#include <Eigen/Core>

namespace marsh_model
{
class root_allocation_model
{
public:
    virtual ~root_allocation_model() = default;

    virtual Eigen::ArrayXXd compute_root_mass_change(
        const column_state& state,
        const biomass_fluxes& biomass,
        const material_catalog& catalog,
        const parameter_set& parameters) const = 0;
};
}
