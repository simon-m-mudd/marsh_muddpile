#pragma once

#include "marsh_model/core/column_state.hpp"
#include "marsh_model/core/forcing_step.hpp"
#include "marsh_model/core/hydrology_diagnostics.hpp"
#include "marsh_model/core/parameter_set.hpp"
#include "marsh_model/core/site_properties.hpp"

namespace marsh_model
{
class water_level_model
{
public:
    virtual ~water_level_model() = default;

    virtual hydrology_diagnostics compute_hydrology(
        const column_state& state,
        const forcing_step& forcing,
        const site_properties& site,
        const parameter_set& parameters) const = 0;
};
}
