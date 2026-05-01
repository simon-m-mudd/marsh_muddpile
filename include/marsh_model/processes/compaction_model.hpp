#pragma once

#include "marsh_model/core/column_state.hpp"
#include "marsh_model/core/forcing_step.hpp"
#include "marsh_model/core/material_catalog.hpp"
#include "marsh_model/core/parameter_set.hpp"

namespace marsh_model
{
class compaction_model
{
public:
    virtual ~compaction_model() = default;

    virtual void update_compaction(
        column_state& state,
        const forcing_step& forcing,
        const material_catalog& catalog,
        const parameter_set& parameters) const = 0;
};
}
