#pragma once

#include "marsh_model/core/column_state.hpp"
#include "marsh_model/core/decay_context.hpp"
#include "marsh_model/core/forcing_step.hpp"
#include "marsh_model/core/material_catalog.hpp"
#include "marsh_model/core/parameter_set.hpp"

namespace marsh_model
{
class decay_model
{
public:
    virtual ~decay_model() = default;

    void apply_decay(
        column_state& state,
        const forcing_step& forcing,
        const material_catalog& catalog,
        const parameter_set& parameters) const
    {
        apply_decay(
            state,
            forcing,
            catalog,
            parameters,
            decay_context{});
    }

    virtual void apply_decay(
        column_state& state,
        const forcing_step& forcing,
        const material_catalog& catalog,
        const parameter_set& parameters,
        const decay_context& context) const = 0;
};
}
