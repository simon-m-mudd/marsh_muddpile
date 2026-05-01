#pragma once

#include "marsh_model/processes/decay_model.hpp"

namespace marsh_model
{
class identity_decay_model : public decay_model
{
public:
    void apply_decay(
        column_state& state,
        const forcing_step& forcing,
        const material_catalog& catalog,
        const parameter_set& parameters) const override;
};
}
