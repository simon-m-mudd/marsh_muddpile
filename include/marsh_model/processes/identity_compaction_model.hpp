#pragma once

#include "marsh_model/processes/compaction_model.hpp"

namespace marsh_model
{
class identity_compaction_model : public compaction_model
{
public:
    void update_compaction(
        column_state& state,
        const forcing_step& forcing,
        const material_catalog& catalog,
        const parameter_set& parameters) const override;
};
}
