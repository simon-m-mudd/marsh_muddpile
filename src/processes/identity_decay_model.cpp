#include "marsh_model/processes/identity_decay_model.hpp"

namespace marsh_model
{
void identity_decay_model::apply_decay(
    column_state&,
    const forcing_step&,
    const material_catalog&,
    const parameter_set&,
    const decay_context&) const
{
}
}
