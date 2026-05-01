#include "marsh_model/processes/null_biomass_model.hpp"

namespace marsh_model
{
biomass_fluxes null_biomass_model::compute_biomass_fluxes(
    const column_state&,
    const forcing_step&,
    const material_catalog&,
    const parameter_set&) const
{
    return biomass_fluxes{};
}
}
