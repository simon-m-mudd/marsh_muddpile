#include "marsh_model/processes/null_root_allocation_model.hpp"

namespace marsh_model
{
Eigen::ArrayXXd null_root_allocation_model::compute_root_mass_change(
    const column_state& state,
    const biomass_fluxes&,
    const material_catalog&,
    const parameter_set&) const
{
    return Eigen::ArrayXXd::Zero(state.n_layers(), state.n_materials());
}
}
