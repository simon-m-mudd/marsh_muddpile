#pragma once

#include "marsh_model/core/decomposition_modifiers.hpp"
#include "marsh_model/core/ecohydrology_state.hpp"
#include "marsh_model/core/forcing_step.hpp"
#include "marsh_model/core/hydrology_diagnostics.hpp"
#include "marsh_model/core/parameter_set.hpp"
#include "marsh_model/core/sediment_surface_properties.hpp"
#include "marsh_model/core/site_properties.hpp"
#include "marsh_model/processes/decomposition_modifier_model.hpp"

namespace marsh_model
{
class hydro_salinity_decomposition_modifier_model
    : public decomposition_modifier_model
{
public:
    decomposition_modifiers compute_modifiers(
        const ecohydrology_state& eco_state,
        const hydrology_diagnostics& hydro,
        const sediment_surface_properties& surface,
        const forcing_step& forcing,
        const site_properties& site,
        const parameter_set& parameters) const override;
};
}
