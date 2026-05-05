#pragma once

#include "marsh_model/core/ecohydrology_state.hpp"
#include "marsh_model/core/hydrology_diagnostics.hpp"
#include "marsh_model/core/sediment_surface_properties.hpp"
#include "marsh_model/core/site_properties.hpp"

namespace marsh_model
{
struct decay_context
{
    const ecohydrology_state* eco_state = nullptr;
    const hydrology_diagnostics* hydrology = nullptr;
    const sediment_surface_properties* surface = nullptr;
    const site_properties* site = nullptr;
};
}
