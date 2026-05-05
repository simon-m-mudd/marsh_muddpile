#pragma once

#include "marsh_model/core/column_state.hpp"
#include "marsh_model/core/material_catalog.hpp"
#include "marsh_model/core/parameter_set.hpp"
#include "marsh_model/core/sediment_surface_properties.hpp"

namespace marsh_model
{
class surface_property_summariser
{
public:
    static sediment_surface_properties summarize(
        const column_state& state,
        const material_catalog& catalog,
        const parameter_set& parameters);
};
}
