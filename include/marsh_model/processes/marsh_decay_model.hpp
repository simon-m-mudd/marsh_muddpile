#pragma once

// marsh_decay_model.hpp
//
// First-order organic matter decay for each layer of the sediment column.
//
// The effective decay rate combines a catalogue surface rate (k_0), an
// exponential depth-attenuation term, an optional temperature sensitivity,
// and a dimensionless modifier from hydro_salinity_decomposition_modifier_model
// that accounts for anoxia (inundation fraction) and salinity inhibition
// separately for labile, refractory, and root-derived organic pools.
//
// See marsh_decay_model.cpp for the full formulation and references.

#include "marsh_model/core/material_properties.hpp"
#include "marsh_model/processes/decay_model.hpp"

namespace marsh_model
{
class marsh_decay_model : public decay_model
{
public:
    marsh_decay_model() = default;

    // Applies one time step of decay to all organic layers in state.
    void apply_decay(
        column_state& state,
        const forcing_step& forcing,
        const material_catalog& catalog,
        const parameter_set& parameters,
        const decay_context& context) const override;

private:
    // Depth of layer midpoint below the sediment surface (m).
    double compute_layer_midpoint_depth_m(
        const column_state& state,
        int layer_index) const;

    // Effective first-order decay rate (d-1) with depth and temperature corrections.
    double compute_decay_rate_per_day(
        const material_properties& material,
        double midpoint_depth_m,
        double temperature,
        const parameter_set& parameters) const;

    // Mass survival fraction for one time step: exp(-k*dt) or Taylor approximation.
    double compute_decay_multiplier(
        double decay_rate_per_day,
        double dt_days,
        const parameter_set& parameters) const;
};
}
