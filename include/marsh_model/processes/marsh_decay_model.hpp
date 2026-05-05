#pragma once

#include "marsh_model/core/material_properties.hpp"
#include "marsh_model/processes/decay_model.hpp"

namespace marsh_model
{
class marsh_decay_model : public decay_model
{
public:
    marsh_decay_model() = default;

    void apply_decay(
        column_state& state,
        const forcing_step& forcing,
        const material_catalog& catalog,
        const parameter_set& parameters,
        const decay_context& context) const override;

private:
    double compute_layer_midpoint_depth_m(
        const column_state& state,
        int layer_index) const;

    double compute_decay_rate_per_day(
        const material_properties& material,
        double midpoint_depth_m,
        double temperature,
        const parameter_set& parameters) const;

    double compute_decay_multiplier(
        double decay_rate_per_day,
        double dt_days,
        const parameter_set& parameters) const;
};
}
