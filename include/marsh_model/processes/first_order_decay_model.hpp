#pragma once

// first_order_decay_model.hpp
//
// Part of marsh_muddpile-- https://github.com/simon-m-mudd/marsh_muddpile
//
// Copyright (C) 2026 Simon M. Mudd
// Released under the GNU General Public Licence v3 (GPL-3.0)
// See LICENSE file or https://www.gnu.org/licenses/gpl-3.0.html
//
// -----------------------------------------------------------------------------
// this component implements a column-scale first-order decay model for organic
// matter pools, isotopes, and other tracers that have decay properties.
//
// the implementation generalises the older decay logic so that any material
// with decay parameters in the material catalog can be decayed without using
// hard-coded material indices.
//
// decay can depend on:
//   - a base rate constant k_0
//   - depth attenuation using gamma
//   - optional temperature sensitivity
//
// -----------------------------------------------------------------------------

#include "marsh_model/core/material_properties.hpp"
#include "marsh_model/processes/decay_model.hpp"

namespace marsh_model
{
class first_order_decay_model : public decay_model
{
public:
    first_order_decay_model() = default;

    void apply_decay(
        column_state& state,
        const forcing_step& forcing,
        const material_catalog& catalog,
        const parameter_set& parameters) const override;

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
