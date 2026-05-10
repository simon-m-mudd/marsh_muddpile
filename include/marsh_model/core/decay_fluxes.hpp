#pragma once

// decay_fluxes.hpp
//
// Part of marsh_muddpile-- https://github.com/simon-m-mudd/marsh_muddpile
//
// Copyright (C) 2026 Simon M. Mudd
// Released under the GNU General Public Licence v3 (GPL-3.0)
// See LICENSE file or https://www.gnu.org/licenses/gpl-3.0.html
//
// -----------------------------------------------------------------------------
// Output struct optionally filled by decay_model::apply_decay so that
// downstream porewater chemistry models can consume the per-layer,
// per-material mass that was decomposed in each time step.
//
// Filled only when decay_context::out_fluxes is non-null; otherwise the
// decay model writes nothing here and the porewater model is not called.
// -----------------------------------------------------------------------------

#include <Eigen/Core>

namespace marsh_model
{
struct decay_fluxes
{
    // Mass decomposed this time step, kg m-2 per layer per material.
    // Dimensions: rows = layers (index 0 = deepest), cols = materials.
    // Values are non-negative.
    Eigen::ArrayXXd mass_decomposed_kg_m2;
};
}
