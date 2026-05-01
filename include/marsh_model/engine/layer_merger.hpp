#pragma once

// layer_merger.hpp
//
// Part of marsh_muddpile-- https://github.com/simon-m-mudd/marsh_muddpile
//
// Copyright (C) 2026 Simon M. Mudd
// Released under the GNU General Public Licence v3 (GPL-3.0)
// See LICENSE file or https://www.gnu.org/licenses/gpl-3.0.html
//
// -----------------------------------------------------------------------------
// this component coarsens the marsh stratigraphic column by merging deeper
// adjacent layers when the number of layers becomes too large.
//
// the purpose is to reduce computational cost during long forward runs while
// preserving high resolution near the marsh surface, where deposition,
// root input, decay, and recent stratigraphy are most dynamic.
//
// this stronger version can:
//
//   1. force layer count down to a maximum by merging the deepest eligible
//      layers below a protected shallow zone
//   2. optionally continue coarsening deep thin layers toward a target deep
//      layer thickness
//
// -----------------------------------------------------------------------------

#include "marsh_model/core/column_state.hpp"
#include "marsh_model/core/parameter_set.hpp"

namespace marsh_model
{
class layer_merger
{
public:
    static void merge_layers_if_needed(
        column_state& state,
        const parameter_set& parameters);

private:
    static bool is_below_protected_surface_zone(
        const column_state& state,
        int upper_layer_index,
        const parameter_set& parameters);

    static bool should_merge_for_target_thickness(
        const column_state& state,
        int lower_layer_index,
        int upper_layer_index,
        const parameter_set& parameters);

    static void merge_adjacent_layers(
        column_state& state,
        int lower_layer_index,
        int upper_layer_index);

    static void recompute_top_elevations(
        column_state& state);
};
}
