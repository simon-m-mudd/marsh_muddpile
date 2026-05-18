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
// Hierarchical layer merging: coarsens the stratigraphic column incrementally
// as layers are buried, using three depth thresholds.
//
// Each layer carries a merge_generation counter (0 = never merged).
// A pair of adjacent layers can merge at level g only when:
//   - both layers have merge_generation == g, AND
//   - the top of the shallower layer is at least depth_threshold[g] below
//     the current surface.
// The resulting layer gets merge_generation = g + 1, preventing it from
// being re-merged at the same level.
//
// Merging happens in a single forward scan per generation per time step, so
// at most one merge event per pair per call — no batch collapses.
//
// Parameters (all optional, with defaults):
//   layer_merging_enable        default 0  (set to 1 to enable)
//   layer_merging_depth_1_m     default 0.5   first merge threshold (m)
//   layer_merging_depth_2_m     default 1.0   second merge threshold (m)
//   layer_merging_depth_3_m     default 2.0   third merge threshold (m)
//   layer_merging_max_per_pass  default 5     max merges per generation per timestep
//                               Rate-limits merges so a large cohort of same-
//                               generation layers crossing a threshold simultaneously
//                               does not produce a burst of compaction-driven
//                               surface elevation changes (sawtooth artifact).
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
    static void merge_one_pass(
        column_state& state,
        int generation,
        double depth_threshold_m,
        int max_merges);

    static void merge_adjacent_layers(
        column_state& state,
        int lower_layer_index,
        int upper_layer_index,
        int result_generation);

    static void recompute_top_elevations(
        column_state& state,
        double column_base_elevation);
};
}
