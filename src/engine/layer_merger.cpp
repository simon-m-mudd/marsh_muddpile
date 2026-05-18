// layer_merger.cpp
//
// Part of marsh_muddpile-- https://github.com/simon-m-mudd/marsh_muddpile
//
// Copyright (C) 2026 Simon M. Mudd
// Released under the GNU General Public Licence v3 (GPL-3.0)
// See LICENSE file or https://www.gnu.org/licenses/gpl-3.0.html
//
// -----------------------------------------------------------------------------
// Hierarchical layer merging.
//
// merge_layers_if_needed runs three single-pass scans per call (one per
// depth threshold).  Each scan walks from the bottom of the column upward.
// When it finds an adjacent pair where both layers have merge_generation == g
// AND the top of the shallower layer is at least depth_threshold[g] below
// the surface, it merges the pair into a single layer with
// merge_generation == g+1, then advances past the merged result.
//
// Because each scan is a single forward pass (not a while loop), at most
// floor(N/2) merges occur per generation per time step — merging is spread
// across many time steps rather than firing as a single batch event.
// -----------------------------------------------------------------------------

#include "marsh_model/engine/layer_merger.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace marsh_model
{
namespace
{
double get_param(const parameter_set& p, const std::string& name, double def)
{
    return p.has(name) ? p.get(name) : def;
}
} // anonymous namespace

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

void layer_merger::merge_layers_if_needed(
    column_state& state,
    const parameter_set& parameters)
{
    const double enable =
        get_param(parameters, "layer_merging_enable", 0.0);
    if (enable <= 0.5)
        return;

    if (state.n_layers() <= 1)
        return;

    const double d1 = get_param(parameters, "layer_merging_depth_1_m", 0.5);
    const double d2 = get_param(parameters, "layer_merging_depth_2_m", 1.0);
    const double d3 = get_param(parameters, "layer_merging_depth_3_m", 2.0);

    // Cap merges per generation per timestep so newly-eligible layers enter
    // the merge queue gradually rather than all at once.  Without this limit
    // a large cohort of same-generation layers can cross a depth threshold
    // simultaneously and cause a multi-year burst of compaction-driven
    // surface elevation changes.
    const int max_per_pass = static_cast<int>(
        get_param(parameters, "layer_merging_max_per_pass", 5.0));

    merge_one_pass(state, 0, d1, max_per_pass);
    merge_one_pass(state, 1, d2, max_per_pass);
    merge_one_pass(state, 2, d3, max_per_pass);
}

// ---------------------------------------------------------------------------
// Single-pass merge for one generation level
// ---------------------------------------------------------------------------

void layer_merger::merge_one_pass(
    column_state& state,
    int generation,
    double depth_threshold_m,
    int max_merges)
{
    const double surf = state.get_surface_elevation();

    int i = 0;
    int n_merged = 0;
    while (i < state.n_layers() - 1 && n_merged < max_merges)
    {
        const int lower = i;
        const int upper = i + 1;

        // Depth of the top of the shallower (upper) layer.
        const double depth_upper_top =
            surf - state.layer_top_elevation()(upper);

        const bool both_deep_enough = depth_upper_top >= depth_threshold_m;
        const bool lower_gen_match  = state.merge_generation()(lower) == generation;
        const bool upper_gen_match  = state.merge_generation()(upper) == generation;

        if (both_deep_enough && lower_gen_match && upper_gen_match)
        {
            merge_adjacent_layers(state, lower, upper, generation + 1);
            // Advance past the merged result; it cannot be re-merged at this
            // generation level (its merge_generation is now generation+1).
            i = lower + 1;
            ++n_merged;
        }
        else
        {
            ++i;
        }
    }
}

// ---------------------------------------------------------------------------
// Merge a pair of adjacent layers into one
// ---------------------------------------------------------------------------

void layer_merger::merge_adjacent_layers(
    column_state& state,
    int lower_layer_index,
    int upper_layer_index,
    int result_generation)
{
    if (upper_layer_index != lower_layer_index + 1)
        throw std::invalid_argument("merge_adjacent_layers requires adjacent indices");

    const int old_n   = state.n_layers();
    const int new_n   = old_n - 1;
    const int n_mats  = state.n_materials();

    Eigen::ArrayXXd new_mass          = Eigen::ArrayXXd::Zero(new_n, n_mats);
    Eigen::ArrayXd  new_thickness     = Eigen::ArrayXd::Zero(new_n);
    Eigen::ArrayXd  new_porosity      = Eigen::ArrayXd::Zero(new_n);
    Eigen::ArrayXd  new_top_elevation = Eigen::ArrayXd::Zero(new_n);
    Eigen::ArrayXd  new_age           = Eigen::ArrayXd::Zero(new_n);
    Eigen::ArrayXd  new_nh4           = Eigen::ArrayXd::Zero(new_n);
    Eigen::ArrayXd  new_so4           = Eigen::ArrayXd::Zero(new_n);
    Eigen::ArrayXd  new_ch4           = Eigen::ArrayXd::Zero(new_n);
    Eigen::ArrayXi  new_gen           = Eigen::ArrayXi::Zero(new_n);

    const bool has_nh4 = state.porewater_nh4().size() == old_n;
    const bool has_so4 = state.porewater_so4().size() == old_n;
    const bool has_ch4 = state.porewater_ch4().size() == old_n;

    int ni = 0;
    for (int oi = 0; oi < old_n; ++oi)
    {
        if (oi == lower_layer_index)
        {
            const double t_lo = state.layer_thickness()(lower_layer_index);
            const double t_up = state.layer_thickness()(upper_layer_index);
            const double t_combined = t_lo + t_up;

            const double pv_lo = state.layer_porosity()(lower_layer_index) * t_lo;
            const double pv_up = state.layer_porosity()(upper_layer_index) * t_up;
            const double pv_combined = pv_lo + pv_up;

            new_mass.row(ni) =
                state.mass().row(lower_layer_index) +
                state.mass().row(upper_layer_index);

            new_thickness(ni) = t_combined;
            new_porosity(ni)  = (t_combined > 0.0) ? pv_combined / t_combined : 0.0;
            new_age(ni)       = (t_combined > 0.0)
                ? (state.layer_age()(lower_layer_index) * t_lo +
                   state.layer_age()(upper_layer_index) * t_up) / t_combined
                : 0.0;
            new_gen(ni) = result_generation;

            if (has_nh4)
                new_nh4(ni) = (pv_combined > 0.0)
                    ? (state.porewater_nh4()(lower_layer_index) * pv_lo +
                       state.porewater_nh4()(upper_layer_index) * pv_up) / pv_combined
                    : 0.0;
            if (has_so4)
                new_so4(ni) = (pv_combined > 0.0)
                    ? (state.porewater_so4()(lower_layer_index) * pv_lo +
                       state.porewater_so4()(upper_layer_index) * pv_up) / pv_combined
                    : 0.0;
            if (has_ch4)
                new_ch4(ni) = (pv_combined > 0.0)
                    ? (state.porewater_ch4()(lower_layer_index) * pv_lo +
                       state.porewater_ch4()(upper_layer_index) * pv_up) / pv_combined
                    : 0.0;

            ++ni;
            ++oi; // skip upper_layer_index
            continue;
        }

        new_mass.row(ni)  = state.mass().row(oi);
        new_thickness(ni) = state.layer_thickness()(oi);
        new_porosity(ni)  = state.layer_porosity()(oi);
        new_age(ni)       = state.layer_age()(oi);
        new_gen(ni)       = state.merge_generation()(oi);
        if (has_nh4) new_nh4(ni) = state.porewater_nh4()(oi);
        if (has_so4) new_so4(ni) = state.porewater_so4()(oi);
        if (has_ch4) new_ch4(ni) = state.porewater_ch4()(oi);
        ++ni;
    }

    const double base_elev =
        state.layer_top_elevation()(0) - state.layer_thickness()(0);

    state.resize(new_n, n_mats);
    state.mass()             = new_mass;
    state.layer_thickness()  = new_thickness;
    state.layer_porosity()   = new_porosity;
    state.layer_age()        = new_age;
    state.merge_generation() = new_gen;
    if (has_nh4) state.porewater_nh4() = new_nh4;
    if (has_so4) state.porewater_so4() = new_so4;
    if (has_ch4) state.porewater_ch4() = new_ch4;

    recompute_top_elevations(state, base_elev);
}

// ---------------------------------------------------------------------------
// Rebuild top elevations from the fixed column base
// ---------------------------------------------------------------------------

void layer_merger::recompute_top_elevations(
    column_state& state,
    double column_base_elevation)
{
    double running = column_base_elevation;
    for (int i = 0; i < state.n_layers(); ++i)
    {
        running += state.layer_thickness()(i);
        state.layer_top_elevation()(i) = running;
    }
}

} // namespace marsh_model
