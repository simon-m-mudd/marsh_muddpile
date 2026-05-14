// layer_merger.cpp
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
// the current implementation preserves:
//
//   - total mass by material
//   - total thickness
//   - total pore volume
//   - representative layer age
//
// and then recomputes layer top elevations from the updated thickness profile.
//
// this stronger version does two passes:
//
//   1. if the number of layers exceeds the configured maximum, it force-merges
//      the deepest eligible adjacent pair below the protected shallow zone
//      until the target layer count is reached or no eligible merge remains
//
//   2. after the layer-count constraint is satisfied, it optionally continues
//      merging deep thin layers toward a target coarse thickness
//
// -----------------------------------------------------------------------------

#include "marsh_model/engine/layer_merger.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace marsh_model
{
namespace
{
double get_parameter_or_default(
    const parameter_set& parameters,
    const std::string& name,
    double default_value)
{
    if (parameters.has(name))
    {
        return parameters.get(name);
    }

    return default_value;
}
}

void layer_merger::merge_layers_if_needed(
    column_state& state,
    const parameter_set& parameters)
{
    const double merging_enable =
        get_parameter_or_default(parameters, "layer_merging_enable", 0.0);

    if (merging_enable <= 0.5)
    {
        return;
    }

    if (state.n_layers() <= 1)
    {
        return;
    }

    const int max_layers =
        static_cast<int>(get_parameter_or_default(parameters, "layer_merging_max_layers", 200.0));

    const double target_thickness_m =
        get_parameter_or_default(parameters, "layer_merging_target_thickness_m", 0.05);

    // -------------------------------------------------------------------------
    // pass 1:
    // force the layer count down to the configured maximum by merging the
    // deepest eligible pair below the protected shallow zone
    // -------------------------------------------------------------------------
    while (state.n_layers() > max_layers)
    {
        bool merged_any_layer = false;

        for (int lower_layer_index = 0; lower_layer_index < state.n_layers() - 1; ++lower_layer_index)
        {
            const int upper_layer_index = lower_layer_index + 1;

            if (!is_below_protected_surface_zone(state, upper_layer_index, parameters))
            {
                continue;
            }

            merge_adjacent_layers(state, lower_layer_index, upper_layer_index);
            merged_any_layer = true;
            break;
        }

        if (!merged_any_layer)
        {
            break;
        }
    }

    // -------------------------------------------------------------------------
    // pass 2:
    // optionally coarsen deep thin layers further toward the target thickness
    // -------------------------------------------------------------------------
    if (target_thickness_m > 0.0)
    {
        bool merged_any_layer = true;

        while (merged_any_layer)
        {
            merged_any_layer = false;

            for (int lower_layer_index = 0; lower_layer_index < state.n_layers() - 1; ++lower_layer_index)
            {
                const int upper_layer_index = lower_layer_index + 1;

                if (!should_merge_for_target_thickness(
                        state,
                        lower_layer_index,
                        upper_layer_index,
                        parameters))
                {
                    continue;
                }

                merge_adjacent_layers(state, lower_layer_index, upper_layer_index);
                merged_any_layer = true;
                break;
            }
        }
    }
}

bool layer_merger::is_below_protected_surface_zone(
    const column_state& state,
    int upper_layer_index,
    const parameter_set& parameters)
{
    if (upper_layer_index < 0 || upper_layer_index >= state.n_layers())
    {
        return false;
    }

    const double minimum_unmerged_depth_m =
        get_parameter_or_default(parameters, "layer_merging_minimum_unmerged_depth_m", 0.30);

    const double surface_elevation =
        state.get_surface_elevation();

    const double upper_layer_top_elevation =
        state.layer_top_elevation()(upper_layer_index);

    const double upper_layer_depth_below_surface_m =
        surface_elevation - upper_layer_top_elevation;

    return upper_layer_depth_below_surface_m >= minimum_unmerged_depth_m;
}

bool layer_merger::should_merge_for_target_thickness(
    const column_state& state,
    int lower_layer_index,
    int upper_layer_index,
    const parameter_set& parameters)
{
    if (lower_layer_index < 0 ||
        upper_layer_index < 0 ||
        lower_layer_index >= state.n_layers() ||
        upper_layer_index >= state.n_layers() ||
        upper_layer_index != lower_layer_index + 1)
    {
        return false;
    }

    if (!is_below_protected_surface_zone(state, upper_layer_index, parameters))
    {
        return false;
    }

    const double target_thickness_m =
        get_parameter_or_default(parameters, "layer_merging_target_thickness_m", 0.05);

    const double combined_thickness =
        state.layer_thickness()(lower_layer_index) +
        state.layer_thickness()(upper_layer_index);

    return combined_thickness <= target_thickness_m;
}

void layer_merger::merge_adjacent_layers(
    column_state& state,
    int lower_layer_index,
    int upper_layer_index)
{
    if (upper_layer_index != lower_layer_index + 1)
    {
        throw std::invalid_argument("merge_adjacent_layers requires adjacent indices");
    }

    const int old_n_layers = state.n_layers();
    const int n_materials = state.n_materials();

    Eigen::ArrayXXd new_mass = Eigen::ArrayXXd::Zero(old_n_layers - 1, n_materials);
    Eigen::ArrayXd new_thickness = Eigen::ArrayXd::Zero(old_n_layers - 1);
    Eigen::ArrayXd new_porosity = Eigen::ArrayXd::Zero(old_n_layers - 1);
    Eigen::ArrayXd new_top_elevation = Eigen::ArrayXd::Zero(old_n_layers - 1);
    Eigen::ArrayXd new_age = Eigen::ArrayXd::Zero(old_n_layers - 1);
    Eigen::ArrayXd new_nh4 = Eigen::ArrayXd::Zero(old_n_layers - 1);
    Eigen::ArrayXd new_so4 = Eigen::ArrayXd::Zero(old_n_layers - 1);
    Eigen::ArrayXd new_ch4 = Eigen::ArrayXd::Zero(old_n_layers - 1);

    const bool has_nh4 =
        state.porewater_nh4().size() == old_n_layers;

    const bool has_so4 =
        state.porewater_so4().size() == old_n_layers;

    const bool has_ch4 =
        state.porewater_ch4().size() == old_n_layers;

    int new_index = 0;

    for (int old_index = 0; old_index < old_n_layers; ++old_index)
    {
        if (old_index == lower_layer_index)
        {
            const double thickness_lower = state.layer_thickness()(lower_layer_index);
            const double thickness_upper = state.layer_thickness()(upper_layer_index);
            const double combined_thickness = thickness_lower + thickness_upper;

            const double pore_volume_lower =
                state.layer_porosity()(lower_layer_index) * thickness_lower;
            const double pore_volume_upper =
                state.layer_porosity()(upper_layer_index) * thickness_upper;

            const double combined_pore_volume =
                pore_volume_lower + pore_volume_upper;

            double combined_porosity = 0.0;
            if (combined_thickness > 0.0)
            {
                combined_porosity = combined_pore_volume / combined_thickness;
            }

            double combined_age = 0.0;
            if (combined_thickness > 0.0)
            {
                combined_age =
                    (state.layer_age()(lower_layer_index) * thickness_lower +
                     state.layer_age()(upper_layer_index) * thickness_upper) /
                    combined_thickness;
            }

            double combined_nh4 = 0.0;
            if (has_nh4 && combined_pore_volume > 0.0)
            {
                combined_nh4 =
                    (state.porewater_nh4()(lower_layer_index) * pore_volume_lower +
                     state.porewater_nh4()(upper_layer_index) * pore_volume_upper) /
                    combined_pore_volume;
            }

            double combined_so4 = 0.0;
            if (has_so4 && combined_pore_volume > 0.0)
            {
                combined_so4 =
                    (state.porewater_so4()(lower_layer_index) * pore_volume_lower +
                     state.porewater_so4()(upper_layer_index) * pore_volume_upper) /
                    combined_pore_volume;
            }

            double combined_ch4 = 0.0;
            if (has_ch4 && combined_pore_volume > 0.0)
            {
                combined_ch4 =
                    (state.porewater_ch4()(lower_layer_index) * pore_volume_lower +
                     state.porewater_ch4()(upper_layer_index) * pore_volume_upper) /
                    combined_pore_volume;
            }

            new_mass.row(new_index) =
                state.mass().row(lower_layer_index) +
                state.mass().row(upper_layer_index);

            new_thickness(new_index) = combined_thickness;
            new_porosity(new_index) = combined_porosity;
            new_age(new_index) = combined_age;
            new_nh4(new_index) = combined_nh4;
            new_so4(new_index) = combined_so4;
            new_ch4(new_index) = combined_ch4;

            ++new_index;
            ++old_index;
            continue;
        }

        if (old_index == upper_layer_index)
        {
            continue;
        }

        new_mass.row(new_index) = state.mass().row(old_index);
        new_thickness(new_index) = state.layer_thickness()(old_index);
        new_porosity(new_index) = state.layer_porosity()(old_index);
        new_age(new_index) = state.layer_age()(old_index);
        if (has_nh4)
        {
            new_nh4(new_index) = state.porewater_nh4()(old_index);
        }
        if (has_so4)
        {
            new_so4(new_index) = state.porewater_so4()(old_index);
        }
        if (has_ch4)
        {
            new_ch4(new_index) = state.porewater_ch4()(old_index);
        }

        ++new_index;
    }

    const double column_base_elevation =
        state.layer_top_elevation()(0) - state.layer_thickness()(0);

    state.resize(old_n_layers - 1, n_materials);
    state.mass() = new_mass;
    state.layer_thickness() = new_thickness;
    state.layer_porosity() = new_porosity;
    state.layer_age() = new_age;
    state.layer_top_elevation() = new_top_elevation;
    state.porewater_nh4() = new_nh4;
    state.porewater_so4() = new_so4;
    state.porewater_ch4() = new_ch4;

    recompute_top_elevations(state, column_base_elevation);
}

void layer_merger::recompute_top_elevations(
    column_state& state,
    double column_base_elevation)
{
    double cumulative_top_elevation = column_base_elevation;

    for (int layer_index = 0; layer_index < state.n_layers(); ++layer_index)
    {
        cumulative_top_elevation += state.layer_thickness()(layer_index);
        state.layer_top_elevation()(layer_index) = cumulative_top_elevation;
    }
}
}
