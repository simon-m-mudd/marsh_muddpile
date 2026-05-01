#pragma once

// exponential_root_allocation_model.hpp
//
// Part of marsh_muddpile-- https://github.com/simon-m-mudd/marsh_muddpile
//
// Copyright (C) 2026 Simon M. Mudd
// Released under the GNU General Public Licence v3 (GPL-3.0)
// See LICENSE file or https://www.gnu.org/licenses/gpl-3.0.html
//
// -----------------------------------------------------------------------------
// this component allocates live roots and dead root inputs through the marsh
// sediment column using an exponential depth profile.
//
// the design follows the role of the older growth-index logic in the original
// marsh code, but is refactored into a modular process that returns a matrix of
// mass changes by layer and material.
//
// it does two things:
//
// 1. computes a target profile of live root biomass through the sediment column
// 2. routes belowground mortality into organic sediment pools
//
// the depth profile is:
//
//   weight(z_top, z_bottom) = exp( -z_top / gamma ) - exp( -z_bottom / gamma )
//
// where gamma is the root e-folding depth.
//
// -----------------------------------------------------------------------------

#include "marsh_model/processes/root_allocation_model.hpp"

#include <Eigen/Core>
#include <string>
#include <vector>

namespace marsh_model
{
class exponential_root_allocation_model : public root_allocation_model
{
public:
    exponential_root_allocation_model() = default;

    Eigen::ArrayXXd compute_root_mass_change(
        const column_state& state,
        const biomass_fluxes& biomass,
        const material_catalog& catalog,
        const parameter_set& parameters) const override;

private:
    Eigen::ArrayXd compute_root_profile_weights(
        const column_state& state,
        const parameter_set& parameters) const;

    int find_material_index(
        const material_catalog& catalog,
        const std::vector<std::string>& candidate_names) const;
};
}
