#pragma once

// zero_deposition_model.hpp
//
// Part of marsh_muddpile-- https://github.com/simon-m-mudd/marsh_muddpile
//
// Copyright (C) 2026 Simon M. Mudd
// Released under the GNU General Public Licence v3 (GPL-3.0)
// See LICENSE file or https://www.gnu.org/licenses/gpl-3.0.html
//
// -----------------------------------------------------------------------------
// this component implements a no-op surface deposition model.
//
// it always returns zero deposited mass for every material.
//
// -----------------------------------------------------------------------------

#include "marsh_model/processes/deposition_model.hpp"

namespace marsh_model
{
class zero_deposition_model : public deposition_model
{
public:
    Eigen::ArrayXd compute_surface_flux(
        const column_state& state,
        const forcing_step& forcing,
        const biomass_fluxes& biomass,
        const material_catalog& catalog,
        const parameter_set& parameters) const override;
};
}
