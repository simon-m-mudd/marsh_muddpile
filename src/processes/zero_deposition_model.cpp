// zero_deposition_model.cpp
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
// -----------------------------------------------------------------------------

#include "marsh_model/processes/zero_deposition_model.hpp"

namespace marsh_model
{
Eigen::ArrayXd zero_deposition_model::compute_surface_flux(
    const column_state&,
    const forcing_step&,
    const biomass_fluxes&,
    const material_catalog& catalog,
    const parameter_set&) const
{
    return Eigen::ArrayXd::Zero(catalog.size());
}
}
