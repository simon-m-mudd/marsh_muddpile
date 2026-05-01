#pragma once

// deposition_model.hpp
//
// Part of marsh_muddpile-- https://github.com/simon-m-mudd/marsh_muddpile
//
// Copyright (C) 2026 Simon M. Mudd
// Released under the GNU General Public Licence v3 (GPL-3.0)
// See LICENSE file or https://www.gnu.org/licenses/gpl-3.0.html
//
// -----------------------------------------------------------------------------
// this file declares the interface for surface deposition models.
//
// deposition models compute material input to the marsh surface over one model
// timestep. Surface deposition may depend on:
//
//   - current column geometry
//   - hydrodynamic forcing
//   - material properties
//   - current vegetation biomass
//
// -----------------------------------------------------------------------------

#include "marsh_model/core/column_state.hpp"
#include "marsh_model/core/forcing_step.hpp"
#include "marsh_model/core/material_catalog.hpp"
#include "marsh_model/core/parameter_set.hpp"
#include "marsh_model/processes/biomass_model.hpp"

#include <Eigen/Core>

namespace marsh_model
{
class deposition_model
{
public:
    virtual ~deposition_model() = default;

    virtual Eigen::ArrayXd compute_surface_flux(
        const column_state& state,
        const forcing_step& forcing,
        const biomass_fluxes& biomass,
        const material_catalog& catalog,
        const parameter_set& parameters) const = 0;
};
}
