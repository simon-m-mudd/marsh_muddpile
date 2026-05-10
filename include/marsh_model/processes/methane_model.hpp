#pragma once

// methane_model.hpp
//
// Part of marsh_muddpile-- https://github.com/simon-m-mudd/marsh_muddpile
//
// Copyright (C) 2026 Simon M. Mudd
// Released under the GNU General Public Licence v3 (GPL-3.0)
// See LICENSE file or https://www.gnu.org/licenses/gpl-3.0.html
//
// -----------------------------------------------------------------------------
// Abstract interface for sediment methane process models.
//
// Implementations update column_state::porewater_so4() and
// column_state::porewater_ch4() each time step and return the total surface
// CH4 flux (μmol m-2 s-1) including diffusive escape and plant-mediated
// transport.
//
// Called once per time step immediately after the porewater chemistry model
// (or directly after apply_decay if no porewater chemistry model is active),
// using the decay fluxes recorded in that same step.
// -----------------------------------------------------------------------------

#include "marsh_model/core/column_state.hpp"
#include "marsh_model/core/decay_fluxes.hpp"
#include "marsh_model/core/ecohydrology_state.hpp"
#include "marsh_model/core/forcing_step.hpp"
#include "marsh_model/core/hydrology_diagnostics.hpp"
#include "marsh_model/core/material_catalog.hpp"
#include "marsh_model/core/parameter_set.hpp"
#include "marsh_model/core/site_properties.hpp"

namespace marsh_model
{
class methane_model
{
public:
    virtual ~methane_model() = default;

    // Updates porewater SO4 and CH4 in state and returns the total surface
    // CH4 flux in μmol m-2 s-1 (diffusive + plant-mediated + ebullition).
    virtual double update_methane(
        column_state& state,
        const decay_fluxes& fluxes,
        const forcing_step& forcing,
        const site_properties& site,
        const ecohydrology_state& eco_state,
        const hydrology_diagnostics& hydro,
        const material_catalog& catalog,
        const parameter_set& parameters) const = 0;
};
}
