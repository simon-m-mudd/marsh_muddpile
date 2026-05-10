#pragma once

// porewater_chemistry_model.hpp
//
// Part of marsh_muddpile-- https://github.com/simon-m-mudd/marsh_muddpile
//
// Copyright (C) 2026 Simon M. Mudd
// Released under the GNU General Public Licence v3 (GPL-3.0)
// See LICENSE file or https://www.gnu.org/licenses/gpl-3.0.html
//
// -----------------------------------------------------------------------------
// Abstract interface for porewater chemistry process models.
//
// Implementations receive the per-layer decay fluxes computed by the decay
// model and update column_state::porewater_nh4() (and any other porewater
// chemistry fields added in future).
//
// Called once per time step immediately after apply_decay, using the decay
// fluxes recorded in that same step.
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
class porewater_chemistry_model
{
public:
    virtual ~porewater_chemistry_model() = default;

    virtual void update_porewater(
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
