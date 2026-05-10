#pragma once

// sulfate_methane_model.hpp
//
// Part of marsh_muddpile-- https://github.com/simon-m-mudd/marsh_muddpile
//
// Copyright (C) 2026 Simon M. Mudd
// Released under the GNU General Public Licence v3 (GPL-3.0)
// See LICENSE file or https://www.gnu.org/licenses/gpl-3.0.html
//
// -----------------------------------------------------------------------------
// Coupled porewater sulfate and methane model for the marsh sediment column.
//
// Organic-matter decomposition drives microbial sulfate reduction (SRB) and
// methanogenesis in competition.  SRB outcompete methanogens when sulfate is
// available (Michaelis-Menten competition); methanogenesis dominates once the
// sulfate pool is depleted.  Tidal inundation replenishes porewater sulfate
// from creek water.  Surface CH4 escapes by Fickian diffusion and through
// plant aerenchyma.
//
// See sulfate_methane_model.cpp for the full formulation and parameters.
// -----------------------------------------------------------------------------

#include "marsh_model/processes/methane_model.hpp"

namespace marsh_model
{
class sulfate_methane_model : public methane_model
{
public:
    sulfate_methane_model() = default;

    double update_methane(
        column_state& state,
        const decay_fluxes& fluxes,
        const forcing_step& forcing,
        const site_properties& site,
        const ecohydrology_state& eco_state,
        const hydrology_diagnostics& hydro,
        const material_catalog& catalog,
        const parameter_set& parameters) const override;
};
}
