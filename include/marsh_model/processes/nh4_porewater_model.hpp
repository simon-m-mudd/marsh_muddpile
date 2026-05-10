#pragma once

// nh4_porewater_model.hpp
//
// Part of marsh_muddpile-- https://github.com/simon-m-mudd/marsh_muddpile
//
// Copyright (C) 2026 Simon M. Mudd
// Released under the GNU General Public Licence v3 (GPL-3.0)
// See LICENSE file or https://www.gnu.org/licenses/gpl-3.0.html
//
// -----------------------------------------------------------------------------
// Porewater ammonium (NH4+) model for the marsh sediment column.
//
// Each time step the concentration in each layer is updated by:
//
//   1. Production from organic-matter decomposition
//      NH4 is released during ammonification at a rate proportional to the
//      organic-C decomposed and the molar C:N ratio of the litter.
//
//   2. Tidal flushing toward creek NH4
//      During inundation, tidal exchange exchanges porewater with the
//      overlying water column.  The exchange is exponentially attenuated
//      with depth below the sediment surface.
//
//   3. Fickian inter-layer diffusion
//      Concentration gradients drive porewater NH4 upward.
//      Explicit Euler is used; D_eff should be ≤ 1e-5 m2/d for stability
//      with the default 30-day time step and 5-cm layers.
//
//   4. Plant root uptake (Michaelis-Menten)
//      Live root mass in each layer takes up NH4 proportional to
//      root density and concentration / (Km + concentration).
//
// See nh4_porewater_model.cpp for the full formulation and parameters.
// -----------------------------------------------------------------------------

#include "marsh_model/processes/porewater_chemistry_model.hpp"

namespace marsh_model
{
class nh4_porewater_model : public porewater_chemistry_model
{
public:
    nh4_porewater_model() = default;

    void update_porewater(
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
