// mixing_compaction_model.hpp
//
// Part of marsh_muddpile -- https://github.com/simon-m-mudd/marsh_muddpile
//
// Copyright (C) 2026 Simon M. Mudd
// Released under the GNU General Public Licence v3 (GPL-3.0)
// See LICENSE file or https://www.gnu.org/licenses/gpl-3.0.html
//
// -----------------------------------------------------------------------------
// Mixing-law compaction model.
//
// Computes the target dry bulk density (DBD) for each layer from its LOI and
// depth below surface using a depth-dependent form of the Morris et al. (2016)
// mixing model:
//
//   DBD = 1 / ( LOI / k1(d) + (1 - LOI) / k2(d) )       [g cm^-3]
//
// where d is depth below the current surface (m), and k1, k2 are the organic
// and mineral end-member densities fitted as second-order polynomials:
//
//   k1(d) = k1_a2 * d^2 + k1_a1 * d + k1_a0
//   k2(d) = k2_a2 * d^2 + k2_a1 * d + k2_a0
//
// Default polynomial coefficients are calibrated by OLS through depth-binned
// CCN synthesis data (all countries; six 20 cm bins 0–150 cm; n=85,122 layers):
//
//   k1: a0=0.092574  a1=+0.035029  a2=0.0  (R^2=0.72)
//   k2: a0=1.552584  a1=-0.375221  a2=0.0  (R^2=0.80)
//
// The quadratic term a2 defaults to 0 (linear); set non-zero in the YAML for
// a quadratic fit. Values are clamped at max_depth (default 1.5 m).
//
// Layer thickness is then back-calculated from the target DBD and the total
// dry solid mass, and porosity is updated accordingly.
//
// The model can be used wherever two_stage_compaction is used; it is simpler
// (no effective-stress overburden calculation) and calibrated directly against
// global marsh-core data.
// -----------------------------------------------------------------------------

#pragma once

#include "marsh_model/processes/compaction_model.hpp"

namespace marsh_model
{

class mixing_compaction_model : public compaction_model
{
public:
    mixing_compaction_model() = default;

    void update_compaction(
        column_state& state,
        const forcing_step& forcing,
        const material_catalog& catalog,
        const parameter_set& parameters) const override;

    // Equilibrate the initial column to a target surface elevation.
    //
    // Applies one pass of the mixing-law DBD formula to bring each layer to its
    // equilibrium thickness (keeping the column base fixed), then shifts the
    // entire column so the surface lands exactly at target_surface_elevation_m.
    //
    // Call this once during initialization when the YAML specifies
    // initial_state.equilibrate_surface_m.  Without it, columns whose
    // specified porosity is inconsistent with the mixing model will jump to an
    // unintended elevation on the first compaction step.
    void equilibrate_initial_column(
        column_state& state,
        double target_surface_elevation_m,
        const material_catalog& catalog,
        const parameter_set& parameters) const;

};

} // namespace marsh_model
