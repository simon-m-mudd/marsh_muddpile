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
// Default polynomial coefficients are calibrated by least-squares fit through
// depth-binned CCN synthesis data (all countries; 91,780 observations):
//
//   k1: a0=0.105944  a1=-0.012864  a2=0.027568    (R^2=0.852)
//   k2: a0=1.399719  a1= 0.170120  a2=-0.313556   (R^2=0.949)
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

private:
    // Compute LOI (organic mass fraction) for one layer.
    double compute_loi(
        const column_state& state,
        int layer_index,
        const material_catalog& catalog) const;

    // Evaluate depth-dependent end-member densities [g cm^-3].
    double k1_at_depth(double depth_m, const parameter_set& parameters) const;
    double k2_at_depth(double depth_m, const parameter_set& parameters) const;
};

} // namespace marsh_model
