// two_stage_compaction_model.cpp
//
// Part of marsh_muddpile-- https://github.com/simon-m-mudd/marsh_muddpile
//
// Copyright (C) 2026 Simon M. Mudd
// Released under the GNU General Public Licence v3 (GPL-3.0)
// See LICENSE file or https://www.gnu.org/licenses/gpl-3.0.html
//
// -----------------------------------------------------------------------------
// this component implements a two stage compaction model derived from Brain et al 2012
//
// -----------------------------------------------------------------------------

#pragma once

#include "marsh_model/processes/compaction_model.hpp"

#include <Eigen/Core>
#include <vector>

namespace marsh_model
{
/*
 * Brain et al. (2012) show that low-energy intertidal and marsh sediments
 * are better represented by a two-stage compression model than by a single
 * compression curve.
 *
 * The key conceptual ingredients used here are:
 *   - a reference void ratio e_1
 *   - a recompression index c_r
 *   - a virgin compression index c_c
 *   - a yield stress sigma_y
 *
 * In Brain et al. (2012), e_1 / c_r / c_c are strongly related to LOI
 * (loss on ignition), while yield stress is more site-specific and linked
 * to marsh elevation / desiccation state / depositional setting.
 *
 * This class follows that structure, but also exposes a grain-size-driven
 * extension point because your new model design wants both carbon content
 * and particle-size effects to influence compressibility.
 */

struct layer_composition_summary
{
    double solid_mass_kg_m2 = 0.0;
    double solid_volume_m3_m2 = 0.0;
    double bulk_solid_density_kg_m3 = 2650.0;

    double organic_mass_fraction = 0.0;
    double estimated_loi_percent = 0.0;

    /*
     * Grain size extension:
     * Brain et al. (2012) focus most strongly on LOI-based controls, but
     * particle-size effects are physically important and can be added here.
     *
     * This draft uses a weighted mean mineral grain size, based on the
     * mineral components present in the layer. The implementation uses the
     * material settling diameter as a proxy if available.
     */
    double mean_mineral_grain_size_m = 0.0;
};

struct layer_compression_properties
{
    /*
     * e_1: reference void ratio at 1 kPa, following the notation used in
     * Brain et al. (2012).
     */
    double e_1 = 1.0;

    /*
     * c_r: recompression index
     * c_c: virgin compression index
     * sigma_y_kpa: yield stress in kPa
     */
    double c_r = 0.05;
    double c_c = 0.50;
    double sigma_y_kpa = 3.0;
};

class two_stage_compaction_model : public compaction_model
{
public:
    two_stage_compaction_model() = default;

    void update_compaction(
        column_state& state,
        const forcing_step& forcing,
        const material_catalog& catalog,
        const parameter_set& parameters) const override;

private:
    layer_composition_summary summarize_layer_composition(
        const column_state& state,
        int layer_index,
        const material_catalog& catalog,
        const parameter_set& parameters) const;

    layer_compression_properties estimate_compression_properties(
        const layer_composition_summary& composition,
        const parameter_set& parameters) const;

    Eigen::ArrayXd compute_buoyant_stress_increment_kpa(
        const column_state& state,
        const material_catalog& catalog,
        const parameter_set& parameters) const;

    Eigen::ArrayXd compute_representative_effective_stress_kpa(
        const Eigen::ArrayXd& buoyant_stress_increment_kpa,
        const parameter_set& parameters) const;

    double compute_void_ratio(
        double effective_stress_kpa,
        const layer_compression_properties& properties,
        const parameter_set& parameters) const;

    double compute_layer_thickness_m(
        double solid_volume_m3_m2,
        double void_ratio) const;

    double compute_porosity(
        double void_ratio) const;
};
}
