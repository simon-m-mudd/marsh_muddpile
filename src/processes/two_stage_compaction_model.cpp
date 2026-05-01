// two_stage_compaction_model.cpp
//
// Part of marsh_muddpile-- https://github.com/simon-m-mudd/marsh_muddpile
//
// Copyright (C) 2026 Simon M. Mudd
// Released under the GNU General Public Licence v3 (GPL-3.0)
// See LICENSE file or https://www.gnu.org/licenses/gpl-3.0.html
//
// -----------------------------------------------------------------------------
// this component implements a two stage compaction model derived from
// Brain et al. (2012).
//
// this version includes safe OpenMP parallelisation over layer-wise work where
// each layer can be processed independently.
//
// -----------------------------------------------------------------------------

#include "marsh_model/processes/two_stage_compaction_model.hpp"

#include "marsh_model/core/material_properties.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifdef marsh_muddpile_use_openmp
#include <omp.h>
#endif

namespace marsh_model
{
namespace
{
double get_parameter_or_default(
    const parameter_set& parameters,
    const std::string& name,
    double default_value)
{
    if (parameters.has(name))
    {
        return parameters.get(name);
    }

    return default_value;
}

bool is_organic_material(const material_properties& material)
{
    return material.category == material_category::organic_labile ||
           material.category == material_category::organic_refractory ||
           material.category == material_category::live_root;
}

bool is_mineral_material(const material_properties& material)
{
    return material.category == material_category::mineral;
}
}

// This is the main compaction entry point.
//
// The overall flow follows the structure implied by Brain et al. (2012):
//
//   1. summarize the composition of each layer
//   2. estimate compression parameters from composition
//   3. calculate an effective stress profile for the full column
//   4. calculate void ratio using a two-stage compression law
//   5. convert void ratio to porosity and thickness
//   6. update layer top elevations
//
// Important note:
// Brain et al. (2012) emphasize that compaction is a column-scale process.
// Therefore this method treats the full stratigraphic stack together rather
// than applying an isolated single-layer rule.
void two_stage_compaction_model::update_compaction(
    column_state& state,
    const forcing_step&,
    const material_catalog& catalog,
    const parameter_set& parameters) const
{
    const int n_layers = state.n_layers();

    if (n_layers == 0)
    {
        return;
    }

    Eigen::ArrayXd buoyant_stress_increment_kpa =
        compute_buoyant_stress_increment_kpa(state, catalog, parameters);

    Eigen::ArrayXd effective_stress_kpa =
        compute_representative_effective_stress_kpa(
            buoyant_stress_increment_kpa,
            parameters);

    Eigen::ArrayXd new_layer_thickness = Eigen::ArrayXd::Zero(n_layers);
    Eigen::ArrayXd new_layer_porosity = Eigen::ArrayXd::Zero(n_layers);
    Eigen::ArrayXd new_layer_top_elevation = Eigen::ArrayXd::Zero(n_layers);

    // This loop is safe to parallelise because each layer writes only to its
    // own entry in the output arrays.
#ifdef marsh_muddpile_use_openmp
#pragma omp parallel for schedule(static) if(n_layers > 32)
#endif
    for (int layer_index = 0; layer_index < n_layers; ++layer_index)
    {
        const layer_composition_summary composition =
            summarize_layer_composition(state, layer_index, catalog, parameters);

        const layer_compression_properties properties =
            estimate_compression_properties(composition, parameters);

        const double void_ratio =
            compute_void_ratio(
                effective_stress_kpa(layer_index),
                properties,
                parameters);

        const double layer_thickness_m =
            compute_layer_thickness_m(
                composition.solid_volume_m3_m2,
                void_ratio);

        const double porosity =
            compute_porosity(void_ratio);

        new_layer_thickness(layer_index) = layer_thickness_m;
        new_layer_porosity(layer_index) = porosity;
    }

    // Layer ordering assumption:
    //   - layer 0 is the deepest / oldest layer
    //   - layer n_layers - 1 is the surface / youngest layer
    //
    // This matches the scaffold behaviour where append_surface_layer()
    // appends the newest layer at the end.
    //
    // This cumulative sum remains serial because each layer top elevation
    // depends on the thickness of all deeper layers.
    double cumulative_top_elevation = 0.0;
    for (int layer_index = 0; layer_index < n_layers; ++layer_index)
    {
        cumulative_top_elevation += new_layer_thickness(layer_index);
        new_layer_top_elevation(layer_index) = cumulative_top_elevation;
    }

    state.set_layer_geometry(
        new_layer_thickness,
        new_layer_porosity,
        new_layer_top_elevation);
}

// Summarize the solid composition of one layer.
//
// Brain et al. (2012) show that LOI is a strong predictor of e_1, c_r and c_c.
// In this draft, LOI is estimated from the modelled layer composition, using
// organic mass fraction as a proxy. This is a practical modelling choice,
// because the state variable available in the forward model is mass by material.
//
// The mapping from organic mass fraction to LOI should be calibrated to your
// own sediment data. The default mapping below is intentionally simple.
layer_composition_summary two_stage_compaction_model::summarize_layer_composition(
    const column_state& state,
    int layer_index,
    const material_catalog& catalog,
    const parameter_set& parameters) const
{
    if (layer_index < 0 || layer_index >= state.n_layers())
    {
        throw std::out_of_range("layer index out of range in summarize_layer_composition");
    }

    const auto& mass = state.mass();

    const double loi_intercept_percent =
        get_parameter_or_default(parameters, "compaction_loi_intercept_percent", 0.0);

    const double loi_slope_percent_per_organic_fraction =
        get_parameter_or_default(
            parameters,
            "compaction_loi_slope_percent_per_organic_fraction",
            100.0);

    const double default_grain_size_m =
        get_parameter_or_default(parameters, "compaction_default_grain_size_m", 2.0e-5);

    layer_composition_summary summary;

    double organic_mass_kg_m2 = 0.0;
    double mineral_mass_kg_m2 = 0.0;
    double weighted_grain_size_sum = 0.0;

    for (int material_index = 0; material_index < catalog.size(); ++material_index)
    {
        const material_properties& material = catalog.get_material(material_index);
        const double mass_kg_m2 = mass(layer_index, material_index);

        if (mass_kg_m2 <= 0.0)
        {
            continue;
        }

        if (material.density <= 0.0)
        {
            throw std::runtime_error("material density must be positive in compaction model");
        }

        summary.solid_mass_kg_m2 += mass_kg_m2;
        summary.solid_volume_m3_m2 += mass_kg_m2 / material.density;

        if (is_organic_material(material))
        {
            organic_mass_kg_m2 += mass_kg_m2;
        }

        if (is_mineral_material(material))
        {
            mineral_mass_kg_m2 += mass_kg_m2;

            if (material.settling.has_value())
            {
                weighted_grain_size_sum +=
                    mass_kg_m2 * material.settling->diameter;
            }
        }
    }

    if (summary.solid_mass_kg_m2 > 0.0)
    {
        summary.organic_mass_fraction =
            organic_mass_kg_m2 / summary.solid_mass_kg_m2;
    }
    else
    {
        summary.organic_mass_fraction = 0.0;
    }

    summary.estimated_loi_percent =
        loi_intercept_percent +
        loi_slope_percent_per_organic_fraction * summary.organic_mass_fraction;

    summary.estimated_loi_percent =
        std::max(0.0, summary.estimated_loi_percent);

    if (mineral_mass_kg_m2 > 0.0 && weighted_grain_size_sum > 0.0)
    {
        summary.mean_mineral_grain_size_m =
            weighted_grain_size_sum / mineral_mass_kg_m2;
    }
    else
    {
        summary.mean_mineral_grain_size_m = default_grain_size_m;
    }

    if (summary.solid_volume_m3_m2 > 0.0)
    {
        summary.bulk_solid_density_kg_m3 =
            summary.solid_mass_kg_m2 / summary.solid_volume_m3_m2;
    }

    return summary;
}

// Estimate layer compression properties from composition.
//
// Brain et al. (2012):
//   - e_1, c_r and c_c are strongly related to LOI.
//   - yield stress sigma_y is more site-specific and not controlled
//     by LOI alone.
//
// Therefore:
//   - this draft treats e_1 / c_r / c_c primarily as LOI-driven
//   - sigma_y is also composition-sensitive here, but should be viewed
//     as a provisional calibration relation, not a final physical law
//
// Grain-size terms have been added as an extension beyond the simplest
// LOI-only implementation, to support your next-generation model design.
// If you want a strict Brain-et-al.-style implementation first, set all
// grain-size slopes to zero.
layer_compression_properties two_stage_compaction_model::estimate_compression_properties(
    const layer_composition_summary& composition,
    const parameter_set& parameters) const
{
    const double grain_size_um = composition.mean_mineral_grain_size_m * 1.0e6;
    const double loi = composition.estimated_loi_percent;

    const double e_1_intercept =
        get_parameter_or_default(parameters, "compaction_e_1_intercept", 0.50);
    const double e_1_loi_slope =
        get_parameter_or_default(parameters, "compaction_e_1_loi_slope", 0.040);
    const double e_1_grain_size_slope =
        get_parameter_or_default(parameters, "compaction_e_1_grain_size_slope", 0.0);

    const double c_r_intercept =
        get_parameter_or_default(parameters, "compaction_c_r_intercept", 0.005);
    const double c_r_loi_slope =
        get_parameter_or_default(parameters, "compaction_c_r_loi_slope", 0.0015);
    const double c_r_grain_size_slope =
        get_parameter_or_default(parameters, "compaction_c_r_grain_size_slope", 0.0);

    const double c_c_intercept =
        get_parameter_or_default(parameters, "compaction_c_c_intercept", 0.10);
    const double c_c_loi_slope =
        get_parameter_or_default(parameters, "compaction_c_c_loi_slope", 0.020);
    const double c_c_grain_size_slope =
        get_parameter_or_default(parameters, "compaction_c_c_grain_size_slope", 0.0);

    const double sigma_y_intercept_kpa =
        get_parameter_or_default(parameters, "compaction_sigma_y_intercept_kpa", 12.0);
    const double sigma_y_loi_slope =
        get_parameter_or_default(parameters, "compaction_sigma_y_loi_slope", -0.15);
    const double sigma_y_grain_size_slope =
        get_parameter_or_default(parameters, "compaction_sigma_y_grain_size_slope", 0.0);

    const double min_e_1 =
        get_parameter_or_default(parameters, "compaction_min_e_1", 0.10);
    const double min_c_r =
        get_parameter_or_default(parameters, "compaction_min_c_r", 1.0e-4);
    const double min_c_c =
        get_parameter_or_default(parameters, "compaction_min_c_c", 1.0e-3);
    const double min_sigma_y_kpa =
        get_parameter_or_default(parameters, "compaction_min_sigma_y_kpa", 3.0);

    layer_compression_properties properties;

    properties.e_1 =
        e_1_intercept +
        e_1_loi_slope * loi +
        e_1_grain_size_slope * grain_size_um;

    properties.c_r =
        c_r_intercept +
        c_r_loi_slope * loi +
        c_r_grain_size_slope * grain_size_um;

    properties.c_c =
        c_c_intercept +
        c_c_loi_slope * loi +
        c_c_grain_size_slope * grain_size_um;

    properties.sigma_y_kpa =
        sigma_y_intercept_kpa +
        sigma_y_loi_slope * loi +
        sigma_y_grain_size_slope * grain_size_um;

    properties.e_1 = std::max(properties.e_1, min_e_1);
    properties.c_r = std::max(properties.c_r, min_c_r);
    properties.c_c = std::max(properties.c_c, min_c_c);
    properties.sigma_y_kpa = std::max(properties.sigma_y_kpa, min_sigma_y_kpa);

    // Brain et al. (2012) distinguish a low-compressibility recompression
    // branch and a higher-compressibility virgin compression branch.
    // Therefore it is generally expected that c_c >= c_r.
    if (properties.c_c < properties.c_r)
    {
        properties.c_c = properties.c_r;
    }

    return properties;
}

// Compute buoyant stress increment contributed by each layer.
//
// This follows the same basic physical idea used in your old code:
// the self-weight relevant to effective loading is the buoyant weight
// of the solids. Brain et al. (2012) formulate compaction in terms of
// effective stress, so this is the stress-loading step needed before
// applying the constitutive law.
//
// Units:
//   mass      -> kg m^-2
//   stress    -> kPa
//   area      -> assumed to be 1 m^2 for the 1D column
Eigen::ArrayXd two_stage_compaction_model::compute_buoyant_stress_increment_kpa(
    const column_state& state,
    const material_catalog& catalog,
    const parameter_set& parameters) const
{
    const int n_layers = state.n_layers();
    const auto& mass = state.mass();

    const double water_density_kg_m3 =
        get_parameter_or_default(parameters, "water_density_kg_m3", 1000.0);
    const double gravity_m_s2 =
        get_parameter_or_default(parameters, "gravity_m_s2", 9.81);

    Eigen::ArrayXd buoyant_stress_increment_kpa = Eigen::ArrayXd::Zero(n_layers);

    // This loop is safe to parallelise because each layer writes only to its
    // own output element.
#ifdef marsh_muddpile_use_openmp
#pragma omp parallel for schedule(static) if(n_layers > 32)
#endif
    for (int layer_index = 0; layer_index < n_layers; ++layer_index)
    {
        double layer_buoyant_force_pa = 0.0;

        for (int material_index = 0; material_index < catalog.size(); ++material_index)
        {
            const material_properties& material = catalog.get_material(material_index);
            const double mass_kg_m2 = mass(layer_index, material_index);

            if (mass_kg_m2 <= 0.0)
            {
                continue;
            }

            if (material.density <= 0.0)
            {
                throw std::runtime_error("material density must be positive in compaction model");
            }

            layer_buoyant_force_pa +=
                mass_kg_m2 * gravity_m_s2 * (1.0 - water_density_kg_m3 / material.density);
        }

        buoyant_stress_increment_kpa(layer_index) = layer_buoyant_force_pa / 1000.0;
    }

    return buoyant_stress_increment_kpa;
}

// Compute a representative effective stress for each layer.
//
// Brain et al. (2012) are fundamentally concerned with the effective stress
// state within the stratigraphy. For a 1D forward model, a practical and
// stable approximation is to use the mid-layer effective stress:
//
//   sigma'_mid = sigma'_top + 0.5 * delta_sigma'_layer
//
// where sigma'_top is the cumulative stress from the overburden above.
//
// The topmost layer is assigned only its self-weight contribution; a minimum
// effective stress cutoff is applied to avoid singular behaviour in log10().
Eigen::ArrayXd two_stage_compaction_model::compute_representative_effective_stress_kpa(
    const Eigen::ArrayXd& buoyant_stress_increment_kpa,
    const parameter_set& parameters) const
{
    const int n_layers = static_cast<int>(buoyant_stress_increment_kpa.size());

    const double min_effective_stress_kpa =
        get_parameter_or_default(parameters, "compaction_min_effective_stress_kpa", 0.01);

    Eigen::ArrayXd top_effective_stress_kpa = Eigen::ArrayXd::Zero(n_layers);
    Eigen::ArrayXd representative_effective_stress_kpa = Eigen::ArrayXd::Zero(n_layers);

    if (n_layers == 0)
    {
        return representative_effective_stress_kpa;
    }

    // Because the surface layer is the last row, we accumulate overburden
    // from top to bottom.
    for (int layer_index = n_layers - 2; layer_index >= 0; --layer_index)
    {
        top_effective_stress_kpa(layer_index) =
            top_effective_stress_kpa(layer_index + 1) +
            buoyant_stress_increment_kpa(layer_index + 1);
    }

    for (int layer_index = 0; layer_index < n_layers; ++layer_index)
    {
        representative_effective_stress_kpa(layer_index) =
            top_effective_stress_kpa(layer_index) +
            0.5 * buoyant_stress_increment_kpa(layer_index);

        representative_effective_stress_kpa(layer_index) =
            std::max(
                representative_effective_stress_kpa(layer_index),
                min_effective_stress_kpa);
    }

    return representative_effective_stress_kpa;
}

// Two-stage constitutive law.
//
// This follows the spirit of the Brain et al. (2012) framework:
//
//   - use a reference void ratio e_1 at 1 kPa
//   - use c_r below yield stress
//   - use c_c above yield stress
//
// The form implemented here is continuous at sigma_y:
//
//   if sigma' <= sigma_y:
//       e = e_1 - c_r * log10(sigma' / sigma_ref)
//
//   if sigma' > sigma_y:
//       e_y = e_1 - c_r * log10(sigma_y / sigma_ref)
//       e   = e_y - c_c * log10(sigma' / sigma_y)
//
// where sigma_ref = 1 kPa.
//
// This is a practical forward-model form consistent with the two-stage
// marsh compression concept described by Brain et al. (2012).
double two_stage_compaction_model::compute_void_ratio(
    double effective_stress_kpa,
    const layer_compression_properties& properties,
    const parameter_set& parameters) const
{
    const double reference_stress_kpa =
        get_parameter_or_default(parameters, "compaction_reference_stress_kpa", 1.0);

    const double min_void_ratio =
        get_parameter_or_default(parameters, "compaction_min_void_ratio", 0.01);

    const double sigma =
        std::max(effective_stress_kpa, 1.0e-12);

    const double sigma_y =
        std::max(properties.sigma_y_kpa, 1.0e-12);

    double void_ratio = 0.0;

    if (sigma <= sigma_y)
    {
        void_ratio =
            properties.e_1 -
            properties.c_r * std::log10(sigma / reference_stress_kpa);
    }
    else
    {
        const double e_y =
            properties.e_1 -
            properties.c_r * std::log10(sigma_y / reference_stress_kpa);

        void_ratio =
            e_y -
            properties.c_c * std::log10(sigma / sigma_y);
    }

    return std::max(void_ratio, min_void_ratio);
}

// Convert solid volume and void ratio to total layer thickness.
//
// Brain et al. (2012) work with void ratio, and thickness then follows from:
//
//   e = V_v / V_s
//   V_t = V_s + V_v = V_s * (1 + e)
//
// For a unit plan area column, volume and thickness are numerically identical.
double two_stage_compaction_model::compute_layer_thickness_m(
    double solid_volume_m3_m2,
    double void_ratio) const
{
    if (solid_volume_m3_m2 <= 0.0)
    {
        return 0.0;
    }

    return solid_volume_m3_m2 * (1.0 + void_ratio);
}

// Convert void ratio to porosity:
//
//   n = e / (1 + e)
//
// This is the standard relation also used in marsh compression work such as
// Brain et al. (2012).
double two_stage_compaction_model::compute_porosity(
    double void_ratio) const
{
    return void_ratio / (1.0 + void_ratio);
}
}
