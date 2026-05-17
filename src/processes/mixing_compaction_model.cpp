// mixing_compaction_model.cpp
//
// Part of marsh_muddpile -- https://github.com/simon-m-mudd/marsh_muddpile
//
// Copyright (C) 2026 Simon M. Mudd
// Released under the GNU General Public Licence v3 (GPL-3.0)
// See LICENSE file or https://www.gnu.org/licenses/gpl-3.0.html
//
// -----------------------------------------------------------------------------
// Implementation of the depth-dependent mixing-law compaction model.
//
// Physical approach
// -----------------
// For each layer the model:
//   1. Computes LOI = organic_mass / total_dry_mass.
//   2. Evaluates k1(depth) and k2(depth) from the calibrated polynomials.
//   3. Applies the Morris mixing model to obtain the target DBD.
//   4. Back-calculates layer thickness = total_dry_mass / (DBD * 1000)
//      (converting g cm^-3 to kg m^-3).
//   5. Updates porosity from the new solid and total volumes.
//
// Depth is measured from the current surface to the mid-point of each layer.
//
// Default polynomial coefficients (depth in metres, DBD in g cm^-3):
//   k1(d) = 0.027568*d^2 - 0.012864*d + 0.105944
//   k2(d) = -0.313556*d^2 + 0.170120*d + 1.399719
// Fitted to CCN synthesis depth-binned data (all countries, 7 bins, R^2 0.85/0.95).
//
// Parameters exposed to the YAML config
// --------------------------------------
//   mixing_compaction_k1_a0   [g cm^-3]  default 0.105944
//   mixing_compaction_k1_a1   [g cm^-3 m^-1]  default -0.012864
//   mixing_compaction_k1_a2   [g cm^-3 m^-2]  default 0.027568
//   mixing_compaction_k2_a0   [g cm^-3]  default 1.399719
//   mixing_compaction_k2_a1   [g cm^-3 m^-1]  default 0.170120
//   mixing_compaction_k2_a2   [g cm^-3 m^-2]  default -0.313556
//   mixing_compaction_k1_min  [g cm^-3]  default 0.02   (prevents non-physical k1)
//   mixing_compaction_k2_min  [g cm^-3]  default 0.20   (prevents non-physical k2)
//   mixing_compaction_max_depth_m        default 2.0    (polynomial clamped beyond this)
// -----------------------------------------------------------------------------

#include "marsh_model/processes/mixing_compaction_model.hpp"
#include "marsh_model/core/material_properties.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace marsh_model
{

namespace
{

double get_param(const parameter_set& p, const std::string& name, double def)
{
    return p.has(name) ? p.get(name) : def;
}

bool is_organic(const material_properties& m)
{
    return m.category == material_category::organic_labile ||
           m.category == material_category::organic_refractory ||
           m.category == material_category::live_root;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Depth-dependent end-member densities
// ---------------------------------------------------------------------------

double mixing_compaction_model::k1_at_depth(
    double depth_m,
    const parameter_set& parameters) const
{
    const double max_depth = get_param(parameters, "mixing_compaction_max_depth_m", 2.0);
    const double d = std::min(depth_m, max_depth);

    const double a0 = get_param(parameters, "mixing_compaction_k1_a0",  0.105944);
    const double a1 = get_param(parameters, "mixing_compaction_k1_a1", -0.012864);
    const double a2 = get_param(parameters, "mixing_compaction_k1_a2",  0.027568);
    const double k_min = get_param(parameters, "mixing_compaction_k1_min", 0.02);

    const double k = a0 + a1 * d + a2 * d * d;
    return std::max(k, k_min);
}

double mixing_compaction_model::k2_at_depth(
    double depth_m,
    const parameter_set& parameters) const
{
    const double max_depth = get_param(parameters, "mixing_compaction_max_depth_m", 2.0);
    const double d = std::min(depth_m, max_depth);

    const double a0 = get_param(parameters, "mixing_compaction_k2_a0",  1.399719);
    const double a1 = get_param(parameters, "mixing_compaction_k2_a1",  0.170120);
    const double a2 = get_param(parameters, "mixing_compaction_k2_a2", -0.313556);
    const double k_min = get_param(parameters, "mixing_compaction_k2_min", 0.20);

    const double k = a0 + a1 * d + a2 * d * d;
    return std::max(k, k_min);
}

// ---------------------------------------------------------------------------
// LOI for one layer
// ---------------------------------------------------------------------------

double mixing_compaction_model::compute_loi(
    const column_state& state,
    int layer_index,
    const material_catalog& catalog) const
{
    const auto& mass = state.mass();
    double organic_mass = 0.0;
    double total_mass   = 0.0;

    for (int m = 0; m < catalog.size(); ++m)
    {
        const double kg = mass(layer_index, m);
        if (kg <= 0.0) continue;
        total_mass += kg;
        if (is_organic(catalog.get_material(m)))
            organic_mass += kg;
    }

    return (total_mass > 0.0) ? organic_mass / total_mass : 0.0;
}

// ---------------------------------------------------------------------------
// Main compaction entry point
// ---------------------------------------------------------------------------

void mixing_compaction_model::update_compaction(
    column_state& state,
    const forcing_step&,
    const material_catalog& catalog,
    const parameter_set& parameters) const
{
    const int n = state.n_layers();
    if (n == 0) return;

    // Surface elevation = top of the highest (last) layer.
    const double surface_elev = state.layer_top_elevation()(n - 1);

    const auto& thick  = state.layer_thickness();
    const auto& top_el = state.layer_top_elevation();
    const auto& mass   = state.mass();

    Eigen::ArrayXd new_thickness  = Eigen::ArrayXd::Zero(n);
    Eigen::ArrayXd new_porosity   = Eigen::ArrayXd::Zero(n);
    Eigen::ArrayXd new_top_elev   = Eigen::ArrayXd::Zero(n);

    for (int i = 0; i < n; ++i)
    {
        // Depth of layer mid-point below current surface.
        const double mid_elev = top_el(i) - 0.5 * thick(i);
        const double depth_m  = std::max(0.0, surface_elev - mid_elev);

        // Total dry mass and solid volume for this layer.
        double total_mass_kg_m2   = 0.0;
        double solid_vol_m3_m2    = 0.0;

        for (int m = 0; m < catalog.size(); ++m)
        {
            const double kg = mass(i, m);
            if (kg <= 0.0) continue;
            const double density = catalog.get_material(m).density;
            if (density <= 0.0)
                throw std::runtime_error(
                    "mixing_compaction_model: material density must be positive");
            total_mass_kg_m2 += kg;
            solid_vol_m3_m2  += kg / density;
        }

        if (total_mass_kg_m2 <= 0.0)
        {
            // Empty layer — preserve current geometry.
            new_thickness(i) = thick(i);
            new_porosity(i)  = (thick(i) > 0.0 && solid_vol_m3_m2 < thick(i))
                               ? (thick(i) - solid_vol_m3_m2) / thick(i) : 0.0;
            continue;
        }

        // LOI and depth-dependent end-member densities.
        const double loi = compute_loi(state, i, catalog);
        const double k1  = k1_at_depth(depth_m, parameters);   // g cm^-3
        const double k2  = k2_at_depth(depth_m, parameters);   // g cm^-3

        // Mixing model DBD [g cm^-3] → [kg m^-3].
        // DBD = 1 / (loi/k1 + (1-loi)/k2)
        const double denom = loi / k1 + (1.0 - loi) / k2;
        // denom is always positive because k1,k2>0 and loi in [0,1]
        const double dbd_gcm3  = 1.0 / denom;
        const double dbd_kgm3  = dbd_gcm3 * 1000.0;

        // New thickness from mass and DBD.
        const double new_thick = total_mass_kg_m2 / dbd_kgm3;

        // Porosity from solid volume and total volume.
        double porosity = 0.0;
        if (new_thick > solid_vol_m3_m2)
            porosity = (new_thick - solid_vol_m3_m2) / new_thick;
        // If solid volume exceeds the mixing-model thickness (shouldn't happen
        // with valid data), clamp porosity to zero.

        new_thickness(i) = new_thick;
        new_porosity(i)  = porosity;
    }

    // Rebuild top elevations from the fixed column base.
    const double base_elev = top_el(0) - thick(0);
    double running_top = base_elev;
    for (int i = 0; i < n; ++i)
    {
        running_top += new_thickness(i);
        new_top_elev(i) = running_top;
    }

    state.set_layer_geometry(new_thickness, new_porosity, new_top_elev);
}

} // namespace marsh_model
