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
// Default linear coefficients (depth in metres, DBD in g cm^-3):
//   k1(d) = 0.035029*d + 0.092574
//   k2(d) = -0.375221*d + 1.552584
// Fitted by OLS through the All-country CCN depth-binned data (six 20 cm bins
// from 0 to 150 cm, 150+ bin excluded; n=85,122 layers; R^2 k1=0.72, k2=0.80).
// Beyond 1.5 m the values are held constant at their 1.5 m level:
//   k1(1.5m) = 0.1452 g/cm^3,  k2(1.5m) = 0.9898 g/cm^3.
// The quadratic term (a2) is retained in the parameter interface for
// backward-compatibility; set it to 0 (the default) to recover linear behaviour.
//
// Parameters exposed to the YAML config
// --------------------------------------
//   mixing_compaction_k1_a0   [g cm^-3]       default 0.092574
//   mixing_compaction_k1_a1   [g cm^-3 m^-1]  default 0.035029
//   mixing_compaction_k1_a2   [g cm^-3 m^-2]  default 0.0  (set non-zero for quadratic)
//   mixing_compaction_k2_a0   [g cm^-3]       default 1.552584
//   mixing_compaction_k2_a1   [g cm^-3 m^-1]  default -0.375221
//   mixing_compaction_k2_a2   [g cm^-3 m^-2]  default 0.0  (set non-zero for quadratic)
//   mixing_compaction_k1_min  [g cm^-3]  default 0.02   (prevents non-physical k1)
//   mixing_compaction_k2_min  [g cm^-3]  default 0.20   (prevents non-physical k2)
//   mixing_compaction_max_depth_m        default 1.5    (values clamped beyond this)
// -----------------------------------------------------------------------------

#include "marsh_model/processes/mixing_compaction_model.hpp"
#include "marsh_model/core/material_properties.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

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

    // Hoist all parameter reads out of the per-layer loop.
    const double max_depth = get_param(parameters, "mixing_compaction_max_depth_m", 1.5);
    const double k1_a0    = get_param(parameters, "mixing_compaction_k1_a0",   0.092574);
    const double k1_a1    = get_param(parameters, "mixing_compaction_k1_a1",   0.035029);
    const double k1_a2    = get_param(parameters, "mixing_compaction_k1_a2",   0.0);
    const double k1_min   = get_param(parameters, "mixing_compaction_k1_min",  0.02);
    const double k2_a0    = get_param(parameters, "mixing_compaction_k2_a0",   1.552584);
    const double k2_a1    = get_param(parameters, "mixing_compaction_k2_a1",  -0.375221);
    const double k2_a2    = get_param(parameters, "mixing_compaction_k2_a2",   0.0);
    const double k2_min   = get_param(parameters, "mixing_compaction_k2_min",  0.20);

    // Surface elevation = top of the highest (last) layer.
    const double surface_elev = state.layer_top_elevation()(n - 1);

    const auto& thick  = state.layer_thickness();
    const auto& top_el = state.layer_top_elevation();
    const auto& mass   = state.mass();
    const int   n_mat  = catalog.size();

    Eigen::ArrayXd new_thickness = Eigen::ArrayXd::Zero(n);
    Eigen::ArrayXd new_porosity  = Eigen::ArrayXd::Zero(n);
    Eigen::ArrayXd new_top_elev  = Eigen::ArrayXd::Zero(n);

    // Flag set inside OpenMP region if any material has non-positive density.
    bool bad_density = false;

#pragma omp parallel for schedule(static) if(n > 32)
    for (int i = 0; i < n; ++i)
    {
        // Depth of layer mid-point below current surface.
        const double mid_elev = top_el(i) - 0.5 * thick(i);
        const double depth_m  = std::max(0.0, surface_elev - mid_elev);

        // Single pass: accumulate total mass, solid volume, and organic mass.
        double total_mass_kg_m2 = 0.0;
        double solid_vol_m3_m2  = 0.0;
        double organic_mass     = 0.0;
        bool   layer_bad        = false;

        for (int m = 0; m < n_mat; ++m)
        {
            const double kg = mass(i, m);
            if (kg <= 0.0) continue;
            const auto& mat = catalog.get_material(m);
            if (mat.density <= 0.0) { layer_bad = true; break; }
            total_mass_kg_m2 += kg;
            solid_vol_m3_m2  += kg / mat.density;
            if (is_organic(mat)) organic_mass += kg;
        }

        if (layer_bad)
        {
#pragma omp atomic write
            bad_density = true;
            // Leave new_thickness/new_porosity at zero; error thrown below.
            continue;
        }

        if (total_mass_kg_m2 <= 0.0)
        {
            // Empty layer — preserve current geometry.
            new_thickness(i) = thick(i);
            new_porosity(i)  = (thick(i) > 0.0 && solid_vol_m3_m2 < thick(i))
                               ? (thick(i) - solid_vol_m3_m2) / thick(i) : 0.0;
            continue;
        }

        // Depth-dependent end-member densities [g cm^-3].
        const double d  = std::min(depth_m, max_depth);
        const double k1 = std::max(k1_a0 + k1_a1 * d + k1_a2 * d * d, k1_min);
        const double k2 = std::max(k2_a0 + k2_a1 * d + k2_a2 * d * d, k2_min);

        // LOI from the single-pass organic/total accumulation.
        const double loi = organic_mass / total_mass_kg_m2;

        // Mixing model DBD [g cm^-3] → [kg m^-3].
        // Algebraic form with 1 division instead of 3:
        //   DBD = k1*k2 / (loi*k2 + (1-loi)*k1)
        const double mineral_frac = 1.0 - loi;
        const double dbd_gcm3 = k1 * k2 / (loi * k2 + mineral_frac * k1);
        const double dbd_kgm3 = dbd_gcm3 * 1000.0;

        // New thickness from mass and DBD.
        const double new_thick = total_mass_kg_m2 / dbd_kgm3;

        // Porosity from solid volume and total volume.
        double porosity = 0.0;
        if (new_thick > solid_vol_m3_m2)
            porosity = (new_thick - solid_vol_m3_m2) / new_thick;

        new_thickness(i) = new_thick;
        new_porosity(i)  = porosity;
    }

    if (bad_density)
        throw std::runtime_error(
            "mixing_compaction_model: material density must be positive");

    // Rebuild top elevations from the fixed column base (serial — sequential dependency).
    const double base_elev = top_el(0) - thick(0);
    double running_top = base_elev;
    for (int i = 0; i < n; ++i)
    {
        running_top += new_thickness(i);
        new_top_elev(i) = running_top;
    }

    state.set_layer_geometry(new_thickness, new_porosity, new_top_elev);
}

// ---------------------------------------------------------------------------
// Initial-column equilibration
// ---------------------------------------------------------------------------

void mixing_compaction_model::equilibrate_initial_column(
    column_state& state,
    double target_surface_elevation_m,
    const material_catalog& catalog,
    const parameter_set& parameters) const
{
    if (state.n_layers() == 0) return;

    // One compaction pass: equilibrates layer thicknesses from the current
    // geometry (base fixed, surface floats to its mixing-model position).
    forcing_step dummy{};
    update_compaction(state, dummy, catalog, parameters);

    // Shift the whole column so the surface is at target_surface_elevation_m.
    const double shift = target_surface_elevation_m - state.get_surface_elevation();
    if (std::abs(shift) < 1.0e-9) return;

    state.set_layer_geometry(
        state.layer_thickness(),
        state.layer_porosity(),
        state.layer_top_elevation() + shift);
}

} // namespace marsh_model
