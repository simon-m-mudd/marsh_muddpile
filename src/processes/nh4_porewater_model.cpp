// nh4_porewater_model.cpp
//
// Part of marsh_muddpile -- https://github.com/simon-m-mudd/marsh_muddpile
//
// Copyright (C) 2026 Simon M. Mudd
// Released under the GNU General Public Licence v3 (GPL-3.0)
// See LICENSE file or https://www.gnu.org/licenses/gpl-3.0.html
//
// -----------------------------------------------------------------------------
// Porewater NH4+ model for the marsh sediment column.
//
// Per-layer NH4 concentration (μmol L-1) in column_state::porewater_nh4() is
// advanced one time step by four sequential explicit updates:
//
// 1. PRODUCTION
//    Organic-matter decomposition mineralises organic N to NH4.  Only
//    materials with decay properties contribute (identified via the material
//    catalog).  The conversion is:
//
//       ΔC_prod [μmol/L] =
//           mass_decomposed [kg m-2]
//           × carbon_fraction [–]
//           × 1000 [g kg-1]
//           / (M_C [g mol-1] × C:N [-])
//           × 1e6 [μmol mol-1]
//           / (porosity × thickness × 1000 [L m-1])
//
//    Simplifying (the two 1000 factors cancel):
//       ΔC_prod = mass_decomposed × carbon_frac × 1e6
//                 / (M_C × C_to_N × porosity × thickness)
//
// 2. TIDAL FLUSHING
//    During inundation the overlying water exchanges with porewater.  The
//    exchange rate decreases exponentially with depth below the surface to
//    represent limited tidal penetration into dense sediments:
//
//       ΔC_flush = nh4_tidal_flushing_rate_per_d
//                  × inundation_fraction
//                  × exp(-depth_m / nh4_flushing_depth_scale_m)
//                  × (creek_nh4 − C_layer)
//                  × dt
//
// 3. FICKIAN DIFFUSION
//    Explicit Euler 1-D diffusion between adjacent layers:
//
//       J_{i→i+1} = D_eff × porosity_mean × (C[i] − C[i+1]) / dz_CC [μmol m-2 d-1]
//
//    where dz_CC = ½(thick_i + thick_{i+1}) is the centre-to-centre distance.
//    The 1000-factor conversions (L↔m3) cancel in ΔC = J / pore_vol_L.
//
//    Stability condition (D_eff × dt / dz^2 < 0.5) is guaranteed for the
//    default D_eff = 1e-5 m2 d-1 with 5-cm layers and monthly steps.
//
// 4. PLANT UPTAKE  (Michaelis-Menten)
//
//       ΔC_uptake = Vmax × root_density_kg_m3 × C / (Km + C) × dt
//
//    where root_density is derived from the live-root material mass per layer.
//
// References:
//   Natelhoffer, K.J., Fry, B., 1988. Controls on natural 15N and 13C
//     abundances in forest soil organic matter.  Soil Science Society of
//     America Journal 52, 1633-1640.
//     (Soil C:N ratios and N mineralisation)
//
//   Reddy, K.R., DeLaune, R.D., 2008. Biogeochemistry of Wetlands.
//     CRC Press, Boca Raton FL.
//     (Porewater NH4 in salt marsh sediments; Sections 6.3 and 8.2)
//
//   Dacey, J.W.H., Drake, B.G., Klug, M.J., 1994. Stimulation of methane
//     emission by carbon dioxide enrichment of marsh vegetation.  Nature 370,
//     47-49. (Root-zone nutrient cycling)
// -----------------------------------------------------------------------------

#include "marsh_model/processes/nh4_porewater_model.hpp"

#include "marsh_model/core/material_properties.hpp"

#include <algorithm>
#include <cmath>
#include <string>

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

// Returns depth of the layer midpoint below the sediment surface (m).
double layer_midpoint_depth_m(
    const column_state& state,
    int layer_index)
{
    const double surface_elev = state.get_surface_elevation();
    const double top_elev = state.layer_top_elevation()(layer_index);
    const double thickness = state.layer_thickness()(layer_index);
    return surface_elev - top_elev + 0.5 * thickness;
}

bool is_organic_material(material_category cat)
{
    return cat == material_category::organic_labile ||
           cat == material_category::organic_refractory ||
           cat == material_category::live_root;
}
}

void nh4_porewater_model::update_porewater(
    column_state& state,
    const decay_fluxes& fluxes,
    const forcing_step& forcing,
    const site_properties& /*site*/,
    const ecohydrology_state& /*eco_state*/,
    const hydrology_diagnostics& hydro,
    const material_catalog& catalog,
    const parameter_set& parameters) const
{
    const int n_layers = state.n_layers();
    const int n_materials = state.n_materials();

    if (n_layers == 0 || forcing.dt_days <= 0.0)
    {
        return;
    }

    // -------------------------------------------------------------------------
    // Read parameters
    // -------------------------------------------------------------------------
    const double carbon_frac =
        get_parameter_or_default(parameters, "nh4_carbon_fraction", 0.45);

    const double c_to_n =
        get_parameter_or_default(parameters, "nh4_c_to_n_molar_ratio", 25.0);

    const double creek_nh4_umol_L =
        get_parameter_or_default(parameters, "nh4_creek_umol_L", 5.0);

    const double tidal_flushing_rate =
        get_parameter_or_default(parameters, "nh4_tidal_flushing_rate_per_d", 0.5);

    const double flushing_depth_scale_m =
        get_parameter_or_default(parameters, "nh4_flushing_depth_scale_m", 0.05);

    const double D_eff =
        get_parameter_or_default(parameters, "nh4_diffusion_coeff_m2_per_d", 1.0e-5);

    const double uptake_vmax =
        get_parameter_or_default(
            parameters,
            "nh4_uptake_vmax_umol_L_d_per_kg_m3",
            10.0);

    const double km =
        get_parameter_or_default(parameters, "nh4_km_umol_L", 50.0);

    const double initial_umol_L =
        get_parameter_or_default(parameters, "nh4_initial_umol_L", 100.0);

    const double min_umol_L =
        get_parameter_or_default(parameters, "nh4_min_umol_L", 0.0);

    const double max_umol_L =
        get_parameter_or_default(parameters, "nh4_max_umol_L", 2000.0);

    const double dt = forcing.dt_days;

    // -------------------------------------------------------------------------
    // Pre-compute material indices once to avoid repeated catalog lookups
    // -------------------------------------------------------------------------
    std::vector<int> organic_indices;
    std::vector<int> root_indices;
    organic_indices.reserve(n_materials);
    root_indices.reserve(n_materials);
    for (int m = 0; m < n_materials; ++m)
    {
        const material_category cat = catalog.get_material(m).category;
        if (is_organic_material(cat))
            organic_indices.push_back(m);
        if (cat == material_category::live_root)
            root_indices.push_back(m);
    }

    // -------------------------------------------------------------------------
    // Initialise NH4 array
    // -------------------------------------------------------------------------
    auto& nh4 = state.porewater_nh4();

    if (nh4.size() != n_layers)
    {
        // Size mismatch (safety guard; should not occur in normal operation)
        nh4 = Eigen::ArrayXd::Constant(n_layers, initial_umol_L);
    }
    else if (nh4.maxCoeff() <= 0.0)
    {
        // First call: entire array is uninitialised zeros
        nh4 = Eigen::ArrayXd::Constant(n_layers, initial_umol_L);
    }
    else if (n_layers > 0 && nh4(n_layers - 1) <= 0.0)
    {
        // New surface layer just appended — initialise to background
        nh4(n_layers - 1) = initial_umol_L;
    }

    // -------------------------------------------------------------------------
    // Step 1: production from organic matter decomposition
    // mass_decomposed × carbon_frac × 1e6 / (M_C [g/mol] × C_to_N × pore_vol)
    // The 1000 g/kg and 1000 L/m3 factors cancel exactly.
    // -------------------------------------------------------------------------
    const double M_C_g_per_mol = 12.011;

    const bool has_fluxes =
        (fluxes.mass_decomposed_kg_m2.rows() == n_layers &&
         fluxes.mass_decomposed_kg_m2.cols() == n_materials);

    if (has_fluxes && c_to_n > 0.0 && carbon_frac > 0.0 && !organic_indices.empty())
    {
        for (int i = 0; i < n_layers; ++i)
        {
            const double pore_vol_frac =
                state.layer_porosity()(i) * state.layer_thickness()(i);

            if (pore_vol_frac <= 0.0)
                continue;

            double decomposed_kg_m2 = 0.0;
            for (int m : organic_indices)
                decomposed_kg_m2 += fluxes.mass_decomposed_kg_m2(i, m);

            if (decomposed_kg_m2 > 0.0)
            {
                nh4(i) +=
                    decomposed_kg_m2 * carbon_frac * 1.0e6
                    / (M_C_g_per_mol * c_to_n * pore_vol_frac);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Step 2: tidal flushing toward creek NH4
    //
    // Analytical exponential solution — stable for any rate × dt:
    //   C_new = creek + (C - creek) × exp(-rate × dt)
    // Equivalent: ΔC = (creek - C) × (1 - exp(-rate × dt))
    // -------------------------------------------------------------------------
    if (tidal_flushing_rate > 0.0 && hydro.inundation_fraction > 0.0)
    {
        const double inund = std::max(0.0, hydro.inundation_fraction);
        const double surface_elev = state.get_surface_elevation();

        for (int i = 0; i < n_layers; ++i)
        {
            const double depth_m =
                surface_elev - state.layer_top_elevation()(i)
                + 0.5 * state.layer_thickness()(i);

            const double depth_factor =
                (flushing_depth_scale_m > 0.0)
                    ? std::exp(-depth_m / flushing_depth_scale_m)
                    : 1.0;

            const double rate = tidal_flushing_rate * inund * depth_factor;
            nh4(i) += (creek_nh4_umol_L - nh4(i)) * (1.0 - std::exp(-rate * dt));
        }
    }

    // -------------------------------------------------------------------------
    // Step 3: Fickian diffusion between adjacent layers (explicit Euler)
    //
    // Upward flux from layer i to layer i+1 (i+1 is shallower):
    //   J = D_eff × phi_avg × (C[i] - C[i+1]) / dz_CC
    // -------------------------------------------------------------------------
    if (D_eff > 0.0 && n_layers > 1)
    {
        const Eigen::ArrayXd nh4_old = nh4;

        for (int i = 0; i < n_layers - 1; ++i)
        {
            const int upper = i + 1;

            const double phi_i   = state.layer_porosity()(i);
            const double phi_u   = state.layer_porosity()(upper);
            const double phi_avg = 0.5 * (phi_i + phi_u);
            const double dz_cc   =
                0.5 * (state.layer_thickness()(i) + state.layer_thickness()(upper));

            if (dz_cc <= 0.0 || phi_i <= 0.0 || phi_u <= 0.0)
                continue;

            const double J_up =
                D_eff * phi_avg * (nh4_old(i) - nh4_old(upper)) / dz_cc;

            const double pv_i = phi_i * state.layer_thickness()(i);
            const double pv_u = phi_u * state.layer_thickness()(upper);

            nh4(i)     -= J_up * dt / pv_i;
            nh4(upper) += J_up * dt / pv_u;
        }
    }

    // -------------------------------------------------------------------------
    // Step 4: plant root uptake (Michaelis-Menten)
    // -------------------------------------------------------------------------
    if (uptake_vmax > 0.0 && km > 0.0 && !root_indices.empty())
    {
        for (int i = 0; i < n_layers; ++i)
        {
            if (nh4(i) <= 0.0)
                continue;

            double root_mass_kg_m2 = 0.0;
            for (int m : root_indices)
                root_mass_kg_m2 += state.mass()(i, m);

            if (root_mass_kg_m2 <= 0.0)
                continue;

            const double thickness = state.layer_thickness()(i);
            const double root_density_kg_m3 =
                (thickness > 0.0) ? root_mass_kg_m2 / thickness : 0.0;

            nh4(i) -= uptake_vmax * root_density_kg_m3 * nh4(i) / (km + nh4(i)) * dt;
        }
    }

    // -------------------------------------------------------------------------
    // Clamp to valid range
    // -------------------------------------------------------------------------
    nh4 = nh4.max(min_umol_L).min(max_umol_L);
}
}
