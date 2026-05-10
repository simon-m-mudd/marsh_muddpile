// sulfate_methane_model.cpp
//
// Part of marsh_muddpile -- https://github.com/simon-m-mudd/marsh_muddpile
//
// Copyright (C) 2026 Simon M. Mudd
// Released under the GNU General Public Licence v3 (GPL-3.0)
// See LICENSE file or https://www.gnu.org/licenses/gpl-3.0.html
//
// -----------------------------------------------------------------------------
// Coupled porewater sulfate and methane model.
//
// Per-layer SO4 and CH4 concentrations (μmol L-1) in column_state are
// advanced one time step by the following sequential explicit updates:
//
// 1. INITIALISATION
//    Arrays are initialised to ch4_initial_so4_umol_L and
//    ch4_initial_ch4_umol_L respectively on first call (zero indicates
//    uninitialised, consistent with the NH4 model convention).
//
// 2. SULFATE-METHANOGENESIS COMPETITION
//    Organic decomposition flux from decay_fluxes is converted to C_umol/L
//    (same unit conversion as the NH4 production step).  The fraction going
//    to SRB vs. methanogens is determined by a Michaelis-Menten expression:
//
//       srb_frac  = SO4 / (SO4 + Km_SO4)
//       meth_frac = 1 − srb_frac
//
//    Sulfate is consumed by SRB (stoichiometry 2 mol C per mol SO4):
//       ΔSO4 = −srb_frac × C_umol_L / 2
//
//    CH4 is produced by methanogens (default 2 mol C per mol CH4,
//    reflecting the mixed acetoclastic + hydrogenotrophic community):
//       ΔCH4 = +meth_frac × C_umol_L / ch4_c_to_ch4_molar_ratio
//
// 3. TIDAL FLUSHING OF SULFATE
//    Tidal inundation restores porewater SO4 toward the creek concentration
//    with an exponential depth-attenuation (same structure as NH4 flushing):
//
//       ΔSO4 = so4_flushing_rate × inundation_frac
//              × exp(−depth / so4_flushing_depth_scale) × (creek_SO4 − SO4) × dt
//
//    No tidal flushing is applied to CH4 (creek CH4 ≈ 0).
//
// 4. FICKIAN DIFFUSION (explicit Euler)
//    Applied independently to SO4 and CH4 with the same inter-layer formula
//    used by the NH4 model:
//
//       J_{i→i+1} = D × phi_mean × (C[i] − C[i+1]) / dz_CC
//
//    Stability: D × dt / dz² < 0.5.  With default D = 1e-5 m² d-1,
//    dt = 30 d, dz = 0.05 m the ratio is ≈ 0.12.
//
//    At the sediment surface (top boundary) CH4 concentration is taken as
//    zero (atmospheric equilibrium is negligible in porewater units), giving
//    a diffusive escape flux from the top layer:
//
//       J_surface [μmol m-2 d-1] = D_CH4 × phi_top × CH4_top / (thick_top/2)
//
// 5. CH4 OXIDATION
//    First-order oxidation decays CH4 in near-surface sediment:
//
//       ΔCH4 = −ch4_oxidation_rate × exp(−depth / ch4_oxidation_depth_scale_m)
//              × CH4 × dt
//
// 6. EBULLITION
//    If CH4 exceeds the saturation threshold, the excess is released
//    instantly as bubbles and added to the surface flux.
//
// 7. SURFACE FLUX
//    Total CH4 surface flux (μmol m-2 s-1) =
//       diffusive surface flux (from step 4 top boundary)
//       × (1 + ch4_plant_transport_factor)           [plant aerenchyma]
//       + ebullition flux (from step 6)
//    The plant transport factor of 2.0 (default) doubles the diffusive flux,
//    matching the edi.1828.3 observation that whole-plant chambers yield
//    ~3× the root-only (= sediment diffusion) flux.
//    The surface flux is returned in μmol m-2 s-1.
//
// Parameters (all prefixed ch4_):
//   ch4_carbon_fraction              = 0.45
//   ch4_c_to_ch4                     = 2.0
//   ch4_km_so4_umol_L                = 1000.0
//   ch4_creek_so4_umol_L             = 19600.0   (≈ 24.5 ppt × 0.8 mM/ppt × 1000)
//   ch4_tidal_flushing_rate_per_d    = 0.3
//   ch4_flushing_depth_scale_m       = 0.10
//   ch4_so4_diffusion_coeff_m2_per_d = 1.0e-5
//   ch4_diffusion_coeff_m2_per_d     = 1.0e-5
//   ch4_oxidation_rate_per_d         = 0.05
//   ch4_oxidation_depth_scale_m      = 0.05
//   ch4_plant_transport_factor       = 2.0
//   ch4_ebullition_threshold_umol_L  = 500.0
//   ch4_initial_so4_umol_L           = 19600.0
//   ch4_initial_ch4_umol_L           = 0.0
//   ch4_min_umol_L                   = 0.0
//   ch4_max_umol_L                   = 2000.0
//
// References:
//   Capone, D.G., Kiene, R.P., 1988. Comparison of microbial dynamics in
//     marine and freshwater sediments. Limnology and Oceanography 33, 725-749.
//     (Sulfate reduction vs. methanogenesis competition)
//
//   Howarth, R.W., Marino, R., 1984. Sulfate reduction in salt marshes.
//     Limnology and Oceanography 29, 598-608.
//     (SRB stoichiometry and rates in Spartina marshes)
//
//   Seyfried et al., 2026. Different mechanisms underlie plant-mediated
//     methane transport in a tidal salt marsh versus an impounded brackish
//     wetland. JGR Biogeosciences.  doi:10.1029/2025JG008987
//     (Plant transport factor; edi.1828.3 data)
//
//   Reddy, K.R., DeLaune, R.D., 2008. Biogeochemistry of Wetlands.
//     CRC Press, Boca Raton FL.
//     (Sulfate-methane transition zone; CH4 production stoichiometry)
// -----------------------------------------------------------------------------

#include "marsh_model/processes/sulfate_methane_model.hpp"

#include "marsh_model/core/material_properties.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace marsh_model
{
namespace
{
double get_param(
    const parameter_set& parameters,
    const std::string& name,
    double default_value)
{
    return parameters.has(name) ? parameters.get(name) : default_value;
}

bool is_organic_material(material_category cat)
{
    return cat == material_category::organic_labile ||
           cat == material_category::organic_refractory ||
           cat == material_category::live_root;
}
}

double sulfate_methane_model::update_methane(
    column_state& state,
    const decay_fluxes& fluxes,
    const forcing_step& forcing,
    const site_properties& /*site*/,
    const ecohydrology_state& /*eco_state*/,
    const hydrology_diagnostics& hydro,
    const material_catalog& catalog,
    const parameter_set& parameters) const
{
    const int n_layers   = state.n_layers();
    const int n_materials = state.n_materials();

    if (n_layers == 0 || forcing.dt_days <= 0.0)
    {
        return 0.0;
    }

    // -------------------------------------------------------------------------
    // Read parameters
    // -------------------------------------------------------------------------
    const double carbon_frac =
        get_param(parameters, "ch4_carbon_fraction",              0.45);
    const double c_to_ch4 =
        get_param(parameters, "ch4_c_to_ch4",                     2.0);
    const double km_so4 =
        get_param(parameters, "ch4_km_so4_umol_L",                1000.0);
    const double creek_so4 =
        get_param(parameters, "ch4_creek_so4_umol_L",             19600.0);
    const double so4_flush_rate =
        get_param(parameters, "ch4_tidal_flushing_rate_per_d",    0.3);
    const double so4_flush_depth =
        get_param(parameters, "ch4_flushing_depth_scale_m",       0.10);
    const double D_so4 =
        get_param(parameters, "ch4_so4_diffusion_coeff_m2_per_d", 1.0e-5);
    const double D_ch4 =
        get_param(parameters, "ch4_diffusion_coeff_m2_per_d",     1.0e-5);
    const double oxidation_rate =
        get_param(parameters, "ch4_oxidation_rate_per_d",         0.05);
    const double oxidation_depth =
        get_param(parameters, "ch4_oxidation_depth_scale_m",      0.05);
    const double plant_factor =
        get_param(parameters, "ch4_plant_transport_factor",       2.0);
    const double ebullition_threshold =
        get_param(parameters, "ch4_ebullition_threshold_umol_L",  500.0);
    const double initial_so4 =
        get_param(parameters, "ch4_initial_so4_umol_L",           19600.0);
    const double initial_ch4 =
        get_param(parameters, "ch4_initial_ch4_umol_L",           0.0);
    const double min_conc =
        get_param(parameters, "ch4_min_umol_L",                   0.0);
    const double max_ch4 =
        get_param(parameters, "ch4_max_umol_L",                   2000.0);

    const double dt = forcing.dt_days;
    const double M_C = 12.011; // g mol-1

    // -------------------------------------------------------------------------
    // Pre-compute organic material indices once to avoid per-layer catalog lookups
    // -------------------------------------------------------------------------
    std::vector<int> organic_indices;
    organic_indices.reserve(n_materials);
    for (int m = 0; m < n_materials; ++m)
    {
        if (is_organic_material(catalog.get_material(m).category))
            organic_indices.push_back(m);
    }

    // -------------------------------------------------------------------------
    // Initialise SO4 and CH4 arrays
    //
    // Convention: 0.0 == uninitialised (column_state initialises to zero).
    // Depletable SO4 must not be re-initialised once the model has run — only
    // the first call (all zeros) and newly-appended surface layers are handled.
    // -------------------------------------------------------------------------
    auto& so4 = state.porewater_so4();
    auto& ch4 = state.porewater_ch4();

    if (so4.size() != n_layers)
    {
        so4 = Eigen::ArrayXd::Constant(n_layers, initial_so4);
    }
    else if (so4.maxCoeff() <= 0.0)
    {
        // First call: entire array is uninitialised zeros
        so4 = Eigen::ArrayXd::Constant(n_layers, initial_so4);
    }
    else if (n_layers > 0 && so4(n_layers - 1) <= 0.0)
    {
        // New surface layer just appended — seed with creek SO4
        so4(n_layers - 1) = creek_so4;
    }

    if (ch4.size() != n_layers)
    {
        ch4 = Eigen::ArrayXd::Constant(n_layers, initial_ch4);
    }
    else if (initial_ch4 > 0.0 && ch4.maxCoeff() <= 0.0)
    {
        // First call with non-zero initial CH4
        ch4 = Eigen::ArrayXd::Constant(n_layers, initial_ch4);
    }
    // New surface layers naturally start at 0 CH4 — no per-layer check needed

    const bool has_fluxes =
        (fluxes.mass_decomposed_kg_m2.rows() == n_layers &&
         fluxes.mass_decomposed_kg_m2.cols() == n_materials);

    // -------------------------------------------------------------------------
    // Pre-compute per-layer pore volumes (reused in multiple steps)
    // -------------------------------------------------------------------------
    const Eigen::ArrayXd pore_vol = state.layer_porosity() * state.layer_thickness();

    // -------------------------------------------------------------------------
    // Step 2: Competition — SRB consumes SO4, methanogens produce CH4
    //
    // C_umol_L = mass_kg_m2 × carbon_frac × 1e6 / (M_C × pore_vol_frac)
    // (1000 g/kg and 1000 L/m3 cancel)
    // SO4 consumed: 2 mol C : 1 mol SO4
    // CH4 produced: c_to_ch4 mol C : 1 mol CH4
    // -------------------------------------------------------------------------
    if (has_fluxes && c_to_ch4 > 0.0 && carbon_frac > 0.0 && !organic_indices.empty())
    {
        for (int i = 0; i < n_layers; ++i)
        {
            if (pore_vol(i) <= 0.0) continue;

            double decomposed_kg_m2 = 0.0;
            for (int m : organic_indices)
                decomposed_kg_m2 += fluxes.mass_decomposed_kg_m2(i, m);

            if (decomposed_kg_m2 <= 0.0) continue;

            const double C_umol_L =
                decomposed_kg_m2 * carbon_frac * 1.0e6
                / (M_C * pore_vol(i));

            const double srb_frac  = so4(i) / (so4(i) + km_so4);
            const double meth_frac = 1.0 - srb_frac;

            so4(i) -= srb_frac  * C_umol_L / 2.0;
            ch4(i) += meth_frac * C_umol_L / c_to_ch4;
        }
    }

    // -------------------------------------------------------------------------
    // Step 3: Tidal flushing of SO4 toward creek concentration
    //
    // Analytical exponential solution — stable for any rate × dt:
    //   C_new = creek + (C - creek) × exp(-rate × dt)
    // -------------------------------------------------------------------------
    if (so4_flush_rate > 0.0 && hydro.inundation_fraction > 0.0)
    {
        const double inund = std::max(0.0, hydro.inundation_fraction);
        const double surface_elev = state.get_surface_elevation();

        for (int i = 0; i < n_layers; ++i)
        {
            const double depth_m =
                surface_elev - state.layer_top_elevation()(i)
                + 0.5 * state.layer_thickness()(i);

            const double depth_factor =
                (so4_flush_depth > 0.0)
                    ? std::exp(-depth_m / so4_flush_depth)
                    : 1.0;

            const double rate = so4_flush_rate * inund * depth_factor;
            so4(i) += (creek_so4 - so4(i)) * (1.0 - std::exp(-rate * dt));
        }
    }

    // -------------------------------------------------------------------------
    // Step 4a: Fickian diffusion of SO4 (explicit Euler)
    // -------------------------------------------------------------------------
    if (D_so4 > 0.0 && n_layers > 1)
    {
        const Eigen::ArrayXd so4_old = so4;

        for (int i = 0; i < n_layers - 1; ++i)
        {
            const int upper = i + 1;
            const double phi_avg =
                0.5 * (state.layer_porosity()(i) + state.layer_porosity()(upper));
            const double dz_cc =
                0.5 * (state.layer_thickness()(i) + state.layer_thickness()(upper));

            if (dz_cc <= 0.0 || pore_vol(i) <= 0.0 || pore_vol(upper) <= 0.0)
                continue;

            const double J = D_so4 * phi_avg * (so4_old(i) - so4_old(upper)) / dz_cc;
            so4(i)     -= J * dt / pore_vol(i);
            so4(upper) += J * dt / pore_vol(upper);
        }
    }

    // -------------------------------------------------------------------------
    // Step 4b: Fickian diffusion of CH4 (explicit Euler)
    // Top boundary: CH4 at sediment surface = 0 (atmosphere)
    // -------------------------------------------------------------------------
    double diffusive_surface_flux_umol_m2_d = 0.0;

    if (D_ch4 > 0.0)
    {
        const Eigen::ArrayXd ch4_old = ch4;

        // Surface escape flux from the top layer (index n_layers-1)
        const int top = n_layers - 1;
        if (pore_vol(top) > 0.0 && state.layer_thickness()(top) > 0.0)
        {
            const double J_surface =
                D_ch4 * state.layer_porosity()(top) * ch4_old(top)
                / (0.5 * state.layer_thickness()(top));

            diffusive_surface_flux_umol_m2_d = J_surface;
            ch4(top) -= J_surface * dt / pore_vol(top);
        }

        // Inter-layer diffusion
        if (n_layers > 1)
        {
            for (int i = 0; i < n_layers - 1; ++i)
            {
                const int upper = i + 1;
                const double phi_avg =
                    0.5 * (state.layer_porosity()(i) + state.layer_porosity()(upper));
                const double dz_cc =
                    0.5 * (state.layer_thickness()(i) + state.layer_thickness()(upper));

                if (dz_cc <= 0.0 || pore_vol(i) <= 0.0 || pore_vol(upper) <= 0.0)
                    continue;

                const double J =
                    D_ch4 * phi_avg * (ch4_old(i) - ch4_old(upper)) / dz_cc;
                ch4(i)     -= J * dt / pore_vol(i);
                ch4(upper) += J * dt / pore_vol(upper);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Step 5: CH4 oxidation (exponential depth decay from surface)
    // Depth array reuses surface_elev cached in Step 3 if flushing was active;
    // compute it here unconditionally.
    // -------------------------------------------------------------------------
    if (oxidation_rate > 0.0)
    {
        const double surface_elev = state.get_surface_elevation();

        for (int i = 0; i < n_layers; ++i)
        {
            if (ch4(i) <= 0.0) continue;

            const double depth_m =
                surface_elev - state.layer_top_elevation()(i)
                + 0.5 * state.layer_thickness()(i);

            const double depth_factor =
                (oxidation_depth > 0.0)
                    ? std::exp(-depth_m / oxidation_depth)
                    : 1.0;

            ch4(i) -= oxidation_rate * depth_factor * ch4(i) * dt;
        }
    }

    // -------------------------------------------------------------------------
    // Step 6: Ebullition — release CH4 above saturation threshold
    // -------------------------------------------------------------------------
    double ebullition_flux_umol_m2_d = 0.0;

    if (ebullition_threshold > 0.0)
    {
        for (int i = 0; i < n_layers; ++i)
        {
            if (ch4(i) > ebullition_threshold)
            {
                const double pore_vol_frac =
                    state.layer_porosity()(i) * state.layer_thickness()(i);

                // excess concentration × pore volume (L m-2) → μmol m-2
                const double excess_umol_m2 =
                    (ch4(i) - ebullition_threshold) * pore_vol_frac * 1000.0;

                ebullition_flux_umol_m2_d += excess_umol_m2 / dt;
                ch4(i) = ebullition_threshold;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Clamp to valid range
    // -------------------------------------------------------------------------
    so4 = so4.max(min_conc);
    ch4 = ch4.max(min_conc).min(max_ch4);

    // -------------------------------------------------------------------------
    // Step 7: Total surface flux (μmol m-2 s-1)
    // Diffusive flux is amplified by (1 + plant_factor) to account for
    // plant-mediated aerenchyma transport; ebullition is added directly.
    // -------------------------------------------------------------------------
    const double total_flux_umol_m2_d =
        diffusive_surface_flux_umol_m2_d * (1.0 + plant_factor) +
        ebullition_flux_umol_m2_d;

    return total_flux_umol_m2_d / 86400.0; // convert d-1 to s-1
}
}
