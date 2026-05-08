// exponential_root_allocation_model.cpp
//
// Part of marsh_muddpile-- https://github.com/simon-m-mudd/marsh_muddpile
//
// Copyright (C) 2026 Simon M. Mudd
// Released under the GNU General Public Licence v3 (GPL-3.0)
// See LICENSE file or https://www.gnu.org/licenses/gpl-3.0.html
//
// -----------------------------------------------------------------------------
// this component allocates live roots and dead-root-derived organic matter
// through the marsh sediment column using an exponential depth profile.
//
// this is the modular replacement for the older growth-index logic used in the
// original marsh model. The older model computed a depth weighting function for
// each layer and then:
//
//   - set the live root mass in each layer
//   - added dead root material to one or more carbon pools
//
// this implementation follows the same basic idea, but returns a matrix of mass
// changes so that the simulator can apply those updates to the central
// column_state.
//
// the live-root profile is represented as a target standing biomass profile.
// the dead-root flux is distributed using the same depth weights and partitioned
// between labile and refractory organic pools.
//
// -----------------------------------------------------------------------------

#include "marsh_model/processes/exponential_root_allocation_model.hpp"

#include <algorithm>
#include <cmath>

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
}

Eigen::ArrayXXd exponential_root_allocation_model::compute_root_mass_change(
    const column_state& state,
    const biomass_fluxes& biomass,
    const material_catalog& catalog,
    const parameter_set& parameters) const
{
    const int n_layers = state.n_layers();
    const int n_materials = state.n_materials();

    Eigen::ArrayXXd delta_mass =
        Eigen::ArrayXXd::Zero(n_layers, n_materials);

    if (n_layers == 0 || n_materials == 0)
    {
        return delta_mass;
    }

    const Eigen::ArrayXd weights =
        compute_root_profile_weights(state, parameters);

    if (weights.size() == 0)
    {
        return delta_mass;
    }

    const int root_material_index =
        find_material_index(catalog, {"roots", "live_roots", "root"});

    const int labile_material_index =
        find_material_index(catalog, {"labile_organic", "labile_carbon", "labile_om"});

    const int refractory_material_index =
        find_material_index(catalog, {"refractory_organic", "refractory_carbon", "refractory_om"});

    const double labile_fraction =
        get_parameter_or_default(parameters, "root_allocation_to_labile_fraction", 0.5);

    const double refractory_fraction =
        get_parameter_or_default(parameters, "root_allocation_to_refractory_fraction", 0.5);

    const double total_partition_fraction =
        labile_fraction + refractory_fraction;

    double safe_labile_fraction = 0.0;
    double safe_refractory_fraction = 0.0;

    if (total_partition_fraction > 0.0)
    {
        safe_labile_fraction = labile_fraction / total_partition_fraction;
        safe_refractory_fraction = refractory_fraction / total_partition_fraction;
    }

    // -------------------------------------------------------------------------
    // live root standing stock
    //
    // the biomass model provides the total belowground standing biomass for the
    // column. This root allocation component translates that total into a target
    // mass profile across the layers.
    //
    // the return value is the adjustment needed to move the existing live root
    // profile toward that target.
    // -------------------------------------------------------------------------
    if (root_material_index >= 0)
    {
        for (int layer_index = 0; layer_index < n_layers; ++layer_index)
        {
            const double target_root_mass =
                biomass.belowground_biomass * weights(layer_index);

            const double current_root_mass =
                state.mass()(layer_index, root_material_index);

            const double root_delta = target_root_mass - current_root_mass;
            delta_mass(layer_index, root_material_index) = root_delta;

            // Root profile redistribution is mass-neutral with respect to
            // organic pools.  Dead root carbon enters labile/refractory only
            // through the explicit belowground mortality flux below (Route 2).
            // Adding the standing-stock decrease here as well would double-count
            // mortality when biomass is declining: the ODE already incorporates
            // mortality into the new belowground_biomass, so route_delta already
            // reflects (growth - mortality).  Route 2 then adds the gross
            // mortality flux to organic, giving the correct column mass balance:
            //   root_delta + mortality_flux = (growth - mortality) + mortality
            //                               = growth  (new mass from NPP) ✓
        }
    }

    // -------------------------------------------------------------------------
    // belowground mortality routed to organic pools
    //
    // this follows the same overall modelling logic as the old code:
    // mortality is treated as a separate flux from the target standing biomass.
    // it is distributed vertically using the root profile and partitioned
    // between carbon pools.
    // -------------------------------------------------------------------------
    for (int layer_index = 0; layer_index < n_layers; ++layer_index)
    {
        const double layer_mortality_input =
            biomass.belowground_mortality * weights(layer_index);

        if (labile_material_index >= 0)
        {
            delta_mass(layer_index, labile_material_index) +=
                safe_labile_fraction * layer_mortality_input;
        }

        if (refractory_material_index >= 0)
        {
            delta_mass(layer_index, refractory_material_index) +=
                safe_refractory_fraction * layer_mortality_input;
        }
    }

    return delta_mass;
}

Eigen::ArrayXd exponential_root_allocation_model::compute_root_profile_weights(
    const column_state& state,
    const parameter_set& parameters) const
{
    const int n_layers = state.n_layers();

    if (n_layers == 0)
    {
        return Eigen::ArrayXd();
    }

    const double gamma_m =
        get_parameter_or_default(parameters, "root_efolding_depth_m", 0.10);

    const double normalize_to_column =
        get_parameter_or_default(parameters, "root_profile_normalize_to_column", 1.0);

    Eigen::ArrayXd weights = Eigen::ArrayXd::Zero(n_layers);

    if (gamma_m <= 0.0)
    {
        return weights;
    }

    const double surface_elevation =
        state.get_surface_elevation();

    for (int layer_index = 0; layer_index < n_layers; ++layer_index)
    {
        const double layer_top_elevation =
            state.layer_top_elevation()(layer_index);

        const double layer_thickness =
            state.layer_thickness()(layer_index);

        const double top_depth =
            surface_elevation - layer_top_elevation;

        const double bottom_depth =
            top_depth + layer_thickness;

        const double layer_weight =
            std::exp(-top_depth / gamma_m) -
            std::exp(-bottom_depth / gamma_m);

        weights(layer_index) = std::max(0.0, layer_weight);
    }

    // -------------------------------------------------------------------------
    // optional renormalization
    //
    // the old model sometimes used weights that summed to nearly one if the
    // modeled column extended deep enough. In the modular version, it is safer
    // to allow explicit normalization over the represented column.
    // -------------------------------------------------------------------------
    if (normalize_to_column > 0.5)
    {
        const double sum_weights = weights.sum();

        if (sum_weights > 0.0)
        {
            weights /= sum_weights;
        }
    }

    return weights;
}

int exponential_root_allocation_model::find_material_index(
    const material_catalog& catalog,
    const std::vector<std::string>& candidate_names) const
{
    for (const std::string& name : candidate_names)
    {
        if (catalog.has_material(name))
        {
            return catalog.get_material_index(name);
        }
    }

    return -1;
}
}
