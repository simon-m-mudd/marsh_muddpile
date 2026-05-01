// simulator.cpp
//
// Part of marsh_muddpile-- https://github.com/simon-m-mudd/marsh_muddpile
//
// Copyright (C) 2026 Simon M. Mudd
// Released under the GNU General Public Licence v3 (GPL-3.0)
// See LICENSE file or https://www.gnu.org/licenses/gpl-3.0.html
//
// -----------------------------------------------------------------------------
// this component coordinates the marsh model processes through time.
//
// the simulator computes biomass first, then passes the resulting vegetation
// state to both root allocation and surface deposition.
//
// this version also includes a small safe performance improvement:
// time-series storage is reserved up front, and new surface layers are only
// appended if deposited mass is actually non-zero.
//
// -----------------------------------------------------------------------------

#include "marsh_model/engine/simulator.hpp"
#include "marsh_model/engine/layer_merger.hpp"

#include <stdexcept>
#include <utility>

namespace marsh_model
{
simulator::simulator(
    std::shared_ptr<deposition_model> deposition,
    std::shared_ptr<biomass_model> biomass,
    std::shared_ptr<root_allocation_model> root_allocation,
    std::shared_ptr<decay_model> decay,
    std::shared_ptr<compaction_model> compaction)
    : deposition_(std::move(deposition)),
      biomass_(std::move(biomass)),
      root_allocation_(std::move(root_allocation)),
      decay_(std::move(decay)),
      compaction_(std::move(compaction))
{
    if (!deposition_ || !biomass_ || !root_allocation_ || !decay_ || !compaction_)
    {
        throw std::invalid_argument("simulator requires all process modules to be non-null");
    }
}

simulation_result simulator::run_forward(
    const simulation_config&,
    const material_catalog& catalog,
    const parameter_set& parameters,
    const forcing_series& forcing,
    column_state initial_state) const
{
    column_state state = std::move(initial_state);
    simulation_result result;

    const int n_steps = forcing.size();

    auto& model_time_days_ts = result.time_series["model_time_days"];
    auto& surface_elevation_ts = result.time_series["surface_elevation"];
    auto& peak_biomass_ts = result.time_series["peak_biomass"];
    auto& aboveground_biomass_ts = result.time_series["aboveground_biomass"];
    auto& belowground_biomass_ts = result.time_series["belowground_biomass"];
    auto& belowground_mortality_ts = result.time_series["belowground_mortality"];

    model_time_days_ts.reserve(n_steps);
    surface_elevation_ts.reserve(n_steps);
    peak_biomass_ts.reserve(n_steps);
    aboveground_biomass_ts.reserve(n_steps);
    belowground_biomass_ts.reserve(n_steps);
    belowground_mortality_ts.reserve(n_steps);

    for (int i = 0; i < forcing.size(); ++i)
    {
        const forcing_step& step = forcing.at(i);

        if (state.n_layers() > 0)
        {
            state.layer_age() += step.dt_days;
        }

        const biomass_fluxes biomass_flux =
            biomass_->compute_biomass_fluxes(state, step, catalog, parameters);

        const Eigen::ArrayXXd root_mass_change =
            root_allocation_->compute_root_mass_change(state, biomass_flux, catalog, parameters);

        if (root_mass_change.size() > 0)
        {
            state.add_mass_to_layers(root_mass_change);
        }

        const Eigen::ArrayXd surface_flux =
            deposition_->compute_surface_flux(state, step, biomass_flux, catalog, parameters);

        if (surface_flux.size() > 0 && surface_flux.abs().sum() > 1.0e-12)
        {
            state.append_surface_layer(surface_flux);
        }

        decay_->apply_decay(state, step, catalog, parameters);
        compaction_->update_compaction(state, step, catalog, parameters);
        layer_merger::merge_layers_if_needed(state, parameters);



        model_time_days_ts.push_back(step.model_time_days);
        surface_elevation_ts.push_back(state.get_surface_elevation());
        peak_biomass_ts.push_back(biomass_flux.peak_biomass);
        aboveground_biomass_ts.push_back(biomass_flux.aboveground_biomass);
        belowground_biomass_ts.push_back(biomass_flux.belowground_biomass);
        belowground_mortality_ts.push_back(biomass_flux.belowground_mortality);
    }

    result.final_state = std::move(state);
    return result;
}
}
