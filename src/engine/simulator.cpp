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
// this version reserves time-series storage up front and can also collect
// optional in-memory column snapshots at requested times or regular intervals.
//
// output writing is intentionally kept outside the solver itself.
//
// -----------------------------------------------------------------------------

#include "marsh_model/engine/simulator.hpp"
#include "marsh_model/engine/layer_merger.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace marsh_model
{
namespace
{
bool should_save_snapshot(
    const output_config& output,
    int time_step_index,
    double model_time_days)
{
    if (!output.write_column_snapshots)
    {
        return false;
    }

    if (output.snapshot_every_n_steps > 0)
    {
        if ((time_step_index + 1) % output.snapshot_every_n_steps == 0)
        {
            return true;
        }
    }

    const double tolerance_days = 1.0e-9;

    for (double requested_time_days : output.snapshot_times_days)
    {
        if (std::abs(model_time_days - requested_time_days) <= tolerance_days)
        {
            return true;
        }
    }

    return false;
}
}

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
    column_state initial_state,
    const output_config& output) const
{
    column_state state = std::move(initial_state);
    simulation_result result;

    const int n_steps = forcing.size();

    std::vector<double>* model_time_days_ts = nullptr;
    std::vector<double>* surface_elevation_ts = nullptr;
    std::vector<double>* peak_biomass_ts = nullptr;
    std::vector<double>* aboveground_biomass_ts = nullptr;
    std::vector<double>* belowground_biomass_ts = nullptr;
    std::vector<double>* belowground_mortality_ts = nullptr;
    std::vector<double>* n_layers_ts = nullptr;

    if (output.write_time_series)
    {
        model_time_days_ts = &result.time_series["model_time_days"];
        surface_elevation_ts = &result.time_series["surface_elevation"];
        peak_biomass_ts = &result.time_series["peak_biomass"];
        aboveground_biomass_ts = &result.time_series["aboveground_biomass"];
        belowground_biomass_ts = &result.time_series["belowground_biomass"];
        belowground_mortality_ts = &result.time_series["belowground_mortality"];
        n_layers_ts = &result.time_series["n_layers"];

        model_time_days_ts->reserve(n_steps);
        surface_elevation_ts->reserve(n_steps);
        peak_biomass_ts->reserve(n_steps);
        aboveground_biomass_ts->reserve(n_steps);
        belowground_biomass_ts->reserve(n_steps);
        belowground_mortality_ts->reserve(n_steps);
        n_layers_ts->reserve(n_steps);

        result.total_mass_by_material_time_series.reserve(n_steps);
    }

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

        if (output.write_time_series)
        {
            model_time_days_ts->push_back(step.model_time_days);
            surface_elevation_ts->push_back(state.get_surface_elevation());
            peak_biomass_ts->push_back(biomass_flux.peak_biomass);
            aboveground_biomass_ts->push_back(biomass_flux.aboveground_biomass);
            belowground_biomass_ts->push_back(biomass_flux.belowground_biomass);
            belowground_mortality_ts->push_back(biomass_flux.belowground_mortality);
            n_layers_ts->push_back(static_cast<double>(state.n_layers()));

            const Eigen::ArrayXd total_mass =
                state.get_total_mass_by_material();

            result.total_mass_by_material_time_series.emplace_back(
                total_mass.data(),
                total_mass.data() + total_mass.size());
        }

        if (should_save_snapshot(output, i, step.model_time_days))
        {
            column_snapshot snapshot;
            snapshot.model_time_days = step.model_time_days;
            snapshot.state = state;
            result.column_snapshots.push_back(std::move(snapshot));
        }
    }

    result.final_state = std::move(state);
    return result;
}
}
