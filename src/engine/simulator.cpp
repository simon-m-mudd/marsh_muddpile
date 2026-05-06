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
// clean architecture version:
//   - water level / inundation
//   - ET
//   - salinity
//   - vegetation / GPP / biomass
//   - root allocation
//   - deposition
//   - decay
//   - compaction
//   - layer merging
//
// ----------------------------------------------------------------------------- */

#include "marsh_model/engine/simulator.hpp"

#include "marsh_model/engine/layer_merger.hpp"
#include "marsh_model/engine/surface_property_summariser.hpp"

#include <functional>
#include <stdexcept>
#include <utility>

namespace marsh_model
{
simulator::simulator(
    std::shared_ptr<water_level_model> water_level,
    std::shared_ptr<evapotranspiration_model> evapotranspiration,
    std::shared_ptr<salinity_model> salinity,
    std::shared_ptr<vegetation_model> vegetation,
    std::shared_ptr<deposition_model> deposition,
    std::shared_ptr<root_allocation_model> root_allocation,
    std::shared_ptr<decay_model> decay,
    std::shared_ptr<compaction_model> compaction)
    : water_level_(std::move(water_level)),
      evapotranspiration_(std::move(evapotranspiration)),
      salinity_(std::move(salinity)),
      vegetation_(std::move(vegetation)),
      deposition_(std::move(deposition)),
      root_allocation_(std::move(root_allocation)),
      decay_(std::move(decay)),
      compaction_(std::move(compaction))
{
    if (!water_level_ ||
        !evapotranspiration_ ||
        !salinity_ ||
        !vegetation_ ||
        !deposition_ ||
        !root_allocation_ ||
        !decay_ ||
        !compaction_)
    {
        throw std::invalid_argument(
            "simulator requires all process modules to be non-null");
    }
}

simulation_result simulator::run_forward(
    const simulation_config&,
    const material_catalog& catalog,
    const parameter_set& parameters,
    const forcing_series& forcing,
    const site_properties& site,
    column_state initial_state,
    ecohydrology_state initial_ecohydrology_state,
    std::function<void(double, double)> progress_cb) const
{
    column_state state = std::move(initial_state);
    ecohydrology_state eco_state = std::move(initial_ecohydrology_state);

    simulation_result result;

    const int n_steps = forcing.size();
    const double total_time_days =
        (n_steps > 0) ? forcing.at(n_steps - 1).model_time_days : 0.0;
    constexpr double progress_interval_days = 91.25;  // ~3 months
    double next_progress_days = progress_interval_days;

    // legacy / existing diagnostics kept for CLI compatibility
    auto& model_time_days_ts = result.time_series["model_time_days"];
    auto& surface_elevation_ts = result.time_series["surface_elevation"];
    auto& peak_biomass_ts = result.time_series["peak_biomass"];
    auto& aboveground_biomass_ts = result.time_series["aboveground_biomass"];
    auto& belowground_biomass_ts = result.time_series["belowground_biomass"];
    auto& belowground_mortality_ts = result.time_series["belowground_mortality"];

    // new ecohydrology diagnostics
    auto& salinity_ts = result.time_series["root_zone_salinity_ppt"];
    auto& lai_ts = result.time_series["lai"];
    auto& above_bio_kg_ts = result.time_series["aboveground_biomass_kg_m2"];
    auto& below_bio_kg_ts = result.time_series["belowground_biomass_kg_m2"];
    auto& gpp_ts = result.time_series["gpp_gC_m2_d"];
    auto& npp_ts = result.time_series["npp_gC_m2_d"];
    auto& aboveground_growth_ts = result.time_series["aboveground_growth_kg_m2_d"];
    auto& aboveground_mortality_ts = result.time_series["aboveground_mortality_kg_m2_d"];
    auto& belowground_growth_ts = result.time_series["belowground_growth_kg_m2_d"];
    auto& belowground_mortality_kg_ts = result.time_series["belowground_mortality_kg_m2_d"];
    auto& et_total_ts = result.time_series["et_total_mm_d"];
    auto& et_transpiration_ts = result.time_series["et_transpiration_mm_d"];
    auto& et_evaporation_ts = result.time_series["et_evaporation_mm_d"];
    auto& inundation_fraction_ts = result.time_series["inundation_fraction"];
    auto& inundation_depth_ts = result.time_series["mean_inundation_depth_m"];
    auto& mean_water_level_ts = result.time_series["mean_water_level_m"];
    auto& max_water_level_ts = result.time_series["max_water_level_m"];

    result.total_mass_by_material_time_series.reserve(n_steps);

    model_time_days_ts.reserve(n_steps);
    surface_elevation_ts.reserve(n_steps);
    peak_biomass_ts.reserve(n_steps);
    aboveground_biomass_ts.reserve(n_steps);
    belowground_biomass_ts.reserve(n_steps);
    belowground_mortality_ts.reserve(n_steps);

    salinity_ts.reserve(n_steps);
    lai_ts.reserve(n_steps);
    above_bio_kg_ts.reserve(n_steps);
    below_bio_kg_ts.reserve(n_steps);
    gpp_ts.reserve(n_steps);
    npp_ts.reserve(n_steps);
    aboveground_growth_ts.reserve(n_steps);
    aboveground_mortality_ts.reserve(n_steps);
    belowground_growth_ts.reserve(n_steps);
    belowground_mortality_kg_ts.reserve(n_steps);
    et_total_ts.reserve(n_steps);
    et_transpiration_ts.reserve(n_steps);
    et_evaporation_ts.reserve(n_steps);
    inundation_fraction_ts.reserve(n_steps);
    inundation_depth_ts.reserve(n_steps);
    mean_water_level_ts.reserve(n_steps);
    max_water_level_ts.reserve(n_steps);

    for (int i = 0; i < forcing.size(); ++i)
    {
        const forcing_step& step = forcing.at(i);

        if (state.n_layers() > 0)
        {
            state.layer_age() += step.dt_days;
        }

        const sediment_surface_properties surface =
            surface_property_summariser::summarize(
                state,
                catalog,
                parameters);

        const hydrology_diagnostics hydro =
            water_level_->compute_hydrology(
                state,
                step,
                site,
                parameters);

        const et_fluxes et =
            evapotranspiration_->compute_et(
                eco_state,
                hydro,
                surface,
                step,
                site,
                parameters);

        salinity_->update_salinity(
            eco_state,
            hydro,
            et,
            surface,
            step,
            site,
            parameters);

        const vegetation_diagnostics vegetation_diag =
            vegetation_->update_vegetation(
                eco_state,
                hydro,
                surface,
                step,
                site,
                parameters);

        // ---------------------------------------------------------------------
        // Bridge the new vegetation state/diagnostics to the pre-existing
        // biomass_fluxes interface used by root allocation and deposition.
        //
        // Current convention retained for compatibility:
        //   - peak_biomass and aboveground_biomass in g m^-2
        //   - belowground_biomass and belowground_mortality in kg m^-2
        // ---------------------------------------------------------------------
        biomass_fluxes biomass_flux;
        biomass_flux.peak_biomass =
            eco_state.aboveground_biomass_kg_m2 * 1000.0;

        biomass_flux.aboveground_biomass =
            eco_state.aboveground_biomass_kg_m2 * 1000.0;

        biomass_flux.belowground_biomass =
            eco_state.belowground_biomass_kg_m2;

        biomass_flux.belowground_mortality =
            vegetation_diag.belowground_mortality_kg_m2_d * step.dt_days;

        const Eigen::ArrayXXd root_mass_change =
            root_allocation_->compute_root_mass_change(
                state,
                biomass_flux,
                catalog,
                parameters);

        if (root_mass_change.size() > 0)
        {
            state.add_mass_to_layers(root_mass_change);
        }

        const Eigen::ArrayXd surface_flux =
            deposition_->compute_surface_flux(
                state,
                step,
                biomass_flux,
                catalog,
                parameters);

        if (surface_flux.size() > 0 && surface_flux.abs().sum() > 1.0e-12)
        {
            state.append_surface_layer(surface_flux);
        }

        decay_context decay_ctx;
        decay_ctx.eco_state = &eco_state;
        decay_ctx.hydrology = &hydro;
        decay_ctx.surface = &surface;
        decay_ctx.site = &site;

        decay_->apply_decay(
            state,
            step,
            catalog,
            parameters,
            decay_ctx);

        compaction_->update_compaction(
            state,
            step,
            catalog,
            parameters);

        layer_merger::merge_layers_if_needed(state, parameters);

        // ---------------------------------------------------------------------
        // Save diagnostics
        // ---------------------------------------------------------------------
        model_time_days_ts.push_back(step.model_time_days);
        surface_elevation_ts.push_back(state.get_surface_elevation());

        peak_biomass_ts.push_back(biomass_flux.peak_biomass);
        aboveground_biomass_ts.push_back(biomass_flux.aboveground_biomass);
        belowground_biomass_ts.push_back(biomass_flux.belowground_biomass);
        belowground_mortality_ts.push_back(biomass_flux.belowground_mortality);

        salinity_ts.push_back(eco_state.root_zone_salinity_ppt);
        lai_ts.push_back(eco_state.lai);
        above_bio_kg_ts.push_back(eco_state.aboveground_biomass_kg_m2);
        below_bio_kg_ts.push_back(eco_state.belowground_biomass_kg_m2);

        gpp_ts.push_back(vegetation_diag.gpp_gC_m2_d);
        npp_ts.push_back(vegetation_diag.npp_gC_m2_d);
        aboveground_growth_ts.push_back(vegetation_diag.aboveground_growth_kg_m2_d);
        aboveground_mortality_ts.push_back(vegetation_diag.aboveground_mortality_kg_m2_d);
        belowground_growth_ts.push_back(vegetation_diag.belowground_growth_kg_m2_d);
        belowground_mortality_kg_ts.push_back(vegetation_diag.belowground_mortality_kg_m2_d);

        et_total_ts.push_back(et.total_et_mm_d);
        et_transpiration_ts.push_back(et.transpiration_mm_d);
        et_evaporation_ts.push_back(et.evaporation_mm_d);

        inundation_fraction_ts.push_back(hydro.inundation_fraction);
        inundation_depth_ts.push_back(hydro.mean_inundation_depth_m);
        mean_water_level_ts.push_back(hydro.mean_water_level_m);
        max_water_level_ts.push_back(hydro.max_water_level_m);

        const Eigen::ArrayXd mass_by_mat = state.get_total_mass_by_material();
        result.total_mass_by_material_time_series.emplace_back(
            mass_by_mat.data(), mass_by_mat.data() + mass_by_mat.size());

        if (progress_cb && step.model_time_days >= next_progress_days)
        {
            progress_cb(step.model_time_days, total_time_days);
            next_progress_days += progress_interval_days;
        }
    }

    result.final_state = std::move(state);
    result.final_ecohydrology_state = std::move(eco_state);

    return result;
}
}
