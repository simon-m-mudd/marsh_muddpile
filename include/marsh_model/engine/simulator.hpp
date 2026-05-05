#pragma once

// simulator.hpp
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
// it is responsible for evolving the column state and collecting aggregate
// diagnostics and optional in-memory column snapshots.
//
// output writing itself is intentionally not handled here, so the solver remains
// reusable for inversion and ensemble workflows.
//
// -----------------------------------------------------------------------------
#include "marsh_model/core/column_state.hpp"
#include "marsh_model/core/ecohydrology_state.hpp"
#include "marsh_model/core/forcing_series.hpp"
#include "marsh_model/core/material_catalog.hpp"
#include "marsh_model/core/parameter_set.hpp"
#include "marsh_model/core/simulation_config.hpp"
#include "marsh_model/core/simulation_result.hpp"
#include "marsh_model/core/site_properties.hpp"
#include "marsh_model/processes/compaction_model.hpp"
#include "marsh_model/processes/decay_model.hpp"
#include "marsh_model/processes/deposition_model.hpp"
#include "marsh_model/processes/evapotranspiration_model.hpp"
#include "marsh_model/processes/root_allocation_model.hpp"
#include "marsh_model/processes/salinity_model.hpp"
#include "marsh_model/processes/vegetation_model.hpp"
#include "marsh_model/processes/water_level_model.hpp"

#include <memory>

namespace marsh_model
{
class simulator
{
public:
    simulator(
        std::shared_ptr<water_level_model> water_level,
        std::shared_ptr<evapotranspiration_model> evapotranspiration,
        std::shared_ptr<salinity_model> salinity,
        std::shared_ptr<vegetation_model> vegetation,
        std::shared_ptr<deposition_model> deposition,
        std::shared_ptr<root_allocation_model> root_allocation,
        std::shared_ptr<decay_model> decay,
        std::shared_ptr<compaction_model> compaction);

    simulation_result run_forward(
        const simulation_config& config,
        const material_catalog& catalog,
        const parameter_set& parameters,
        const forcing_series& forcing,
        const site_properties& site,
        column_state initial_state,
        ecohydrology_state initial_ecohydrology_state) const;

private:
    std::shared_ptr<water_level_model> water_level_;
    std::shared_ptr<evapotranspiration_model> evapotranspiration_;
    std::shared_ptr<salinity_model> salinity_;
    std::shared_ptr<vegetation_model> vegetation_;
    std::shared_ptr<deposition_model> deposition_;
    std::shared_ptr<root_allocation_model> root_allocation_;
    std::shared_ptr<decay_model> decay_;
    std::shared_ptr<compaction_model> compaction_;
};
}
