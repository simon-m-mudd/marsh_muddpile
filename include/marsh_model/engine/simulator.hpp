#pragma once

#include "marsh_model/core/column_state.hpp"
#include "marsh_model/core/forcing_series.hpp"
#include "marsh_model/core/material_catalog.hpp"
#include "marsh_model/core/parameter_set.hpp"
#include "marsh_model/core/simulation_config.hpp"
#include "marsh_model/core/simulation_result.hpp"
#include "marsh_model/processes/biomass_model.hpp"
#include "marsh_model/processes/compaction_model.hpp"
#include "marsh_model/processes/decay_model.hpp"
#include "marsh_model/processes/deposition_model.hpp"
#include "marsh_model/processes/root_allocation_model.hpp"
#include <memory>

namespace marsh_model
{
class simulator
{
public:
    simulator(
        std::shared_ptr<deposition_model> deposition,
        std::shared_ptr<biomass_model> biomass,
        std::shared_ptr<root_allocation_model> root_allocation,
        std::shared_ptr<decay_model> decay,
        std::shared_ptr<compaction_model> compaction);

    simulation_result run_forward(
        const simulation_config& config,
        const material_catalog& catalog,
        const parameter_set& parameters,
        const forcing_series& forcing,
        column_state initial_state) const;

private:
    std::shared_ptr<deposition_model> deposition_;
    std::shared_ptr<biomass_model> biomass_;
    std::shared_ptr<root_allocation_model> root_allocation_;
    std::shared_ptr<decay_model> decay_;
    std::shared_ptr<compaction_model> compaction_;
};
}
