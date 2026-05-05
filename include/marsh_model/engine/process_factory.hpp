#pragma once

// process_factory.hpp
//
// Part of marsh_muddpile-- https://github.com/simon-m-mudd/marsh_muddpile
//
// Copyright (C) 2026 Simon M. Mudd
// Released under the GNU General Public Licence v3 (GPL-3.0)
// See LICENSE file or https://www.gnu.org/licenses/gpl-3.0.html
//
// -----------------------------------------------------------------------------
// this component creates process-model objects from string names.
//
// it allows the yaml configuration to choose which deposition, biomass,
// root-allocation, decay, and compaction models are used in a simulation.
//
// -----------------------------------------------------------------------------

#include "marsh_model/processes/biomass_model.hpp"
#include "marsh_model/processes/compaction_model.hpp"
#include "marsh_model/processes/decay_model.hpp"
#include "marsh_model/processes/deposition_model.hpp"
#include "marsh_model/processes/root_allocation_model.hpp"
#include "marsh_model/processes/evapotranspiration_model.hpp"
#include "marsh_model/processes/salinity_model.hpp"
#include "marsh_model/processes/vegetation_model.hpp"
#include "marsh_model/processes/water_level_model.hpp"

#include <memory>
#include <string>

namespace marsh_model
{
class process_factory
{
public:
    static std::shared_ptr<deposition_model> create_deposition_model(
        const std::string& name);

    static std::shared_ptr<biomass_model> create_biomass_model(
        const std::string& name);

    static std::shared_ptr<root_allocation_model> create_root_allocation_model(
        const std::string& name);

    static std::shared_ptr<decay_model> create_decay_model(
        const std::string& name);

    static std::shared_ptr<compaction_model> create_compaction_model(
        const std::string& name);

    static std::shared_ptr<water_level_model> create_water_level_model(
        const std::string& name);

    static std::shared_ptr<salinity_model> create_salinity_model(
        const std::string& name);

    static std::shared_ptr<evapotranspiration_model> create_evapotranspiration_model(
        const std::string& name);

    static std::shared_ptr<vegetation_model> create_vegetation_model(
        const std::string& name);

};
}
