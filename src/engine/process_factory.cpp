// process_factory.cpp
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
// it is the bridge between yaml configuration and modular process selection.
//
// -----------------------------------------------------------------------------
#include "marsh_model/engine/process_factory.hpp"

#include "marsh_model/processes/composite_water_level_model.hpp"
#include "marsh_model/processes/distance_flushing_salinity_model.hpp"
#include "marsh_model/processes/nh4_porewater_model.hpp"
#include "marsh_model/processes/sulfate_methane_model.hpp"
#include "marsh_model/processes/simple_canopy_et_model.hpp"
#include "marsh_model/processes/marsh_gpp_biomass_model.hpp"

#include "marsh_model/processes/zero_deposition_model.hpp"
#include "marsh_model/processes/tke_deposition_model.hpp"
#include "marsh_model/processes/edge_distance_deposition_model.hpp"

#include "marsh_model/processes/null_biomass_model.hpp"
#include "marsh_model/processes/seasonal_biomass_model.hpp"

#include "marsh_model/processes/null_root_allocation_model.hpp"
#include "marsh_model/processes/exponential_root_allocation_model.hpp"

#include "marsh_model/processes/identity_decay_model.hpp"
#include "marsh_model/processes/first_order_decay_model.hpp"
#include "marsh_model/processes/marsh_decay_model.hpp"

#include "marsh_model/processes/identity_compaction_model.hpp"
#include "marsh_model/processes/two_stage_compaction_model.hpp"
#include "marsh_model/processes/mixing_compaction_model.hpp"

#include <stdexcept>

namespace marsh_model
{
std::shared_ptr<water_level_model> process_factory::create_water_level_model(
    const std::string& name)
{
    if (name == "composite_water_level")
    {
        return std::make_shared<composite_water_level_model>();
    }

    throw std::invalid_argument("unknown water level model: " + name);
}

std::shared_ptr<salinity_model> process_factory::create_salinity_model(
    const std::string& name)
{
    if (name == "distance_flushing_salinity")
    {
        return std::make_shared<distance_flushing_salinity_model>();
    }

    throw std::invalid_argument("unknown salinity model: " + name);
}

std::shared_ptr<evapotranspiration_model>
process_factory::create_evapotranspiration_model(
    const std::string& name)
{
    if (name == "simple_canopy_et")
    {
        return std::make_shared<simple_canopy_et_model>();
    }

    throw std::invalid_argument("unknown evapotranspiration model: " + name);
}

std::shared_ptr<vegetation_model> process_factory::create_vegetation_model(
    const std::string& name)
{
    if (name == "marsh_gpp_biomass")
    {
        return std::make_shared<marsh_gpp_biomass_model>();
    }

    throw std::invalid_argument("unknown vegetation model: " + name);
}

std::shared_ptr<deposition_model> process_factory::create_deposition_model(
    const std::string& name)
{
    if (name == "zero_deposition")
    {
        return std::make_shared<zero_deposition_model>();
    }

    if (name == "tke_deposition")
    {
        return std::make_shared<tke_deposition_model>();
    }

    if (name == "edge_distance_deposition")
    {
        return std::make_shared<edge_distance_deposition_model>();
    }

    throw std::invalid_argument("unknown deposition model: " + name);
}

std::shared_ptr<biomass_model> process_factory::create_biomass_model(
    const std::string& name)
{
    if (name == "null_biomass")
    {
        return std::make_shared<null_biomass_model>();
    }

    if (name == "seasonal_biomass")
    {
        return std::make_shared<seasonal_biomass_model>();
    }

    throw std::invalid_argument("unknown biomass model: " + name);
}

std::shared_ptr<root_allocation_model> process_factory::create_root_allocation_model(
    const std::string& name)
{
    if (name == "null_root_allocation")
    {
        return std::make_shared<null_root_allocation_model>();
    }

    if (name == "exponential_root_allocation")
    {
        return std::make_shared<exponential_root_allocation_model>();
    }

    throw std::invalid_argument("unknown root allocation model: " + name);
}

std::shared_ptr<decay_model> process_factory::create_decay_model(
    const std::string& name)
{
    if (name == "identity_decay")
    {
        return std::make_shared<identity_decay_model>();
    }

    if (name == "first_order_decay")
    {
        return std::make_shared<first_order_decay_model>();
    }

    if (name == "marsh_decay")
    {
        return std::make_shared<marsh_decay_model>();
    }

    throw std::invalid_argument("unknown decay model: " + name);
}

std::shared_ptr<compaction_model> process_factory::create_compaction_model(
    const std::string& name)
{
    if (name == "identity_compaction")
    {
        return std::make_shared<identity_compaction_model>();
    }

    if (name == "two_stage_compaction")
    {
        return std::make_shared<two_stage_compaction_model>();
    }

    if (name == "mixing_compaction")
    {
        return std::make_shared<mixing_compaction_model>();
    }

    throw std::invalid_argument("unknown compaction model: " + name);
}

std::shared_ptr<porewater_chemistry_model>
process_factory::create_porewater_chemistry_model(
    const std::string& name)
{
    if (name == "none" || name.empty())
    {
        return nullptr;
    }

    if (name == "nh4_porewater")
    {
        return std::make_shared<nh4_porewater_model>();
    }

    throw std::invalid_argument("unknown porewater chemistry model: " + name);
}

std::shared_ptr<methane_model>
process_factory::create_methane_model(
    const std::string& name)
{
    if (name == "none" || name.empty())
    {
        return nullptr;
    }

    if (name == "sulfate_methane")
    {
        return std::make_shared<sulfate_methane_model>();
    }

    throw std::invalid_argument("unknown methane model: " + name);
}
}
