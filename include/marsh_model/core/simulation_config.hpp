#pragma once

#include <string>

namespace marsh_model
{
struct simulation_config
{
    int start_year = 0;
    int end_year = 100;
    double dt_days = 30.0;

    int initial_layer_count = 1;
    double initial_mineral_mass = 25.0;

    std::string deposition_model_name = "zero_deposition";
    std::string biomass_model_name = "null_biomass";
    std::string root_allocation_model_name = "null_root_allocation";
    std::string decay_model_name = "identity_decay";
    std::string compaction_model_name = "identity_compaction";
};
}
