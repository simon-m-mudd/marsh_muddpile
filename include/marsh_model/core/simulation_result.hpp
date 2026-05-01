#pragma once

#include "marsh_model/core/column_state.hpp"
#include <string>
#include <unordered_map>
#include <vector>

namespace marsh_model
{
struct simulation_result
{
    column_state final_state;
    std::unordered_map<std::string, std::vector<double>> time_series;
};
}
