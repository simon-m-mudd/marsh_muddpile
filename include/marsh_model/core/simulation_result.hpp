#pragma once

// simulation_result.hpp
//
// Part of marsh_muddpile-- https://github.com/simon-m-mudd/marsh_muddpile
//
// Copyright (C) 2026 Simon M. Mudd
// Released under the GNU General Public Licence v3 (GPL-3.0)
// See LICENSE file or https://www.gnu.org/licenses/gpl-3.0.html
//
// -----------------------------------------------------------------------------
// this file defines the output of a forward marsh simulation.
//
// it stores:
//
//   - the final column state
//   - aggregate time series through the run
//   - total mass by material through time
//   - optional saved layer-resolved column snapshots at selected times
//
// -----------------------------------------------------------------------------

#include "marsh_model/core/column_state.hpp"
#include "marsh_model/core/ecohydrology_state.hpp"


#include <string>
#include <unordered_map>
#include <vector>

namespace marsh_model
{
struct column_snapshot
{
    double model_time_days = 0.0;
    column_state state;
};

struct simulation_result
{
    column_state final_state;
    std::unordered_map<std::string, std::vector<double>> time_series;
    std::vector<std::vector<double>> total_mass_by_material_time_series;
    std::vector<column_snapshot> column_snapshots;
    ecohydrology_state final_ecohydrology_state;

};
}
