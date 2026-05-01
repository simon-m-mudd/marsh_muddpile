#pragma once

// result_io.hpp
//
// Part of marsh_muddpile-- https://github.com/simon-m-mudd/marsh_muddpile
//
// Copyright (C) 2026 Simon M. Mudd
// Released under the GNU General Public Licence v3 (GPL-3.0)
// See LICENSE file or https://www.gnu.org/licenses/gpl-3.0.html
//
// -----------------------------------------------------------------------------
// this component writes simulation outputs to disk.
//
// the target format is netcdf and supports:
//
//   - aggregate time series output through the run
//   - total mass by material through time
//   - layer-resolved column snapshots at selected times
//
// output writing is intentionally separated from the solver so the forward
// model remains reusable in inversion workflows.
//
// -----------------------------------------------------------------------------

#include "marsh_model/core/material_catalog.hpp"
#include "marsh_model/core/simulation_result.hpp"

#include <string>

namespace marsh_model
{
class result_io
{
public:
    static void write_netcdf(
        const std::string& file_name,
        const simulation_result& result,
        const material_catalog& materials);
};
}
