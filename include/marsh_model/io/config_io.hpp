#pragma once

// config_io.hpp
//
// Part of marsh_muddpile-- https://github.com/simon-m-mudd/marsh_muddpile
//
// Copyright (C) 2026 Simon M. Mudd
// Released under the GNU General Public Licence v3 (GPL-3.0)
// See LICENSE file or https://www.gnu.org/licenses/gpl-3.0.html
//
// -----------------------------------------------------------------------------
// this component loads marsh_muddpile model configuration from yaml files.
//
// it loads:
//   - simulation settings
//   - model parameters
//   - material definitions
//   - forcing time series
//   - initial column state
//
// this is the first step in moving hard-coded setup out of marsh_cli.cpp.
//
// -----------------------------------------------------------------------------

#include "marsh_model/core/column_state.hpp"
#include "marsh_model/core/forcing_series.hpp"
#include "marsh_model/core/material_catalog.hpp"
#include "marsh_model/core/parameter_set.hpp"
#include "marsh_model/core/simulation_config.hpp"

#include <string>

namespace marsh_model
{
struct loaded_run_config
{
    simulation_config simulation;
    parameter_set parameters;
    material_catalog materials;
    forcing_series forcing;
    column_state initial_state;
};

class config_io
{
public:
    static loaded_run_config load_run_config(const std::string& yaml_file);
};
}
