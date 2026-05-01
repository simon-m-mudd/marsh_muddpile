#pragma once

// output_config.hpp
//
// Part of marsh_muddpile-- https://github.com/simon-m-mudd/marsh_muddpile
//
// Copyright (C) 2026 Simon M. Mudd
// Released under the GNU General Public Licence v3 (GPL-3.0)
// See LICENSE file or https://www.gnu.org/licenses/gpl-3.0.html
//
// -----------------------------------------------------------------------------
// this file defines output and snapshot scheduling options for a simulation.
//
// the key design choice is that these options are separate from the solver
// physics, so the forward model can be reused cleanly in inversion workflows.
//
// -----------------------------------------------------------------------------

#include <string>
#include <vector>

namespace marsh_model
{
struct output_config
{
    std::string file = "run_output.nc";

    bool write_time_series = true;
    bool write_column_snapshots = false;

    std::vector<double> snapshot_times_days;
    int snapshot_every_n_steps = 0;
};
}
