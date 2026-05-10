// marsh_cli.cpp
//
// Part of marsh_muddpile-- https://github.com/simon-m-mudd/marsh_muddpile
//
// Copyright (C) 2026 Simon M. Mudd
// Released under the GNU General Public Licence v3 (GPL-3.0)
// See LICENSE file or https://www.gnu.org/licenses/gpl-3.0.html
//
// -----------------------------------------------------------------------------
// this application is the command-line interface for marsh_muddpile.
//
// it loads a yaml run configuration, constructs the requested process modules,
// runs the forward model, and prints a summary of model diagnostics.
//
// -----------------------------------------------------------------------------

#include "marsh_model/core/simulation_result.hpp"
#include "marsh_model/engine/process_factory.hpp"
#include "marsh_model/engine/simulator.hpp"
#include "marsh_model/io/config_io.hpp"
#include "marsh_model/io/result_io.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
bool has_flag(int argc, char** argv, const std::string& flag)
{
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == flag) return true;
    return false;
}

void run_from_yaml(const std::string& config_file, bool silent)
{
    using namespace marsh_model;

    const loaded_run_config loaded =
        config_io::load_run_config(config_file);

    const int n_forcing = loaded.forcing.size();
    const double total_days =
        (n_forcing > 0) ? loaded.forcing.at(n_forcing - 1).model_time_days : 0.0;
    const double total_years = total_days / 365.25;

    if (!silent)
    {
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "total run duration: " << total_years << " yr ("
                  << total_days << " days)\n";
    }

    const auto water_level =
        process_factory::create_water_level_model(
            loaded.simulation.water_level_model_name);

    const auto evapotranspiration =
        process_factory::create_evapotranspiration_model(
            loaded.simulation.evapotranspiration_model_name);

    const auto salinity =
        process_factory::create_salinity_model(
            loaded.simulation.salinity_model_name);

    const auto vegetation =
        process_factory::create_vegetation_model(
            loaded.simulation.vegetation_model_name);

    const auto deposition =
        process_factory::create_deposition_model(
            loaded.simulation.deposition_model_name);

    const auto root_allocation =
        process_factory::create_root_allocation_model(
            loaded.simulation.root_allocation_model_name);

    const auto decay =
        process_factory::create_decay_model(
            loaded.simulation.decay_model_name);

    const auto compaction =
        process_factory::create_compaction_model(
            loaded.simulation.compaction_model_name);

    simulator model(
        water_level,
        evapotranspiration,
        salinity,
        vegetation,
        deposition,
        root_allocation,
        decay,
        compaction);

    const auto porewater_chemistry =
        process_factory::create_porewater_chemistry_model(
            loaded.simulation.porewater_chemistry_model_name);

    if (porewater_chemistry)
    {
        model.set_porewater_chemistry(porewater_chemistry);
    }

    const auto methane =
        process_factory::create_methane_model(
            loaded.simulation.methane_model_name);

    if (methane)
    {
        model.set_methane_model(methane);
    }

    std::function<void(double, double)> progress_cb = nullptr;
    if (!silent)
    {
        progress_cb = [](double current_days, double total_days_cb)
        {
            std::cout << std::fixed << std::setprecision(2)
                      << "  " << current_days / 365.25 << " yr"
                      << " / " << total_days_cb / 365.25 << " yr\n";
        };
    }

    const simulation_result result =
        model.run_forward(
            loaded.simulation,
            loaded.materials,
            loaded.parameters,
            loaded.forcing,
            loaded.site,
            loaded.initial_state,
            loaded.initial_ecohydrology_state,
            progress_cb);

    result_io::write_netcdf(
        loaded.output.file,
        result,
        loaded.materials);

    if (!silent)
        std::cout << "wrote netcdf output: " << loaded.output.file << "\n";
}
}

int main(int argc, char** argv)
{
    try
    {
        const bool silent = has_flag(argc, argv, "--silent");

        std::string config_file = "example_run.yaml";
        for (int i = 1; i < argc; ++i)
        {
            if (std::string(argv[i]) != "--silent")
            {
                config_file = argv[i];
                break;
            }
        }

        if (!silent)
            std::cout << "loading config: " << config_file << "\n";

        run_from_yaml(config_file, silent);
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
