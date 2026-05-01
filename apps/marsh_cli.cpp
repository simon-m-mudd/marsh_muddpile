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

#include <Eigen/Core>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
void print_run_summary(
    const marsh_model::loaded_run_config& loaded,
    const marsh_model::simulation_result& result)
{
    using namespace marsh_model;

    std::cout << "\n";
    std::cout << "run complete\n";
    std::cout << "selected models:\n";
    std::cout << "  deposition: " << loaded.simulation.deposition_model_name << "\n";
    std::cout << "  biomass: " << loaded.simulation.biomass_model_name << "\n";
    std::cout << "  root allocation: " << loaded.simulation.root_allocation_model_name << "\n";
    std::cout << "  decay: " << loaded.simulation.decay_model_name << "\n";
    std::cout << "  compaction: " << loaded.simulation.compaction_model_name << "\n";

    std::cout << "\n";
    std::cout << "final state summary:\n";
    std::cout << "  number of layers: " << result.final_state.n_layers() << "\n";
    std::cout << "  number of materials: " << result.final_state.n_materials() << "\n";
    std::cout << "  surface elevation: " << result.final_state.get_surface_elevation() << "\n";

    if (result.final_state.n_layers() > 0)
    {
        const int top_index = result.final_state.n_layers() - 1;

        std::cout << "  top layer thickness: "
                  << result.final_state.layer_thickness()(top_index)
                  << "\n";

        std::cout << "  top layer porosity: "
                  << result.final_state.layer_porosity()(top_index)
                  << "\n";

        std::cout << "  top layer age_days: "
                  << result.final_state.layer_age()(top_index)
                  << "\n";
    }

    std::cout << "\n";
    std::cout << "final total mass by material:\n";

    const Eigen::ArrayXd total_mass =
        result.final_state.get_total_mass_by_material();

    for (int i = 0; i < loaded.materials.size(); ++i)
    {
        std::cout << "  "
                  << loaded.materials.get_material(i).name
                  << ": "
                  << total_mass(i)
                  << "\n";
    }

    std::cout << "\n";
    std::cout << "time series:\n";

    const auto& model_time_days =
        result.time_series.at("model_time_days");
    const auto& surface_elevation =
        result.time_series.at("surface_elevation");
    const auto& peak_biomass =
        result.time_series.at("peak_biomass");
    const auto& aboveground_biomass =
        result.time_series.at("aboveground_biomass");
    const auto& belowground_biomass =
        result.time_series.at("belowground_biomass");
    const auto& belowground_mortality =
        result.time_series.at("belowground_mortality");

    for (std::size_t i = 0; i < model_time_days.size(); ++i)
    {
        std::cout << "  step " << i
                  << " time_days=" << model_time_days[i]
                  << " surface_elevation=" << surface_elevation[i]
                  << " peak_biomass=" << peak_biomass[i]
                  << " aboveground_biomass=" << aboveground_biomass[i]
                  << " belowground_biomass=" << belowground_biomass[i]
                  << " belowground_mortality=" << belowground_mortality[i]
                  << "\n";
    }
}

void run_from_yaml(const std::string& config_file)
{
    using namespace marsh_model;

    const loaded_run_config loaded =
        config_io::load_run_config(config_file);

    const auto deposition =
        process_factory::create_deposition_model(
            loaded.simulation.deposition_model_name);

    const auto biomass =
        process_factory::create_biomass_model(
            loaded.simulation.biomass_model_name);

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
        deposition,
        biomass,
        root_allocation,
        decay,
        compaction);

    const simulation_result result =
        model.run_forward(
            loaded.simulation,
            loaded.materials,
            loaded.parameters,
            loaded.forcing,
            loaded.initial_state);

    print_run_summary(loaded, result);
}
}

int main(int argc, char** argv)
{
    try
    {
        const std::string config_file =
            (argc > 1) ? std::string(argv[1]) : "example_run.yaml";

        std::cout << "loading config: " << config_file << "\n";
        run_from_yaml(config_file);
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
