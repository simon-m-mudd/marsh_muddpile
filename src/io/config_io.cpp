// config_io.cpp
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
// this first implementation supports inline yaml definitions for:
//
//   - simulation
//   - parameters
//   - materials
//   - forcing steps
//   - initial state layers
//
// layer ordering convention:
//   - layers are listed deepest to surface
//
// if top_elevation_m is not provided for an initial-state layer, it is computed
// from cumulative thickness.
//
// -----------------------------------------------------------------------------

#include "marsh_model/io/config_io.hpp"

#include "marsh_model/core/material_properties.hpp"

#include <yaml-cpp/yaml.h>

#include <Eigen/Core>

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace marsh_model
{
namespace
{
template <typename t>
t get_required_scalar(const YAML::Node& node, const std::string& key)
{
    if (!node[key])
    {
        throw std::runtime_error("missing required key: " + key);
    }

    return node[key].as<t>();
}

template <typename t>
t get_optional_scalar(const YAML::Node& node, const std::string& key, const t& default_value)
{
    if (!node[key])
    {
        return default_value;
    }

    return node[key].as<t>();
}

material_category parse_material_category(const std::string& name)
{
    if (name == "mineral")
    {
        return material_category::mineral;
    }
    if (name == "organic_labile")
    {
        return material_category::organic_labile;
    }
    if (name == "organic_refractory")
    {
        return material_category::organic_refractory;
    }
    if (name == "live_root")
    {
        return material_category::live_root;
    }
    if (name == "isotope")
    {
        return material_category::isotope;
    }
    if (name == "tracer")
    {
        return material_category::tracer;
    }

    return material_category::other;
}

material_properties parse_material(const YAML::Node& node)
{
    material_properties material;

    material.name = get_required_scalar<std::string>(node, "name");
    material.category = parse_material_category(
        get_optional_scalar<std::string>(node, "category", "other"));
    material.density = get_required_scalar<double>(node, "density");

    material.allow_surface_deposition =
        get_optional_scalar<bool>(node, "allow_surface_deposition", false);

    material.allow_root_input =
        get_optional_scalar<bool>(node, "allow_root_input", false);

    material.track_age =
        get_optional_scalar<bool>(node, "track_age", false);

    material.track_osl_age =
        get_optional_scalar<bool>(node, "track_osl_age", false);

    if (node["decay"])
    {
        decay_properties decay;
        decay.k_0 = get_optional_scalar<double>(node["decay"], "k_0", 0.0);
        decay.gamma = get_optional_scalar<double>(node["decay"], "gamma", 0.0);
        decay.temperature_sensitive =
            get_optional_scalar<bool>(node["decay"], "temperature_sensitive", false);
        material.decay = decay;
    }

    if (node["compaction"])
    {
        compaction_properties compaction;
        compaction.e_0 = get_optional_scalar<double>(node["compaction"], "e_0", 0.0);
        compaction.compression_index =
            get_optional_scalar<double>(node["compaction"], "compression_index", 0.0);
        compaction.sigma_0 = get_optional_scalar<double>(node["compaction"], "sigma_0", 1.0);
        material.compaction = compaction;
    }

    if (node["settling"])
    {
        settling_properties settling;
        settling.diameter = get_optional_scalar<double>(node["settling"], "diameter", 0.0);
        settling.settling_velocity =
            get_optional_scalar<double>(node["settling"], "settling_velocity", 0.0);
        material.settling = settling;
    }

    return material;
}

simulation_config parse_simulation_config(const YAML::Node& node)
{
    simulation_config config;

    config.start_year = get_optional_scalar<int>(node, "start_year", 0);
    config.end_year = get_optional_scalar<int>(node, "end_year", 100);
    config.dt_days = get_optional_scalar<double>(node, "dt_days", 30.0);

    config.initial_layer_count = get_optional_scalar<int>(node, "initial_layer_count", 1);
    config.initial_mineral_mass = get_optional_scalar<double>(node, "initial_mineral_mass", 25.0);

    config.deposition_model_name =
        get_optional_scalar<std::string>(node, "deposition_model_name", "tke_deposition");
    config.biomass_model_name =
        get_optional_scalar<std::string>(node, "biomass_model_name", "seasonal_biomass");
    config.root_allocation_model_name =
        get_optional_scalar<std::string>(node, "root_allocation_model_name", "exponential_root_allocation");
    config.decay_model_name =
        get_optional_scalar<std::string>(node, "decay_model_name", "first_order_decay");
    config.compaction_model_name =
        get_optional_scalar<std::string>(node, "compaction_model_name", "two_stage_compaction");

    return config;
}

parameter_set parse_parameter_set(const YAML::Node& node)
{
    parameter_set parameters;

    if (!node)
    {
        return parameters;
    }

    for (auto it = node.begin(); it != node.end(); ++it)
    {
        const std::string key = it->first.as<std::string>();
        const double value = it->second.as<double>();
        parameters.set(key, value);
    }

    return parameters;
}

forcing_series parse_forcing_series(const YAML::Node& node)
{
    forcing_series forcing;

    if (!node)
    {
        return forcing;
    }

    if (!node["steps"])
    {
        throw std::runtime_error("forcing section must contain 'steps'");
    }

    for (const auto& step_node : node["steps"])
    {
        forcing_step step;

        step.model_time_days =
            get_optional_scalar<double>(step_node, "model_time_days", 0.0);

        step.dt_days =
            get_optional_scalar<double>(step_node, "dt_days", 0.0);

        step.mean_sea_level =
            get_optional_scalar<double>(step_node, "mean_sea_level", 0.0);

        step.mean_high_tide =
            get_optional_scalar<double>(step_node, "mean_high_tide", 0.0);

        step.tidal_amplitude =
            get_optional_scalar<double>(step_node, "tidal_amplitude", 0.0);

        step.tidal_period_hours =
            get_optional_scalar<double>(step_node, "tidal_period_hours", 0.0);

        step.temperature =
            get_optional_scalar<double>(step_node, "temperature", 20.0);

        step.suspended_sediment_concentration =
            get_optional_scalar<double>(step_node, "suspended_sediment_concentration", 0.0);

        step.fine_sediment_concentration =
            get_optional_scalar<double>(step_node, "fine_sediment_concentration", 0.0);

        step.external_pb210_supply =
            get_optional_scalar<double>(step_node, "external_pb210_supply", 0.0);

        forcing.add_step(step);
    }

    return forcing;
}

column_state parse_initial_state(
    const YAML::Node& node,
    const material_catalog& materials)
{
    if (!node || !node["layers"])
    {
        throw std::runtime_error("initial_state section must contain 'layers'");
    }

    const YAML::Node layers_node = node["layers"];
    const int n_layers = static_cast<int>(layers_node.size());
    const int n_materials = materials.size();

    column_state state(n_layers, n_materials);

    Eigen::ArrayXd thickness = Eigen::ArrayXd::Zero(n_layers);
    Eigen::ArrayXd porosity = Eigen::ArrayXd::Zero(n_layers);
    Eigen::ArrayXd top_elevation = Eigen::ArrayXd::Zero(n_layers);

    double cumulative_top_elevation = 0.0;

    for (int layer_index = 0; layer_index < n_layers; ++layer_index)
    {
        const YAML::Node layer_node = layers_node[layer_index];

        thickness(layer_index) =
            get_optional_scalar<double>(layer_node, "thickness_m", 0.0);

        porosity(layer_index) =
            get_optional_scalar<double>(layer_node, "porosity", 0.0);

        cumulative_top_elevation += thickness(layer_index);

        if (layer_node["top_elevation_m"])
        {
            top_elevation(layer_index) =
                layer_node["top_elevation_m"].as<double>();
        }
        else
        {
            top_elevation(layer_index) = cumulative_top_elevation;
        }

        state.layer_age()(layer_index) =
            get_optional_scalar<double>(layer_node, "age_days", 0.0);

        if (layer_node["mass_kg_m2"])
        {
            const YAML::Node mass_node = layer_node["mass_kg_m2"];

            for (auto it = mass_node.begin(); it != mass_node.end(); ++it)
            {
                const std::string material_name = it->first.as<std::string>();
                const double mass_value = it->second.as<double>();

                if (!materials.has_material(material_name))
                {
                    throw std::runtime_error(
                        "initial_state references unknown material: " + material_name);
                }

                const int material_index =
                    materials.get_material_index(material_name);

                state.mass()(layer_index, material_index) = mass_value;
            }
        }
    }

    state.set_layer_geometry(thickness, porosity, top_elevation);
    return state;
}

material_catalog parse_material_catalog(const YAML::Node& node)
{
    material_catalog materials;

    if (!node)
    {
        return materials;
    }

    for (const auto& material_node : node)
    {
        materials.add_material(parse_material(material_node));
    }

    return materials;
}
}

loaded_run_config config_io::load_run_config(const std::string& yaml_file)
{
    const YAML::Node root = YAML::LoadFile(yaml_file);

    loaded_run_config config;

    if (root["simulation"])
    {
        config.simulation = parse_simulation_config(root["simulation"]);
    }

    config.parameters = parse_parameter_set(root["parameters"]);
    config.materials = parse_material_catalog(root["materials"]);
    config.forcing = parse_forcing_series(root["forcing"]);
    config.initial_state = parse_initial_state(root["initial_state"], config.materials);

    return config;
}
}
