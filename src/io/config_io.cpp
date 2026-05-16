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
// supported yaml definitions:
//
//   - simulation
//   - parameters
//   - materials
//   - forcing.steps
//   - forcing.constant
//   - initial_state.layers
//   - initial_state.uniform_column
//   - initial_ecohydrology_state
//   - site
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

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace marsh_model
{
namespace
{
template <typename T>
T get_required_scalar(const YAML::Node& node, const std::string& key)
{
    if (!node[key])
    {
        throw std::runtime_error("missing required key: " + key);
    }

    return node[key].as<T>();
}

template <typename T>
T get_optional_scalar(
    const YAML::Node& node,
    const std::string& key,
    const T& default_value)
{
    if (!node[key])
    {
        return default_value;
    }

    return node[key].as<T>();
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
        compaction.sigma_0 =
            get_optional_scalar<double>(node["compaction"], "sigma_0", 1.0);
        material.compaction = compaction;
    }

    if (node["settling"])
    {
        settling_properties settling;
        settling.diameter =
            get_optional_scalar<double>(node["settling"], "diameter", 0.0);
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

    config.water_level_model_name =
        get_optional_scalar<std::string>(
            node,
            "water_level_model_name",
            "composite_water_level");

    config.salinity_model_name =
        get_optional_scalar<std::string>(
            node,
            "salinity_model_name",
            "distance_flushing_salinity");

    config.evapotranspiration_model_name =
        get_optional_scalar<std::string>(
            node,
            "evapotranspiration_model_name",
            "simple_canopy_et");

    config.vegetation_model_name =
        get_optional_scalar<std::string>(
            node,
            "vegetation_model_name",
            "marsh_gpp_biomass");

    config.deposition_model_name =
        get_optional_scalar<std::string>(
            node,
            "deposition_model_name",
            "tke_deposition");

    // legacy field retained for compatibility if older code still reads it
    config.biomass_model_name =
        get_optional_scalar<std::string>(
            node,
            "biomass_model_name",
            "seasonal_biomass");

    config.root_allocation_model_name =
        get_optional_scalar<std::string>(
            node,
            "root_allocation_model_name",
            "exponential_root_allocation");

    config.decay_model_name =
        get_optional_scalar<std::string>(
            node,
            "decay_model_name",
            "marsh_decay");

    config.compaction_model_name =
        get_optional_scalar<std::string>(
            node,
            "compaction_model_name",
            "two_stage_compaction");

    config.porewater_chemistry_model_name =
        get_optional_scalar<std::string>(
            node,
            "porewater_chemistry_model_name",
            "none");

    config.methane_model_name =
        get_optional_scalar<std::string>(
            node,
            "methane_model_name",
            "none");

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

forcing_step parse_forcing_step(const YAML::Node& step_node)
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

    step.precipitation_mm_d =
        get_optional_scalar<double>(step_node, "precipitation_mm_d", 0.0);

    step.par_umol_m2_d =
        get_optional_scalar<double>(step_node, "par_umol_m2_d", 0.0);

    step.creek_salinity_ppt =
        get_optional_scalar<double>(step_node, "creek_salinity_ppt", 0.0);

    step.freshwater_input_mm_d =
        get_optional_scalar<double>(step_node, "freshwater_input_mm_d", 0.0);

    if (step_node["observed_water_level_m"])
    {
        step.observed_water_level_m =
            step_node["observed_water_level_m"].as<double>();
        step.has_observed_water_level = true;
    }

    step.storm_surge_residual_m =
        get_optional_scalar<double>(step_node, "storm_surge_residual_m", 0.0);

    step.suspended_sediment_concentration =
        get_optional_scalar<double>(
            step_node,
            "suspended_sediment_concentration",
            0.0);

    step.fine_sediment_concentration =
        get_optional_scalar<double>(
            step_node,
            "fine_sediment_concentration",
            0.0);

    step.external_pb210_supply =
        get_optional_scalar<double>(step_node, "external_pb210_supply", 0.0);

    return step;
}

forcing_series parse_forcing_series(const YAML::Node& node)
{
    forcing_series forcing;

    if (!node)
    {
        return forcing;
    }

    if (node["steps"])
    {
        for (const auto& step_node : node["steps"])
        {
            forcing.add_step(parse_forcing_step(step_node));
        }

        return forcing;
    }

    if (node["constant"])
    {
        const YAML::Node constant = node["constant"];

        const int n_steps =
            get_required_scalar<int>(constant, "n_steps");

        const double dt_days =
            get_optional_scalar<double>(constant, "dt_days", 0.0);

        const double start_time_days =
            get_optional_scalar<double>(constant, "start_time_days", 0.0);

        if (n_steps < 0)
        {
            throw std::runtime_error("forcing.constant.n_steps must be non-negative");
        }

        for (int i = 0; i < n_steps; ++i)
        {
            forcing_step step = parse_forcing_step(constant);
            step.dt_days = dt_days;
            step.model_time_days = start_time_days + i * dt_days;
            forcing.add_step(step);
        }

        return forcing;
    }

    if (node["generated"])
    {
        const YAML::Node& gen = node["generated"];

        const int n_steps = get_required_scalar<int>(gen, "n_steps");
        const double dt_days = get_optional_scalar<double>(gen, "dt_days", 30.4375);
        const double start_time_days =
            get_optional_scalar<double>(gen, "start_time_days", 0.0);

        // sea level
        const double initial_msl =
            get_optional_scalar<double>(gen, "initial_mean_sea_level", 0.0);
        const double slr_rate =
            get_optional_scalar<double>(gen, "sea_level_rise_rate_m_per_yr", 0.0);

        // temperature: sinusoidal seasonal cycle + optional long-term trend
        const double temp_mean =
            get_optional_scalar<double>(gen, "temperature_mean_c", 20.0);
        const double temp_amp =
            get_optional_scalar<double>(gen, "temperature_amplitude_c", 0.0);
        const double temp_peak =
            get_optional_scalar<double>(gen, "temperature_peak_day", 200.0);
        const double temp_trend =
            get_optional_scalar<double>(gen, "temperature_trend_c_per_yr", 0.0);

        // PAR: sinusoidal seasonal cycle + optional long-term trend
        const double par_mean =
            get_optional_scalar<double>(gen, "par_mean_umol_m2_d", 0.0);
        const double par_amp =
            get_optional_scalar<double>(gen, "par_amplitude_umol_m2_d", 0.0);
        const double par_peak =
            get_optional_scalar<double>(gen, "par_peak_day", 172.0);
        const double par_trend =
            get_optional_scalar<double>(gen, "par_trend_umol_m2_d_per_yr", 0.0);

        // all other fields (salinity, SSC, tidal params, etc.) come from the
        // template parsed by parse_forcing_step; temperature, par, sea level,
        // model_time_days, and dt_days are overridden per step below.
        const forcing_step tmpl = parse_forcing_step(gen);

        if (n_steps < 0)
        {
            throw std::runtime_error(
                "forcing.generated.n_steps must be non-negative");
        }

        constexpr double two_pi = 2.0 * 3.14159265358979323846;
        constexpr double days_per_year = 365.25;

        for (int i = 0; i < n_steps; ++i)
        {
            // SLR applied at start of step (matches Python forcing_builder convention).
            // T and PAR evaluated at mid-step for a more representative average.
            const double step_start_days = start_time_days + i * dt_days;
            const double step_mid_days   = step_start_days + 0.5 * dt_days;

            const double year_at_start = step_start_days / days_per_year;
            const double year_at_mid   = step_mid_days   / days_per_year;

            // day-of-year drives the seasonal cycle
            const double doy = std::fmod(step_mid_days, days_per_year);

            const double temperature =
                temp_mean +
                temp_amp * std::cos(two_pi * (doy - temp_peak) / days_per_year) +
                year_at_mid * temp_trend;

            const double par = std::max(
                0.0,
                par_mean +
                    par_amp * std::cos(two_pi * (doy - par_peak) / days_per_year) +
                    year_at_mid * par_trend);

            forcing_step step       = tmpl;
            step.model_time_days    = start_time_days + (i + 1) * dt_days;
            step.dt_days            = dt_days;
            step.mean_sea_level     = initial_msl + year_at_start * slr_rate;
            step.temperature        = temperature;
            step.par_umol_m2_d      = par;

            forcing.add_step(step);
        }

        return forcing;
    }

    throw std::runtime_error(
        "forcing section must contain either 'steps', 'constant', or 'generated'");
}

site_properties parse_site_properties(const YAML::Node& node)
{
    site_properties site;

    if (!node)
    {
        return site;
    }

    site.distance_from_creek_m =
        get_optional_scalar<double>(node, "distance_from_creek_m", 0.0);

    site.creek_bank_elevation_m =
        get_optional_scalar<double>(node, "creek_bank_elevation_m", 0.0);

    site.local_tidal_offset_m =
        get_optional_scalar<double>(node, "local_tidal_offset_m", 0.0);

    return site;
}

ecohydrology_state parse_initial_ecohydrology_state(const YAML::Node& node)
{
    ecohydrology_state state;

    if (!node)
    {
        return state;
    }

    state.root_zone_salinity_ppt =
        get_optional_scalar<double>(node, "root_zone_salinity_ppt", 0.0);

    state.aboveground_biomass_kg_m2 =
        get_optional_scalar<double>(node, "aboveground_biomass_kg_m2", 0.0);

    state.belowground_biomass_kg_m2 =
        get_optional_scalar<double>(node, "belowground_biomass_kg_m2", 0.0);

    state.lai =
        get_optional_scalar<double>(node, "lai", 0.0);

    state.litter_kg_m2 =
        get_optional_scalar<double>(node, "litter_kg_m2", 0.0);

    return state;
}

output_config parse_output_config(const YAML::Node& node)
{
    output_config output;

    if (!node)
    {
        return output;
    }

    output.file =
        get_optional_scalar<std::string>(node, "file", "run_output.nc");

    output.write_time_series =
        get_optional_scalar<bool>(node, "write_time_series", true);

    output.write_column_snapshots =
        get_optional_scalar<bool>(node, "write_column_snapshots", false);

    if (node["snapshot_times_days"])
    {
        for (const auto& value : node["snapshot_times_days"])
        {
            output.snapshot_times_days.push_back(value.as<double>());
        }
    }

    output.snapshot_every_n_steps =
        get_optional_scalar<int>(node, "snapshot_every_n_steps", 0);

    return output;
}



column_state parse_layered_initial_state(
    const YAML::Node& layers_node,
    const material_catalog& materials)
{
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

column_state parse_uniform_column_initial_state(
    const YAML::Node& node,
    const material_catalog& materials)
{
    const int n_layers =
        get_required_scalar<int>(node, "n_layers");

    const double total_thickness_m =
        get_required_scalar<double>(node, "total_thickness_m");

    const double porosity_value =
        get_optional_scalar<double>(node, "porosity", 0.0);

    const double age_days =
        get_optional_scalar<double>(node, "age_days", 0.0);

    const std::string fill_material_name =
        get_required_scalar<std::string>(node, "fill_material");

    if (n_layers <= 0)
    {
        throw std::runtime_error(
            "initial_state.uniform_column.n_layers must be positive");
    }

    if (total_thickness_m < 0.0)
    {
        throw std::runtime_error(
            "initial_state.uniform_column.total_thickness_m must be non-negative");
    }

    if (!materials.has_material(fill_material_name))
    {
        throw std::runtime_error(
            "initial_state.uniform_column.fill_material references unknown material: " +
            fill_material_name);
    }

    const int fill_material_index =
        materials.get_material_index(fill_material_name);

    const material_properties& fill_material =
        materials.get_material(fill_material_index);

    if (fill_material.density <= 0.0)
    {
        throw std::runtime_error(
            "initial_state.uniform_column.fill_material must have positive density");
    }

    const int n_materials = materials.size();

    column_state state(n_layers, n_materials);

    Eigen::ArrayXd thickness =
        Eigen::ArrayXd::Constant(
            n_layers,
            total_thickness_m / static_cast<double>(n_layers));

    Eigen::ArrayXd porosity =
        Eigen::ArrayXd::Constant(n_layers, porosity_value);

    Eigen::ArrayXd top_elevation = Eigen::ArrayXd::Zero(n_layers);

    double cumulative_top_elevation = 0.0;
    for (int layer_index = 0; layer_index < n_layers; ++layer_index)
    {
        cumulative_top_elevation += thickness(layer_index);
        top_elevation(layer_index) = cumulative_top_elevation;
        state.layer_age()(layer_index) = age_days;
    }

    for (int layer_index = 0; layer_index < n_layers; ++layer_index)
    {
        const double solid_volume_m3_m2 =
            thickness(layer_index) * (1.0 - porosity(layer_index));

        const double mass_kg_m2 =
            solid_volume_m3_m2 * fill_material.density;

        state.mass()(layer_index, fill_material_index) = mass_kg_m2;
    }

    state.set_layer_geometry(thickness, porosity, top_elevation);
    return state;
}

column_state parse_initial_state(
    const YAML::Node& node,
    const material_catalog& materials)
{
    if (!node)
    {
        throw std::runtime_error(
            "initial_state section must be provided");
    }

    if (node["layers"])
    {
        return parse_layered_initial_state(node["layers"], materials);
    }

    if (node["uniform_column"])
    {
        return parse_uniform_column_initial_state(
            node["uniform_column"],
            materials);
    }

    throw std::runtime_error(
        "initial_state section must contain either 'layers' or 'uniform_column'");
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
    config.initial_state =
        parse_initial_state(root["initial_state"], config.materials);
    config.initial_ecohydrology_state =
        parse_initial_ecohydrology_state(root["initial_ecohydrology_state"]);
    config.site =
        parse_site_properties(root["site"]);
    config.output =
        parse_output_config(root["output"]);

    return config;
}
}
