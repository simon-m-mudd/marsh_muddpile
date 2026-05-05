// result_io.cpp
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
// the current implementation writes a netcdf file containing:
//
//   - aggregate time series
//   - total mass by material through time
//   - layer-resolved column snapshots at selected times
//   - material names and densities
//
// snapshot layer arrays are padded to the maximum number of layers found among
// the saved snapshots. The valid layer count for each snapshot is stored in
// snapshot_n_layers.
//
// -----------------------------------------------------------------------------

#include "marsh_model/io/result_io.hpp"

#include <netcdf>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace marsh_model
{
namespace
{
constexpr std::size_t material_name_strlen = 64;
constexpr double missing_value_double = -9999.0;

const std::vector<double>* find_series(
    const simulation_result& result,
    const std::string& name)
{
    auto it = result.time_series.find(name);

    if (it == result.time_series.end())
    {
        return nullptr;
    }

    return &it->second;
}

std::vector<char> build_material_name_buffer(
    const material_catalog& materials)
{
    std::vector<char> buffer(
        materials.size() * material_name_strlen,
        '\0');

    for (int i = 0; i < materials.size(); ++i)
    {
        const std::string& name = materials.get_material(i).name;
        const std::size_t n_copy =
            std::min(name.size(), material_name_strlen - 1);

        for (std::size_t j = 0; j < n_copy; ++j)
        {
            buffer[i * material_name_strlen + j] = name[j];
        }
    }

    return buffer;
}

std::size_t compute_max_snapshot_layers(
    const simulation_result& result)
{
    std::size_t max_layers = 0;

    for (const auto& snapshot : result.column_snapshots)
    {
        max_layers = std::max(
            max_layers,
            static_cast<std::size_t>(snapshot.state.n_layers()));
    }

    return max_layers;
}
}

void result_io::write_netcdf(
    const std::string& file_name,
    const simulation_result& result,
    const material_catalog& materials)
{
    try
    {
        using namespace netCDF;

        NcFile file(file_name, NcFile::replace);

        file.putAtt("title", "marsh_muddpile simulation output");
        file.putAtt("source", "marsh_muddpile");
        file.putAtt("history", "written by result_io::write_netcdf");

        const std::size_t n_materials =
            static_cast<std::size_t>(materials.size());

        const std::vector<char> material_name_buffer =
            build_material_name_buffer(materials);

        std::vector<double> material_density_kg_m3(n_materials, 0.0);
        for (std::size_t i = 0; i < n_materials; ++i)
        {
            material_density_kg_m3[i] =
                materials.get_material(static_cast<int>(i)).density;
        }

        NcDim material_dim =
            file.addDim("material", n_materials);

        NcDim name_strlen_dim =
            file.addDim("name_strlen", material_name_strlen);

        NcVar material_name_var =
            file.addVar("material_name", ncChar, {material_dim, name_strlen_dim});

        material_name_var.putVar(material_name_buffer.data());

        NcVar material_density_var =
            file.addVar("material_density_kg_m3", ncDouble, material_dim);

        material_density_var.putAtt("units", "kg m-3");
        material_density_var.putVar(material_density_kg_m3.data());

        // ---------------------------------------------------------------------
        // write aggregate time series
        // ---------------------------------------------------------------------
        const std::vector<double>* model_time_days =
            find_series(result, "model_time_days");

        if (model_time_days && !model_time_days->empty())
        {
            const std::size_t n_time = model_time_days->size();

            NcDim time_dim =
                file.addDim("time", n_time);

            auto write_scalar_time_series =
                [&](const std::string& name, const std::string& units)
                {
                    const std::vector<double>* series =
                        find_series(result, name);

                    if (!series)
                    {
                        return;
                    }

                    if (series->size() != n_time)
                    {
                        throw std::runtime_error(
                            "time series length mismatch for variable: " + name);
                    }

                    NcVar var =
                        file.addVar(name, ncDouble, time_dim);

                    if (!units.empty())
                    {
                        var.putAtt("units", units);
                    }

                    var.putVar(series->data());
                };

            write_scalar_time_series("model_time_days", "days");
            write_scalar_time_series("surface_elevation", "m");
            write_scalar_time_series("peak_biomass", "g m-2");
            write_scalar_time_series("aboveground_biomass", "g m-2");
            write_scalar_time_series("belowground_biomass", "kg m-2");
            write_scalar_time_series("belowground_mortality", "kg m-2");
            write_scalar_time_series("n_layers", "count");
            write_scalar_time_series("root_zone_salinity_ppt", "ppt");
            write_scalar_time_series("lai", "1");
            write_scalar_time_series("aboveground_biomass_kg_m2", "kg m-2");
            write_scalar_time_series("belowground_biomass_kg_m2", "kg m-2");
            write_scalar_time_series("gpp_gC_m2_d", "gC m-2 d-1");
            write_scalar_time_series("npp_gC_m2_d", "gC m-2 d-1");
            write_scalar_time_series("et_total_mm_d", "mm d-1");
            write_scalar_time_series("et_transpiration_mm_d", "mm d-1");
            write_scalar_time_series("et_evaporation_mm_d", "mm d-1");
            write_scalar_time_series("inundation_fraction", "1");
            write_scalar_time_series("mean_inundation_depth_m", "m");
            write_scalar_time_series("mean_water_level_m", "m");
            write_scalar_time_series("max_water_level_m", "m");



            if (!result.total_mass_by_material_time_series.empty())
            {
                if (result.total_mass_by_material_time_series.size() != n_time)
                {
                    throw std::runtime_error(
                        "total_mass_by_material_time_series length does not match time dimension");
                }

                std::vector<double> total_mass_flat(
                    n_time * n_materials,
                    0.0);

                for (std::size_t t = 0; t < n_time; ++t)
                {
                    if (result.total_mass_by_material_time_series[t].size() != n_materials)
                    {
                        throw std::runtime_error(
                            "total_mass_by_material_time_series row width does not match number of materials");
                    }

                    for (std::size_t m = 0; m < n_materials; ++m)
                    {
                        total_mass_flat[t * n_materials + m] =
                            result.total_mass_by_material_time_series[t][m];
                    }
                }

                NcVar total_mass_var =
                    file.addVar(
                        "total_mass_by_material",
                        ncDouble,
                        {time_dim, material_dim});

                total_mass_var.putAtt("units", "kg m-2");
                total_mass_var.putVar(total_mass_flat.data());
            }
        }

        // ---------------------------------------------------------------------
        // write column snapshots
        // ---------------------------------------------------------------------
        const std::size_t n_snapshots =
            result.column_snapshots.size();

        if (n_snapshots > 0)
        {
            const std::size_t max_layers =
                compute_max_snapshot_layers(result);

            NcDim snapshot_dim =
                file.addDim("snapshot", n_snapshots);

            NcVar snapshot_time_var =
                file.addVar("snapshot_time_days", ncDouble, snapshot_dim);

            snapshot_time_var.putAtt("units", "days");

            std::vector<double> snapshot_time_days(n_snapshots, 0.0);
            std::vector<int> snapshot_n_layers(n_snapshots, 0);

            for (std::size_t s = 0; s < n_snapshots; ++s)
            {
                snapshot_time_days[s] =
                    result.column_snapshots[s].model_time_days;

                snapshot_n_layers[s] =
                    result.column_snapshots[s].state.n_layers();
            }

            snapshot_time_var.putVar(snapshot_time_days.data());

            NcVar snapshot_n_layers_var =
                file.addVar("snapshot_n_layers", ncInt, snapshot_dim);

            snapshot_n_layers_var.putAtt("units", "count");
            snapshot_n_layers_var.putVar(snapshot_n_layers.data());

            if (max_layers > 0)
            {
                NcDim layer_dim =
                    file.addDim("layer", max_layers);

                std::vector<double> layer_thickness(
                    n_snapshots * max_layers,
                    missing_value_double);

                std::vector<double> layer_porosity(
                    n_snapshots * max_layers,
                    missing_value_double);

                std::vector<double> layer_top_elevation(
                    n_snapshots * max_layers,
                    missing_value_double);

                std::vector<double> layer_age(
                    n_snapshots * max_layers,
                    missing_value_double);

                std::vector<double> layer_mass(
                    n_snapshots * max_layers * n_materials,
                    0.0);

                for (std::size_t s = 0; s < n_snapshots; ++s)
                {
                    const column_state& state =
                        result.column_snapshots[s].state;

                    const std::size_t n_layers =
                        static_cast<std::size_t>(state.n_layers());

                    for (std::size_t l = 0; l < n_layers; ++l)
                    {
                        const std::size_t flat_index =
                            s * max_layers + l;

                        layer_thickness[flat_index] =
                            state.layer_thickness()(static_cast<int>(l));

                        layer_porosity[flat_index] =
                            state.layer_porosity()(static_cast<int>(l));

                        layer_top_elevation[flat_index] =
                            state.layer_top_elevation()(static_cast<int>(l));

                        layer_age[flat_index] =
                            state.layer_age()(static_cast<int>(l));

                        for (std::size_t m = 0; m < n_materials; ++m)
                        {
                            const std::size_t flat_mass_index =
                                (s * max_layers + l) * n_materials + m;

                            layer_mass[flat_mass_index] =
                                state.mass()(static_cast<int>(l), static_cast<int>(m));
                        }
                    }
                }

                NcVar layer_thickness_var =
                    file.addVar("layer_thickness", ncDouble, {snapshot_dim, layer_dim});
                layer_thickness_var.putAtt("units", "m");
                layer_thickness_var.putAtt("_FillValue", ncDouble, missing_value_double);
                layer_thickness_var.putVar(layer_thickness.data());

                NcVar layer_porosity_var =
                    file.addVar("layer_porosity", ncDouble, {snapshot_dim, layer_dim});
                layer_porosity_var.putAtt("units", "1");
                layer_porosity_var.putAtt("_FillValue", ncDouble, missing_value_double);
                layer_porosity_var.putVar(layer_porosity.data());

                NcVar layer_top_elevation_var =
                    file.addVar("layer_top_elevation", ncDouble, {snapshot_dim, layer_dim});
                layer_top_elevation_var.putAtt("units", "m");
                layer_top_elevation_var.putAtt("_FillValue", ncDouble, missing_value_double);
                layer_top_elevation_var.putVar(layer_top_elevation.data());

                NcVar layer_age_var =
                    file.addVar("layer_age", ncDouble, {snapshot_dim, layer_dim});
                layer_age_var.putAtt("units", "days");
                layer_age_var.putAtt("_FillValue", ncDouble, missing_value_double);
                layer_age_var.putVar(layer_age.data());

                NcVar layer_mass_var =
                    file.addVar("layer_mass", ncDouble, {snapshot_dim, layer_dim, material_dim});
                layer_mass_var.putAtt("units", "kg m-2");
                layer_mass_var.putVar(layer_mass.data());
            }
        }
    }
    catch (const netCDF::exceptions::NcException& error)
    {
        throw std::runtime_error(
            "failed to write NetCDF file '" + file_name + "': " + error.what());
    }
}
}
