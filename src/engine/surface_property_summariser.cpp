#include "marsh_model/engine/surface_property_summariser.hpp"

#include "marsh_model/core/material_properties.hpp"

#include <algorithm>
#include <stdexcept>

namespace marsh_model
{
namespace
{
double get_parameter_or_default(
    const parameter_set& parameters,
    const std::string& name,
    double default_value)
{
    if (parameters.has(name))
    {
        return parameters.get(name);
    }

    return default_value;
}

bool is_organic_material(const material_properties& material)
{
    return material.category == material_category::organic_labile ||
           material.category == material_category::organic_refractory ||
           material.category == material_category::live_root;
}

bool is_mineral_material(const material_properties& material)
{
    return material.category == material_category::mineral;
}
}

sediment_surface_properties surface_property_summariser::summarize(
    const column_state& state,
    const material_catalog& catalog,
    const parameter_set& parameters)
{
    sediment_surface_properties properties;

    if (state.n_layers() == 0 || catalog.size() == 0)
    {
        return properties;
    }

    const double active_depth_m =
        std::max(
            1.0e-6,
            get_parameter_or_default(
                parameters,
                "surface_property_active_depth_m",
                0.10));

    const double fine_grain_threshold_m =
        get_parameter_or_default(
            parameters,
            "surface_property_fine_grain_size_threshold_m",
            6.3e-5);

    const double default_grain_size_m =
        get_parameter_or_default(
            parameters,
            "surface_property_default_grain_size_m",
            2.0e-5);

    const double default_organic_carbon_fraction =
        get_parameter_or_default(
            parameters,
            "surface_property_default_organic_carbon_fraction",
            0.45);

    const double surface_elevation = state.get_surface_elevation();

    double included_thickness_sum = 0.0;
    double porosity_weighted_sum = 0.0;

    double total_solid_mass_kg_m2 = 0.0;
    double mineral_mass_kg_m2 = 0.0;
    double organic_mass_kg_m2 = 0.0;
    double organic_carbon_mass_kg_m2 = 0.0;

    double weighted_grain_size_sum = 0.0;
    double fine_mineral_mass_kg_m2 = 0.0;
    double coarse_mineral_mass_kg_m2 = 0.0;

    for (int layer_index = 0; layer_index < state.n_layers(); ++layer_index)
    {
        const double thickness_m = state.layer_thickness()(layer_index);

        if (thickness_m <= 0.0)
        {
            continue;
        }

        const double layer_top_elevation =
            state.layer_top_elevation()(layer_index);

        const double top_depth_m =
            std::max(0.0, surface_elevation - layer_top_elevation);

        const double bottom_depth_m =
            top_depth_m + thickness_m;

        const double included_thickness_m =
            std::max(
                0.0,
                std::min(bottom_depth_m, active_depth_m) -
                std::max(top_depth_m, 0.0));

        if (included_thickness_m <= 0.0)
        {
            continue;
        }

        const double layer_fraction =
            included_thickness_m / thickness_m;

        included_thickness_sum += included_thickness_m;
        porosity_weighted_sum +=
            included_thickness_m * state.layer_porosity()(layer_index);

        for (int material_index = 0; material_index < catalog.size(); ++material_index)
        {
            const material_properties& material =
                catalog.get_material(material_index);

            const double mass_kg_m2 =
                state.mass()(layer_index, material_index) * layer_fraction;

            if (mass_kg_m2 <= 0.0)
            {
                continue;
            }

            if (material.density <= 0.0)
            {
                throw std::runtime_error(
                    "surface_property_summariser requires positive material density");
            }

            total_solid_mass_kg_m2 += mass_kg_m2;

            if (is_organic_material(material))
            {
                organic_mass_kg_m2 += mass_kg_m2;
                organic_carbon_mass_kg_m2 +=
                    default_organic_carbon_fraction * mass_kg_m2;
            }

            if (is_mineral_material(material))
            {
                mineral_mass_kg_m2 += mass_kg_m2;

                double grain_size_m = default_grain_size_m;
                if (material.settling.has_value() &&
                    material.settling->diameter > 0.0)
                {
                    grain_size_m = material.settling->diameter;
                }

                weighted_grain_size_sum += mass_kg_m2 * grain_size_m;

                if (grain_size_m <= fine_grain_threshold_m)
                {
                    fine_mineral_mass_kg_m2 += mass_kg_m2;
                }
                else
                {
                    coarse_mineral_mass_kg_m2 += mass_kg_m2;
                }
            }
        }
    }

    if (included_thickness_sum > 0.0)
    {
        properties.porosity =
            porosity_weighted_sum / included_thickness_sum;

        properties.bulk_density_kg_m3 =
            total_solid_mass_kg_m2 / included_thickness_sum;
    }

    if (total_solid_mass_kg_m2 > 0.0)
    {
        properties.organic_mass_fraction =
            organic_mass_kg_m2 / total_solid_mass_kg_m2;

        properties.organic_carbon_fraction =
            organic_carbon_mass_kg_m2 / total_solid_mass_kg_m2;
    }

    if (mineral_mass_kg_m2 > 0.0)
    {
        properties.representative_grain_size_m =
            weighted_grain_size_sum / mineral_mass_kg_m2;

        properties.fine_fraction =
            fine_mineral_mass_kg_m2 / mineral_mass_kg_m2;

        properties.coarse_fraction =
            coarse_mineral_mass_kg_m2 / mineral_mass_kg_m2;
    }
    else
    {
        properties.representative_grain_size_m = default_grain_size_m;
    }

    return properties;
}
}
