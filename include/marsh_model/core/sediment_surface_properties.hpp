#pragma once

namespace marsh_model
{
struct sediment_surface_properties
{
    double porosity = 0.0;
    double bulk_density_kg_m3 = 0.0;

    double fine_fraction = 0.0;
    double coarse_fraction = 0.0;

    double organic_mass_fraction = 0.0;
    double organic_carbon_fraction = 0.0;

    double representative_grain_size_m = 0.0;
};
}
