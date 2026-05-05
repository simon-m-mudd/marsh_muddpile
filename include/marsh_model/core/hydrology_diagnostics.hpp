#pragma once

namespace marsh_model
{
struct hydrology_diagnostics
{
    double mean_water_level_m = 0.0;
    double max_water_level_m = 0.0;

    double inundation_fraction = 0.0;
    double mean_inundation_depth_m = 0.0;
    double inundation_duration_hours = 0.0;

    double storm_surge_residual_m = 0.0;
};
}
