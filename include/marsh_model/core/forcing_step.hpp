#pragma once

namespace marsh_model
{
struct forcing_step
{
    double model_time_days = 0.0;
    double dt_days = 0.0;

    double mean_sea_level = 0.0;
    double mean_high_tide = 0.0;
    double tidal_amplitude = 0.0;
    double tidal_period_hours = 0.0;

    double temperature = 0.0;

    double precipitation_mm_d = 0.0;
    double par_umol_m2_d = 0.0;

    double creek_salinity_ppt = 0.0;
    double freshwater_input_mm_d = 0.0;

    double observed_water_level_m = 0.0;
    bool has_observed_water_level = false;
    double storm_surge_residual_m = 0.0;

    double suspended_sediment_concentration = 0.0;
    double fine_sediment_concentration = 0.0;

    double external_pb210_supply = 0.0;
};
}
