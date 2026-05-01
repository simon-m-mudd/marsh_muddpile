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
    double suspended_sediment_concentration = 0.0;
    double fine_sediment_concentration = 0.0;

    double external_pb210_supply = 0.0;
};
}
