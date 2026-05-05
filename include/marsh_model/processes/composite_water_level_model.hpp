#pragma once

#include "marsh_model/processes/water_level_model.hpp"

namespace marsh_model
{
class composite_water_level_model : public water_level_model
{
public:
    hydrology_diagnostics compute_hydrology(
        const column_state& state,
        const forcing_step& forcing,
        const site_properties& site,
        const parameter_set& parameters) const override;

private:
    double compute_astronomical_water_level_m(
        double time_hours,
        const forcing_step& forcing,
        const site_properties& site,
        const parameter_set& parameters) const;
};
}
