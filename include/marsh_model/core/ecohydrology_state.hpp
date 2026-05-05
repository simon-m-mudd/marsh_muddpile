#pragma once

namespace marsh_model
{
struct ecohydrology_state
{
    double root_zone_salinity_ppt = 0.0;

    double aboveground_biomass_kg_m2 = 0.0;
    double belowground_biomass_kg_m2 = 0.0;

    double lai = 0.0;
    double litter_kg_m2 = 0.0;
};
}
