#pragma once

namespace marsh_model
{
struct vegetation_diagnostics
{
    double gpp_gC_m2_d = 0.0;
    double npp_gC_m2_d = 0.0;

    double aboveground_growth_kg_m2_d = 0.0;
    double belowground_growth_kg_m2_d = 0.0;

    double aboveground_mortality_kg_m2_d = 0.0;
    double belowground_mortality_kg_m2_d = 0.0;

    double lai = 0.0;
};
}
