#pragma once

// simple_canopy_et_model.hpp
//
// Canopy-based evapotranspiration model that splits potential ET into
// transpiration and soil evaporation.
//
// Potential ET is estimated empirically from air temperature and PAR.
// Transpiration is scaled by LAI-driven canopy cover and reduced by salinity
// and inundation stress. Soil evaporation is scaled by the exposed (non-inundated)
// surface fraction and modified by surface texture (fine fraction, organic content,
// porosity, and coarse fraction).
//
// References:
//   Allen, R.G., Pereira, L.S., Raes, D., Smith, M., 1998. Crop
//     evapotranspiration: guidelines for computing crop water requirements.
//     FAO Irrigation and Drainage Paper 56. FAO, Rome.
//     https://www.fao.org/3/x0490e/x0490e00.htm
//
//   Monsi, M., Saeki, T., 1953. Über den Lichtfaktor in den Pflanzengesellschaften
//     und seine Bedeutung für die Stoffproduktion. Japanese Journal of Botany 14,
//     22-52. (English translation: Annals of Botany 95, 549-567, 2005.
//     https://doi.org/10.1093/aob/mci052)

#include "marsh_model/core/ecohydrology_state.hpp"
#include "marsh_model/core/et_fluxes.hpp"
#include "marsh_model/core/forcing_step.hpp"
#include "marsh_model/core/hydrology_diagnostics.hpp"
#include "marsh_model/core/parameter_set.hpp"
#include "marsh_model/core/sediment_surface_properties.hpp"
#include "marsh_model/core/site_properties.hpp"
#include "marsh_model/processes/evapotranspiration_model.hpp"

namespace marsh_model
{
class simple_canopy_et_model : public evapotranspiration_model
{
public:
    // Computes transpiration and evaporation for the time step, returning both
    // components and the total in et_fluxes (mm d^-1).
    et_fluxes compute_et(
        const ecohydrology_state& eco_state,
        const hydrology_diagnostics& hydro,
        const sediment_surface_properties& surface,
        const forcing_step& forcing,
        const site_properties& site,
        const parameter_set& parameters) const override;

private:
    // Returns LAI from state if available; otherwise estimates from biomass.
    double compute_effective_lai(
        const ecohydrology_state& eco_state,
        const parameter_set& parameters) const;

    // Empirical potential ET (mm d^-1) from temperature and PAR inputs.
    double compute_potential_et_mm_d(
        const forcing_step& forcing,
        const parameter_set& parameters) const;
};
}
