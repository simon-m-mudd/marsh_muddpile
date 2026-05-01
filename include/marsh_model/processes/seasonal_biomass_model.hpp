#pragma once

// seasonal_biomass_model.hpp
//
// Part of marsh_muddpile-- https://github.com/simon-m-mudd/marsh_muddpile
//
// Copyright (C) 2026 Simon M. Mudd
// Released under the GNU General Public Licence v3 (GPL-3.0)
// See LICENSE file or https://www.gnu.org/licenses/gpl-3.0.html
//
// -----------------------------------------------------------------------------
// this component implements a seasonal marsh biomass model.
//
// the model follows the logic of the older marsh code:
//
// 1. peak aboveground biomass depends on marsh elevation relative to mean high
//    tide
// 2. aboveground biomass varies seasonally through the year
// 3. belowground biomass is estimated from the aboveground biomass using a
//    depth-dependent belowground-to-aboveground ratio
// 4. belowground mortality is returned as a flux for use by the root allocation
//    component
//
// the implementation supports a simple linear temperature effect on peak
// biomass, similar in spirit to the older model, but the interface is modular
// so this relationship can be replaced later.
//
// -----------------------------------------------------------------------------

#include "marsh_model/processes/biomass_model.hpp"

namespace marsh_model
{
class seasonal_biomass_model : public biomass_model
{
public:
    seasonal_biomass_model() = default;

    biomass_fluxes compute_biomass_fluxes(
        const column_state& state,
        const forcing_step& forcing,
        const material_catalog& catalog,
        const parameter_set& parameters) const override;

private:
    double compute_day_of_year_days(
        const forcing_step& forcing,
        const parameter_set& parameters) const;

    double compute_peak_aboveground_biomass_g_m2(
        const column_state& state,
        const forcing_step& forcing,
        const parameter_set& parameters) const;

    double compute_aboveground_biomass_g_m2(
        double day_of_year_days,
        double peak_biomass_g_m2,
        const parameter_set& parameters) const;

    double compute_aboveground_mortality_g_m2_over_timestep(
        double day_of_year_days,
        double peak_biomass_g_m2,
        double dt_days,
        const parameter_set& parameters) const;

    double compute_belowground_to_aboveground_ratio(
        const column_state& state,
        const forcing_step& forcing,
        const parameter_set& parameters) const;
};
}
