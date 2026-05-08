#pragma once

// tidal_utilities.hpp
//
// Shared harmonic tidal reconstruction used by deposition models to integrate
// deposition over a full timestep with spring-neap variability rather than
// treating every tidal cycle as identical.
//
// Performance note: call prepare_tidal_data() once per forcing step to avoid
// repeated parameter map lookups inside the cycle/sample loops.

#include "marsh_model/core/forcing_step.hpp"
#include "marsh_model/core/parameter_set.hpp"

#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace marsh_model
{
namespace tidal_utils
{

struct constituent_info
{
    const char* name;
    double speed_deg_per_hour;
};

constexpr std::array<constituent_info, 11> k_standard_constituents = {{
    {"M2",  28.9841042},
    {"S2",  30.0000000},
    {"N2",  28.4397295},
    {"K1",  15.0410686},
    {"O1",  13.9430356},
    {"P1",  14.9589314},
    {"K2",  30.0821373},
    {"Q1",  13.3986609},
    {"2N2", 27.8953548},
    {"M4",  57.9682084},
    {"MS4", 58.9841042}
}};

// Pre-read constituent data from parameters so the inner cycle/sample loops
// only do arithmetic without any string map lookups.
struct active_constituent
{
    double speed_rad_per_hour;
    double amplitude;
    double phase_rad;
    double nodal_factor;
    double eq_arg_rad;
};

struct tidal_data
{
    double mean_wl_m;                        // mean_sea_level + storm_surge
    double ref_hours;                         // reference time offset
    std::vector<active_constituent> constituents;

    // Simple-sinusoid fallback (used when no harmonic constituents are present)
    bool use_simple_sinusoid = false;
    double simple_amplitude = 0.0;
    double simple_phase_offset = 0.0;        // in cycles (not radians)
    double simple_period_hours = 12.42;
};

// Call once per forcing step. Reads all harmonic constituent data from the
// parameter set so subsequent water-level evaluations do not touch the map.
inline tidal_data prepare_tidal_data(
    const forcing_step& forcing,
    const parameter_set& parameters)
{
    constexpr double deg_to_rad = 3.14159265358979323846 / 180.0;

    tidal_data td;
    td.mean_wl_m   = forcing.mean_sea_level + forcing.storm_surge_residual_m;
    td.ref_hours   = parameters.has("water_level_reference_time_hours")
                         ? parameters.get("water_level_reference_time_hours")
                         : 0.0;

    for (const auto& c : k_standard_constituents)
    {
        const std::string amp_name =
            "water_level_constituent_" + std::string(c.name) + "_amplitude_m";
        if (!parameters.has(amp_name))
            continue;
        const double amp = parameters.get(amp_name);
        if (amp <= 0.0)
            continue;

        const std::string phase_name =
            "water_level_constituent_" + std::string(c.name) + "_phase_deg";
        const double phase_deg =
            parameters.has(phase_name) ? parameters.get(phase_name) : 0.0;

        const std::string nf_name =
            "water_level_constituent_" + std::string(c.name) + "_nodal_factor";
        const double nf = parameters.has(nf_name) ? parameters.get(nf_name) : 1.0;

        const std::string eq_name =
            "water_level_constituent_" + std::string(c.name) + "_equilibrium_argument_deg";
        const double eq_deg =
            parameters.has(eq_name) ? parameters.get(eq_name) : 0.0;

        active_constituent ac;
        ac.speed_rad_per_hour = c.speed_deg_per_hour * deg_to_rad;
        ac.amplitude          = amp;
        ac.phase_rad          = phase_deg * deg_to_rad;
        ac.nodal_factor       = nf;
        ac.eq_arg_rad         = eq_deg * deg_to_rad;
        td.constituents.push_back(ac);
    }

    if (td.constituents.empty())
    {
        td.use_simple_sinusoid = true;
        td.simple_amplitude    = forcing.tidal_amplitude;
        td.simple_phase_offset = parameters.has("water_level_phase_offset_cycles")
                                     ? parameters.get("water_level_phase_offset_cycles")
                                     : -0.25;
        td.simple_period_hours = forcing.tidal_period_hours;
    }

    return td;
}

// Evaluate water level at time_hours using pre-read tidal data (no map lookups).
inline double eval_water_level_m(double time_hours, const tidal_data& td)
{
    constexpr double two_pi = 2.0 * 3.14159265358979323846;

    if (td.use_simple_sinusoid)
    {
        if (td.simple_period_hours > 0.0)
        {
            const double phase_rad =
                two_pi * (time_hours / td.simple_period_hours + td.simple_phase_offset);
            return td.mean_wl_m + td.simple_amplitude * std::sin(phase_rad);
        }
        return td.mean_wl_m;
    }

    double astro = 0.0;
    for (const auto& ac : td.constituents)
    {
        const double arg = ac.speed_rad_per_hour * (time_hours - td.ref_hours)
                           + ac.eq_arg_rad - ac.phase_rad;
        astro += ac.nodal_factor * ac.amplitude * std::cos(arg);
    }
    return td.mean_wl_m + astro;
}

// Effective envelope for one tidal cycle starting at cycle_start_hours.
// Returns amplitude = (max-min)/2 and mean = (max+min)/2 using pre-read data.
struct cycle_envelope
{
    double amplitude_m;
    double mean_m;
};

inline cycle_envelope compute_cycle_envelope(
    double cycle_start_hours,
    double tidal_period_hours,
    const tidal_data& td,
    int n_samples = 60)
{
    double wl_max = -1.0e30;
    double wl_min =  1.0e30;

    for (int i = 0; i < n_samples; ++i)
    {
        const double t = cycle_start_hours +
            tidal_period_hours * (i + 0.5) / static_cast<double>(n_samples);
        const double wl = eval_water_level_m(t, td);
        if (wl > wl_max) wl_max = wl;
        if (wl < wl_min) wl_min = wl;
    }

    return {(wl_max - wl_min) * 0.5, (wl_max + wl_min) * 0.5};
}

} // namespace tidal_utils
} // namespace marsh_model
