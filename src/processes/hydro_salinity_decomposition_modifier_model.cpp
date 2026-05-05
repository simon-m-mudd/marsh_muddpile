#include "marsh_model/processes/hydro_salinity_decomposition_modifier_model.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace marsh_model
{
namespace
{
double get_parameter_or_default(
    const parameter_set& parameters,
    const std::string& name,
    double default_value)
{
    if (parameters.has(name))
    {
        return parameters.get(name);
    }

    return default_value;
}

double clamp(double value, double lower, double upper)
{
    return std::max(lower, std::min(value, upper));
}

double compute_hydrologic_oxygen_multiplier(
    double inundation_fraction,
    const sediment_surface_properties& surface,
    double hydro_sensitivity,
    double fine_sensitivity,
    double organic_sensitivity,
    double porosity_sensitivity,
    double min_multiplier)
{
    const double resistance_factor =
        1.0 +
        fine_sensitivity * surface.fine_fraction +
        organic_sensitivity * surface.organic_mass_fraction +
        porosity_sensitivity * surface.porosity;

    const double raw_multiplier =
        std::exp(-hydro_sensitivity * inundation_fraction * resistance_factor);

    return std::max(min_multiplier, raw_multiplier);
}

double compute_salinity_multiplier(
    double salinity_ppt,
    double threshold_ppt,
    double salinity_sensitivity,
    double min_multiplier)
{
    const double salinity_excess =
        std::max(0.0, salinity_ppt - threshold_ppt);

    const double raw_multiplier =
        std::exp(-salinity_sensitivity * salinity_excess);

    return std::max(min_multiplier, raw_multiplier);
}
}

decomposition_modifiers
hydro_salinity_decomposition_modifier_model::compute_modifiers(
    const ecohydrology_state& eco_state,
    const hydrology_diagnostics& hydro,
    const sediment_surface_properties& surface,
    const forcing_step&,
    const site_properties&,
    const parameter_set& parameters) const
{
    decomposition_modifiers modifiers;

    const double global_min_multiplier =
        clamp(
            get_parameter_or_default(
                parameters,
                "decomposition_modifier_global_min_multiplier",
                0.05),
            0.0,
            1.0);

    const double global_max_multiplier =
        std::max(
            global_min_multiplier,
            get_parameter_or_default(
                parameters,
                "decomposition_modifier_global_max_multiplier",
                2.0));

    // -------------------------------------------------------------------------
    // Labile organic matter
    // More sensitive to flooding/anoxia and salinity than refractory material.
    // -------------------------------------------------------------------------
    const double labile_hydro_sensitivity =
        std::max(
            0.0,
            get_parameter_or_default(
                parameters,
                "decomposition_labile_hydro_sensitivity",
                2.5));

    const double labile_fine_sensitivity =
        std::max(
            0.0,
            get_parameter_or_default(
                parameters,
                "decomposition_labile_fine_sensitivity",
                0.8));

    const double labile_organic_sensitivity =
        std::max(
            0.0,
            get_parameter_or_default(
                parameters,
                "decomposition_labile_organic_sensitivity",
                0.6));

    const double labile_porosity_sensitivity =
        std::max(
            0.0,
            get_parameter_or_default(
                parameters,
                "decomposition_labile_porosity_sensitivity",
                0.4));

    const double labile_salinity_threshold_ppt =
        get_parameter_or_default(
            parameters,
            "decomposition_labile_salinity_threshold_ppt",
            18.0);

    const double labile_salinity_sensitivity =
        std::max(
            0.0,
            get_parameter_or_default(
                parameters,
                "decomposition_labile_salinity_sensitivity",
                0.035));

    const double labile_min_multiplier =
        clamp(
            get_parameter_or_default(
                parameters,
                "decomposition_labile_min_multiplier",
                global_min_multiplier),
            0.0,
            1.0);

    const double labile_hydro_multiplier =
        compute_hydrologic_oxygen_multiplier(
            hydro.inundation_fraction,
            surface,
            labile_hydro_sensitivity,
            labile_fine_sensitivity,
            labile_organic_sensitivity,
            labile_porosity_sensitivity,
            labile_min_multiplier);

    const double labile_salinity_multiplier =
        compute_salinity_multiplier(
            eco_state.root_zone_salinity_ppt,
            labile_salinity_threshold_ppt,
            labile_salinity_sensitivity,
            labile_min_multiplier);

    modifiers.labile_multiplier =
        clamp(
            labile_hydro_multiplier * labile_salinity_multiplier,
            global_min_multiplier,
            global_max_multiplier);

    // -------------------------------------------------------------------------
    // Refractory organic matter
    // Usually less sensitive than labile material.
    // -------------------------------------------------------------------------
    const double refractory_hydro_sensitivity =
        std::max(
            0.0,
            get_parameter_or_default(
                parameters,
                "decomposition_refractory_hydro_sensitivity",
                1.2));

    const double refractory_fine_sensitivity =
        std::max(
            0.0,
            get_parameter_or_default(
                parameters,
                "decomposition_refractory_fine_sensitivity",
                0.5));

    const double refractory_organic_sensitivity =
        std::max(
            0.0,
            get_parameter_or_default(
                parameters,
                "decomposition_refractory_organic_sensitivity",
                0.4));

    const double refractory_porosity_sensitivity =
        std::max(
            0.0,
            get_parameter_or_default(
                parameters,
                "decomposition_refractory_porosity_sensitivity",
                0.2));

    const double refractory_salinity_threshold_ppt =
        get_parameter_or_default(
            parameters,
            "decomposition_refractory_salinity_threshold_ppt",
            25.0);

    const double refractory_salinity_sensitivity =
        std::max(
            0.0,
            get_parameter_or_default(
                parameters,
                "decomposition_refractory_salinity_sensitivity",
                0.015));

    const double refractory_min_multiplier =
        clamp(
            get_parameter_or_default(
                parameters,
                "decomposition_refractory_min_multiplier",
                global_min_multiplier),
            0.0,
            1.0);

    const double refractory_hydro_multiplier =
        compute_hydrologic_oxygen_multiplier(
            hydro.inundation_fraction,
            surface,
            refractory_hydro_sensitivity,
            refractory_fine_sensitivity,
            refractory_organic_sensitivity,
            refractory_porosity_sensitivity,
            refractory_min_multiplier);

    const double refractory_salinity_multiplier =
        compute_salinity_multiplier(
            eco_state.root_zone_salinity_ppt,
            refractory_salinity_threshold_ppt,
            refractory_salinity_sensitivity,
            refractory_min_multiplier);

    modifiers.refractory_multiplier =
        clamp(
            refractory_hydro_multiplier * refractory_salinity_multiplier,
            global_min_multiplier,
            global_max_multiplier);

    // -------------------------------------------------------------------------
    // Root-derived organic matter
    // Often somewhat protected physically, but still sensitive to prolonged
    // flooding and salt stress.
    // -------------------------------------------------------------------------
    const double root_hydro_sensitivity =
        std::max(
            0.0,
            get_parameter_or_default(
                parameters,
                "decomposition_root_hydro_sensitivity",
                1.8));

    const double root_fine_sensitivity =
        std::max(
            0.0,
            get_parameter_or_default(
                parameters,
                "decomposition_root_fine_sensitivity",
                0.7));

    const double root_organic_sensitivity =
        std::max(
            0.0,
            get_parameter_or_default(
                parameters,
                "decomposition_root_organic_sensitivity",
                0.5));

    const double root_porosity_sensitivity =
        std::max(
            0.0,
            get_parameter_or_default(
                parameters,
                "decomposition_root_porosity_sensitivity",
                0.3));

    const double root_salinity_threshold_ppt =
        get_parameter_or_default(
            parameters,
            "decomposition_root_salinity_threshold_ppt",
            22.0);

    const double root_salinity_sensitivity =
        std::max(
            0.0,
            get_parameter_or_default(
                parameters,
                "decomposition_root_salinity_sensitivity",
                0.025));

    const double root_min_multiplier =
        clamp(
            get_parameter_or_default(
                parameters,
                "decomposition_root_min_multiplier",
                global_min_multiplier),
            0.0,
            1.0);

    const double root_hydro_multiplier =
        compute_hydrologic_oxygen_multiplier(
            hydro.inundation_fraction,
            surface,
            root_hydro_sensitivity,
            root_fine_sensitivity,
            root_organic_sensitivity,
            root_porosity_sensitivity,
            root_min_multiplier);

    const double root_salinity_multiplier =
        compute_salinity_multiplier(
            eco_state.root_zone_salinity_ppt,
            root_salinity_threshold_ppt,
            root_salinity_sensitivity,
            root_min_multiplier);

    modifiers.root_multiplier =
        clamp(
            root_hydro_multiplier * root_salinity_multiplier,
            global_min_multiplier,
            global_max_multiplier);

    return modifiers;
}
}
