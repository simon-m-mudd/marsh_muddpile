#pragma once

#include <optional>
#include <string>

namespace marsh_model
{
enum class material_category
{
    mineral,
    organic_labile,
    organic_refractory,
    live_root,
    isotope,
    tracer,
    other
};

struct decay_properties
{
    double k_0 = 0.0;
    double gamma = 0.0;
    bool temperature_sensitive = false;
};

struct compaction_properties
{
    double e_0 = 0.0;
    double compression_index = 0.0;
    double sigma_0 = 1.0;
};

struct settling_properties
{
    double diameter = 0.0;
    double settling_velocity = 0.0;
};

struct material_properties
{
    int id = -1;
    std::string name;
    material_category category = material_category::other;

    double density = 0.0;

    std::optional<decay_properties> decay;
    std::optional<compaction_properties> compaction;
    std::optional<settling_properties> settling;

    bool allow_surface_deposition = false;
    bool allow_root_input = false;
    bool track_age = false;
    bool track_osl_age = false;
};
}
