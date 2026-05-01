#pragma once

#include "marsh_model/core/material_properties.hpp"
#include <string>
#include <unordered_map>
#include <vector>

namespace marsh_model
{
class material_catalog
{
public:
    material_catalog() = default;

    int add_material(const material_properties& material);

    bool has_material(const std::string& name) const;
    int get_material_index(const std::string& name) const;

    const material_properties& get_material(int index) const;
    const material_properties& get_material(const std::string& name) const;

    int size() const;
    const std::vector<material_properties>& materials() const;

private:
    std::vector<material_properties> materials_;
    std::unordered_map<std::string, int> index_by_name_;
};
}
