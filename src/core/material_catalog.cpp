#include "marsh_model/core/material_catalog.hpp"

#include <stdexcept>

namespace marsh_model
{
int material_catalog::add_material(const material_properties& material)
{
    if (material.name.empty())
    {
        throw std::invalid_argument("material name cannot be empty");
    }

    if (has_material(material.name))
    {
        throw std::invalid_argument("duplicate material name: " + material.name);
    }

    material_properties stored = material;
    stored.id = static_cast<int>(materials_.size());

    materials_.push_back(stored);
    index_by_name_[stored.name] = stored.id;

    return stored.id;
}

bool material_catalog::has_material(const std::string& name) const
{
    return index_by_name_.find(name) != index_by_name_.end();
}

int material_catalog::get_material_index(const std::string& name) const
{
    auto it = index_by_name_.find(name);
    if (it == index_by_name_.end())
    {
        throw std::out_of_range("unknown material name: " + name);
    }

    return it->second;
}

const material_properties& material_catalog::get_material(int index) const
{
    if (index < 0 || index >= static_cast<int>(materials_.size()))
    {
        throw std::out_of_range("material index out of range");
    }

    return materials_[index];
}

const material_properties& material_catalog::get_material(const std::string& name) const
{
    return get_material(get_material_index(name));
}

int material_catalog::size() const
{
    return static_cast<int>(materials_.size());
}

const std::vector<material_properties>& material_catalog::materials() const
{
    return materials_;
}
}
