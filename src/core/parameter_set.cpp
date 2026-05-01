#include "marsh_model/core/parameter_set.hpp"

#include <stdexcept>

namespace marsh_model
{
void parameter_set::set(const std::string& name, double value)
{
    values_[name] = value;
}

double parameter_set::get(const std::string& name) const
{
    auto it = values_.find(name);
    if (it == values_.end())
    {
        throw std::out_of_range("missing parameter: " + name);
    }

    return it->second;
}

bool parameter_set::has(const std::string& name) const
{
    return values_.find(name) != values_.end();
}
}
