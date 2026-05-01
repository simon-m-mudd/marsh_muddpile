#pragma once

#include <string>
#include <unordered_map>

namespace marsh_model
{
class parameter_set
{
public:
    void set(const std::string& name, double value);
    double get(const std::string& name) const;
    bool has(const std::string& name) const;

private:
    std::unordered_map<std::string, double> values_;
};
}
