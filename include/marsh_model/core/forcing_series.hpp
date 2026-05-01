#pragma once

#include "marsh_model/core/forcing_step.hpp"
#include <vector>

namespace marsh_model
{
class forcing_series
{
public:
    forcing_series() = default;

    void add_step(const forcing_step& step);
    const forcing_step& at(int index) const;
    int size() const;

private:
    std::vector<forcing_step> steps_;
};
}
