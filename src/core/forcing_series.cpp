#include "marsh_model/core/forcing_series.hpp"

#include <stdexcept>

namespace marsh_model
{
void forcing_series::add_step(const forcing_step& step)
{
    steps_.push_back(step);
}

const forcing_step& forcing_series::at(int index) const
{
    if (index < 0 || index >= static_cast<int>(steps_.size()))
    {
        throw std::out_of_range("forcing index out of range");
    }

    return steps_[index];
}

int forcing_series::size() const
{
    return static_cast<int>(steps_.size());
}
}
