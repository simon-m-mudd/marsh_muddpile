#include "marsh_model/core/column_state.hpp"

#include <stdexcept>

namespace marsh_model
{
column_state::column_state(int n_layers, int n_materials)
{
    resize(n_layers, n_materials);
}

int column_state::n_layers() const
{
    return static_cast<int>(mass_.rows());
}

int column_state::n_materials() const
{
    return static_cast<int>(mass_.cols());
}

void column_state::resize(int n_layers, int n_materials)
{
    if (n_layers < 0 || n_materials < 0)
    {
        throw std::invalid_argument("negative dimensions are not allowed");
    }

    mass_ = Eigen::ArrayXXd::Zero(n_layers, n_materials);
    layer_thickness_ = Eigen::ArrayXd::Zero(n_layers);
    layer_porosity_ = Eigen::ArrayXd::Zero(n_layers);
    layer_top_elevation_ = Eigen::ArrayXd::Zero(n_layers);
    layer_age_ = Eigen::ArrayXd::Zero(n_layers);
}

void column_state::append_surface_layer(const Eigen::ArrayXd& deposited_mass)
{
    if (mass_.size() == 0)
    {
        resize(1, static_cast<int>(deposited_mass.size()));
        mass_.row(0) = deposited_mass.transpose();
        return;
    }

    if (deposited_mass.size() != mass_.cols())
    {
        throw std::invalid_argument("deposited_mass size does not match number of materials");
    }

    const int old_layers = n_layers();
    mass_.conservativeResize(old_layers + 1, Eigen::NoChange);
    mass_.row(old_layers) = deposited_mass.transpose();

    layer_thickness_.conservativeResize(old_layers + 1);
    layer_porosity_.conservativeResize(old_layers + 1);
    layer_top_elevation_.conservativeResize(old_layers + 1);
    layer_age_.conservativeResize(old_layers + 1);

    layer_thickness_(old_layers) = 0.0;
    layer_porosity_(old_layers) = 0.0;
    layer_top_elevation_(old_layers) = get_surface_elevation();
    layer_age_(old_layers) = 0.0;
}

void column_state::add_mass_to_layers(const Eigen::ArrayXXd& delta_mass)
{
    if (delta_mass.rows() != mass_.rows() || delta_mass.cols() != mass_.cols())
    {
        throw std::invalid_argument("delta_mass shape does not match state mass shape");
    }

    mass_ += delta_mass;
}

double column_state::get_total_mass(int material_index) const
{
    if (material_index < 0 || material_index >= n_materials())
    {
        throw std::out_of_range("material index out of range");
    }

    return mass_.col(material_index).sum();
}

Eigen::ArrayXd column_state::get_total_mass_by_material() const
{
    if (n_materials() == 0)
    {
        return Eigen::ArrayXd();
    }

    return mass_.colwise().sum().transpose();
}

double column_state::get_surface_elevation() const
{
    if (layer_top_elevation_.size() == 0)
    {
        return 0.0;
    }

    return layer_top_elevation_(layer_top_elevation_.size() - 1);
}

void column_state::set_layer_geometry(
    const Eigen::ArrayXd& layer_thickness,
    const Eigen::ArrayXd& layer_porosity,
    const Eigen::ArrayXd& layer_top_elevation)
{
    if (layer_thickness.size() != n_layers() ||
        layer_porosity.size() != n_layers() ||
        layer_top_elevation.size() != n_layers())
    {
        throw std::invalid_argument("geometry arrays must match number of layers");
    }

    layer_thickness_ = layer_thickness;
    layer_porosity_ = layer_porosity;
    layer_top_elevation_ = layer_top_elevation;
}

Eigen::ArrayXXd& column_state::mass()
{
    return mass_;
}

const Eigen::ArrayXXd& column_state::mass() const
{
    return mass_;
}

Eigen::ArrayXd& column_state::layer_thickness()
{
    return layer_thickness_;
}

const Eigen::ArrayXd& column_state::layer_thickness() const
{
    return layer_thickness_;
}

Eigen::ArrayXd& column_state::layer_porosity()
{
    return layer_porosity_;
}

const Eigen::ArrayXd& column_state::layer_porosity() const
{
    return layer_porosity_;
}

Eigen::ArrayXd& column_state::layer_top_elevation()
{
    return layer_top_elevation_;
}

const Eigen::ArrayXd& column_state::layer_top_elevation() const
{
    return layer_top_elevation_;
}

Eigen::ArrayXd& column_state::layer_age()
{
    return layer_age_;
}

const Eigen::ArrayXd& column_state::layer_age() const
{
    return layer_age_;
}
}
