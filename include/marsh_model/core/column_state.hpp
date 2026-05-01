#pragma once

#include <Eigen/Core>

namespace marsh_model
{
class column_state
{
public:
    column_state() = default;
    column_state(int n_layers, int n_materials);

    int n_layers() const;
    int n_materials() const;

    void resize(int n_layers, int n_materials);
    void append_surface_layer(const Eigen::ArrayXd& deposited_mass);
    void add_mass_to_layers(const Eigen::ArrayXXd& delta_mass);

    double get_total_mass(int material_index) const;
    Eigen::ArrayXd get_total_mass_by_material() const;
    double get_surface_elevation() const;

    void set_layer_geometry(
        const Eigen::ArrayXd& layer_thickness,
        const Eigen::ArrayXd& layer_porosity,
        const Eigen::ArrayXd& layer_top_elevation);

    Eigen::ArrayXXd& mass();
    const Eigen::ArrayXXd& mass() const;

    Eigen::ArrayXd& layer_thickness();
    const Eigen::ArrayXd& layer_thickness() const;

    Eigen::ArrayXd& layer_porosity();
    const Eigen::ArrayXd& layer_porosity() const;

    Eigen::ArrayXd& layer_top_elevation();
    const Eigen::ArrayXd& layer_top_elevation() const;

    Eigen::ArrayXd& layer_age();
    const Eigen::ArrayXd& layer_age() const;

private:
    Eigen::ArrayXXd mass_;
    Eigen::ArrayXd layer_thickness_;
    Eigen::ArrayXd layer_porosity_;
    Eigen::ArrayXd layer_top_elevation_;
    Eigen::ArrayXd layer_age_;
};
}
