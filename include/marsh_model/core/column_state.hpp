
// column_state.hpp
//
// Part of marsh_muddpile-- https://github.com/simon-m-mudd/marsh_muddpile
//
// Copyright (C) 2026 Simon M. Mudd
// Released under the GNU General Public Licence v3 (GPL-3.0)
// See LICENSE file or https://www.gnu.org/licenses/gpl-3.0.html
//
// -----------------------------------------------------------------------------

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

    // Porewater ammonium concentration, μmol L-1, per layer.
    // Zero indicates uninitialised; the NH4 porewater model initialises
    // from nh4_initial_umol_L on first use.
    Eigen::ArrayXd& porewater_nh4();
    const Eigen::ArrayXd& porewater_nh4() const;

    // Porewater sulfate concentration, μmol L-1, per layer.
    // Zero indicates uninitialised; the sulfate-methane model initialises
    // from ch4_initial_so4_umol_L on first use.
    Eigen::ArrayXd& porewater_so4();
    const Eigen::ArrayXd& porewater_so4() const;

    // Porewater methane concentration, μmol L-1, per layer.
    // Zero indicates uninitialised; the sulfate-methane model initialises
    // from ch4_initial_ch4_umol_L on first use.
    Eigen::ArrayXd& porewater_ch4();
    const Eigen::ArrayXd& porewater_ch4() const;

private:
    Eigen::ArrayXXd mass_;
    Eigen::ArrayXd layer_thickness_;
    Eigen::ArrayXd layer_porosity_;
    Eigen::ArrayXd layer_top_elevation_;
    Eigen::ArrayXd layer_age_;
    Eigen::ArrayXd porewater_nh4_umol_per_L_;
    Eigen::ArrayXd porewater_so4_umol_per_L_;
    Eigen::ArrayXd porewater_ch4_umol_per_L_;
};
}
