#!/usr/bin/env python3

# inspect_marsh_output.py
#
# Part of marsh_muddpile-- https://github.com/simon-m-mudd/marsh_muddpile
#
# Copyright (C) 2026 Simon M. Mudd
# Released under the GNU General Public Licence v3 (GPL-3.0)
# See LICENSE file or https://www.gnu.org/licenses/gpl-3.0.html
#
# -----------------------------------------------------------------------------
# this script inspects marsh_muddpile netcdf output and writes png figures.
#
# it is designed for headless environments and uses the matplotlib Agg backend.
#
# it will:
#   - print a summary of the dataset to screen
#   - save time-series figures to png
#   - save total-mass-by-material plots to png
#   - save snapshot profile figures to png
#
# -----------------------------------------------------------------------------

import argparse
import os
import sys

import numpy as np
from netCDF4 import Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def decode_material_names(var):
    data = var[:]

    # Expected shape: (material, name_strlen)
    names = []
    for row in data:
        chars = []
        for item in row:
            if isinstance(item, bytes):
                c = item.decode("utf-8", errors="ignore")
            else:
                c = chr(item) if isinstance(item, (int, np.integer)) else str(item)
            chars.append(c)
        name = "".join(chars).split("\x00")[0].strip()
        names.append(name)
    return names


def read_optional_variable(ds, name):
    if name in ds.variables:
        return ds.variables[name][:]
    return None


def ensure_output_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def print_summary(ds, material_names):
    print("\n=== marsh_muddpile output summary ===\n")

    print("dimensions:")
    for name, dim in ds.dimensions.items():
        print(f"  {name}: {len(dim)}")

    print("\nvariables:")
    for name, var in ds.variables.items():
        dims = ", ".join(var.dimensions)
        print(f"  {name}: dims=({dims})")

    print("\nmaterials:")
    for i, name in enumerate(material_names):
        print(f"  {i}: {name}")

    if "model_time_days" in ds.variables:
        time = ds.variables["model_time_days"][:]
        print(f"\ntime series length: {len(time)}")
        if len(time) > 0:
            print(f"  start time_days: {time[0]}")
            print(f"  end time_days:   {time[-1]}")

    if "snapshot_time_days" in ds.variables:
        snapshot_time = ds.variables["snapshot_time_days"][:]
        print(f"\nsnapshots saved: {len(snapshot_time)}")
        for i, t in enumerate(snapshot_time):
            print(f"  snapshot {i}: time_days={t}")

    print("")


def plot_time_series(ds, output_dir, prefix):
    time = read_optional_variable(ds, "model_time_days")
    if time is None or len(time) == 0:
        print("no time series found; skipping time-series plots")
        return

    surface_elevation = read_optional_variable(ds, "surface_elevation")
    peak_biomass = read_optional_variable(ds, "peak_biomass")
    aboveground_biomass = read_optional_variable(ds, "aboveground_biomass")
    belowground_biomass = read_optional_variable(ds, "belowground_biomass")
    belowground_mortality = read_optional_variable(ds, "belowground_mortality")
    n_layers = read_optional_variable(ds, "n_layers")

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.ravel()

    plots = [
        ("surface_elevation", surface_elevation, "surface elevation", "m"),
        ("peak_biomass", peak_biomass, "peak biomass", "g m$^{-2}$"),
        ("aboveground_biomass", aboveground_biomass, "aboveground biomass", "g m$^{-2}$"),
        ("belowground_biomass", belowground_biomass, "belowground biomass", "kg m$^{-2}$"),
        ("belowground_mortality", belowground_mortality, "belowground mortality", "kg m$^{-2}$"),
        ("n_layers", n_layers, "number of layers", "count"),
    ]

    for ax, (var_name, values, title, ylabel) in zip(axes, plots):
        if values is not None:
            ax.plot(time, values, lw=2)
            ax.set_title(title)
            ax.set_xlabel("model time (days)")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
        else:
            ax.set_visible(False)

    fig.tight_layout()
    file_name = os.path.join(output_dir, f"{prefix}_time_series.png")
    fig.savefig(file_name, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {file_name}")


def plot_total_mass_by_material(ds, material_names, output_dir, prefix):
    time = read_optional_variable(ds, "model_time_days")
    total_mass = read_optional_variable(ds, "total_mass_by_material")

    if time is None or total_mass is None:
        print("no total_mass_by_material found; skipping material mass plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    n_materials = total_mass.shape[1]
    for i in range(n_materials):
        label = material_names[i] if i < len(material_names) else f"material_{i}"
        ax.plot(time, total_mass[:, i], lw=2, label=label)

    ax.set_xlabel("model time (days)")
    ax.set_ylabel("total mass (kg m$^{-2}$)")
    ax.set_title("total mass by material through time")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    file_name = os.path.join(output_dir, f"{prefix}_total_mass_by_material.png")
    fig.savefig(file_name, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {file_name}")


def compute_snapshot_midpoint_depths(layer_top_elevation, layer_thickness, n_layers):
    valid_top = layer_top_elevation[:n_layers]
    valid_thickness = layer_thickness[:n_layers]

    if n_layers == 0:
        return np.array([])

    surface_elevation = valid_top[-1]
    midpoint_depth = surface_elevation - valid_top + 0.5 * valid_thickness
    return midpoint_depth


def plot_snapshot_profiles(ds, material_names, output_dir, prefix, max_snapshots=None):
    snapshot_time = read_optional_variable(ds, "snapshot_time_days")
    snapshot_n_layers = read_optional_variable(ds, "snapshot_n_layers")
    layer_thickness = read_optional_variable(ds, "layer_thickness")
    layer_porosity = read_optional_variable(ds, "layer_porosity")
    layer_top_elevation = read_optional_variable(ds, "layer_top_elevation")
    layer_age = read_optional_variable(ds, "layer_age")
    layer_mass = read_optional_variable(ds, "layer_mass")

    if snapshot_time is None or snapshot_n_layers is None:
        print("no snapshots found; skipping snapshot profile plots")
        return

    n_snapshots = len(snapshot_time)
    indices = list(range(n_snapshots))

    if max_snapshots is not None and max_snapshots > 0:
        indices = indices[:max_snapshots]

    for s in indices:
        n_layers = int(snapshot_n_layers[s])

        if n_layers <= 0:
            continue

        depth = compute_snapshot_midpoint_depths(
            layer_top_elevation[s, :],
            layer_thickness[s, :],
            n_layers)

        porosity = layer_porosity[s, :n_layers]
        age_days = layer_age[s, :n_layers]

        fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True)

        # Porosity profile
        axes[0].plot(porosity, depth, lw=2)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("porosity")
        axes[0].set_ylabel("depth below surface (m)")
        axes[0].set_title("porosity profile")
        axes[0].grid(True, alpha=0.3)

        # Age profile
        axes[1].plot(age_days, depth, lw=2)
        axes[1].set_xlabel("layer age (days)")
        axes[1].set_title("age profile")
        axes[1].grid(True, alpha=0.3)

        # Material mass profiles
        if layer_mass is not None:
            n_materials = layer_mass.shape[2]
            for m in range(n_materials):
                label = material_names[m] if m < len(material_names) else f"material_{m}"
                axes[2].plot(layer_mass[s, :n_layers, m], depth, lw=1.5, label=label)

        axes[2].set_xlabel("layer mass (kg m$^{-2}$)")
        axes[2].set_title("material mass profiles")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="best", fontsize=7)

        fig.suptitle(f"snapshot {s} at time_days={snapshot_time[s]}")
        fig.tight_layout()

        file_name = os.path.join(output_dir, f"{prefix}_snapshot_{s:03d}.png")
        fig.savefig(file_name, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"wrote {file_name}")


def main():
    parser = argparse.ArgumentParser(
        description="inspect and visualise marsh_muddpile NetCDF output")
    parser.add_argument("input_file", help="NetCDF file produced by marsh_muddpile")
    parser.add_argument(
        "--output_dir",
        default="marsh_output_figures",
        help="directory for png outputs")
    parser.add_argument(
        "--prefix",
        default="marsh_output",
        help="prefix for output png files")
    parser.add_argument(
        "--max_snapshots",
        type=int,
        default=0,
        help="maximum number of snapshot figures to write (0 = all)")

    args = parser.parse_args()

    ensure_output_dir(args.output_dir)

    with Dataset(args.input_file, "r") as ds:
        if "material_name" in ds.variables:
            material_names = decode_material_names(ds.variables["material_name"])
        else:
            n_materials = len(ds.dimensions["material"]) if "material" in ds.dimensions else 0
            material_names = [f"material_{i}" for i in range(n_materials)]

        print_summary(ds, material_names)
        plot_time_series(ds, args.output_dir, args.prefix)
        plot_total_mass_by_material(ds, material_names, args.output_dir, args.prefix)

        max_snapshots = None if args.max_snapshots == 0 else args.max_snapshots
        plot_snapshot_profiles(
            ds,
            material_names,
            args.output_dir,
            args.prefix,
            max_snapshots=max_snapshots)

    print("\nfinished")


if __name__ == "__main__":
    main()
