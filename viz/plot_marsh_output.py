#!/usr/bin/env python3

import argparse
import pathlib
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def pick_var(ds, candidates):
    for name in candidates:
        if name in ds.variables:
            return name
    return None


def get_material_names(ds):
    if "material_name" not in ds.variables:
        n = ds.sizes.get("material", 0)
        return [f"material_{i}" for i in range(n)]

    arr = ds["material_name"].values

    # Case 1: NC_STRING or already decoded string array
    if arr.dtype.kind in ("U", "O"):
        return [str(x) for x in arr]

    # Case 2: char array shape (material, name_strlen)
    names = []
    if arr.ndim == 2:
        for row in arr:
            chars = []
            for c in row:
                if isinstance(c, bytes):
                    c = c.decode("utf-8", errors="ignore")
                else:
                    c = str(c)
                if c == "\x00":
                    continue
                chars.append(c)
            names.append("".join(chars).strip())
        return names

    n = ds.sizes.get("material", 0)
    return [f"material_{i}" for i in range(n)]


def ensure_outdir(path):
    path.mkdir(parents=True, exist_ok=True)


def print_summary(ds):
    print("\nDataset summary")
    print("================")
    print(ds)
    print("\nVariables:")
    for v in ds.data_vars:
        print(f"  {v}: dims={ds[v].dims}, shape={ds[v].shape}")
    print("\nCoordinates:")
    for c in ds.coords:
        print(f"  {c}: dims={ds[c].dims}, shape={ds[c].shape}")
    print()


def plot_time_series(ds, outdir):
    time_name = pick_var(ds, ["time", "model_time_days"])
    if time_name is None:
        print("No time variable found; skipping time-series plots.")
        return

    t = ds[time_name].values

    series_to_plot = [
        ("surface_elevation", "Surface elevation", "m"),
        ("root_zone_salinity_ppt", "Root-zone salinity", "ppt"),
        ("lai", "LAI", "-"),
        ("gpp_gC_m2_d", "GPP", "gC m$^{-2}$ d$^{-1}$"),
        ("npp_gC_m2_d", "NPP", "gC m$^{-2}$ d$^{-1}$"),
        ("et_total_mm_d", "ET total", "mm d$^{-1}$"),
        ("et_transpiration_mm_d", "ET transpiration", "mm d$^{-1}$"),
        ("et_evaporation_mm_d", "ET evaporation", "mm d$^{-1}$"),
        ("inundation_fraction", "Inundation fraction", "-"),
        ("mean_inundation_depth_m", "Mean inundation depth", "m"),
        ("mean_water_level_m", "Mean water level", "m"),
        ("max_water_level_m", "Max water level", "m"),
        ("peak_biomass", "Peak biomass", "g m$^{-2}$"),
        ("aboveground_biomass", "Aboveground biomass", "g m$^{-2}$"),
        ("belowground_biomass", "Belowground biomass", "kg m$^{-2}$"),
        ("belowground_mortality", "Belowground mortality", "kg m$^{-2}$"),
        ("aboveground_biomass_kg_m2", "Aboveground biomass", "kg m$^{-2}$"),
        ("belowground_biomass_kg_m2", "Belowground biomass", "kg m$^{-2}$"),
    ]

    available = [(name, title, units) for name, title, units in series_to_plot if name in ds.variables]
    if not available:
        print("No recognised time-series variables found.")
        return

    n = len(available)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax, (name, title, units) in zip(axes, available):
        y = ds[name].values
        ax.plot(t, y, lw=1.8)
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel(units)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(available):]:
        ax.axis("off")

    fig.tight_layout()
    outfile = outdir / "time_series_overview.png"
    fig.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {outfile}")


def plot_total_mass_by_material(ds, outdir):
    varname = pick_var(ds, ["total_mass_by_material", "final_total_mass_by_material_kg_m2"])
    if varname is None:
        print("No total-mass-by-material variable found; skipping material-mass plot.")
        return

    material_names = get_material_names(ds)

    arr = ds[varname].values
    if arr.ndim == 1:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(arr.shape[0])
        ax.bar(x, arr)
        ax.set_xticks(x)
        ax.set_xticklabels(material_names[: len(x)], rotation=45, ha="right")
        ax.set_ylabel("kg m$^{-2}$")
        ax.set_title("Final total mass by material")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        outfile = outdir / "final_total_mass_by_material.png"
        fig.savefig(outfile, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {outfile}")
        return

    time_name = pick_var(ds, ["time", "model_time_days"])
    if time_name is None:
        print("No time variable found for total_mass_by_material; skipping.")
        return

    t = ds[time_name].values
    fig, ax = plt.subplots(figsize=(10, 5))

    for i in range(arr.shape[1]):
        label = material_names[i] if i < len(material_names) else f"material_{i}"
        ax.plot(t, arr[:, i], label=label, lw=1.5)

    ax.set_title("Total mass by material through time")
    ax.set_xlabel("Time")
    ax.set_ylabel("kg m$^{-2}$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    outfile = outdir / "total_mass_by_material_timeseries.png"
    fig.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {outfile}")


def plot_final_state_profiles(ds, outdir):
    thickness_name = pick_var(ds, ["final_layer_thickness_m"])
    porosity_name = pick_var(ds, ["final_layer_porosity"])
    age_name = pick_var(ds, ["final_layer_age_days"])
    mass_name = pick_var(ds, ["final_mass_kg_m2"])

    if thickness_name is None:
        print("No final-state layer profile found; skipping final-state plots.")
        return

    thickness = ds[thickness_name].values
    n_layers = len(thickness)

    top = np.cumsum(thickness)
    bottom = np.concatenate(([0.0], top[:-1]))
    mid = 0.5 * (top + bottom)
    depth = top[-1] - mid

    material_names = get_material_names(ds)

    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)

    if porosity_name is not None:
        porosity = ds[porosity_name].values
        axes[0].plot(porosity, depth, marker="o")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Porosity")
    axes[0].set_ylabel("Depth below surface (m)")
    axes[0].set_title("Final porosity profile")
    axes[0].grid(True, alpha=0.3)

    if age_name is not None:
        age = ds[age_name].values
        axes[1].plot(age, depth, marker="o")
    axes[1].set_xlabel("Age (days)")
    axes[1].set_title("Final age profile")
    axes[1].grid(True, alpha=0.3)

    if mass_name is not None:
        mass = ds[mass_name].values
        for i in range(min(mass.shape[1], len(material_names))):
            axes[2].plot(mass[:, i], depth, marker="o", label=material_names[i])
        if mass.shape[1] > 0:
            axes[2].legend(fontsize=8)
    axes[2].set_xlabel("Mass (kg m$^{-2}$)")
    axes[2].set_title("Final layer mass profile")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    outfile = outdir / "final_state_profiles.png"
    fig.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {outfile}")


def plot_latest_snapshot(ds, outdir):
    time_name = pick_var(ds, ["snapshot_time_days"])
    count_name = pick_var(ds, ["snapshot_n_layers", "snapshot_layer_count"])
    thickness_name = pick_var(ds, ["layer_thickness", "snapshot_layer_thickness_m"])
    porosity_name = pick_var(ds, ["layer_porosity", "snapshot_layer_porosity"])
    age_name = pick_var(ds, ["layer_age", "snapshot_layer_age_days"])
    mass_name = pick_var(ds, ["layer_mass", "snapshot_mass_kg_m2"])

    if time_name is None or count_name is None or thickness_name is None:
        print("No snapshot data found; skipping snapshot plots.")
        return

    snapshot_times = ds[time_name].values
    layer_counts = ds[count_name].values.astype(int)
    s = len(snapshot_times) - 1
    nl = layer_counts[s]

    thickness = ds[thickness_name].values[s, :nl]
    top = np.cumsum(thickness)
    bottom = np.concatenate(([0.0], top[:-1]))
    mid = 0.5 * (top + bottom)
    depth = top[-1] - mid

    material_names = get_material_names(ds)

    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)

    if porosity_name is not None:
        porosity = ds[porosity_name].values[s, :nl]
        axes[0].plot(porosity, depth, marker="o")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Porosity")
    axes[0].set_ylabel("Depth below surface (m)")
    axes[0].set_title(f"Snapshot porosity\n t={snapshot_times[s]:.1f} d")
    axes[0].grid(True, alpha=0.3)

    if age_name is not None:
        age = ds[age_name].values[s, :nl]
        axes[1].plot(age, depth, marker="o")
    axes[1].set_xlabel("Age (days)")
    axes[1].set_title("Snapshot age")
    axes[1].grid(True, alpha=0.3)

    if mass_name is not None:
        mass = ds[mass_name].values[s, :nl, :]
        for i in range(min(mass.shape[1], len(material_names))):
            axes[2].plot(mass[:, i], depth, marker="o", label=material_names[i])
        if mass.shape[1] > 0:
            axes[2].legend(fontsize=8)
    axes[2].set_xlabel("Mass (kg m$^{-2}$)")
    axes[2].set_title("Snapshot layer mass")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    outfile = outdir / "latest_snapshot_profiles.png"
    fig.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {outfile}")


def main():
    parser = argparse.ArgumentParser(description="Visualise marsh_muddpile NetCDF output.")
    parser.add_argument("nc_file", help="Path to NetCDF output file")
    parser.add_argument("--outdir", default="figures", help="Directory to save figures")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    args = parser.parse_args()

    nc_path = pathlib.Path(args.nc_file)
    outdir = pathlib.Path(args.outdir)
    ensure_outdir(outdir)

    ds = xr.open_dataset(nc_path, decode_times=False)


    print_summary(ds)
    plot_time_series(ds, outdir)
    plot_total_mass_by_material(ds, outdir)
    plot_final_state_profiles(ds, outdir)
    plot_latest_snapshot(ds, outdir)

    if args.show:
        plt.show()

    ds.close()


if __name__ == "__main__":
    main()
