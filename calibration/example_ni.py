# example_ni.py
#
# End-to-end example: run the model for one North Inlet plot and compare
# the simulated annual ANPP against LTER observations.
#
# PLACEHOLDERS throughout - replace values marked PLACEHOLDER before use.
# Run from the calibration/ directory:
#
#   cd calibration
#   python example_ni.py
#
# Prerequisites:
#   - marsh_cli on PATH (or edit CLI_BINARY below)
#   - Python packages: pyyaml, netCDF4, numpy, pandas

import os
import sys

# Allow imports from this directory regardless of working directory.
sys.path.insert(0, os.path.dirname(__file__))

from site_config import (
    PlotConfig,
    north_inlet_default_tides,
    north_inlet_default_met,
)
from yaml_writer import write_config
from model_runner import run_model
from output_reader import annual_anpp_g_m2_yr, summarise_run
from lter_reader import read_ni_annual_productivity

# ---------------------------------------------------------------------------
# Configuration - edit these to match a real NI plot
# ---------------------------------------------------------------------------

CLI_BINARY = "marsh_cli"   # path to the compiled marsh_cli binary

# Surface elevation from lidar (m, relative to MSL).
# PLACEHOLDER: derive from site lidar data for the specific plot.
PLOT_SURFACE_ELEVATION_M = 0.30   # PLACEHOLDER

# Distance from the nearest tidal creek (m).
# PLACEHOLDER: measure from a map or lidar-derived creek network.
PLOT_DISTANCE_FROM_CREEK_M = 75.0   # PLACEHOLDER

RUN_YEARS = 30
OUTPUT_DIR = "calibration_runs"
RUN_ID = "ni_example_plot"

# ---------------------------------------------------------------------------
# Build and run
# ---------------------------------------------------------------------------

def main() -> None:
    config = PlotConfig(
        site_name="north_inlet",
        plot_id=RUN_ID,
        surface_elevation_m=PLOT_SURFACE_ELEVATION_M,
        distance_from_creek_m=PLOT_DISTANCE_FROM_CREEK_M,
        tides=north_inlet_default_tides(),
        met=north_inlet_default_met(),
        n_years=RUN_YEARS,
        output_dir=OUTPUT_DIR,
    )

    yaml_path = os.path.join(OUTPUT_DIR, f"{RUN_ID}.yaml")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Writing config: {yaml_path}")
    write_config(config, yaml_path)

    print(f"Running model...")
    nc_path = run_model(yaml_path, cli_binary=CLI_BINARY, silent=False)
    print(f"Output: {nc_path}")

    # -----------------------------------------------------------------------
    # Compare modelled ANPP against NI observations
    # -----------------------------------------------------------------------
    modelled_anpp = annual_anpp_g_m2_yr(nc_path)
    print("\nModelled annual ANPP (g m-2 yr-1):")
    for yr, val in enumerate(modelled_anpp):
        print(f"  year {yr:2d}: {val:.1f}")

    diag = summarise_run(nc_path, skip_spinup_years=2)
    print("\nPost-spinup mean diagnostics:")
    for k, v in diag.items():
        if v is not None:
            print(f"  {k}: {v:.3f}")

    # Load NI control ANPP for comparison (all years, all locations).
    obs = read_ni_annual_productivity(treatment="C")
    print(f"\nNI observed ANPP (control, all locations):")
    print(obs[["SITE", "LOCATION", "YEAR", "PRODUCTIVITY", "STDERR"]].to_string(index=False))

    # Simple comparison: mean modelled vs mean observed
    mean_mod = float(modelled_anpp[2:].mean()) if len(modelled_anpp) > 2 else float(modelled_anpp.mean())
    mean_obs = float(obs["PRODUCTIVITY"].mean())
    print(f"\nMean modelled ANPP (post-spinup): {mean_mod:.1f} g m-2 yr-1")
    print(f"Mean observed ANPP (all NI control): {mean_obs:.1f} g m-2 yr-1")
    print(f"Ratio (mod/obs): {mean_mod / mean_obs:.2f}")


if __name__ == "__main__":
    main()
