#!/usr/bin/env python3
"""
write_yaml.py  —  Generate a marsh_muddpile run YAML from site_config.py

Usage
-----
# Single plot, specifying surface elevation (required):
python write_yaml.py --site ni --elevation 0.30

# Multiple elevations (one YAML per elevation):
python write_yaml.py --site ni --elevation 0.15 0.25 0.35 0.45

# All three sites at one elevation each:
python write_yaml.py --site ni  --elevation 0.30
python write_yaml.py --site pie --elevation 0.80
python write_yaml.py --site gce --elevation 0.50

# Override optional fields:
python write_yaml.py --site ni --elevation 0.30 \
    --distance 100 --years 20 --outdir runs/

# Run from the calibration/ directory:
cd calibration && python write_yaml.py --site ni --elevation 0.30

Sites
-----
  ni   North Inlet, SC      (NOAA 8661070, ERA5 1985-2024)
  pie  Plum Island Est., MA (NOAA 8419870, ERA5 1985-2024)
  gce  GCE Sapelo Island, GA(NOAA 8670870, ERA5 1985-2024)
"""
from __future__ import annotations

import argparse
import os
import sys

# Allow running from the project root or from calibration/
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from site_config import (
    PlotConfig,
    north_inlet_default_tides, north_inlet_default_met,
    pie_default_tides,         pie_default_met,
    gce_default_tides,         gce_default_met,
)
from yaml_writer import write_config


# Map site key → (site_name string, tides factory, met factory)
SITES = {
    "ni": (
        "north_inlet",
        north_inlet_default_tides,
        north_inlet_default_met,
    ),
    "pie": (
        "pie",
        pie_default_tides,
        pie_default_met,
    ),
    "gce": (
        "gce",
        gce_default_tides,
        gce_default_met,
    ),
}


def make_config(
    site_key: str,
    elevation_m: float,
    distance_m: float,
    n_years: int,
    outdir: str,
) -> PlotConfig:
    site_name, tides_fn, met_fn = SITES[site_key]
    plot_id = f"{site_key}_elev{elevation_m:.3f}".replace(".", "p")
    return PlotConfig(
        site_name=site_name,
        plot_id=plot_id,
        surface_elevation_m=elevation_m,
        distance_from_creek_m=distance_m,
        tides=tides_fn(),
        met=met_fn(),
        n_years=n_years,
        output_dir=outdir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write marsh_muddpile YAML run files from site_config.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--site",
        choices=list(SITES),
        required=True,
        help="Site key: ni, pie, or gce.",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        nargs="+",
        required=True,
        metavar="M",
        help="Surface elevation(s) relative to MSL (m).  One YAML per value.",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=50.0,
        metavar="M",
        help="Distance from nearest tidal creek (m, default: 50).",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=10,
        metavar="N",
        help="Number of model years to simulate (default: 10).",
    )
    parser.add_argument(
        "--outdir",
        default=os.path.join(_here, "calibration_runs"),
        metavar="DIR",
        help="Directory for YAML (and NetCDF) output (default: calibration/calibration_runs/).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    for elev in args.elevation:
        config = make_config(
            site_key=args.site,
            elevation_m=elev,
            distance_m=args.distance,
            n_years=args.years,
            outdir=args.outdir,
        )
        yaml_path = os.path.join(args.outdir, f"{config.plot_id}.yaml")
        write_config(config, yaml_path)
        print(f"Wrote: {yaml_path}")
        print(f"  site={config.site_name}  elevation={elev:.3f} m  "
              f"distance={args.distance:.0f} m  years={args.years}")


if __name__ == "__main__":
    main()
