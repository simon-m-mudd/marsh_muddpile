# output_reader.py
#
# Reads marsh_muddpile NetCDF output and computes diagnostics used for
# calibration against LTER observations.
#
# Key function: annual_anpp_g_m2_yr() computes annual aboveground net primary
# productivity (g m-2 yr-1) from the aboveground_growth_kg_m2_d time series.
# This matches the Smalley (1968) method used to measure ANPP at NI and PIE:
# both accumulate the sum of positive biomass increments over a growing season.
#
# Reference:
#   Smalley, A.E., 1958. The role of two invertebrate populations, Littorina
#   irrorata and Orchelimum fidicinium, in the energy flow of a salt marsh
#   ecosystem. PhD thesis, University of Georgia.

from __future__ import annotations
from typing import Dict, Optional

import numpy as np

try:
    import netCDF4 as nc
except ImportError as exc:
    raise ImportError("netCDF4 is required: pip install netCDF4") from exc


def read_time_series(nc_path: str) -> Dict[str, np.ndarray]:
    """Return all scalar time series from a marsh_muddpile NetCDF file.

    Keys match the variable names written by result_io::write_netcdf.
    Returns arrays as numpy float64.
    """
    series: Dict[str, np.ndarray] = {}

    with nc.Dataset(nc_path, "r") as ds:
        for name, var in ds.variables.items():
            if var.ndim == 1 and "time" in var.dimensions:
                series[name] = np.array(var[:], dtype=np.float64)

    return series


def annual_anpp_g_m2_yr(nc_path: str) -> np.ndarray:
    """Compute calendar-year ANPP (g m-2 yr-1) from model output.

    ANPP is computed as the annual integral of aboveground_growth_kg_m2_d,
    converted to g m-2.  Partial final years are included.

    Returns an array of length n_complete_years (or n_complete_years + 1 if
    there is a partial year at the end).
    """
    series = read_time_series(nc_path)

    time_days = series["model_time_days"]
    growth_kg_m2_d = series["aboveground_growth_kg_m2_d"]
    dt_days = np.diff(time_days, prepend=0.0)

    # Assign each step to a calendar year (0-based).
    year_index = (time_days / 365.25).astype(int)

    n_years = int(year_index[-1]) + 1
    anpp = np.zeros(n_years, dtype=np.float64)

    for i, yi in enumerate(year_index):
        # growth (kg m-2 d-1) * dt (d) * 1000 -> g m-2
        anpp[yi] += growth_kg_m2_d[i] * dt_days[i] * 1000.0

    return anpp


def mean_annual_anpp_g_m2_yr(
    nc_path: str,
    skip_spinup_years: int = 2,
) -> float:
    """Return the mean annual ANPP (g m-2 yr-1) after a spin-up period."""
    anpp = annual_anpp_g_m2_yr(nc_path)

    if len(anpp) <= skip_spinup_years:
        return float(np.mean(anpp))

    return float(np.mean(anpp[skip_spinup_years:]))


def summarise_run(nc_path: str, skip_spinup_years: int = 2) -> Dict[str, float]:
    """Return a dict of key scalar diagnostics averaged over the post-spinup period."""
    series = read_time_series(nc_path)

    time_days = series["model_time_days"]
    mask = time_days >= skip_spinup_years * 365.25

    def _mean(key: str) -> Optional[float]:
        if key not in series:
            return None
        return float(np.mean(series[key][mask]))

    return {
        "mean_anpp_g_m2_yr": mean_annual_anpp_g_m2_yr(nc_path, skip_spinup_years),
        "mean_surface_elevation_m": _mean("surface_elevation"),
        "mean_aboveground_biomass_kg_m2": _mean("aboveground_biomass_kg_m2"),
        "mean_belowground_biomass_kg_m2": _mean("belowground_biomass_kg_m2"),
        "mean_root_zone_salinity_ppt": _mean("root_zone_salinity_ppt"),
        "mean_inundation_fraction": _mean("inundation_fraction"),
        "mean_gpp_gC_m2_d": _mean("gpp_gC_m2_d"),
        "mean_lai": _mean("lai"),
    }
