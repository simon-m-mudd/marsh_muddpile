# download_era5_met.py
#
# Downloads ERA5-Land 2 m air temperature and surface solar radiation for
# the three LTER marsh sites (North Inlet SC, PIE MA, GCE GA).
#
# Prerequisites:
#   pip install cdsapi
#
# You also need a free Copernicus CDS account and a ~/.cdsapirc file:
#   url: https://cds.climate.copernicus.eu/api/v2
#   key: <your-uid>:<your-api-key>
# Get your key at: https://cds.climate.copernicus.eu/user/login
#
# Usage:
#   python download_era5_met.py                  # download all three sites
#   python download_era5_met.py --site ni        # one site only
#   python download_era5_met.py --start 2000 --end 2020
#   python download_era5_met.py --outdir /data/era5
#
# Output:
#   One NetCDF file per site, e.g. era5_land_ni_1985_2024.nc
#   Variables:
#     t2m   - 2 m air temperature (K)
#     ssrd  - surface solar radiation downwards (J m-2, accumulated per hour)
#
# Post-processing helpers at the bottom of this file convert those variables
# to the units expected by marsh_muddpile forcing steps:
#     temperature        degC
#     par_umol_m2_d      umol m-2 d-1

from __future__ import annotations
import argparse
import os
import sys
from dataclasses import dataclass
from typing import List


@dataclass
class SiteBox:
    """Bounding box and metadata for one LTER site."""
    key: str           # short identifier used in filenames
    label: str         # human-readable name
    # CDS area order: [north, west, south, east] in decimal degrees
    north: float
    west: float
    south: float
    east: float


# 1-degree buffer around each site's approximate centre coordinate.
SITES: List[SiteBox] = [
    SiteBox(
        key="ni",
        label="North Inlet, SC",
        north=34.0, west=-80.0, south=33.0, east=-79.0,
    ),
    SiteBox(
        key="pie",
        label="Plum Island Estuary, MA",
        north=43.0, west=-71.5, south=42.0, east=-70.5,
    ),
    SiteBox(
        key="gce",
        label="Georgia Coastal Ecosystems, GA",
        north=32.0, west=-82.0, south=31.0, east=-81.0,
    ),
]

# Variables to download.
ERA5_VARIABLES = [
    "2m_temperature",
    "surface_solar_radiation_downwards",
]


def download_site(
    site: SiteBox,
    start_year: int,
    end_year: int,
    outdir: str,
) -> str:
    """Submit a CDS request for one site and return the output file path."""
    try:
        import cdsapi
    except ImportError:
        sys.exit("cdsapi not found.  Run: pip install cdsapi")

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(
        outdir,
        f"era5_land_{site.key}_{start_year}_{end_year}.nc",
    )

    if os.path.exists(out_path):
        print(f"[{site.key}] already exists, skipping: {out_path}")
        return out_path

    print(f"[{site.key}] Requesting ERA5-Land for {site.label} "
          f"({start_year}-{end_year})...")

    years  = [str(y) for y in range(start_year, end_year + 1)]
    months = [f"{m:02d}" for m in range(1, 13)]
    days   = [f"{d:02d}" for d in range(1, 32)]
    # Hourly data - download every hour so daily means/totals can be computed.
    times  = [f"{h:02d}:00" for h in range(24)]

    client = cdsapi.Client()
    client.retrieve(
        "reanalysis-era5-land",
        {
            "variable": ERA5_VARIABLES,
            "year":   years,
            "month":  months,
            "day":    days,
            "time":   times,
            "area":   [site.north, site.west, site.south, site.east],
            "format": "netcdf",
        },
        out_path,
    )

    print(f"[{site.key}] Saved to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def era5_to_daily(nc_path: str) -> "pd.DataFrame":
    """Convert an ERA5-Land NetCDF file to a daily DataFrame.

    Returns a DataFrame indexed by date with columns:
        temperature_mean_c   - daily mean 2 m air temperature (degC)
        temperature_max_c    - daily maximum (degC)
        temperature_min_c    - daily minimum (degC)
        par_umol_m2_d        - daily PAR (umol m-2 d-1)

    Spatial averaging is applied across the bounding box so the result is
    a single time series representative of the site.

    Requires: netCDF4, numpy, pandas
    """
    try:
        import netCDF4 as nc
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "netCDF4, numpy, and pandas are required for post-processing"
        ) from exc

    with nc.Dataset(nc_path, "r") as ds:
        # Time axis
        time_var = ds.variables["time"]
        times = nc.num2date(time_var[:], time_var.units, time_var.calendar
                            if hasattr(time_var, "calendar") else "standard")

        # 2 m temperature (K) - spatially averaged
        t2m_K = np.array(ds.variables["t2m"][:])   # (time, lat, lon)
        t2m_K = t2m_K.reshape(t2m_K.shape[0], -1).mean(axis=1)  # spatial mean

        # Surface solar radiation downwards (J m-2, accumulated per hour)
        ssrd = np.array(ds.variables["ssrd"][:])
        ssrd = ssrd.reshape(ssrd.shape[0], -1).mean(axis=1)

    dates = np.array([t.date() for t in times])

    df_hourly = pd.DataFrame({
        "date":   dates,
        "t2m_K":  t2m_K,
        "ssrd_J": ssrd,
    })

    # Daily aggregation
    daily = df_hourly.groupby("date").agg(
        temperature_mean_K=("t2m_K", "mean"),
        temperature_max_K=("t2m_K", "max"),
        temperature_min_K=("t2m_K", "min"),
        ssrd_J_total=("ssrd_J", "sum"),   # total J m-2 d-1
    ).reset_index()

    daily["temperature_mean_c"] = daily["temperature_mean_K"] - 273.15
    daily["temperature_max_c"]  = daily["temperature_max_K"]  - 273.15
    daily["temperature_min_c"]  = daily["temperature_min_K"]  - 273.15

    # ssrd is accumulated J m-2 per hour; sum over 24 hours = J m-2 d-1.
    # Convert to PAR (umol m-2 d-1):
    #   PAR fraction of shortwave ~ 0.45 (dimensionless)
    #   1 W m-2 = 4.57 umol m-2 s-1 (McCree 1972)
    #   1 J m-2 d-1 = 1/86400 W m-2
    daily["par_umol_m2_d"] = (
        daily["ssrd_J_total"] / 86400.0   # -> W m-2
        * 0.45                            # -> PAR W m-2
        * 4.57                            # -> umol m-2 s-1
        * 86400.0                         # -> umol m-2 d-1
    )

    return daily[[
        "date",
        "temperature_mean_c",
        "temperature_max_c",
        "temperature_min_c",
        "par_umol_m2_d",
    ]]


def fit_seasonal_parameters(daily_df: "pd.DataFrame") -> dict:
    """Fit sinusoidal seasonal parameters from a daily DataFrame.

    Returns a dict with keys matching MetForcingParams field names:
        temperature_mean_c, temperature_amplitude_c, temperature_peak_day,
        par_mean_umol_m2_d, par_amplitude_umol_m2_d, par_peak_day

    These can be passed directly to MetForcingParams(**fit_seasonal_parameters(df)).

    Requires: numpy, scipy
    """
    try:
        import numpy as np
        from scipy.optimize import curve_fit
    except ImportError as exc:
        raise ImportError("numpy and scipy are required for seasonal fitting") from exc

    import pandas as pd

    df = daily_df.copy()
    df["doy"] = pd.to_datetime(df["date"]).dt.dayofyear

    def cosine_model(doy, mean, amplitude, peak_day):
        return mean + amplitude * np.cos(2 * np.pi * (doy - peak_day) / 365.25)

    def _fit(values, doy):
        p0 = [float(values.mean()), float(values.std()), 172.0]
        popt, _ = curve_fit(cosine_model, doy, values, p0=p0, maxfev=5000)
        return float(popt[0]), abs(float(popt[1])), float(popt[2]) % 365.25

    t_mean, t_amp, t_peak = _fit(
        df["temperature_mean_c"].values, df["doy"].values
    )
    p_mean, p_amp, p_peak = _fit(
        df["par_umol_m2_d"].values, df["doy"].values
    )

    return {
        "temperature_mean_c":      round(t_mean, 2),
        "temperature_amplitude_c": round(t_amp,  2),
        "temperature_peak_day":    round(t_peak, 1),
        "par_mean_umol_m2_d":      round(p_mean, 0),
        "par_amplitude_umol_m2_d": round(p_amp,  0),
        "par_peak_day":            round(p_peak, 1),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ERA5-Land temperature and solar radiation for LTER sites."
    )
    parser.add_argument(
        "--site",
        choices=[s.key for s in SITES] + ["all"],
        default="all",
        help="Which site to download (default: all).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1985,
        metavar="YEAR",
        help="First year to download (default: 1985).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=2024,
        metavar="YEAR",
        help="Last year to download (default: 2024).",
    )
    parser.add_argument(
        "--outdir",
        default=os.path.join(os.path.dirname(__file__), "..", "era5_data"),
        metavar="DIR",
        help="Output directory (default: ../era5_data/ relative to this script).",
    )
    parser.add_argument(
        "--process",
        action="store_true",
        help="After downloading, convert to daily CSV and print fitted seasonal parameters.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sites_to_run = (
        SITES if args.site == "all"
        else [s for s in SITES if s.key == args.site]
    )

    for site in sites_to_run:
        nc_path = download_site(site, args.start, args.end, args.outdir)

        if args.process and os.path.exists(nc_path):
            print(f"[{site.key}] Converting to daily values...")
            daily = era5_to_daily(nc_path)
            csv_path = nc_path.replace(".nc", "_daily.csv")
            daily.to_csv(csv_path, index=False)
            print(f"[{site.key}] Daily CSV: {csv_path}")

            params = fit_seasonal_parameters(daily)
            print(f"[{site.key}] Fitted seasonal parameters for MetForcingParams:")
            for k, v in params.items():
                print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
