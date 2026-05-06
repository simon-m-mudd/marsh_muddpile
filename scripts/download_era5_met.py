# download_era5_met.py
#
# Downloads ERA5-Land 2 m air temperature, surface solar radiation, and
# precipitation for the three LTER marsh sites via the Open-Meteo archive API.
#
# No account, no API key, no cdsapi required.
# Open-Meteo serves the same ERA5-Land reanalysis (9 km, hourly) for free.
#
# Usage:
#   python download_era5_met.py                  # download all three sites
#   python download_era5_met.py --site ni        # one site only
#   python download_era5_met.py --start 2000 --end 2020
#   python download_era5_met.py --outdir /data/era5
#   python download_era5_met.py --process        # fit seasonal parameters too
#
# Output:
#   One CSV per site: era5_land_<key>_<start>_<end>_daily.csv
#   Columns:
#     date                  YYYY-MM-DD
#     temperature_mean_c    daily mean 2 m air temperature (degC)
#     temperature_max_c     daily maximum (degC)
#     temperature_min_c     daily minimum (degC)
#     par_umol_m2_d         daily PAR (umol m-2 d-1)
#     precipitation_mm_d    daily total precipitation (mm d-1)

from __future__ import annotations
import argparse
import os
import time
from dataclasses import dataclass
from typing import List

import requests
import pandas as pd


OPENMETEO_URL = "https://archive-api.open-meteo.com/v1/archive"

# ERA5-Land serves temperature at 9 km resolution.
# Radiation and precipitation are only available in the base ERA5 model
# (0.25°); two separate requests per chunk are made and then merged.
VARS_ERA5_LAND = ["temperature_2m"]
VARS_ERA5      = ["shortwave_radiation",   # W m-2 (mean over the hour)
                  "precipitation"]         # mm (accumulated in the hour)

# PAR conversion constants
_PAR_FRACTION = 0.45          # fraction of shortwave that is PAR
_PAR_UMOL_PER_W = 4.57        # umol m-2 s-1 per W m-2 (McCree 1972)
_SECONDS_PER_DAY = 86400.0


@dataclass
class SiteBox:
    """Representative point and metadata for one LTER site."""
    key:   str    # short identifier used in filenames
    label: str    # human-readable name
    lat:   float  # representative latitude  (decimal degrees, WGS84)
    lon:   float  # representative longitude (decimal degrees, WGS84)


SITES: List[SiteBox] = [
    SiteBox(
        key="ni",
        label="North Inlet, SC",
        lat=33.35,
        lon=-79.18,
    ),
    SiteBox(
        key="pie",
        label="Plum Island Estuary, MA",
        lat=42.72,
        lon=-70.87,
    ),
    SiteBox(
        key="gce",
        label="Georgia Coastal Ecosystems, GA",
        lat=31.43,
        lon=-81.28,
    ),
]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _get(lat: float, lon: float, start_year: int, end_year: int,
         variables: list, model: str,
         max_retries: int = 4, retry_wait_s: int = 90) -> dict:
    """Single Open-Meteo request with retry on 429.  Returns the hourly dict."""
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": f"{start_year}-01-01",
        "end_date":   f"{end_year}-12-31",
        "hourly":     ",".join(variables),
        "timezone":   "UTC",
        "models":     model,
    }
    for attempt in range(max_retries):
        r = requests.get(OPENMETEO_URL, params=params, timeout=120)
        if r.status_code == 429:
            wait = retry_wait_s * (attempt + 1)
            print(f"  Rate limited (429) — waiting {wait}s before retry "
                  f"{attempt + 1}/{max_retries}...")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()["hourly"]
    raise RuntimeError(
        f"Open-Meteo returned 429 on every attempt for {start_year}–{end_year}. "
        "Try again later or reduce --chunk-years."
    )


def _fetch_year_chunk(lat: float, lon: float,
                      start_year: int, end_year: int,
                      max_retries: int = 4,
                      retry_wait_s: int = 90) -> pd.DataFrame:
    """
    Fetch hourly data for one time chunk using two model requests:
      - temperature_2m        from era5_land  (9 km resolution)
      - shortwave_radiation   from era5        (radiation not in era5_land)
      - precipitation         from era5

    Returns a daily-aggregated DataFrame.
    """
    print(f"  Fetching temperature (era5_land)...")
    h_land = _get(lat, lon, start_year, end_year,
                  VARS_ERA5_LAND, "era5_land", max_retries, retry_wait_s)
    time.sleep(1)   # brief pause between the two requests

    print(f"  Fetching radiation + precipitation (era5)...")
    h_era5 = _get(lat, lon, start_year, end_year,
                  VARS_ERA5, "era5", max_retries, retry_wait_s)

    # Validate that key variables are not all-null
    for name, src in [("shortwave_radiation", h_era5), ("precipitation", h_era5)]:
        vals = [v for v in src.get(name, []) if v is not None]
        if not vals:
            raise RuntimeError(
                f"Open-Meteo returned no valid '{name}' values for "
                f"{start_year}–{end_year}.  Check the variable name or model."
            )

    df = pd.DataFrame({
        "time":                pd.to_datetime(h_land["time"]),
        "temperature_2m":      h_land["temperature_2m"],
        "shortwave_radiation": h_era5["shortwave_radiation"],
        "precipitation":       h_era5["precipitation"],
    })
    df["date"] = df["time"].dt.date

    # Daily aggregation
    daily = df.groupby("date").agg(
        temperature_mean_c=("temperature_2m", "mean"),
        temperature_max_c= ("temperature_2m", "max"),
        temperature_min_c= ("temperature_2m", "min"),
        # shortwave_radiation is mean W m-2 per hour; summing 24 values and
        # multiplying by 3600 s gives total J m-2 d-1
        ssrd_W_mean_sum=   ("shortwave_radiation", "sum"),
        precipitation_mm_d=("precipitation", "sum"),
    ).reset_index()

    # W m-2 (hourly mean) × 3600 s h-1 × 24 h d-1 = J m-2 d-1, then to PAR
    daily["par_umol_m2_d"] = (
        daily["ssrd_W_mean_sum"] * 3600.0   # J m-2 d-1
        * _PAR_FRACTION                      # PAR J m-2 d-1
        / _SECONDS_PER_DAY                   # PAR W m-2
        * _PAR_UMOL_PER_W                    # PAR umol m-2 s-1
        * _SECONDS_PER_DAY                   # PAR umol m-2 d-1
    )
    # Simplifies to:  ssrd_W_mean_sum × 3600 × _PAR_FRACTION × _PAR_UMOL_PER_W

    return daily[[
        "date",
        "temperature_mean_c",
        "temperature_max_c",
        "temperature_min_c",
        "par_umol_m2_d",
        "precipitation_mm_d",
    ]]


def download_site(site: SiteBox, start_year: int, end_year: int,
                  outdir: str, chunk_years: int = 10) -> str:
    """
    Download ERA5-Land data for one site in year-chunks and save as CSV.
    Returns the output file path.
    """
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(
        outdir, f"era5_land_{site.key}_{start_year}_{end_year}_daily.csv"
    )

    if os.path.exists(out_path):
        print(f"[{site.key}] already exists, skipping: {out_path}")
        return out_path

    chunks = []
    year = start_year
    while year <= end_year:
        chunk_end = min(year + chunk_years - 1, end_year)
        print(f"[{site.key}] Fetching {year}–{chunk_end} from Open-Meteo...")
        try:
            chunk = _fetch_year_chunk(site.lat, site.lon, year, chunk_end)
            chunks.append(chunk)
        except Exception as e:
            print(f"[{site.key}] WARNING: chunk {year}–{chunk_end} failed: {e}")
        year = chunk_end + 1
        if year <= end_year:
            time.sleep(0.5)   # be polite to the API

    if not chunks:
        raise RuntimeError(f"[{site.key}] No data retrieved.")

    daily = pd.concat(chunks, ignore_index=True)
    daily.to_csv(out_path, index=False)
    print(f"[{site.key}] Saved {len(daily)} daily records to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Post-processing helpers (called by gather_site_data.py)
# ---------------------------------------------------------------------------

def era5_to_daily(path: str) -> pd.DataFrame:
    """
    Load a daily met CSV produced by download_site() and return a DataFrame
    with the same column layout used throughout this codebase.

    Accepts the daily CSV output from this script.  The function name is kept
    for compatibility with gather_site_data.py.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"temperature_mean_c", "temperature_max_c",
                "temperature_min_c", "par_umol_m2_d"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV {path} is missing columns: {missing}")
    return df


def fit_seasonal_parameters(daily_df: pd.DataFrame) -> dict:
    """Fit sinusoidal seasonal parameters from a daily DataFrame.

    Returns a dict with keys matching MetForcingParams field names:
        temperature_mean_c, temperature_amplitude_c, temperature_peak_day,
        par_mean_umol_m2_d, par_amplitude_umol_m2_d, par_peak_day

    Uses scipy.optimize.curve_fit if available, otherwise falls back to a
    simple FFT-based estimate that needs only numpy.
    """
    import numpy as np

    df = daily_df.copy()
    df["doy"] = pd.to_datetime(df["date"]).dt.dayofyear

    def cosine_model(doy, mean, amplitude, peak_day):
        return mean + amplitude * np.cos(2 * np.pi * (doy - peak_day) / 365.25)

    def _fit_scipy(values, doy):
        from scipy.optimize import curve_fit
        p0 = [float(values.mean()), float(values.std()), 172.0]
        popt, _ = curve_fit(cosine_model, doy, values, p0=p0, maxfev=5000)
        return float(popt[0]), abs(float(popt[1])), float(popt[2]) % 365.25

    def _fit_fft(values, doy):
        """Fallback: estimate using the annual Fourier component."""
        mean = float(np.mean(values))
        # Bin into 365 day-of-year slots, take the mean per slot
        doy_means = (
            pd.DataFrame({"doy": doy, "v": values})
            .groupby("doy")["v"].mean()
            .reindex(range(1, 366))
            .interpolate()
            .values
        )
        fft  = np.fft.rfft(doy_means - mean)
        amp  = float(abs(fft[1])) * 2 / 365
        phase_rad = float(np.angle(fft[1]))
        peak_day  = (-phase_rad / (2 * np.pi) * 365.25) % 365.25
        return mean, amp, peak_day

    try:
        _fit = _fit_scipy
        import scipy  # noqa: just checking availability
    except ImportError:
        print("scipy not found — using FFT-based seasonal fit (less precise).")
        _fit = _fit_fft

    t_mean, t_amp, t_peak = _fit(
        daily_df["temperature_mean_c"].values, df["doy"].values
    )
    p_mean, p_amp, p_peak = _fit(
        daily_df["par_umol_m2_d"].values, df["doy"].values
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
        description=(
            "Download ERA5-Land temperature, solar radiation, and precipitation "
            "for LTER sites via the Open-Meteo archive API (no account required)."
        )
    )
    parser.add_argument(
        "--site",
        choices=[s.key for s in SITES] + ["all"],
        default=None,
        required=True,
        help=(
            "Site to download: "
            + ", ".join(s.key for s in SITES)
            + ", or all.  Run one site at a time to avoid API rate limits."
        ),
    )
    parser.add_argument(
        "--start", type=int, default=1985, metavar="YEAR",
        help="First year (default: 1985).",
    )
    parser.add_argument(
        "--end", type=int, default=2024, metavar="YEAR",
        help="Last year (default: 2024).",
    )
    parser.add_argument(
        "--outdir",
        default=os.path.join(os.path.dirname(__file__), "..", "era5_data"),
        metavar="DIR",
        help="Output directory (default: ../era5_data/ relative to this script).",
    )
    parser.add_argument(
        "--chunk-years", type=int, default=10, metavar="N",
        help="Years per API request (default: 10).  Reduce if requests time out.",
    )
    parser.add_argument(
        "--process", action="store_true",
        help="After downloading, print fitted seasonal parameters.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sites_to_run = (
        SITES if args.site == "all"
        else [s for s in SITES if s.key == args.site]
    )

    for site in sites_to_run:
        csv_path = download_site(site, args.start, args.end,
                                 args.outdir, chunk_years=args.chunk_years)

        if args.process and os.path.exists(csv_path):
            print(f"[{site.key}] Fitting seasonal parameters...")
            daily  = era5_to_daily(csv_path)
            params = fit_seasonal_parameters(daily)
            print(f"[{site.key}] MetForcingParams fields:")
            for k, v in params.items():
                print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
