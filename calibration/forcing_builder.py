# forcing_builder.py
#
# Builds a list of monthly forcing-step dicts from a PlotConfig.
#
# Two paths are available:
#
#   build_monthly_forcing(config)
#       Synthetic sinusoidal forcing derived from MetForcingParams
#       (temperature mean/amplitude/peak_day, PAR mean/amplitude/peak_day).
#       Produces a perfectly repeating annual cycle with no interannual
#       variability.
#
#   build_monthly_forcing_from_era5(config, era5_csv)
#       Aggregates a daily ERA5 CSV (date, temperature_mean_c, par_umol_m2_d,
#       precipitation_mm_d) to calendar-month means and uses those values
#       directly.  Preserves interannual variability.  If the run duration
#       exceeds the ERA5 record, the record is cycled from its start.
#
# ERA5_CSV maps site names to the project-relative CSV paths so callers
# can do:  build_monthly_forcing_from_era5(config, ERA5_CSV["north_inlet"])
#
# The generated steps are written into the YAML 'forcing.steps' section by
# yaml_writer.write_config().  Monthly resolution is a reasonable compromise
# between capturing seasonal variation and keeping run times short.

from __future__ import annotations
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union

from site_config import PlotConfig


_DT_DAYS = 365.25 / 12.0   # nominal month length (days)
_DAYS_IN_YEAR = 365.25

# Project-root-relative paths to the ERA5 daily CSVs.
_ERA5_DIR = Path(__file__).parent.parent / "era5_data"
ERA5_CSV: Dict[str, Path] = {
    "north_inlet": _ERA5_DIR / "era5_land_ni_1985_2024_daily.csv",
    "pie":         _ERA5_DIR / "era5_land_pie_1985_2024_daily.csv",
    "gce":         _ERA5_DIR / "era5_land_gce_1985_2024_daily.csv",
}


def _day_of_year(month_index: int) -> float:
    """Return approximate mid-month day-of-year for a 0-based month index."""
    return (month_index + 0.5) * (_DAYS_IN_YEAR / 12.0)


def _sinusoid(mean: float, amplitude: float, peak_day: float, doy: float) -> float:
    """Seasonal cosine: value = mean + amplitude * cos(2*pi*(doy - peak_day) / 365)."""
    angle = 2.0 * math.pi * (doy - peak_day) / _DAYS_IN_YEAR
    return mean + amplitude * math.cos(angle)


def build_monthly_forcing(config: PlotConfig) -> List[dict]:
    """Return a list of forcing-step dicts covering config.n_years.

    Each dict maps directly to the keys expected in the YAML 'forcing.steps'
    section.  The model_time_days field increases monotonically from dt_days
    (the end of the first step) to n_years * 365.25.
    """
    steps: List[dict] = []
    n_months = config.n_years * 12
    met = config.met
    tides = config.tides

    for i in range(n_months):
        doy = _day_of_year(i % 12)
        model_time_days = (i + 1) * _DT_DAYS

        temperature = _sinusoid(
            met.temperature_mean_c,
            met.temperature_amplitude_c,
            met.temperature_peak_day,
            doy,
        )

        par = max(
            0.0,
            _sinusoid(
                met.par_mean_umol_m2_d,
                met.par_amplitude_umol_m2_d,
                met.par_peak_day,
                doy,
            ),
        )

        year = i / 12.0
        slr = year * config.sea_level_rise_m_yr

        step = {
            "model_time_days": round(model_time_days, 4),
            "dt_days": round(_DT_DAYS, 4),
            "mean_sea_level": round(tides.mean_sea_level_m + slr, 6),
            "mean_high_tide": round(tides.mean_high_tide_m + slr, 6),
            "tidal_amplitude": tides.tidal_amplitude_m,
            "tidal_period_hours": tides.tidal_period_hours,
            "temperature": round(temperature, 3),
            "precipitation_mm_d": met.precipitation_mean_mm_d,
            "par_umol_m2_d": round(par, 1),
            "creek_salinity_ppt": tides.creek_salinity_ppt,
            "freshwater_input_mm_d": met.freshwater_input_mm_d,
            "storm_surge_residual_m": 0.0,
            "suspended_sediment_concentration": tides.suspended_sediment_concentration_kg_m3,
            "fine_sediment_concentration": tides.fine_sediment_concentration_kg_m3,
            "external_pb210_supply": 0.0,
        }
        steps.append(step)

    return steps


# ---------------------------------------------------------------------------
# Compact generated forcing block (preferred for new runs)
# ---------------------------------------------------------------------------

def build_generated_forcing_block(config: PlotConfig) -> dict:
    """Return a compact dict for the YAML 'forcing.generated' block.

    The C++ config_io expands this at load time into a full forcing series,
    computing temperature and PAR from sinusoidal seasonal cycles (with optional
    linear trends) and applying linear sea-level rise per step.  The YAML file
    is therefore independent of run length — changing n_years only updates
    n_steps; no step-list rewrite is needed.

    Parameters
    ----------
    config :
        PlotConfig describing the site.

    Returns
    -------
    dict
        Suitable for ``{"forcing": {"generated": <return value>}}``.
    """
    met = config.met
    tides = config.tides
    n_steps = config.n_years * 12

    return {
        "n_steps": n_steps,
        "dt_days": round(_DT_DAYS, 4),
        "start_time_days": 0.0,
        # sea level
        "initial_mean_sea_level": tides.mean_sea_level_m,
        "sea_level_rise_rate_m_per_yr": config.sea_level_rise_m_yr,
        # temperature seasonal cycle + optional trend
        "temperature_mean_c": met.temperature_mean_c,
        "temperature_amplitude_c": met.temperature_amplitude_c,
        "temperature_peak_day": met.temperature_peak_day,
        "temperature_trend_c_per_yr": met.temperature_trend_c_per_yr,
        # PAR seasonal cycle + optional trend
        "par_mean_umol_m2_d": met.par_mean_umol_m2_d,
        "par_amplitude_umol_m2_d": met.par_amplitude_umol_m2_d,
        "par_peak_day": met.par_peak_day,
        "par_trend_umol_m2_d_per_yr": met.par_trend_umol_m2_d_per_yr,
        # constant fields passed through to each step
        "tidal_amplitude": tides.tidal_amplitude_m,
        "tidal_period_hours": tides.tidal_period_hours,
        "creek_salinity_ppt": tides.creek_salinity_ppt,
        "suspended_sediment_concentration": tides.suspended_sediment_concentration_kg_m3,
        "fine_sediment_concentration": tides.fine_sediment_concentration_kg_m3,
        "precipitation_mm_d": met.precipitation_mean_mm_d,
        "freshwater_input_mm_d": met.freshwater_input_mm_d,
        "storm_surge_residual_m": 0.0,
        "external_pb210_supply": 0.0,
    }


# ---------------------------------------------------------------------------
# ERA5-driven forcing
# ---------------------------------------------------------------------------

def _read_era5_monthly(csv_path: Path) -> List[dict]:
    """Read a daily ERA5 CSV and return a list of calendar-month aggregates.

    Each element covers one (year, month) and contains the mean of all daily
    values in that month.  The list is sorted chronologically.

    Expected CSV columns: date, temperature_mean_c, par_umol_m2_d,
    precipitation_mm_d.  Extra columns (temperature_max_c, etc.) are ignored.
    """
    monthly: Dict = defaultdict(lambda: {"temps": [], "pars": [], "precips": []})

    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            y, m, _ = row["date"].split("-")
            key = (int(y), int(m))
            monthly[key]["temps"].append(float(row["temperature_mean_c"]))
            monthly[key]["pars"].append(float(row["par_umol_m2_d"]))
            monthly[key]["precips"].append(float(row["precipitation_mm_d"]))

    result = []
    for (year, month) in sorted(monthly.keys()):
        d = monthly[(year, month)]
        n = len(d["temps"])
        result.append({
            "year": year,
            "month": month,
            "temperature": sum(d["temps"]) / n,
            "par_umol_m2_d": max(0.0, sum(d["pars"]) / n),
            "precipitation_mm_d": sum(d["precips"]) / n,
        })
    return result


def build_monthly_forcing_from_era5(
    config: PlotConfig,
    era5_csv: Union[str, Path],
) -> List[dict]:
    """Return forcing-step dicts driven by actual ERA5 temperature and PAR.

    Daily ERA5 values are aggregated to calendar-month means.  If
    config.n_years exceeds the length of the ERA5 record the record is
    cycled from its start, preserving within-year correlations.

    Temperature, PAR, and precipitation come from the ERA5 CSV.  Everything
    else (tidal amplitude/period, SSC, salinity, SLR) comes from config,
    identical to build_monthly_forcing().

    Parameters
    ----------
    config :
        PlotConfig describing the site.
    era5_csv :
        Path to the daily ERA5 CSV for this site.  Use the ERA5_CSV dict for
        the pre-built site paths, e.g.::

            build_monthly_forcing_from_era5(config, ERA5_CSV["north_inlet"])
    """
    era5_months = _read_era5_monthly(Path(era5_csv))
    n_era5 = len(era5_months)

    n_months = config.n_years * 12
    tides = config.tides
    steps: List[dict] = []

    for i in range(n_months):
        rec = era5_months[i % n_era5]
        model_time_days = (i + 1) * _DT_DAYS
        year_elapsed = i / 12.0
        slr = year_elapsed * config.sea_level_rise_m_yr

        steps.append({
            "model_time_days": round(model_time_days, 4),
            "dt_days": round(_DT_DAYS, 4),
            "mean_sea_level": round(tides.mean_sea_level_m + slr, 6),
            "mean_high_tide": round(tides.mean_high_tide_m + slr, 6),
            "tidal_amplitude": tides.tidal_amplitude_m,
            "tidal_period_hours": tides.tidal_period_hours,
            "temperature": round(rec["temperature"], 3),
            "precipitation_mm_d": round(rec["precipitation_mm_d"], 4),
            "par_umol_m2_d": round(rec["par_umol_m2_d"], 1),
            "creek_salinity_ppt": tides.creek_salinity_ppt,
            "freshwater_input_mm_d": config.met.freshwater_input_mm_d,
            "storm_surge_residual_m": 0.0,
            "suspended_sediment_concentration": tides.suspended_sediment_concentration_kg_m3,
            "fine_sediment_concentration": tides.fine_sediment_concentration_kg_m3,
            "external_pb210_supply": 0.0,
        })

    return steps
