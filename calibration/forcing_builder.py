# forcing_builder.py
#
# Builds a list of monthly forcing-step dicts from a PlotConfig.
# Each step represents one calendar month using sinusoidal seasonal cycles
# for temperature and PAR derived from the MetForcingParams.
#
# The generated steps are written into the YAML 'forcing.steps' section by
# yaml_writer.write_config().  Monthly resolution is a reasonable compromise
# between capturing seasonal variation and keeping run times short.

from __future__ import annotations
import math
from typing import List

from site_config import PlotConfig


_DT_DAYS = 365.25 / 12.0   # nominal month length (days)
_DAYS_IN_YEAR = 365.25


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

        step = {
            "model_time_days": round(model_time_days, 4),
            "dt_days": round(_DT_DAYS, 4),
            "mean_sea_level": tides.mean_sea_level_m,
            "mean_high_tide": tides.mean_high_tide_m,
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
