#!/usr/bin/env python3
"""
temperature_forcing_sensitivity.py

Compares two temperature-forcing approaches to understand why modelled biomass
rises more rapidly in spring than field observations suggest.

Three 10-year runs (2015-2024 ERA5 period), all other parameters identical:

  A  cosine_T_cosine_PAR  — sinusoidal T and PAR fitted to North Inlet ERA5
                            (the approach used in all other sensitivity scripts)
  B  era5_T_cosine_PAR   — actual ERA5 daily temperature, sinusoidal PAR
                            (isolates the temperature signal)
  C  era5_T_era5_PAR     — actual ERA5 daily temperature and PAR
                            (full real forcing)

Comparing A vs B shows whether the sinusoidal temperature approximation is
responsible for the rapid spring biomass rise.  Comparing B vs C shows
whether PAR is also a factor.

Outputs (2 PNGs)
----------------
  temp_forcing_comparison.png
      Panel 1 — Mean annual temperature cycle: cosine vs ERA5 (10-yr mean ± 1 SD)
      Panel 2 — Mean annual PAR cycle:         cosine vs ERA5

  temp_sensitivity_biomass.png
      Panel 1 — GPP (gC m⁻² d⁻¹), last year of each run
      Panel 2 — Aboveground biomass (kg m⁻²), last year
      Panel 3 — Belowground biomass (kg m⁻²), last year

Usage
-----
    python sensitivity/temperature_forcing_sensitivity.py
    python sensitivity/temperature_forcing_sensitivity.py --plot-only
    python sensitivity/temperature_forcing_sensitivity.py --binary /path/to/marsh_cli
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).parent.resolve()
_CALIB_DIR = _THIS_DIR.parent / "calibration"
_ERA5_DIR  = _THIS_DIR.parent / "era5_data"
sys.path.insert(0, str(_CALIB_DIR))

from site_config import PlotConfig, north_inlet_default_met
from yaml_writer import _DEFAULT_PARAMETERS, _DEFAULT_MATERIALS, _tidal_constituent_parameters
from model_runner import run_model

try:
    import pandas as pd
except ImportError as exc:
    raise ImportError("pandas is required: pip install pandas") from exc

try:
    import yaml
except ImportError as exc:
    raise ImportError("pyyaml is required: pip install pyyaml") from exc

try:
    import netCDF4 as nc4
except ImportError as exc:
    raise ImportError("netCDF4 is required: pip install netCDF4") from exc

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError as exc:
    raise ImportError("matplotlib is required: pip install matplotlib") from exc

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ERA5_CSV    = _ERA5_DIR / "era5_land_ni_1985_2024_daily.csv"
ERA5_START  = "2015-01-01"   # 10-year period (2015-2024)
ERA5_END    = "2024-12-31"

OUTER_DT_DAYS         = 1.0
_DAYS_IN_YEAR         = 365.25

TIDAL_AMPLITUDE_M     = 0.766
TIDAL_PERIOD_HOURS    = 12.42
MSL_M                 = 0.0
SURFACE_ELEVATION_M   = 0.30
DISTANCE_FROM_CREEK_M = 10.0
CREEK_SALINITY_PPT    = 28.0
SSC_KG_M3             = 0.020
FINE_SSC_KG_M3        = 0.005

OUTPUT_DIR = _THIS_DIR / "runs" / "temperature_forcing"
FIGURE_DIR = _THIS_DIR / "figures"

# Run labels and colours
RUNS: List[Tuple[str, str, str]] = [
    # (stem,                  label,                    colour)
    ("cosine_T_cosine_PAR", "Cosine T + Cosine PAR",  "#d73027"),
    ("era5_T_cosine_PAR",   "ERA5 T + Cosine PAR",    "#4575b4"),
    ("era5_T_era5_PAR",     "ERA5 T + ERA5 PAR",      "#1a9641"),
]

# ---------------------------------------------------------------------------
# ERA5 data loading
# ---------------------------------------------------------------------------

def _load_era5() -> pd.DataFrame:
    df = pd.read_csv(ERA5_CSV, parse_dates=["date"])
    mask = (df["date"] >= ERA5_START) & (df["date"] <= ERA5_END)
    df = df.loc[mask].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(
            f"No ERA5 rows found between {ERA5_START} and {ERA5_END} in {ERA5_CSV}"
        )
    return df


# ---------------------------------------------------------------------------
# Tidal record (M2-only sine for all runs)
# ---------------------------------------------------------------------------

def _sine_tidal_record():
    from site_config import TideRecord
    t = TideRecord()
    t.mean_sea_level_m      = MSL_M
    t.mean_high_tide_m      = TIDAL_AMPLITUDE_M
    t.tidal_amplitude_m     = TIDAL_AMPLITUDE_M
    t.tidal_period_hours    = TIDAL_PERIOD_HOURS
    t.M2_amplitude_m        = TIDAL_AMPLITUDE_M
    t.M2_phase_deg          = 0.0
    t.S2_amplitude_m        = 0.0
    t.S2_phase_deg          = 0.0
    t.N2_amplitude_m        = 0.0
    t.N2_phase_deg          = 0.0
    t.K1_amplitude_m        = 0.0
    t.K1_phase_deg          = 0.0
    t.O1_amplitude_m        = 0.0
    t.O1_phase_deg          = 0.0
    t.creek_salinity_ppt    = CREEK_SALINITY_PPT
    t.suspended_sediment_concentration_kg_m3 = SSC_KG_M3
    t.fine_sediment_concentration_kg_m3      = FINE_SSC_KG_M3
    return t


# ---------------------------------------------------------------------------
# Forcing builders
# ---------------------------------------------------------------------------

def _sinusoid(mean: float, amp: float, peak_day: float, doy: float) -> float:
    return mean + amp * math.cos(2.0 * math.pi * (doy - peak_day) / _DAYS_IN_YEAR)


def _common_step_fields(tides, model_time_days: float, met) -> dict:
    """Fields that are the same across all three runs."""
    return {
        "model_time_days":                  round(model_time_days, 4),
        "dt_days":                          round(OUTER_DT_DAYS, 4),
        "mean_sea_level":                   tides.mean_sea_level_m,
        "mean_high_tide":                   tides.mean_high_tide_m,
        "tidal_amplitude":                  tides.tidal_amplitude_m,
        "tidal_period_hours":               tides.tidal_period_hours,
        "creek_salinity_ppt":               tides.creek_salinity_ppt,
        "freshwater_input_mm_d":            met.freshwater_input_mm_d,
        "storm_surge_residual_m":           0.0,
        "suspended_sediment_concentration": tides.suspended_sediment_concentration_kg_m3,
        "fine_sediment_concentration":      tides.fine_sediment_concentration_kg_m3,
        "external_pb210_supply":            0.0,
    }


def _build_cosine_forcing(tides, met) -> List[dict]:
    """Sinusoidal temperature and PAR (current sensitivity approach)."""
    n_days = round(len(_load_era5()))   # match ERA5 length exactly
    steps  = []
    for i in range(n_days):
        doy             = ((i + 0.5) * OUTER_DT_DAYS) % _DAYS_IN_YEAR
        model_time_days = (i + 1) * OUTER_DT_DAYS
        step = _common_step_fields(tides, model_time_days, met)
        step["temperature"] = round(
            _sinusoid(met.temperature_mean_c, met.temperature_amplitude_c,
                      met.temperature_peak_day, doy), 3)
        step["par_umol_m2_d"] = round(
            max(0.0, _sinusoid(met.par_mean_umol_m2_d, met.par_amplitude_umol_m2_d,
                               met.par_peak_day, doy)), 1)
        step["precipitation_mm_d"] = met.precipitation_mean_mm_d
        steps.append(step)
    return steps


def _build_era5_T_cosine_PAR_forcing(tides, met, era5: pd.DataFrame) -> List[dict]:
    """Actual ERA5 daily temperature, sinusoidal PAR."""
    steps = []
    for i, row in era5.iterrows():
        doy             = ((i + 0.5) * OUTER_DT_DAYS) % _DAYS_IN_YEAR
        model_time_days = (i + 1) * OUTER_DT_DAYS
        step = _common_step_fields(tides, model_time_days, met)
        step["temperature"] = round(float(row["temperature_mean_c"]), 3)
        step["par_umol_m2_d"] = round(
            max(0.0, _sinusoid(met.par_mean_umol_m2_d, met.par_amplitude_umol_m2_d,
                               met.par_peak_day, doy)), 1)
        step["precipitation_mm_d"] = round(float(row["precipitation_mm_d"]), 4)
        steps.append(step)
    return steps


def _build_era5_T_era5_PAR_forcing(tides, met, era5: pd.DataFrame) -> List[dict]:
    """Actual ERA5 daily temperature and PAR."""
    steps = []
    for i, row in era5.iterrows():
        model_time_days = (i + 1) * OUTER_DT_DAYS
        step = _common_step_fields(tides, model_time_days, met)
        step["temperature"]     = round(float(row["temperature_mean_c"]), 3)
        step["par_umol_m2_d"]   = round(max(0.0, float(row["par_umol_m2_d"])), 1)
        step["precipitation_mm_d"] = round(float(row["precipitation_mm_d"]), 4)
        steps.append(step)
    return steps


# ---------------------------------------------------------------------------
# YAML writer
# ---------------------------------------------------------------------------

def _write_config(output_path: Path, forcing_steps: List[dict],
                  n_years: int, tides, met) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = PlotConfig(
        site_name="temperature_sensitivity",
        plot_id=output_path.stem,
        surface_elevation_m=SURFACE_ELEVATION_M,
        distance_from_creek_m=DISTANCE_FROM_CREEK_M,
        tides=tides,
        met=met,
        n_years=n_years,
    )

    parameters = dict(_DEFAULT_PARAMETERS)
    parameters.update(_tidal_constituent_parameters(config))
    parameters.update({
        "deposition_distance_from_edge_m": DISTANCE_FROM_CREEK_M,
        "deposition_basin_length_m":       50.0,
        "deposition_length_scale_beta":    3.0,
        "salinity_initial_root_zone_ppt":  CREEK_SALINITY_PPT,
    })

    doc = {
        "simulation": {
            "start_year":                    0,
            "end_year":                      n_years,
            "dt_days":                       365.0,
            "water_level_model_name":        "composite_water_level",
            "salinity_model_name":           "distance_flushing_salinity",
            "evapotranspiration_model_name": "simple_canopy_et",
            "vegetation_model_name":         "marsh_gpp_biomass",
            "deposition_model_name":         "edge_distance_deposition",
            "root_allocation_model_name":    "exponential_root_allocation",
            "decay_model_name":              "marsh_decay",
            "compaction_model_name":           "mixing_compaction",
        },
        "site": {
            "distance_from_creek_m":  DISTANCE_FROM_CREEK_M,
            "creek_bank_elevation_m": MSL_M,
            "local_tidal_offset_m":   0.0,
        },
        "parameters": parameters,
        "materials":  _DEFAULT_MATERIALS,
        "forcing":    {"steps": forcing_steps},
        "initial_state": {
            "layers": [
                {
                    "top_elevation_m": SURFACE_ELEVATION_M,
                    "thickness_m":     config.initial_column_thickness_m,
                    "porosity":        config.initial_porosity,
                    "age_days":        0.0,
                    "fill_material":   config.initial_fill_material,
                }
            ]
        },
        "initial_ecohydrology_state": {
            "root_zone_salinity_ppt":    CREEK_SALINITY_PPT,
            "aboveground_biomass_kg_m2": config.initial_aboveground_biomass_kg_m2,
            "belowground_biomass_kg_m2": config.initial_belowground_biomass_kg_m2,
            "lai":                       config.initial_lai,
            "litter_kg_m2":              0.0,
        },
        "output": {
            "file":               str(output_path.with_suffix(".nc")),
            "write_time_series":  True,
            "write_column_snapshots": False,
        },
    }

    with open(output_path, "w") as fh:
        yaml.dump(doc, fh, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)


# ---------------------------------------------------------------------------
# Run management
# ---------------------------------------------------------------------------

def _run_all(cli_binary: str, force: bool) -> Dict[str, Path]:
    era5 = _load_era5()
    n_years = len(era5) / _DAYS_IN_YEAR   # fractional years — end_year set below
    # Use exact day count as a proxy; end_year is set from forcing length
    n_years_int = int(np.ceil(len(era5) / _DAYS_IN_YEAR))

    tides = _sine_tidal_record()
    met   = north_inlet_default_met()

    forcing_builders = {
        "cosine_T_cosine_PAR": lambda: _build_cosine_forcing(tides, met),
        "era5_T_cosine_PAR":   lambda: _build_era5_T_cosine_PAR_forcing(tides, met, era5),
        "era5_T_era5_PAR":     lambda: _build_era5_T_era5_PAR_forcing(tides, met, era5),
    }

    results: Dict[str, Path] = {}
    total = len(RUNS)
    for idx, (stem, label, _colour) in enumerate(RUNS, 1):
        yaml_path = OUTPUT_DIR / f"{stem}.yaml"
        nc_path   = OUTPUT_DIR / f"{stem}.nc"
        print(f"  [{idx}/{total}] {label}")

        if not force and nc_path.exists():
            print("             output exists, skipping.")
        else:
            forcing_steps = forcing_builders[stem]()
            _write_config(yaml_path, forcing_steps, n_years_int, tides, met)
            try:
                run_model(str(yaml_path), cli_binary=cli_binary, silent=True)
            except RuntimeError as exc:
                print(f"    ERROR: {exc}")
                continue

        if nc_path.exists():
            results[stem] = nc_path

    return results


def _collect_existing() -> Dict[str, Path]:
    results: Dict[str, Path] = {}
    for stem, _label, _colour in RUNS:
        nc_path = OUTPUT_DIR / f"{stem}.nc"
        if nc_path.exists():
            results[stem] = nc_path
    return results


# ---------------------------------------------------------------------------
# Output reader
# ---------------------------------------------------------------------------

def _read_ts(nc_path: Path) -> Dict[str, np.ndarray]:
    series: Dict[str, np.ndarray] = {}
    with nc4.Dataset(str(nc_path), "r") as ds:
        for name, var in ds.variables.items():
            if var.ndim == 1 and "time" in var.dimensions:
                series[name] = np.array(var[:], dtype=np.float64)
    return series


# ---------------------------------------------------------------------------
# Forcing comparison helpers (no model run needed)
# ---------------------------------------------------------------------------

def _mean_annual_cycle(values: np.ndarray, doy_arr: np.ndarray,
                        n_bins: int = 365) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (bin_centres, mean, std) for a DOY-binned annual cycle.
    doy_arr is 1-based day-of-year (1..365).
    """
    bin_centres = np.arange(1, n_bins + 1, dtype=float)
    mean_vals   = np.full(n_bins, np.nan)
    std_vals    = np.full(n_bins, np.nan)
    for d in range(1, n_bins + 1):
        idx = np.where(np.round(doy_arr).astype(int) == d)[0]
        if len(idx) > 0:
            mean_vals[d - 1] = values[idx].mean()
            std_vals[d - 1]  = values[idx].std()
    return bin_centres, mean_vals, std_vals


def _era5_annual_cycles(era5: pd.DataFrame):
    era5 = era5.copy()
    era5["doy"] = era5["date"].dt.dayofyear.astype(float)
    doy = era5["doy"].values
    t_doy, t_mean, t_std = _mean_annual_cycle(era5["temperature_mean_c"].values, doy)
    p_doy, p_mean, p_std = _mean_annual_cycle(era5["par_umol_m2_d"].values, doy)
    return t_doy, t_mean, t_std, p_mean, p_std


def _cosine_annual_cycle(met, n_days: int = 365) -> Tuple[np.ndarray, np.ndarray]:
    """Return (T_cosine, PAR_cosine) evaluated at DOY 1..n_days."""
    doys = np.arange(1, n_days + 1, dtype=float)
    T   = np.array([_sinusoid(met.temperature_mean_c, met.temperature_amplitude_c,
                               met.temperature_peak_day, d) for d in doys])
    PAR = np.array([max(0.0, _sinusoid(met.par_mean_umol_m2_d, met.par_amplitude_umol_m2_d,
                                        met.par_peak_day, d)) for d in doys])
    return T, PAR


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _last_year_mask(t: np.ndarray) -> np.ndarray:
    return t >= (t[-1] - _DAYS_IN_YEAR)


def _colour_map() -> Dict[str, str]:
    return {stem: colour for stem, _label, colour in RUNS}


def _label_map() -> Dict[str, str]:
    return {stem: label for stem, label, _colour in RUNS}


def _save_or_show(fig, outpath: Optional[Path]) -> None:
    if outpath is not None:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        print(f"Saved: {outpath}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Forcing comparison
# ---------------------------------------------------------------------------

def plot_forcing_comparison(era5: pd.DataFrame, met,
                             outpath: Optional[Path]) -> None:
    t_doy, t_mean, t_std, p_mean, p_std = _era5_annual_cycles(era5)
    T_cos, PAR_cos = _cosine_annual_cycle(met)

    fig, (ax_t, ax_p) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # Temperature
    ax_t.fill_between(t_doy, t_mean - t_std, t_mean + t_std,
                       color="#4575b4", alpha=0.20, label="ERA5 ± 1 SD")
    ax_t.plot(t_doy, t_mean, color="#4575b4", linewidth=1.6,
              label=f"ERA5 mean (2015–2024)")
    ax_t.plot(np.arange(1, 366), T_cos, color="#d73027", linewidth=1.6,
              linestyle="--", label="Cosine fit")
    ax_t.set_ylabel("Temperature (°C)", fontsize=10)
    ax_t.set_title("Mean annual temperature cycle — ERA5 vs cosine", fontsize=10)
    ax_t.legend(fontsize=9)
    ax_t.grid(True, linewidth=0.4, alpha=0.5)

    # PAR
    ax_p.fill_between(t_doy, (p_mean - p_std) / 1e6, (p_mean + p_std) / 1e6,
                       color="#1a9641", alpha=0.20, label="ERA5 ± 1 SD")
    ax_p.plot(t_doy, p_mean / 1e6, color="#1a9641", linewidth=1.6,
              label="ERA5 mean (2015–2024)")
    ax_p.plot(np.arange(1, 366), PAR_cos / 1e6, color="#d73027", linewidth=1.6,
              linestyle="--", label="Cosine fit")
    ax_p.set_xlabel("Day of year", fontsize=10)
    ax_p.set_ylabel("PAR (×10⁶ µmol m⁻² d⁻¹)", fontsize=10)
    ax_p.set_title("Mean annual PAR cycle — ERA5 vs cosine", fontsize=10)
    ax_p.legend(fontsize=9)
    ax_p.grid(True, linewidth=0.4, alpha=0.5)
    ax_p.set_xlim(1, 365)

    fig.suptitle(
        "North Inlet ERA5 forcing vs cosine approximation\n"
        f"T mean {met.temperature_mean_c:.1f} °C  amp {met.temperature_amplitude_c:.1f} °C  "
        f"peak DOY {met.temperature_peak_day:.0f}",
        fontsize=11,
    )
    plt.tight_layout()
    _save_or_show(fig, outpath)


# ---------------------------------------------------------------------------
# Figure 2: Biomass and GPP response
# ---------------------------------------------------------------------------

def plot_biomass_response(results: Dict[str, Path],
                           outpath: Optional[Path]) -> None:
    colours = _colour_map()
    labels  = _label_map()

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    ax_gpp, ax_above, ax_below = axes

    for stem, nc_path in results.items():
        ts   = _read_ts(nc_path)
        t    = ts["model_time_days"]
        mask = _last_year_mask(t)

        # DOY within last year (1-based)
        year_start = t[mask][0]
        doy = t[mask] - year_start + 1.0

        colour = colours[stem]
        label  = labels[stem]

        if "gpp_gC_m2_d" in ts:
            ax_gpp.plot(doy, ts["gpp_gC_m2_d"][mask],
                        color=colour, linewidth=1.3, label=label)

        if "aboveground_biomass_kg_m2" in ts:
            ax_above.plot(doy, ts["aboveground_biomass_kg_m2"][mask],
                          color=colour, linewidth=1.3, label=label)

        if "belowground_biomass_kg_m2" in ts:
            ax_below.plot(doy, ts["belowground_biomass_kg_m2"][mask],
                          color=colour, linewidth=1.3, label=label)

    ax_gpp.set_ylabel("GPP (gC m⁻² d⁻¹)", fontsize=10)
    ax_gpp.set_title("GPP — last year of 10-year run", fontsize=10)
    ax_gpp.legend(fontsize=9)
    ax_gpp.grid(True, linewidth=0.4, alpha=0.5)

    ax_above.set_ylabel("Aboveground biomass (kg m⁻²)", fontsize=10)
    ax_above.set_title("Aboveground biomass — last year", fontsize=10)
    ax_above.legend(fontsize=9)
    ax_above.grid(True, linewidth=0.4, alpha=0.5)

    ax_below.set_ylabel("Belowground biomass (kg m⁻²)", fontsize=10)
    ax_below.set_title("Belowground biomass — last year", fontsize=10)
    ax_below.set_xlabel("Day of final year", fontsize=10)
    ax_below.legend(fontsize=9)
    ax_below.grid(True, linewidth=0.4, alpha=0.5)

    fig.suptitle(
        "Temperature forcing sensitivity — North Inlet\n"
        f"elev {SURFACE_ELEVATION_M} m, dist {DISTANCE_FROM_CREEK_M} m, "
        f"M2-only tide {TIDAL_AMPLITUDE_M} m",
        fontsize=11,
    )
    plt.tight_layout()
    _save_or_show(fig, outpath)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--binary",    default="marsh_cli")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--plot-only", action="store_true")
    g.add_argument("--skip-runs", action="store_true")
    p.add_argument("--force",     action="store_true")
    p.add_argument("--no-save",   action="store_true")
    return p.parse_args()


def main() -> None:
    args   = _parse_args()
    era5   = _load_era5()
    met    = north_inlet_default_met()
    save   = not args.no_save

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.plot_only or args.skip_runs:
        results = _collect_existing()
    else:
        print(f"Running {len(RUNS)} simulations "
              f"({ERA5_START} – {ERA5_END}, {len(era5)} daily steps)...")
        results = _run_all(cli_binary=args.binary, force=args.force)

    print("\nPlotting forcing comparison...")
    plot_forcing_comparison(
        era5, met,
        outpath=FIGURE_DIR / "temp_forcing_comparison.png" if save else None,
    )

    if results:
        print("Plotting biomass response...")
        plot_biomass_response(
            results,
            outpath=FIGURE_DIR / "temp_sensitivity_biomass.png" if save else None,
        )
    else:
        print("No model outputs found — run without --plot-only to generate them.")


if __name__ == "__main__":
    main()
