#!/usr/bin/env python3
"""
gather_site_data.py

For each LTER site in calibration/site_config.py, downloads:

  1. NOAA CO-OPS tidal datums (MSL, MHW, MLW) and harmonic constituents
     (M2, S2, N2, K1, O1) for the named gauge — filling TideRecord fields.

  2. ERA5-Land daily temperature and solar radiation, fitted to sinusoidal
     seasonal parameters — filling MetForcingParams fields.
     ERA5 NetCDF files must already exist (run download_era5_met.py first).
     If they are absent the tidal data is still downloaded and the met
     parameters remain as placeholders.

Outputs written to --outdir (default: calibration/):
  site_data_<key>.json        raw values as downloaded
  site_config_filled.py       drop-in replacement for calibration/site_config.py
                              with all PLACEHOLDER values replaced

Usage
-----
# Tidal data only (no ERA5 needed):
python scripts/gather_site_data.py

# Tidal + met (ERA5 NetCDFs must exist first):
python scripts/gather_site_data.py --era5-dir era5_data

# One site only:
python scripts/gather_site_data.py --site ni --era5-dir era5_data

Requirements: requests, numpy (for met fitting)
Optional:     scipy (better seasonal curve fit), netCDF4 (ERA5 processing)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Site definitions
# ---------------------------------------------------------------------------
# Each entry ties one LTER site to:
#   coops_station  — the NOAA CO-OPS gauge used for tidal parameters
#   era5_key       — the key used by download_era5_met.py (and its SITES list)
#   config_prefix  — prefix of the factory functions in site_config.py
#                    (e.g. "north_inlet" → north_inlet_default_tides / _met)

SITE_DEFS = [
    {
        "key":           "ni",
        "label":         "North Inlet, SC",
        "coops_station": "8661070",   # Oyster Landing / North Inlet, SC
        "era5_key":      "ni",
        "config_prefix": "north_inlet",
        "default_salinity_ppt": 28.0,
    },
    {
        "key":           "pie",
        "label":         "Plum Island Estuary, MA",
        "coops_station": "8419870",   # Seavey Island, NH (~41 km north of PIE)
        "era5_key":      "pie",
        "config_prefix": "pie",
        "default_salinity_ppt": 22.0,
    },
    {
        "key":           "gce",
        "label":         "GCE, Sapelo Island GA",
        "coops_station": "8670870",   # Fort Pulaski, GA (~80 km NE of Sapelo)
        "era5_key":      "gce",
        "config_prefix": "gce",
        "default_salinity_ppt": 25.0,
    },
]

NOAA_DATA_URL  = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
NOAA_MDAPI_URL = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi"

# Constituents we want, in the order they appear in TideRecord
CONSTITUENTS = ["M2", "S2", "N2", "K1", "O1"]


# ---------------------------------------------------------------------------
# NOAA CO-OPS: datums
# ---------------------------------------------------------------------------

def fetch_coops_datums(station_id: str) -> dict:
    """
    Return a dict of datum name → value (m, NAVD88 or station datum) for a
    CO-OPS station.  Relevant keys: MSL, MHW, MHHW, MLW, MLLW.
    Uses the mdapi endpoint which returns metric values without requiring a
    date range (unlike the datagetter product=datums endpoint).
    """
    url = f"{NOAA_MDAPI_URL}/stations/{station_id}/datums.json"
    r = requests.get(url, params={"units": "metric"}, timeout=30)
    r.raise_for_status()
    payload = r.json()

    datums = {}
    for d in payload.get("datums", []):
        name = d.get("name", "").strip()
        try:
            datums[name] = float(d["value"])
        except (KeyError, TypeError, ValueError):
            continue
    if not datums:
        raise RuntimeError(
            f"CO-OPS mdapi returned no datums for station {station_id}. "
            f"Response keys: {list(payload.keys())}"
        )
    return datums


# ---------------------------------------------------------------------------
# NOAA CO-OPS: harmonic constituents
# ---------------------------------------------------------------------------

def fetch_coops_harmonics(station_id: str) -> dict[str, dict]:
    """
    Return {constituent_name: {"amplitude_m": float, "phase_deg": float}}
    for all available constituents at a CO-OPS station.
    Phase is referenced to GMT (Greenwich phase lag).
    """
    url = f"{NOAA_MDAPI_URL}/stations/{station_id}/harcon.json"
    r = requests.get(url, params={"units": "metric"}, timeout=30)
    r.raise_for_status()
    payload = r.json()

    result = {}
    for entry in payload.get("HarmonicConstituents", []):
        name = entry.get("name", "").strip().upper()
        try:
            amp   = float(entry["amplitude"])
            phase = float(entry["phase_GMT"])
        except (KeyError, TypeError, ValueError):
            continue
        result[name] = {"amplitude_m": amp, "phase_deg": phase}

    return result


# ---------------------------------------------------------------------------
# Derive TideRecord values from raw API data
# ---------------------------------------------------------------------------

def build_tide_record_values(datums: dict, harmonics: dict,
                             salinity_ppt: float) -> dict:
    """
    Convert raw CO-OPS datum and harmonic data to TideRecord field values.

    Datum reference convention used here:
      - mean_sea_level_m  = 0.0  (model runs relative to MSL)
      - mean_high_tide_m  = MHW  − MSL
      - tidal_amplitude_m = (MHW − MLW) / 2

    All values are in metres.
    """
    msl = datums.get("MSL", 0.0)
    mhw = datums.get("MHW", datums.get("MHHW", msl + 0.8))
    mlw = datums.get("MLW", datums.get("MLLW", msl - 0.8))

    mhw_rel = mhw - msl
    tidal_amp = (mhw - mlw) / 2.0

    record = {
        "mean_sea_level_m":    0.0,
        "mean_high_tide_m":    round(mhw_rel, 3),
        "tidal_amplitude_m":   round(tidal_amp, 3),
        "tidal_period_hours":  12.42,
        "creek_salinity_ppt":  salinity_ppt,
    }

    for name in CONSTITUENTS:
        key = name.upper()
        if key in harmonics:
            record[f"{name}_amplitude_m"] = round(harmonics[key]["amplitude_m"], 4)
            record[f"{name}_phase_deg"]   = round(harmonics[key]["phase_deg"], 2)
        else:
            record[f"{name}_amplitude_m"] = None   # not available
            record[f"{name}_phase_deg"]   = None

    return record


# ---------------------------------------------------------------------------
# ERA5: seasonal met parameters
# ---------------------------------------------------------------------------

def find_era5_file(era5_dir: str, era5_key: str) -> str | None:
    """
    Return path to the ERA5 daily CSV for a site, or None if absent.
    Prefers the daily CSV produced by the Open-Meteo download; falls back
    to a legacy NetCDF if present.
    """
    if not era5_dir:
        return None
    pattern = f"era5_land_{era5_key}_"
    csv_match = nc_match = None
    for name in os.listdir(era5_dir):
        if name.startswith(pattern):
            if name.endswith("_daily.csv"):
                csv_match = os.path.join(era5_dir, name)
            elif name.endswith(".nc") and nc_match is None:
                nc_match = os.path.join(era5_dir, name)
    return csv_match or nc_match


def fit_met_parameters(nc_path: str) -> dict | None:
    """
    Process an ERA5-Land NetCDF using the helpers in download_era5_met.py.
    Returns a dict with MetForcingParams field names, or None on failure.
    """
    # Import processing helpers from the sibling script
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    try:
        from download_era5_met import era5_to_daily, fit_seasonal_parameters
    except ImportError:
        print("  Could not import download_era5_met — skipping ERA5 processing.",
              file=sys.stderr)
        return None

    try:
        daily  = era5_to_daily(nc_path)
        params = fit_seasonal_parameters(daily)
        return params
    except Exception as e:
        print(f"  ERA5 processing failed: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

_TIDE_TEMPLATE = """\
def {prefix}_default_tides() -> TideRecord:
    \"\"\"
    {label}
    Tidal data from NOAA CO-OPS station {station} (datums and harmonics).
    Datum reference: MSL = 0.0 m; all heights relative to MSL.
    \"\"\"
    t = TideRecord()
    t.mean_sea_level_m            = {mean_sea_level_m}
    t.mean_high_tide_m            = {mean_high_tide_m}
    t.tidal_amplitude_m           = {tidal_amplitude_m}
    t.tidal_period_hours          = {tidal_period_hours}
    t.M2_amplitude_m              = {M2_amplitude_m}
    t.M2_phase_deg                = {M2_phase_deg}
    t.S2_amplitude_m              = {S2_amplitude_m}
    t.S2_phase_deg                = {S2_phase_deg}
    t.N2_amplitude_m              = {N2_amplitude_m}
    t.N2_phase_deg                = {N2_phase_deg}
    t.K1_amplitude_m              = {K1_amplitude_m}
    t.K1_phase_deg                = {K1_phase_deg}
    t.O1_amplitude_m              = {O1_amplitude_m}
    t.O1_phase_deg                = {O1_phase_deg}
    t.creek_salinity_ppt          = {creek_salinity_ppt}
    t.suspended_sediment_concentration_kg_m3 = 0.020  # PLACEHOLDER
    t.fine_sediment_concentration_kg_m3      = 0.005  # PLACEHOLDER
    return t
"""

_MET_TEMPLATE_FILLED = """\
def {prefix}_default_met() -> MetForcingParams:
    \"\"\"
    {label}
    Seasonal parameters fitted from ERA5-Land ({era5_years}).
    \"\"\"
    m = MetForcingParams()
    m.temperature_mean_c          = {temperature_mean_c}
    m.temperature_amplitude_c     = {temperature_amplitude_c}
    m.temperature_peak_day        = {temperature_peak_day}
    m.par_mean_umol_m2_d          = {par_mean_umol_m2_d}
    m.par_amplitude_umol_m2_d     = {par_amplitude_umol_m2_d}
    m.par_peak_day                = {par_peak_day}
    m.precipitation_mean_mm_d     = 3.0   # PLACEHOLDER — add ERA5 tp variable
    m.freshwater_input_mm_d       = 0.0
    return m
"""

_MET_TEMPLATE_PLACEHOLDER = """\
def {prefix}_default_met() -> MetForcingParams:
    \"\"\"
    {label}
    PLACEHOLDER — run download_era5_met.py --site {era5_key} then re-run this script.
    \"\"\"
    m = MetForcingParams()
    m.temperature_mean_c          = {temperature_mean_c}
    m.temperature_amplitude_c     = {temperature_amplitude_c}
    m.temperature_peak_day        = {temperature_peak_day}
    m.par_mean_umol_m2_d          = {par_mean_umol_m2_d}
    m.par_amplitude_umol_m2_d     = {par_amplitude_umol_m2_d}
    m.par_peak_day                = {par_peak_day}
    m.precipitation_mean_mm_d     = 3.0   # PLACEHOLDER
    m.freshwater_input_mm_d       = 0.0
    return m
"""

_FILLED_CONFIG_HEADER = '''\
# site_config_filled.py
#
# Auto-generated by gather_site_data.py — do not edit by hand.
# Copy the factory functions below into site_config.py to replace placeholders.
#
# Tidal data:  NOAA CO-OPS datums + harmonic constituents
# Met data:    ERA5-Land sinusoidal seasonal fit
#
# Generated: {timestamp}

from __future__ import annotations
from dataclasses import dataclass, field

# Import dataclasses from the original config so this file is self-contained
from site_config import TideRecord, MetForcingParams, PlotConfig


'''


def _fmt(v, missing="None  # not available from CO-OPS"):
    if v is None:
        return missing
    return str(v)


def render_tide_function(site_def: dict, tide_vals: dict) -> str:
    tv = {k: _fmt(tide_vals.get(k)) for k in [
        "mean_sea_level_m", "mean_high_tide_m", "tidal_amplitude_m",
        "tidal_period_hours",
        "M2_amplitude_m", "M2_phase_deg",
        "S2_amplitude_m", "S2_phase_deg",
        "N2_amplitude_m", "N2_phase_deg",
        "K1_amplitude_m", "K1_phase_deg",
        "O1_amplitude_m", "O1_phase_deg",
        "creek_salinity_ppt",
    ]}
    return _TIDE_TEMPLATE.format(
        prefix=site_def["config_prefix"],
        label=site_def["label"],
        station=site_def["coops_station"],
        **tv,
    )


def render_met_function(site_def: dict, met_vals: dict | None,
                        era5_years: str = "1985–2024") -> str:
    placeholder_met = {
        "temperature_mean_c":      15.0,
        "temperature_amplitude_c": 12.0,
        "temperature_peak_day":    198.0,
        "par_mean_umol_m2_d":      25_000_000.0,
        "par_amplitude_umol_m2_d": 13_000_000.0,
        "par_peak_day":            172.0,
    }
    vals   = met_vals if met_vals else placeholder_met
    filled = met_vals is not None
    tmpl   = _MET_TEMPLATE_FILLED if filled else _MET_TEMPLATE_PLACEHOLDER

    return tmpl.format(
        prefix=site_def["config_prefix"],
        label=site_def["label"],
        era5_key=site_def["era5_key"],
        era5_years=era5_years,
        **{k: _fmt(vals.get(k)) for k in [
            "temperature_mean_c", "temperature_amplitude_c", "temperature_peak_day",
            "par_mean_umol_m2_d", "par_amplitude_umol_m2_d", "par_peak_day",
        ]},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def gather_site(site_def: dict, era5_dir: str | None, outdir: str) -> dict:
    """
    Download tidal and (optionally) met data for one site.
    Returns a results dict suitable for JSON serialisation.
    """
    key     = site_def["key"]
    label   = site_def["label"]
    station = site_def["coops_station"]

    print(f"\n{'='*60}")
    print(f"  {label}  (CO-OPS {station})")
    print(f"{'='*60}")

    result = {
        "site":    key,
        "label":   label,
        "station": station,
        "datums":  {},
        "harmonics": {},
        "tide_record": {},
        "met_params":  None,
    }

    # --- Tidal datums ---
    print("  Fetching CO-OPS datums...")
    try:
        datums = fetch_coops_datums(station)
        result["datums"] = datums
        for name, val in datums.items():
            print(f"    {name:6s}  {val:+.3f} m")
    except Exception as e:
        print(f"  ERROR fetching datums: {e}", file=sys.stderr)
        datums = {}

    # --- Harmonic constituents ---
    print("  Fetching CO-OPS harmonic constituents...")
    try:
        harmonics = fetch_coops_harmonics(station)
        result["harmonics"] = harmonics
        for name in CONSTITUENTS:
            h = harmonics.get(name)
            if h:
                print(f"    {name:3s}  amp={h['amplitude_m']:.4f} m  "
                      f"phase={h['phase_deg']:.1f}°")
            else:
                print(f"    {name:3s}  not available")
    except Exception as e:
        print(f"  ERROR fetching harmonics: {e}", file=sys.stderr)
        harmonics = {}

    # --- Build TideRecord values ---
    tide_vals = build_tide_record_values(
        datums, harmonics, site_def["default_salinity_ppt"]
    )
    result["tide_record"] = tide_vals
    print(f"  MSL={tide_vals['mean_sea_level_m']} m  "
          f"MHW={tide_vals['mean_high_tide_m']} m  "
          f"TidalAmp={tide_vals['tidal_amplitude_m']} m (half-range)")

    # --- ERA5 met ---
    nc_path = find_era5_file(era5_dir, site_def["era5_key"]) if era5_dir else None
    if nc_path:
        print(f"  Processing ERA5: {os.path.basename(nc_path)}")
        met_vals = fit_met_parameters(nc_path)
        if met_vals:
            result["met_params"] = met_vals
            print(f"  T: mean={met_vals['temperature_mean_c']:.1f}°C  "
                  f"amp={met_vals['temperature_amplitude_c']:.1f}°C  "
                  f"peak_day={met_vals['temperature_peak_day']:.0f}")
            print(f"  PAR: mean={met_vals['par_mean_umol_m2_d']:.0f}  "
                  f"amp={met_vals['par_amplitude_umol_m2_d']:.0f}  "
                  f"peak_day={met_vals['par_peak_day']:.0f}")
    else:
        print(f"  ERA5 NetCDF not found — met parameters remain as placeholders.")
        print(f"  Run: python scripts/download_era5_met.py --site {key} --process")

    # --- Write JSON ---
    os.makedirs(outdir, exist_ok=True)
    json_path = os.path.join(outdir, f"site_data_{key}.json")
    with open(json_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"  Saved: {json_path}")

    return result


def write_filled_config(results: list[dict], site_defs: list[dict],
                        outdir: str, era5_years: str) -> None:
    from datetime import datetime

    lines = [_FILLED_CONFIG_HEADER.format(
        timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    )]

    for result, site_def in zip(results, site_defs):
        lines.append(f"# {'─'*56}\n")
        lines.append(f"# {site_def['label']}\n")
        lines.append(f"# CO-OPS station: {site_def['coops_station']}\n")
        lines.append(f"# {'─'*56}\n\n")
        lines.append(render_tide_function(site_def, result["tide_record"]))
        lines.append("\n")
        lines.append(render_met_function(site_def, result.get("met_params"),
                                         era5_years=era5_years))
        lines.append("\n\n")

    out_path = os.path.join(outdir, "site_config_filled.py")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.writelines(lines)
    print(f"\nWrote: {out_path}")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    calib_dir = str(here.parent / "calibration")

    parser = argparse.ArgumentParser(
        description="Gather tidal and met data for LTER marsh calibration sites.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--site",
        choices=[s["key"] for s in SITE_DEFS] + ["all"],
        default="all",
        help="Which site to process (default: all).",
    )
    parser.add_argument(
        "--era5-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory containing ERA5-Land NetCDF files produced by "
            "download_era5_met.py (e.g. era5_data/).  "
            "If omitted, met parameters are left as placeholders."
        ),
    )
    parser.add_argument(
        "--outdir",
        default=calib_dir,
        metavar="DIR",
        help=f"Output directory (default: {calib_dir}).",
    )
    parser.add_argument(
        "--era5-years",
        default="1985–2024",
        metavar="RANGE",
        help="Year range label for the generated docstrings (default: 1985–2024).",
    )
    return parser.parse_args()


def main() -> None:
    args  = parse_args()
    sites = (
        SITE_DEFS if args.site == "all"
        else [s for s in SITE_DEFS if s["key"] == args.site]
    )

    results = []
    for site_def in sites:
        result = gather_site(site_def, args.era5_dir, args.outdir)
        results.append(result)

    # Write combined filled config if we processed anything
    if results:
        write_filled_config(results, sites, args.outdir, args.era5_years)

    print("\nDone.")
    if any(r.get("met_params") is None for r in results):
        print("\nNote: one or more sites still have placeholder met parameters.")
        print("Download ERA5 data then re-run with --era5-dir:")
        print("  python scripts/download_era5_met.py --process")
        print("  python scripts/gather_site_data.py --era5-dir era5_data")


if __name__ == "__main__":
    main()
