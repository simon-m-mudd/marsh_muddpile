#!/usr/bin/env python3
"""
plot_ni_biomass_climate.py

Explores the relationship between observed aboveground biomass at North Inlet
LTER plots and ERA5-derived climate variables, porewater chemistry,
tidal inundation, and suspended sediment concentration.

Data sources
------------
Biomass:   edi.135.12  NILTREB_plants_aboveground_biomass_density.csv
           Units: g m⁻²; control plots only; sites GI and OL, locations HM and LM.
PAR/Temp:  era5_data/era5_land_ni_1985_2024_daily.csv  (daily resolution)
Porewater: edi.136.11  NILTREB_porewater.csv
           SALINITY, NH4, S2 at depths 10/25/50/75/100 cm.
Tides:     NOAA CO-OPS station 8661070 (Springmaid Pier, SC; ~45 km from North Inlet).
           Monthly mean water levels cached in era5_data/noaa_8661070_monthly_means.csv.
           Annual inundation hours computed analytically with the arcsine formula.
SSC:       knb-lter-nin.8.1  LTER.NIN.sedi.csv
           Total suspended sediment concentration (mg/L) at OL, TC, CB transects,
           1981–1992.

Figures saved to scripts/figures/:
  ni_biomass_vs_peak_temperature.png      2×2 by site
  ni_biomass_vs_par.png                   2×2 by site
  ni_biomass_timing_par.png               2×2 by site
  ni_biomass_timing_temp.png              2×2 by site
  ni_biomass_vs_salinity_depths.png       1×5 by depth
  ni_biomass_vs_nh4_depths.png            1×5 by depth
  ni_biomass_vs_s2_depths.png             1×5 by depth
  ni_biomass_vs_inundation.png            Left: r vs elevation threshold; right: scatter
  ni_biomass_vs_ssc.png                   Biomass vs annual mean SSC (1984-1992 overlap)

Usage
-----
    python scripts/plot_ni_biomass_climate.py
    python scripts/plot_ni_biomass_climate.py --no-save   # interactive display
    python scripts/plot_ni_biomass_climate.py --no-download  # skip NOAA API call
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections import defaultdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import requests
except ImportError:
    requests = None  # type: ignore

try:
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from scipy import stats as sp_stats
except ImportError as exc:
    raise ImportError(
        "matplotlib and scipy are required: pip install matplotlib scipy"
    ) from exc

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.parent
_LTER = _ROOT / "lter_data"
_ERA5_DIR = _ROOT / "era5_data"
_ERA5 = _ERA5_DIR / "era5_land_ni_1985_2024_daily.csv"
_BIOMASS_CSV = _LTER / "edi.135.12" / "NILTREB_plants_aboveground_biomass_density.csv"
_POREWATER_CSV = _LTER / "edi.136.11" / "NILTREB_porewater.csv"
_SEDI_CSV = _LTER / "knb-lter-nin.8.1" / "LTER.NIN.sedi.csv"
_SET_CHANGE_CSV = _LTER / "edi.134.12" / "NILTREB_marsh_elevation_change.csv"
_SET_INIT_CSV = _LTER / "edi.134.12" / "NILTREB_marsh_surface_elevation_init.csv"
_NOAA_CACHE = _ERA5_DIR / "noaa_8661070_hourly.csv"
_FIGURE_DIR = Path(__file__).parent / "figures"

# ---------------------------------------------------------------------------
# NOAA gauge constants  (Springmaid Pier, 8661070)
# ---------------------------------------------------------------------------

_NOAA_STATION = "8661070"
_NOAA_START_YEAR = 1985
_NOAA_END_YEAR = 2024

# Offset to convert m NAVD88 → m above MLLW at Springmaid Pier.
# From NOAA CO-OPS datums: NAVD88 = 32.45 ft, MLLW = 29.29 ft above station datum.
# (32.45 − 29.29) × 0.3048 = 0.963 m  (epoch 1983-2001).
_NAVD88_TO_MLLW_M = 0.963

# ---------------------------------------------------------------------------
# Visual encoding
# ---------------------------------------------------------------------------

_SITES = [("GI", "HM"), ("GI", "LM"), ("OL", "HM"), ("OL", "LM")]
_SITE_LABELS = {
    ("GI", "HM"): "GI High Marsh",
    ("GI", "LM"): "GI Low Marsh",
    ("OL", "HM"): "OL High Marsh",
    ("OL", "LM"): "OL Low Marsh",
}
_SITE_COLORS = {
    ("GI", "HM"): "#2166ac",
    ("GI", "LM"): "#74add1",
    ("OL", "HM"): "#d73027",
    ("OL", "LM"): "#f46d43",
}
_SITE_MARKERS = {
    ("GI", "HM"): "o",
    ("GI", "LM"): "^",
    ("OL", "HM"): "o",
    ("OL", "LM"): "^",
}

_DEPTHS = [10, 25, 50, 75, 100]
_ROLLING_WINDOW = 5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _parse_date(s: str) -> date:
    return datetime.strptime(s.strip(), "%Y-%m-%d").date()


def load_biomass() -> Dict[Tuple[str, str, date], float]:
    """Return {(site, location, date): mean_aboveground_biomass_g_m2}."""
    accum: Dict[Tuple[str, str, date], List[float]] = defaultdict(list)
    with open(_BIOMASS_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["TREATMENT"].strip() != "C":
                continue
            val_str = row["ABOVEGROUND_BIOMASS"].strip()
            if val_str in ("", "NA"):
                continue
            try:
                val = float(val_str)
            except ValueError:
                continue
            key = (row["SITE"].strip(), row["LOCATION"].strip(), _parse_date(row["DATE"]))
            accum[key].append(val)
    return {k: float(np.mean(v)) for k, v in accum.items()}


def load_era5_daily() -> Dict[date, dict]:
    """Return {date: {temperature_mean_c, par_umol_m2_d}} from the daily ERA5 CSV."""
    result: Dict[date, dict] = {}
    with open(_ERA5, newline="") as fh:
        for row in csv.DictReader(fh):
            dt = datetime.strptime(row["date"].strip(), "%Y-%m-%d").date()
            result[dt] = {
                "temperature_mean_c": float(row["temperature_mean_c"]),
                "par_umol_m2_d":      max(0.0, float(row["par_umol_m2_d"])),
            }
    return result


def _rolling_backward(
    era5: Dict[date, dict], target: date, key: str
) -> Optional[float]:
    vals = []
    for i in range(_ROLLING_WINDOW):
        rec = era5.get(target - timedelta(days=i))
        if rec is None:
            return None
        vals.append(rec[key])
    return float(np.mean(vals))


def _annual_peak_rolling_doy(
    era5: Dict[date, dict], year: int, key: str
) -> Optional[int]:
    best_doy: Optional[int] = None
    best_val = -np.inf
    d = date(year, 1, 1) + timedelta(days=_ROLLING_WINDOW - 1)
    end = date(year, 12, 31)
    while d <= end:
        val = _rolling_backward(era5, d, key)
        if val is not None and val > best_val:
            best_val = val
            best_doy = d.timetuple().tm_yday
        d += timedelta(days=1)
    return best_doy


def load_porewater_chemistry() -> Dict[Tuple[str, str, date, int], dict]:
    """Return {(site, location, date, depth_cm): {SALINITY, NH4, S2}} means."""
    accum: Dict[Tuple[str, str, date, int], Dict[str, List[float]]] = defaultdict(
        lambda: {"SALINITY": [], "NH4": [], "S2": []}
    )
    with open(_POREWATER_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["TREATMENT"].strip() != "C":
                continue
            try:
                depth = int(row["DEPTH"].strip())
            except ValueError:
                continue
            if depth not in _DEPTHS:
                continue
            key = (
                row["SITE"].strip(),
                row["LOCATION"].strip(),
                _parse_date(row["DATE"]),
                depth,
            )
            for var in ("SALINITY", "NH4", "S2"):
                val_str = row[var].strip()
                if val_str not in ("", "NM", "NA"):
                    try:
                        accum[key][var].append(float(val_str))
                    except ValueError:
                        pass
    result = {}
    for k, d in accum.items():
        result[k] = {
            var: (float(np.mean(vals)) if vals else None)
            for var, vals in d.items()
        }
    return result


# ---------------------------------------------------------------------------
# SET (Surface Elevation Table) data  edi.134.12
# ---------------------------------------------------------------------------

def load_set_annual_elevation() -> Dict[Tuple[str, str, int], float]:
    """Return {(site, location, year): mean_elevation_m_navd88}.

    Combines:
    - NILTREB_marsh_surface_elevation_init.csv  — absolute elevation (m NAVD88)
      at the time of SET installation for each control plot.
    - NILTREB_marsh_elevation_change.csv  — cumulative change since installation
      in cm, sampled several times per year.

    The annual value is the mean of all measurements in that calendar year
    across all control plots at the site-location.  Sites with SET data that
    overlap the biomass record: GI/HM (1996–) and GI/LM (2001–).
    """
    # Read initial elevations
    init: Dict[Tuple[str, str, int], Tuple[date, float]] = {}
    with open(_SET_INIT_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["TREATMENT"].strip() != "C":
                continue
            try:
                elev = float(row["MARSH_SURFACE_ELEVATION"])
                d = _parse_date(row["INIT_DATE"])
                key = (row["SITE"].strip(), row["LOCATION"].strip(), int(row["PLOT"].strip()))
                init[key] = (d, elev)
            except (ValueError, KeyError):
                pass

    # Read cumulative changes and compute absolute elevation at each measurement
    by_site_yr: Dict[Tuple[str, str, int], List[float]] = defaultdict(list)
    with open(_SET_CHANGE_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["TREATMENT"].strip() != "C":
                continue
            try:
                plot_key = (row["SITE"].strip(), row["LOCATION"].strip(), int(row["PLOT"].strip()))
                if plot_key not in init:
                    continue
                d = _parse_date(row["DATE"])
                cum_cm = float(row["MEAN_ELEV_CHANGE"])
                elev_m = init[plot_key][1] + cum_cm / 100.0
                yr_key = (row["SITE"].strip(), row["LOCATION"].strip(), d.year)
                by_site_yr[yr_key].append(elev_m)
            except (ValueError, KeyError):
                pass

    return {k: float(np.mean(v)) for k, v in by_site_yr.items()}


# ---------------------------------------------------------------------------
# NOAA tidal data  (Springmaid Pier, station 8661070)
# ---------------------------------------------------------------------------
# Inundation is computed by directly counting hours above a threshold
# elevation in the observed hourly water level record.  Monthly mean water
# levels are insufficient because they discard the actual tidal oscillations
# — a diurnal tide with a ~24 h period has only ~1 inundation event per day,
# so you must resolve individual tidal cycles to count inundation correctly.
# ---------------------------------------------------------------------------

_NOAA_DATA_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"


def _fetch_noaa_hourly_chunk(
    station: str, begin_date: datetime, end_date: datetime
) -> List[Tuple[str, float]]:
    """Return list of (datetime_str, water_level_m_MLLW) for one month chunk."""
    params = {
        "product": "hourly_height",
        "station": station,
        "begin_date": begin_date.strftime("%Y%m%d"),
        "end_date": end_date.strftime("%Y%m%d"),
        "datum": "MLLW",
        "units": "metric",
        "time_zone": "gmt",
        "format": "json",
    }
    r = requests.get(_NOAA_DATA_URL, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if "error" in payload:
        msg = payload["error"]
        if isinstance(msg, dict):
            msg = msg.get("message", str(msg))
        raise RuntimeError(msg)
    rows = []
    for rec in payload.get("data", []):
        try:
            rows.append((rec["t"], float(rec["v"])))
        except (KeyError, ValueError):
            pass
    return rows


def _month_chunks(
    start_year: int, end_year: int
) -> List[Tuple[datetime, datetime]]:
    chunks = []
    for yr in range(start_year, end_year + 1):
        for mo in range(1, 13):
            t0 = datetime(yr, mo, 1)
            if mo == 12:
                t1 = datetime(yr + 1, 1, 1) - timedelta(days=1)
            else:
                t1 = datetime(yr, mo + 1, 1) - timedelta(days=1)
            chunks.append((t0, t1))
    return chunks


def download_noaa_hourly(
    station: str,
    start_year: int,
    end_year: int,
    out_csv: Path,
) -> None:
    """Download hourly water level observations from NOAA and save to CSV.

    Missing months (e.g. 1990–1991 at Springmaid Pier) are silently skipped.
    The output CSV has two columns: datetime_gmt, wl_m_mllw.
    """
    if requests is None:
        raise RuntimeError("requests library required: pip install requests")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["datetime_gmt", "wl_m_mllw"])
        for t0, t1 in _month_chunks(start_year, end_year):
            try:
                rows = _fetch_noaa_hourly_chunk(station, t0, t1)
                for dt_str, wl in rows:
                    writer.writerow([dt_str, wl])
                total += len(rows)
                time.sleep(0.25)
            except RuntimeError:
                pass   # gap in record — skip silently
    print(f"  Saved {total} hourly records to {out_csv}")


def load_noaa_hourly(csv_path: Path) -> Dict[int, List[float]]:
    """Read cached hourly water level CSV.

    Returns {year: [wl_m_mllw, ...]} with all valid observations for that year.
    """
    by_year: Dict[int, List[float]] = defaultdict(list)
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                yr = int(row["datetime_gmt"][:4])
                wl = float(row["wl_m_mllw"])
                by_year[yr].append(wl)
            except (ValueError, KeyError):
                pass
    return dict(by_year)


def compute_annual_inundation_hours(
    hourly_by_year: Dict[int, List[float]],
    elevation_m_mllw: float,
) -> Dict[int, float]:
    """Return {year: inundation_hours} by directly counting hourly observations
    above the given elevation threshold.

    Years with fewer than 8000 valid hourly observations are excluded
    (~91 % of a complete year; covers stations with short data gaps).
    """
    result = {}
    for yr, wls in hourly_by_year.items():
        if len(wls) < 8000:
            continue
        count = sum(1 for w in wls if w >= elevation_m_mllw)
        result[yr] = float(count)
    return result


# ---------------------------------------------------------------------------
# NIN suspended sediment data
# ---------------------------------------------------------------------------

_GROWING_SEASON_MONTHS = range(4, 10)   # April–September inclusive


def load_nin_ssc(growing_season_only: bool = False) -> Dict[Tuple[str, int], float]:
    """Return {(transect, year): mean_total_ssc_mg_L} from LTER.NIN.sedi.csv.

    If growing_season_only is True, only samples from April–September are used.
    Negative values are treated as missing.  Transects: OL, TC, CB.
    Date format in source file is M/D/YYYY.
    """
    by_tr_yr: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    with open(_SEDI_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            transect = row["Transect"].strip()
            try:
                parts = row["Date"].split("/")
                mo = int(parts[0])
                yr = int(parts[2])
                v = float(row["Total_sedi"])
                if v <= 0:
                    continue
                if growing_season_only and mo not in _GROWING_SEASON_MONTHS:
                    continue
                by_tr_yr[(transect, yr)].append(v)
            except (ValueError, IndexError):
                continue
    return {k: float(np.mean(v)) for k, v in by_tr_yr.items()}


# ---------------------------------------------------------------------------
# Annual peak computation
# ---------------------------------------------------------------------------

def annual_peaks(
    biomass: Dict[Tuple[str, str, date], float],
    era5: Dict[date, dict],
    porewater: Dict[Tuple[str, str, date, int], dict],
    set_elevation: Dict[Tuple[str, str, int], float],
) -> List[dict]:
    """Build one record per (site, location, year) with annual peak biomass,
    ERA5 rolling values, porewater chemistry at all depths, and SET elevation."""
    by_site_yr: Dict[Tuple[str, str, int], List[Tuple[date, float]]] = defaultdict(list)
    for (site, loc, dt), val in biomass.items():
        by_site_yr[(site, loc, dt.year)].append((dt, val))

    records = []
    for (site, loc, yr), obs in sorted(by_site_yr.items()):
        if len(obs) < 3:
            continue

        peak_date, peak_bio = max(obs, key=lambda x: x[1])
        peak_doy = peak_date.timetuple().tm_yday

        temp_roll = _rolling_backward(era5, peak_date, "temperature_mean_c")
        par_roll  = _rolling_backward(era5, peak_date, "par_umol_m2_d")

        annual_peak_temp_doy = _annual_peak_rolling_doy(era5, yr, "temperature_mean_c")
        annual_peak_par_doy  = _annual_peak_rolling_doy(era5, yr, "par_umol_m2_d")

        porewater_by_depth: Dict[int, dict] = {}
        for depth in _DEPTHS:
            pw = porewater.get((site, loc, peak_date, depth))
            porewater_by_depth[depth] = pw if pw is not None else {
                "SALINITY": None, "NH4": None, "S2": None
            }

        # Annual mean surface elevation from SET data (m NAVD88); None if unavailable
        elev_navd88 = set_elevation.get((site, loc, yr))

        records.append({
            "site": site,
            "location": loc,
            "year": yr,
            "peak_date": peak_date,
            "peak_doy": peak_doy,
            "peak_biomass_g_m2": peak_bio,
            "temp_5day_roll_c": temp_roll,
            "par_5day_roll_umol_m2_d": par_roll,
            "annual_peak_temp_doy": annual_peak_temp_doy,
            "annual_peak_par_doy": annual_peak_par_doy,
            "porewater_by_depth": porewater_by_depth,
            "elevation_m_navd88": elev_navd88,
        })

    return records


def attach_inundation(records: List[dict], hourly_by_year: Dict[int, List[float]]) -> None:
    """Add inundation fields to each record.

    'annual_inundation_by_elev': dict mapping threshold elevation (m MLLW) →
        inundation_hours, for 50 thresholds spanning 0.5–2.0 m.  Used for the
        correlation-vs-threshold diagnostic panel.

    'inundation_hours_at_measured_elev': inundation hours using the actual SET
        surface elevation for that year (m NAVD88 + _NAVD88_TO_MLLW_M offset).
        None if no SET data for that site-location-year.
    """
    elevations = np.linspace(0.5, 2.0, 50)
    inundation_tables: Dict[float, Dict[int, float]] = {}
    for elev in elevations:
        inundation_tables[elev] = compute_annual_inundation_hours(hourly_by_year, float(elev))

    for rec in records:
        yr = rec["year"]
        rec["annual_inundation_by_elev"] = {
            elev: inundation_tables[elev].get(yr)
            for elev in elevations
        }

        # Inundation at measured elevation (GI sites only)
        elev_navd88 = rec.get("elevation_m_navd88")
        if elev_navd88 is not None and yr in hourly_by_year:
            elev_mllw = elev_navd88 + _NAVD88_TO_MLLW_M
            wls = hourly_by_year[yr]
            if len(wls) >= 8000:
                rec["inundation_hours_at_measured_elev"] = float(
                    sum(1 for w in wls if w >= elev_mllw)
                )
            else:
                rec["inundation_hours_at_measured_elev"] = None
        else:
            rec["inundation_hours_at_measured_elev"] = None


def attach_elevation_gain(
    records: List[dict],
    set_elevation: Dict[Tuple[str, str, int], float],
) -> None:
    """Add 'elevation_gain_mm_prev_year' to each record.

    The gain is (mean elevation in current year) − (mean elevation in previous
    year), expressed in mm.  Positive = accretion; negative = subsidence/loss.
    Only populated for records where both years have SET data (GI/HM and GI/LM).
    """
    for rec in records:
        site, loc, yr = rec["site"], rec["location"], rec["year"]
        curr = set_elevation.get((site, loc, yr))
        prev = set_elevation.get((site, loc, yr - 1))
        if curr is not None and prev is not None:
            rec["elevation_gain_mm_prev_year"] = (curr - prev) * 1000.0
        else:
            rec["elevation_gain_mm_prev_year"] = None


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _pearson_r(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 3:
        return None
    r, _ = sp_stats.pearsonr(xs, ys)
    return float(r)


def _legend_handles() -> List[mlines.Line2D]:
    return [
        mlines.Line2D([], [], color=_SITE_COLORS[k], marker=_SITE_MARKERS[k],
                      linestyle="None", markersize=6, label=_SITE_LABELS[k])
        for k in _SITES
    ]


def _save_or_show(fig, outpath: Optional[Path]) -> None:
    if outpath:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        print(f"  Saved: {outpath}")
    else:
        plt.show()
    plt.close(fig)


_PANEL_LAYOUT = [
    ("GI", "HM", 0, 0),
    ("GI", "LM", 0, 1),
    ("OL", "HM", 1, 0),
    ("OL", "LM", 1, 1),
]


def _scatter_2x2(
    axes,
    records: List[dict],
    x_key: str,
    y_key: str,
    add_1to1: bool = False,
) -> None:
    for site, loc, row_i, col_i in _PANEL_LAYOUT:
        ax = axes[row_i][col_i]
        xs = [r[x_key] for r in records
              if r["site"] == site and r["location"] == loc
              and r[x_key] is not None and r[y_key] is not None]
        ys = [r[y_key] for r in records
              if r["site"] == site and r["location"] == loc
              and r[x_key] is not None and r[y_key] is not None]
        color = _SITE_COLORS[(site, loc)]
        marker = _SITE_MARKERS[(site, loc)]
        if xs:
            ax.scatter(xs, ys, color=color, marker=marker,
                       s=40, alpha=0.8, edgecolors="none")
            r_val = _pearson_r(xs, ys)
            if r_val is not None:
                ax.annotate(f"r = {r_val:.2f}", xy=(0.05, 0.93),
                            xycoords="axes fraction", fontsize=8,
                            va="top", color="0.3")
        ax.set_title(_SITE_LABELS[(site, loc)], fontsize=9)
        ax.grid(True, linewidth=0.3, alpha=0.4)
        if add_1to1 and xs:
            lo = min(min(xs), min(ys))
            hi = max(max(xs), max(ys))
            ax.plot([lo, hi], [lo, hi], color="0.6", linewidth=0.8,
                    linestyle="--", zorder=0)


# ---------------------------------------------------------------------------
# Climate figures (2×2)
# ---------------------------------------------------------------------------

def plot_biomass_vs_peak_temperature(
    records: List[dict], outpath: Optional[Path]
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    _scatter_2x2(axes, records, "temp_5day_roll_c", "peak_biomass_g_m2")
    xlabel = f"ERA5 {_ROLLING_WINDOW}-day rolling mean temp at peak biomass (°C)"
    ylabel = "Annual peak aboveground biomass (g m⁻²)"
    for ax in axes[1]:
        ax.set_xlabel(xlabel, fontsize=8)
    for ax in [axes[0][0], axes[1][0]]:
        ax.set_ylabel(ylabel, fontsize=8)
    fig.suptitle(
        f"Annual peak biomass vs {_ROLLING_WINDOW}-day mean temperature at peak\n"
        "North Inlet LTER (GI & OL), ERA5 1985–2024", fontsize=11)
    plt.tight_layout()
    _save_or_show(fig, outpath)


def plot_biomass_vs_par(records: List[dict], outpath: Optional[Path]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    _scatter_2x2(axes, records, "par_5day_roll_umol_m2_d", "peak_biomass_g_m2")
    xlabel = f"ERA5 {_ROLLING_WINDOW}-day rolling mean PAR at peak biomass (μmol m⁻² d⁻¹)"
    ylabel = "Annual peak aboveground biomass (g m⁻²)"
    for ax in axes[1]:
        ax.set_xlabel(xlabel, fontsize=8)
    for ax in [axes[0][0], axes[1][0]]:
        ax.set_ylabel(ylabel, fontsize=8)
    fig.suptitle(
        f"Annual peak biomass vs {_ROLLING_WINDOW}-day mean PAR at peak\n"
        "North Inlet LTER, ERA5 1985–2024", fontsize=11)
    plt.tight_layout()
    _save_or_show(fig, outpath)


def plot_biomass_timing_par(records: List[dict], outpath: Optional[Path]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    _scatter_2x2(axes, records, "annual_peak_par_doy", "peak_doy", add_1to1=True)
    xlabel = f"DOY of annual peak {_ROLLING_WINDOW}-day rolling mean PAR (ERA5)"
    ylabel = "DOY of annual peak biomass"
    for ax in axes[1]:
        ax.set_xlabel(xlabel, fontsize=8)
    for ax in [axes[0][0], axes[1][0]]:
        ax.set_ylabel(ylabel, fontsize=8)
    fig.suptitle(
        f"Timing of peak biomass vs timing of peak {_ROLLING_WINDOW}-day PAR\n"
        "North Inlet LTER, 1985–2024", fontsize=11)
    plt.tight_layout()
    _save_or_show(fig, outpath)


def plot_biomass_timing_temp(records: List[dict], outpath: Optional[Path]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    _scatter_2x2(axes, records, "annual_peak_temp_doy", "peak_doy", add_1to1=True)
    xlabel = f"DOY of annual peak {_ROLLING_WINDOW}-day rolling mean temperature (ERA5)"
    ylabel = "DOY of annual peak biomass"
    for ax in axes[1]:
        ax.set_xlabel(xlabel, fontsize=8)
    for ax in [axes[0][0], axes[1][0]]:
        ax.set_ylabel(ylabel, fontsize=8)
    fig.suptitle(
        f"Timing of peak biomass vs timing of peak {_ROLLING_WINDOW}-day temperature\n"
        "North Inlet LTER, 1985–2024", fontsize=11)
    plt.tight_layout()
    _save_or_show(fig, outpath)


# ---------------------------------------------------------------------------
# Porewater depth figures (1×5)
# ---------------------------------------------------------------------------

def _plot_porewater_depths(
    records: List[dict],
    variable: str,
    xlabel: str,
    title: str,
    outpath: Optional[Path],
) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(16, 4), sharey=True)
    for ax, depth in zip(axes, _DEPTHS):
        xs_all, ys_all = [], []
        for site, loc in _SITES:
            xs, ys = [], []
            for r in records:
                if r["site"] != site or r["location"] != loc:
                    continue
                pw = r["porewater_by_depth"].get(depth, {})
                val = pw.get(variable) if pw else None
                bio = r["peak_biomass_g_m2"]
                if val is not None and bio is not None:
                    xs.append(val)
                    ys.append(bio)
            if xs:
                ax.scatter(xs, ys, color=_SITE_COLORS[(site, loc)],
                           marker=_SITE_MARKERS[(site, loc)],
                           s=25, alpha=0.75, edgecolors="none",
                           label=_SITE_LABELS[(site, loc)])
            xs_all.extend(xs)
            ys_all.extend(ys)
        r_val = _pearson_r(xs_all, ys_all)
        n = len(xs_all)
        ann = f"r = {r_val:.2f}\nn = {n}" if r_val is not None else f"n = {n}"
        ax.annotate(ann, xy=(0.05, 0.93), xycoords="axes fraction",
                    fontsize=8, va="top", color="0.3")
        ax.set_title(f"{depth} cm", fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.grid(True, linewidth=0.3, alpha=0.4)
    axes[0].set_ylabel("Annual peak aboveground biomass (g m⁻²)", fontsize=8)
    fig.legend(handles=_legend_handles(), loc="lower center", ncol=4,
               fontsize=8, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    _save_or_show(fig, outpath)


def plot_biomass_vs_salinity_depths(
    records: List[dict], outpath: Optional[Path]
) -> None:
    _plot_porewater_depths(
        records, "SALINITY", "Porewater salinity (ppt)",
        "Annual peak biomass vs porewater salinity by depth\nNorth Inlet LTER 1993–2024",
        outpath)


def plot_biomass_vs_nh4_depths(
    records: List[dict], outpath: Optional[Path]
) -> None:
    _plot_porewater_depths(
        records, "NH4", "Porewater NH₄ (μmol L⁻¹)",
        "Annual peak biomass vs porewater NH₄ by depth\nNorth Inlet LTER 1993–2024",
        outpath)


def plot_biomass_vs_s2_depths(
    records: List[dict], outpath: Optional[Path]
) -> None:
    _plot_porewater_depths(
        records, "S2", "Porewater S²⁻ / FeS (μmol L⁻¹)",
        "Annual peak biomass vs porewater sulphide (S²⁻) by depth\nNorth Inlet LTER 1993–2024",
        outpath)


# ---------------------------------------------------------------------------
# Inundation figure
# ---------------------------------------------------------------------------

def plot_biomass_vs_inundation(
    records: List[dict], outpath: Optional[Path]
) -> None:
    """Three-panel figure.

    Left: Pearson r (inundation hours vs peak biomass) vs elevation threshold
          for all four sites.  Tidal datum reference lines shown.
    Centre: Scatter for GI/HM and GI/LM using measured SET elevation each year.
    Right:  Scatter for OL/HM and OL/LM at the threshold that maximises |r|
            (no SET data available for OL).
    """
    sample = records[0].get("annual_inundation_by_elev", {}) if records else {}
    if not sample:
        print("  No inundation data available; skipping plot.")
        return

    elevations = sorted(sample.keys())
    _DATUMS = {"MSL": 0.826, "MHW": 1.588, "MHHW": 1.707}

    fig, (ax_r, ax_gi, ax_ol) = plt.subplots(1, 3, figsize=(18, 5))

    best_elev_by_site: Dict[Tuple[str, str], float] = {}

    for site, loc in _SITES:
        site_recs = [r for r in records if r["site"] == site and r["location"] == loc]
        rs = []
        for elev in elevations:
            xs = [r["annual_inundation_by_elev"].get(elev) for r in site_recs]
            ys = [r["peak_biomass_g_m2"] for r in site_recs]
            pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
            if len(pairs) >= 3:
                rv = _pearson_r([p[0] for p in pairs], [p[1] for p in pairs])
                rs.append(rv if rv is not None else float("nan"))
            else:
                rs.append(float("nan"))
        ax_r.plot(elevations, rs, color=_SITE_COLORS[(site, loc)],
                  linewidth=1.5, label=_SITE_LABELS[(site, loc)])
        valid = [(e, r) for e, r in zip(elevations, rs) if not math.isnan(r)]
        if valid:
            best_elev_by_site[(site, loc)] = max(valid, key=lambda x: abs(x[1]))[0]

    for name, val in _DATUMS.items():
        ax_r.axvline(val, color="0.5", linewidth=0.8, linestyle=":",
                     label=f"{name} = {val:.2f} m")
    ax_r.axhline(0, color="0.7", linewidth=0.5)
    ax_r.set_xlabel("Elevation threshold (m above MLLW)", fontsize=9)
    ax_r.set_ylabel("Pearson r", fontsize=9)
    ax_r.set_title("r vs elevation threshold\n(all sites)", fontsize=9)
    ax_r.legend(fontsize=7, loc="lower left")
    ax_r.grid(True, linewidth=0.3, alpha=0.4)

    # Centre: GI with measured SET elevation
    for site, loc in [("GI", "HM"), ("GI", "LM")]:
        site_recs = [r for r in records
                     if r["site"] == site and r["location"] == loc
                     and r.get("inundation_hours_at_measured_elev") is not None]
        xs = [r["inundation_hours_at_measured_elev"] for r in site_recs]
        ys = [r["peak_biomass_g_m2"] for r in site_recs]
        if xs:
            ax_gi.scatter(xs, ys, color=_SITE_COLORS[(site, loc)],
                          marker=_SITE_MARKERS[(site, loc)], s=40, alpha=0.8,
                          edgecolors="none", label=_SITE_LABELS[(site, loc)])
            rv = _pearson_r(xs, ys)
            if rv is not None:
                ax_gi.annotate(
                    f"{_SITE_LABELS[(site,loc)]}\nr = {rv:.2f}",
                    xy=(0.04, 0.96 if loc == "HM" else 0.82),
                    xycoords="axes fraction", fontsize=7, va="top", color="0.3"
                )
    ax_gi.set_xlabel("Annual inundation hours at measured SET elevation", fontsize=9)
    ax_gi.set_ylabel("Annual peak aboveground biomass (g m⁻²)", fontsize=9)
    ax_gi.set_title("GI sites — measured elevation\n(edi.134.12 SET data)", fontsize=9)
    ax_gi.legend(fontsize=7)
    ax_gi.grid(True, linewidth=0.3, alpha=0.4)

    # Right: OL at best threshold elevation (no SET data)
    for site, loc in [("OL", "HM"), ("OL", "LM")]:
        best_elev = best_elev_by_site.get((site, loc))
        if best_elev is None:
            continue
        site_recs = [r for r in records if r["site"] == site and r["location"] == loc]
        xs = [r["annual_inundation_by_elev"].get(best_elev) for r in site_recs]
        ys = [r["peak_biomass_g_m2"] for r in site_recs]
        pairs = [(x, y) for x, y in zip(xs, ys) if x is not None]
        if pairs:
            ax_ol.scatter(
                [p[0] for p in pairs], [p[1] for p in pairs],
                color=_SITE_COLORS[(site, loc)], marker=_SITE_MARKERS[(site, loc)],
                s=40, alpha=0.8, edgecolors="none",
                label=f"{_SITE_LABELS[(site, loc)]} (z={best_elev:.2f} m MLLW)",
            )
    ax_ol.set_xlabel("Annual inundation hours at best-r elevation threshold", fontsize=9)
    ax_ol.set_ylabel("Annual peak aboveground biomass (g m⁻²)", fontsize=9)
    ax_ol.set_title("OL sites — no SET data\n(threshold at max |r|)", fontsize=9)
    ax_ol.legend(fontsize=7)
    ax_ol.grid(True, linewidth=0.3, alpha=0.4)

    fig.suptitle(
        "Tidal inundation vs annual peak biomass — North Inlet LTER\n"
        "Tide gauge: NOAA 8661070 Springmaid Pier SC (~45 km), datum MLLW",
        fontsize=10,
    )
    plt.tight_layout()
    _save_or_show(fig, outpath)


def plot_elevation_timeseries(
    records: List[dict], outpath: Optional[Path]
) -> None:
    """Two-panel figure: surface elevation (m NAVD88) and peak biomass vs year
    for GI/HM and GI/LM.  Only years with SET data are plotted.
    """
    fig, (ax_elev, ax_bio) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for site, loc in [("GI", "HM"), ("GI", "LM")]:
        site_recs = sorted(
            [r for r in records
             if r["site"] == site and r["location"] == loc
             and r.get("elevation_m_navd88") is not None],
            key=lambda r: r["year"],
        )
        if not site_recs:
            continue
        yrs = [r["year"] for r in site_recs]
        elevs = [r["elevation_m_navd88"] for r in site_recs]
        bios = [r["peak_biomass_g_m2"] for r in site_recs]
        color = _SITE_COLORS[(site, loc)]
        marker = _SITE_MARKERS[(site, loc)]
        ax_elev.plot(yrs, elevs, color=color, marker=marker, markersize=5,
                     linewidth=1.2, label=_SITE_LABELS[(site, loc)])
        ax_bio.plot(yrs, bios, color=color, marker=marker, markersize=5,
                    linewidth=1.2, label=_SITE_LABELS[(site, loc)])

    ax_elev.set_ylabel("Annual mean surface elevation (m NAVD88)", fontsize=9)
    ax_elev.set_title(
        "Marsh surface elevation from SET data (edi.134.12)\n"
        "North Inlet GI plots, control treatments",
        fontsize=9,
    )
    ax_elev.legend(fontsize=8)
    ax_elev.grid(True, linewidth=0.3, alpha=0.4)

    ax_bio.set_ylabel("Annual peak aboveground biomass (g m⁻²)", fontsize=9)
    ax_bio.set_xlabel("Year", fontsize=9)
    ax_bio.legend(fontsize=8)
    ax_bio.grid(True, linewidth=0.3, alpha=0.4)

    fig.suptitle(
        "Surface elevation and peak biomass over time — GI site, North Inlet LTER",
        fontsize=11,
    )
    plt.tight_layout()
    _save_or_show(fig, outpath)


# ---------------------------------------------------------------------------
# Elevation gain figure
# ---------------------------------------------------------------------------

def plot_biomass_vs_elevation_gain(
    records: List[dict], outpath: Optional[Path]
) -> None:
    """Scatter: annual peak biomass vs elevation gain (mm) over the previous year.

    Only GI/HM and GI/LM have SET data.  Each panel shows one site-location;
    Pearson r is annotated.  The gain is measured elevation in the current year
    minus measured elevation in the prior year (mm), capturing sediment
    accretion and organic matter production during the prior growing season.
    """
    gi_sites = [("GI", "HM"), ("GI", "LM")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, (site, loc) in zip(axes, gi_sites):
        site_recs = [r for r in records
                     if r["site"] == site and r["location"] == loc
                     and r.get("elevation_gain_mm_prev_year") is not None]
        xs = [r["elevation_gain_mm_prev_year"] for r in site_recs]
        ys = [r["peak_biomass_g_m2"] for r in site_recs]
        if xs:
            ax.scatter(xs, ys, color=_SITE_COLORS[(site, loc)],
                       marker=_SITE_MARKERS[(site, loc)],
                       s=45, alpha=0.85, edgecolors="none")
            rv = _pearson_r(xs, ys)
            n = len(xs)
            if rv is not None:
                ax.annotate(f"r = {rv:.2f}\nn = {n}", xy=(0.05, 0.93),
                            xycoords="axes fraction", fontsize=9, va="top", color="0.3")
        ax.axvline(0, color="0.6", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Elevation gain over previous year (mm)", fontsize=9)
        ax.set_ylabel("Annual peak aboveground biomass (g m⁻²)", fontsize=9)
        ax.set_title(_SITE_LABELS[(site, loc)], fontsize=10)
        ax.grid(True, linewidth=0.3, alpha=0.4)

    fig.suptitle(
        "Annual peak biomass vs prior-year surface elevation gain\n"
        "North Inlet LTER — GI plots, SET data edi.134.12 (1997–2025)",
        fontsize=11,
    )
    plt.tight_layout()
    _save_or_show(fig, outpath)


# ---------------------------------------------------------------------------
# Suspended sediment figure
# ---------------------------------------------------------------------------

def _ssc_scatter_panel(
    ax,
    records: List[dict],
    ssc_annual: Dict[Tuple[str, int], float],
    ssc_growing: Dict[Tuple[str, int], float],
    transect: str,
    use_growing: bool,
) -> None:
    """Populate one SSC scatter panel."""
    xs_all, ys_all = [], []
    for site, loc in _SITES:
        if transect == "OL" and site != "OL":
            continue
        site_recs = [r for r in records if r["site"] == site and r["location"] == loc]
        xs, ys = [], []
        for r in site_recs:
            ssc_dict = ssc_growing if use_growing else ssc_annual
            ssc_val = ssc_dict.get((transect, r["year"]))
            if ssc_val is not None:
                xs.append(ssc_val)
                ys.append(r["peak_biomass_g_m2"])
        if xs:
            ax.scatter(xs, ys, color=_SITE_COLORS[(site, loc)],
                       marker=_SITE_MARKERS[(site, loc)],
                       s=50, alpha=0.85, edgecolors="none",
                       label=_SITE_LABELS[(site, loc)])
        xs_all.extend(xs)
        ys_all.extend(ys)
    rv = _pearson_r(xs_all, ys_all)
    n = len(xs_all)
    ann = f"r = {rv:.2f}\nn = {n}" if rv is not None else f"n = {n}"
    ax.annotate(ann, xy=(0.05, 0.93), xycoords="axes fraction",
                fontsize=9, va="top", color="0.3")
    season = "Apr–Sep" if use_growing else "annual"
    ax.set_xlabel(f"{season} mean total SSC (mg L⁻¹)", fontsize=9)
    ax.set_ylabel("Annual peak aboveground biomass (g m⁻²)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.4)


def plot_biomass_vs_ssc(
    records: List[dict],
    ssc_annual: Dict[Tuple[str, int], float],
    ssc_growing: Dict[Tuple[str, int], float],
    outpath: Optional[Path],
) -> None:
    """2×2 scatter: rows = annual / growing-season SSC; cols = OL / TC transect.

    The SSC data (1981–1992) is matched to biomass records by year.
    OL transect is co-located with OL biomass sites; TC (Town Creek, main
    channel) captures the ambient creek sediment load available to all sites.
    Growing season = April–September.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for col, (transect, col_title) in enumerate([
        ("OL", "OL transect"),
        ("TC", "TC transect (main channel)"),
    ]):
        for row, (use_growing, row_title) in enumerate([
            (False, "Annual mean SSC"),
            (True,  "Growing-season SSC (Apr–Sep)"),
        ]):
            ax = axes[row][col]
            _ssc_scatter_panel(ax, records, ssc_annual, ssc_growing, transect, use_growing)
            ax.set_title(f"{col_title}\n{row_title}", fontsize=9)

    fig.suptitle(
        "Annual peak biomass vs suspended sediment concentration\n"
        "NIN LTER knb-lter-nin.8.1 (1981–1992 overlap with biomass record)",
        fontsize=11,
    )
    plt.tight_layout()
    _save_or_show(fig, outpath)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-save", action="store_true",
                   help="Display figures interactively instead of saving to figures/")
    p.add_argument("--no-download", action="store_true",
                   help="Skip NOAA API download; skip inundation figures if cache missing")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    save = not args.no_save

    print("Loading biomass data ...")
    biomass = load_biomass()
    print(f"  {len(biomass)} site-location-date means")

    print("Loading ERA5 daily data ...")
    era5 = load_era5_daily()
    print(f"  {len(era5)} daily records")

    print("Loading porewater chemistry (all depths) ...")
    porewater = load_porewater_chemistry()
    print(f"  {len(porewater)} site-location-date-depth means")

    print("Loading SET surface elevation data ...")
    set_elevation = load_set_annual_elevation()
    n_set = len(set_elevation)
    set_sites = sorted({(s, l) for s, l, y in set_elevation})
    print(f"  {n_set} site-location-year means: {set_sites}")

    print("Computing annual peaks ...")
    records = annual_peaks(biomass, era5, porewater, set_elevation)
    print(f"  {len(records)} site-location-year records")

    # ---- NOAA tidal data ----
    hourly_by_year: Dict[int, List[float]] = {}
    if not args.no_download:
        if not _NOAA_CACHE.exists():
            print(f"Downloading NOAA hourly water levels (station {_NOAA_STATION}) ...")
            print(f"  This downloads ~350k hourly observations; takes ~2 minutes ...")
            download_noaa_hourly(
                _NOAA_STATION, _NOAA_START_YEAR, _NOAA_END_YEAR, _NOAA_CACHE
            )
        else:
            print("Loading NOAA hourly water levels from cache ...")
        hourly_by_year = load_noaa_hourly(_NOAA_CACHE)
        n_total = sum(len(v) for v in hourly_by_year.values())
        print(f"  {n_total} hourly records across {len(hourly_by_year)} years")
        print("Computing annual inundation hours ...")
        attach_inundation(records, hourly_by_year)
    else:
        if _NOAA_CACHE.exists():
            print("Loading NOAA hourly water levels from cache ...")
            hourly_by_year = load_noaa_hourly(_NOAA_CACHE)
            n_total = sum(len(v) for v in hourly_by_year.values())
            print(f"  {n_total} hourly records across {len(hourly_by_year)} years")
            print("Computing annual inundation hours ...")
            attach_inundation(records, hourly_by_year)
        else:
            print("NOAA cache not found and --no-download set; skipping inundation.")

    # ---- SSC data ----
    print("Loading NIN suspended sediment data ...")
    ssc_annual  = load_nin_ssc(growing_season_only=False)
    ssc_growing = load_nin_ssc(growing_season_only=True)
    print(f"  {len(ssc_annual)} transect-year annual SSC values (1981–1992)")
    print(f"  {len(ssc_growing)} transect-year growing-season SSC values")

    # ---- Elevation gain ----
    attach_elevation_gain(records, set_elevation)
    n_gain = sum(1 for r in records if r.get("elevation_gain_mm_prev_year") is not None)
    print(f"  {n_gain} records with prior-year elevation gain")

    print("Plotting ...")

    def _path(name):
        return _FIGURE_DIR / name if save else None

    plot_biomass_vs_peak_temperature(records, _path("ni_biomass_vs_peak_temperature.png"))
    plot_biomass_vs_par(records,              _path("ni_biomass_vs_par.png"))
    plot_biomass_timing_par(records,           _path("ni_biomass_timing_par.png"))
    plot_biomass_timing_temp(records,          _path("ni_biomass_timing_temp.png"))
    plot_biomass_vs_salinity_depths(records,   _path("ni_biomass_vs_salinity_depths.png"))
    plot_biomass_vs_nh4_depths(records,        _path("ni_biomass_vs_nh4_depths.png"))
    plot_biomass_vs_s2_depths(records,         _path("ni_biomass_vs_s2_depths.png"))

    if any("annual_inundation_by_elev" in r for r in records):
        plot_biomass_vs_inundation(records,    _path("ni_biomass_vs_inundation.png"))
        plot_elevation_timeseries(records,     _path("ni_elevation_timeseries.png"))

    plot_biomass_vs_elevation_gain(records,    _path("ni_biomass_vs_elevation_gain.png"))
    plot_biomass_vs_ssc(records, ssc_annual, ssc_growing,
                                               _path("ni_biomass_vs_ssc.png"))


if __name__ == "__main__":
    main()
