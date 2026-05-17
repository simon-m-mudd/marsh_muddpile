#!/usr/bin/env python3
"""
compare_datum_methods.py

Diagnostic script: compare multiple NAVD88 datum estimation methods for
water-level stations located between two bounding NOAA reference gauges.

For a target location (default: North Inlet, SC) the script:
  1. Finds the nearest NOAA tide gauges to the north and south that have
     published NAVD88 datums.
  2. Discovers NOAA CO-OPS and USGS water-level stations in the corridor
     between the two reference gauges.
  3. Queries EPQS in parallel for all candidate stations and filters to the
     2 closest whose land-surface elevation is at or below MSL + 0.1 m
     (coastal/estuarine filter — rejects upland or streamflow stations).
  4. Downloads all records (reference gauges + 2 filtered stations) in a
     single flat ThreadPoolExecutor — all year-sized NOAA chunks and USGS
     requests fire simultaneously.
  5. Derives NAVD88 offsets for each intermediate station by five methods:
       north_2yr       — mean-WL overlap with north reference gauge, full window
       south_2yr       — mean-WL overlap with south reference gauge, full window
       north_1yr       — mean-WL overlap with north gauge, most recent year only
       usgs_alt        — USGS site-metadata alt_va (NAVD88) where available
       published_datum — NOAA datums API (NOAA stations) or USGS alt_va (USGS
                         stations); no WL overlap required
  6. Tests the EPQS tidal estimate:
       epqs_tidal — EPQS elevation minus (MHW_ref − MSL_ref), no WL data needed
  7. Produces five plots (time-series panels + offset comparison) and a
     1-yr vs 2-yr stability bar chart.
  8. Prints a wall-clock timing report for each stage.

Usage
-----
python3 scripts/compare_datum_methods.py \\
    --lat 33.35 --lon -79.18 \\
    --years 2 \\
    --output-dir scripts/figures/datum_comparison
"""

import argparse
import math
import sys
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

NOAA_MDAPI_URL  = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"
NOAA_DATA_URL   = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
NOAA_DATUMS_URL = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{sid}/datums.json"
USGS_SITE_URL   = "https://waterservices.usgs.gov/nwis/site/"
USGS_IV_URL     = "https://waterservices.usgs.gov/nwis/iv/"
EPQS_URL        = "https://epqs.nationalmap.gov/v1/json"

_FT_TO_M       = 0.3048
_TIMEOUT_NOAA  = 30
_TIMEOUT_USGS  = 120
_TIMEOUT_EPQS  = 10
_DELAY         = 0.3
_DL_WORKERS    = 16   # single flat pool for all WL downloads
_EPQS_WORKERS  = 8

# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

class Timer:
    def __init__(self):
        self._t0    = _time.perf_counter()
        self._laps: list[tuple[str, float]] = []

    def lap(self, label: str) -> float:
        t = _time.perf_counter() - self._t0
        self._laps.append((label, t))
        print(f"  ⏱  {label}: {t:.1f} s total", flush=True)
        return t

    def report(self):
        print("\n── Timing report ──────────────────────────────────────────────────")
        prev = 0.0
        for label, t in self._laps:
            print(f"  {t - prev:6.1f} s  {label}")
            prev = t
        if self._laps:
            print(f"  {'─'*52}")
            print(f"  {self._laps[-1][1]:6.1f} s  TOTAL")


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


# ---------------------------------------------------------------------------
# NOAA station catalogue
# ---------------------------------------------------------------------------

def fetch_noaa_stations(cache_path=None):
    if cache_path and Path(cache_path).exists():
        df = pd.read_csv(cache_path, dtype={"station_id": str})
        print(f"  Loaded {len(df)} NOAA stations from cache: {cache_path}", flush=True)
        return df
    print("  Fetching NOAA CO-OPS station catalogue ...", flush=True)
    r = requests.get(NOAA_MDAPI_URL, params={"type": "waterlevels"}, timeout=60)
    r.raise_for_status()
    rows = []
    for s in r.json().get("stations", []):
        try:
            rows.append({
                "station_id": str(s["id"]),
                "name":       s["name"],
                "latitude":   float(s["lat"]),
                "longitude":  float(s["lng"]),
                "status":     s.get("status", ""),
            })
        except (KeyError, ValueError):
            continue
    df = pd.DataFrame(rows)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
    print(f"  {len(df)} NOAA CO-OPS stations")
    return df


# ---------------------------------------------------------------------------
# NOAA datums
# ---------------------------------------------------------------------------

def fetch_noaa_datums(station_id):
    """Return dict of NAVD88-referenced datum values, or {} if NAVD88 unavailable."""
    url = NOAA_DATUMS_URL.format(sid=station_id)
    try:
        r = requests.get(url, params={"units": "metric", "datum": "NAVD"},
                         timeout=_TIMEOUT_NOAA)
        if not r.ok:
            return {}
        raw = {d["name"]: float(d["value"]) for d in r.json().get("datums", [])}
        offset = raw.get("NAVD88", float("nan"))
        if math.isnan(offset):
            return {}
        result = {"NAVD88_offset": offset}
        for key in ("MHW", "MHHW", "MSL", "MTL", "MLW", "MLLW"):
            if key in raw:
                result[key] = round(raw[key] - offset, 5)
        return result
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Bounding reference gauges
# ---------------------------------------------------------------------------

def find_bounding_gauges(lat, lon, noaa_stations, max_candidates=25):
    """Return (north_info, south_info) dicts for nearest NOAA gauges with NAVD88."""
    df = noaa_stations.copy()
    df["dist_km"] = df.apply(
        lambda r: haversine_km(r["longitude"], r["latitude"], lon, lat), axis=1
    )
    north_cands = df[df["latitude"] > lat].sort_values("dist_km").head(max_candidates)
    south_cands = df[df["latitude"] < lat].sort_values("dist_km").head(max_candidates)

    def first_with_datums(cands, direction):
        for _, row in cands.iterrows():
            _time.sleep(_DELAY)
            datums = fetch_noaa_datums(row["station_id"])
            if datums:
                print(f"    {direction}: {row['name']} ({row['station_id']})  "
                      f"{row['dist_km']:.1f} km", flush=True)
                return {
                    "id":      row["station_id"],
                    "name":    row["name"],
                    "lat":     row["latitude"],
                    "lon":     row["longitude"],
                    "dist_km": float(row["dist_km"]),
                    "datums":  datums,
                }
        return None

    n_gauge = first_with_datums(north_cands, "N")
    s_gauge = first_with_datums(south_cands, "S")
    return n_gauge, s_gauge


# ---------------------------------------------------------------------------
# Intermediate station discovery
# ---------------------------------------------------------------------------

def _fetch_usgs_coastal_stations(bbox):
    """Return DataFrame of active USGS ES/ST stations in bbox."""
    xmin, ymin, xmax, ymax = bbox
    rows = []
    for site_type in ("ES", "ST"):
        params = {
            "format":      "rdb",
            "siteStatus":  "active",
            "parameterCd": "00065",
            "siteType":    site_type,
            "bBox":        f"{xmin:.4f},{ymin:.4f},{xmax:.4f},{ymax:.4f}",
        }
        try:
            r = requests.get(USGS_SITE_URL, params=params, timeout=60)
            if not r.ok:
                continue
            lines = [ln for ln in r.text.splitlines()
                     if ln.strip() and not ln.startswith("#")]
            if len(lines) < 3:
                continue
            df = pd.read_csv(StringIO("\n".join(lines)), sep="\t", dtype=str)
            df = df.iloc[1:].copy()
            df["latitude"]  = pd.to_numeric(df.get("dec_lat_va",  pd.Series(dtype=str)), errors="coerce")
            df["longitude"] = pd.to_numeric(df.get("dec_long_va", pd.Series(dtype=str)), errors="coerce")
            df["alt_va"]    = pd.to_numeric(df.get("alt_va",      pd.Series(dtype=str)), errors="coerce")
            df = df.dropna(subset=["latitude", "longitude"])
            for _, row in df.iterrows():
                alt_datum  = str(row.get("alt_datum_cd", "")).strip().upper()
                alt_navd88 = float("nan")
                if alt_datum == "NAVD88" and not pd.isna(row["alt_va"]):
                    alt_navd88 = float(row["alt_va"]) * _FT_TO_M
                rows.append({
                    "source":       "usgs",
                    "id":           str(row.get("site_no", "")).strip(),
                    "name":         str(row.get("station_nm", "")).strip(),
                    "lat":          float(row["latitude"]),
                    "lon":          float(row["longitude"]),
                    "alt_navd88_m": alt_navd88,
                })
        except Exception:
            pass
        _time.sleep(_DELAY)
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows)
            .drop_duplicates(subset=["id"])
            .reset_index(drop=True))


def find_candidate_stations(lat_min, lat_max, lon_min, lon_max, noaa_stations):
    """Return all candidate NOAA + USGS stations in the corridor (unsorted, unfiltered)."""
    stations = []

    noaa_in = noaa_stations[
        noaa_stations["latitude"].between(lat_min, lat_max) &
        noaa_stations["longitude"].between(lon_min, lon_max)
    ]
    for _, row in noaa_in.iterrows():
        stations.append({
            "source":       "noaa",
            "id":           row["station_id"],
            "name":         row["name"],
            "lat":          row["latitude"],
            "lon":          row["longitude"],
            "alt_navd88_m": float("nan"),
        })

    try:
        usgs = _fetch_usgs_coastal_stations((lon_min, lat_min, lon_max, lat_max))
        for _, row in usgs.iterrows():
            stations.append(row.to_dict())
    except Exception as exc:
        print(f"    USGS station discovery failed: {exc}", file=sys.stderr)

    return stations


# ---------------------------------------------------------------------------
# EPQS
# ---------------------------------------------------------------------------

def _query_epqs_one(lat, lon):
    params = {"x": lon, "y": lat, "wkid": 4326,
              "units": "Meters", "includeDate": "false"}
    try:
        r = requests.get(EPQS_URL, params=params, timeout=_TIMEOUT_EPQS)
        if not r.ok:
            return float("nan")
        val = r.json().get("value")
        if val is None:
            return float("nan")
        fval = float(val)
        return fval if fval > -1e5 else float("nan")
    except Exception:
        return float("nan")


def query_epqs_parallel(stations):
    """Return dict station_id → NAVD88 land-surface elevation (m)."""
    def _q(st):
        return st["id"], _query_epqs_one(st["lat"], st["lon"])

    results = {}
    with ThreadPoolExecutor(max_workers=_EPQS_WORKERS) as ex:
        for sid, val in ex.map(_q, stations):
            results[sid] = val
    return results


# ---------------------------------------------------------------------------
# Data download — single flat pool for all stations + chunks
# ---------------------------------------------------------------------------

def _noaa_chunks(start, end, max_days=365):
    current = start
    while current <= end:
        chunk_end = min(end, current + timedelta(days=max_days - 1))
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def _noaa_chunk_job(station_id, datum, chunk_start, chunk_end):
    """Fetch one NOAA year-chunk; returns (station_id, rows_list) or (station_id, None).

    Uses hourly_height product (365-day limit) instead of water_level+interval=h
    (which is capped at 31 days).  Both return hourly NAVD88 or STND water levels.
    """
    params = {
        "product":     "hourly_height",
        "application": "datum_comparison",
        "begin_date":  chunk_start.strftime("%Y%m%d"),
        "end_date":    chunk_end.strftime("%Y%m%d"),
        "station":     station_id,
        "datum":       datum,
        "time_zone":   "gmt",
        "units":       "metric",
        "format":      "json",
    }
    try:
        r = requests.get(NOAA_DATA_URL, params=params, timeout=_TIMEOUT_NOAA)
        if not r.ok:
            return station_id, None
        payload = r.json()
        if "error" in payload:
            # API-level "no data" — expected for gaps, not a warning
            return station_id, []
        rows = []
        for item in payload.get("data", []):
            try:
                rows.append({"t": item["t"], "v_m": float(item["v"])})
            except (KeyError, ValueError):
                pass
        return station_id, rows
    except Exception:
        return station_id, None


def _usgs_job(site_no, start, end):
    """Fetch USGS gage-height for the full period; returns (site_no, rows_list) or None."""
    params = {
        "format":      "json",
        "sites":       str(site_no),
        "startDT":     start.strftime("%Y-%m-%d"),
        "endDT":       end.strftime("%Y-%m-%d"),
        "parameterCd": "00065",
        "siteStatus":  "all",
    }
    try:
        r = requests.get(USGS_IV_URL, params=params, timeout=_TIMEOUT_USGS)
        if not r.ok:
            return site_no, None
        payload = r.json()
        rows = []
        for ts in payload.get("value", {}).get("timeSeries", []):
            unit    = ts.get("variable", {}).get("unit", {}).get("unitCode", "")
            is_feet = "ft" in unit.lower()
            for block in ts.get("values", []):
                for item in block.get("value", []):
                    try:
                        val = float(item["value"])
                        if val <= -999990:        # USGS missing-value sentinel
                            continue
                        if is_feet:
                            val *= _FT_TO_M
                        rows.append({"t": item["dateTime"], "v_m": val})
                    except (KeyError, ValueError):
                        pass
        return site_no, rows
    except Exception:
        return site_no, None


def _assemble_ts(raw_rows, source):
    """Build a clean hourly DataFrame from a list of raw row dicts."""
    if not raw_rows:
        return pd.DataFrame(columns=["t", "v_m"])
    df = pd.DataFrame(raw_rows)
    df["v_m"] = pd.to_numeric(df["v_m"], errors="coerce")
    df["t"]   = pd.to_datetime(df["t"], errors="coerce", utc=True).dt.tz_localize(None)
    df = df.dropna(subset=["t", "v_m"]).sort_values("t")
    # Resample to hourly — normalises USGS 15-min cadence and reduces volume
    df = (df.set_index("t")
            .resample("h")["v_m"]
            .mean()
            .dropna()
            .reset_index())
    return df


def download_all_parallel(ref_gauges, intermediate_stations, start, end):
    """
    Download all WL records in a single flat ThreadPoolExecutor.

    ref_gauges         : list of dicts {id, datum} for reference NOAA gauges
    intermediate_stations : list of station dicts with source, id fields
    start / end        : datetime window

    Returns dict station_id → DataFrame(t, v_m), hourly, UTC-naive.
    """
    # Build job list: (tag, callable_args...)
    # tag is used to map results back to stations
    jobs = []
    for g in ref_gauges:
        for s, e in _noaa_chunks(start, end):
            jobs.append(("noaa", g["id"], g["datum"], s, e))

    for st in intermediate_stations:
        if st["source"] == "noaa":
            for s, e in _noaa_chunks(start, end):
                jobs.append(("noaa", st["id"], "STND", s, e))
        else:
            jobs.append(("usgs", st["id"], None, start, end))

    print(f"  Submitting {len(jobs)} download jobs to {_DL_WORKERS} workers ...",
          flush=True)

    raw: dict[str, list] = {}  # station_id → accumulated rows
    http_errors: set[str] = set()

    def _run(job):
        if job[0] == "noaa":
            _, sid, datum, s, e = job
            return _noaa_chunk_job(sid, datum, s, e)
        else:
            _, sid, _, s, e = job
            return _usgs_job(sid, s, e)

    with ThreadPoolExecutor(max_workers=_DL_WORKERS) as ex:
        futures = {ex.submit(_run, job): job for job in jobs}
        done = 0
        for fut in as_completed(futures):
            done += 1
            print(f"    {done}/{len(jobs)} chunks complete\r",
                  end="", flush=True)
            try:
                sid, rows = fut.result()
            except Exception:
                continue
            if rows is None:
                http_errors.add(sid)
            else:
                raw.setdefault(sid, []).extend(rows)

    print()  # newline after \r progress

    if http_errors:
        print(f"  Note: HTTP errors for {len(http_errors)} station(s): "
              f"{', '.join(sorted(http_errors))}", file=sys.stderr)

    # Assemble per-station DataFrames
    source_map = {st["id"]: st["source"] for st in intermediate_stations}
    for g in ref_gauges:
        source_map[g["id"]] = "noaa"

    result = {}
    all_ids = {j[1] for j in jobs}
    for sid in all_ids:
        result[sid] = _assemble_ts(raw.get(sid, []), source_map.get(sid, "noaa"))

    return result


# ---------------------------------------------------------------------------
# Datum offset computation
# ---------------------------------------------------------------------------

def compute_offset(ref_ts, coastal_ts, min_hours=24 * 30):
    """Compute offset_m = mean(ref_NAVD88) - mean(coastal_raw) over overlap.

    Returns (offset_m, n_overlap_hours).
    """
    if ref_ts.empty or coastal_ts.empty:
        return float("nan"), 0

    ref   = ref_ts.set_index("t")["v_m"].rename("ref")
    coast = coastal_ts.set_index("t")["v_m"].rename("coast")
    merged = pd.concat([ref, coast], axis=1).dropna()

    if len(merged) < min_hours:
        return float("nan"), len(merged)

    return float(merged["ref"].mean() - merged["coast"].mean()), len(merged)


def epqs_tidal_offset(epqs_elevation, datums, coastal_ts):
    """Estimate NAVD88 offset using EPQS + tidal-frame assumption.

    Assumes station elevation ≈ MHW (typical for a dock or bank sensor):
        epqs_msl = epqs_elevation - (MHW_ref - MSL_ref)
        offset_m = epqs_msl - mean(coastal_raw)

    Returns nan on failure.
    """
    if math.isnan(epqs_elevation) or coastal_ts.empty:
        return float("nan")
    mhw = datums.get("MHW", float("nan"))
    msl = datums.get("MSL", float("nan"))
    if math.isnan(mhw) or math.isnan(msl):
        return float("nan")
    epqs_msl    = epqs_elevation - (mhw - msl)
    coast_mean  = float(coastal_ts["v_m"].mean())
    return epqs_msl - coast_mean


def fetch_published_datums_parallel(intermediate_stations):
    """Fetch published NAVD88 datum offsets for all intermediate stations.

    For NOAA stations: queries the NOAA datums API in parallel.
      NAVD88_offset = height of station datum above NAVD88.
      Data downloaded as STND, so offset to add = -NAVD88_offset.
    For USGS stations: reads alt_navd88_m from station dict.
      USGS returns gage height above gauge datum, so offset to add = alt_navd88_m.

    Returns dict {station_id: offset_m} — nan where unavailable.
    """
    results = {st["id"]: float("nan") for st in intermediate_stations}

    noaa_stations = [st for st in intermediate_stations if st["source"] == "noaa"]
    usgs_stations = [st for st in intermediate_stations if st["source"] != "noaa"]

    def _fetch_noaa(st):
        datums = fetch_noaa_datums(st["id"])
        offset = -datums["NAVD88_offset"] if datums else float("nan")
        return st["id"], offset

    if noaa_stations:
        with ThreadPoolExecutor(max_workers=min(8, len(noaa_stations))) as ex:
            for sid, offset in ex.map(_fetch_noaa, noaa_stations):
                results[sid] = offset

    for st in usgs_stations:
        alt = st.get("alt_navd88_m", float("nan"))
        results[st["id"]] = float(alt) if not math.isnan(float(alt if alt is not None else float("nan"))) else float("nan")

    return results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_ts_panel(ax, series_dict, title, ref_ts=None, ref_label=None):
    for label, ts in series_dict.items():
        if ts is None or ts.empty:
            continue
        daily = ts.set_index("t").resample("D")["v_m"].mean().dropna()
        ax.plot(daily.index, daily.values, lw=0.9, alpha=0.85, label=label)
    if ref_ts is not None and not ref_ts.empty:
        daily = ref_ts.set_index("t").resample("D")["v_m"].mean().dropna()
        ax.plot(daily.index, daily.values, lw=1.2, color="k", alpha=0.4,
                ls="--", label=ref_label or "Reference")
    ax.axhline(0, color="k", lw=0.5, ls=":", alpha=0.3)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("m NAVD88", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    if series_dict:
        ax.legend(fontsize=7, ncol=2, loc="upper right")


def _plot_offset_comparison(ax, station_labels, offsets_by_method):
    methods = list(offsets_by_method.keys())
    n       = len(station_labels)
    x       = np.arange(n)
    width   = min(0.8 / max(len(methods), 1), 0.2)
    colors  = plt.cm.tab10(np.linspace(0, 0.6, len(methods)))

    for i, (method, off_dict) in enumerate(offsets_by_method.items()):
        vals = [off_dict.get(lbl, float("nan")) for lbl in station_labels]
        ax.bar(x + i * width, vals, width,
               label=method, color=colors[i], alpha=0.85, edgecolor="white")

    mid = width * (len(methods) - 1) / 2
    ax.set_xticks(x + mid)
    ax.set_xticklabels([lbl[:24] for lbl in station_labels],
                       rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Offset added to raw (m)", fontsize=8)
    ax.set_title("NAVD88 datum offset by method", fontsize=9)
    ax.legend(fontsize=7)
    ax.axhline(0, color="k", lw=0.5)


def _plot_1yr_vs_2yr(ax, station_labels, off_2yr, off_1yr):
    x = np.arange(len(station_labels))
    ax.bar(x - 0.2, off_2yr, 0.35, label="North gauge 2 yr",
           alpha=0.85, color="steelblue")
    ax.bar(x + 0.2, off_1yr, 0.35, label="North gauge 1 yr",
           alpha=0.85, color="darkorange")
    for xi, v2, v1 in zip(x, off_2yr, off_1yr):
        if not (math.isnan(v2) or math.isnan(v1)):
            ymax = max(abs(v2), abs(v1)) + 0.05
            ax.text(xi, ymax, f"Δ{v2-v1:+.3f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl[:24] for lbl in station_labels],
                       rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Offset added to raw (m)", fontsize=8)
    ax.set_title("Datum stability: 1-year vs 2-year record\n"
                 "(Δ = 2yr − 1yr; closer to zero = stable)", fontsize=9)
    ax.legend(fontsize=8)
    ax.axhline(0, color="k", lw=0.5)


def _plot_epqs_scatter(ax, station_labels, epqs_vals, north_offs, datums_n):
    mhw = datums_n.get("MHW", float("nan"))
    msl = datums_n.get("MSL", float("nan"))

    xs, ys, lbls = [], [], []
    for lbl, epqs, off in zip(station_labels, epqs_vals, north_offs):
        if not (math.isnan(epqs) or math.isnan(off)):
            xs.append(epqs)
            ys.append(off)
            lbls.append(lbl[:16])

    if xs:
        ax.scatter(xs, ys, s=60, zorder=3, color="steelblue",
                   edgecolors="k", lw=0.5)
        for x, y, lbl in zip(xs, ys, lbls):
            ax.annotate(lbl, (x, y), textcoords="offset points",
                        xytext=(5, 2), fontsize=7)
        lo = min(min(xs), min(ys)) - 0.2
        hi = max(max(xs), max(ys)) + 0.2
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.7, alpha=0.35, label="1:1")

    if not math.isnan(mhw):
        ax.axvline(mhw, color="tomato",    lw=0.9, ls=":", alpha=0.7,
                   label=f"N-ref MHW = {mhw:+.3f} m")
    if not math.isnan(msl):
        ax.axhline(msl, color="steelblue", lw=0.9, ls=":", alpha=0.7,
                   label=f"N-ref MSL = {msl:+.3f} m")

    ax.set_xlabel("EPQS land-surface (m NAVD88)", fontsize=8)
    ax.set_ylabel("Overlap offset (m)", fontsize=8)
    ax.set_title("EPQS elevation vs overlap-derived datum\n"
                 "(stations near MHW should sit left of MHW line)", fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Compare NAVD88 datum methods for WL stations between "
                    "two bounding NOAA reference gauges.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--lat", type=float, default=33.35,
                   help="Target latitude (default: North Inlet, SC)")
    p.add_argument("--lon", type=float, default=-79.18,
                   help="Target longitude")
    p.add_argument("--years", type=float, default=2.0,
                   help="Years of WL history to download")
    p.add_argument("--corridor-lon-deg", type=float, default=0.6,
                   help="Longitude half-width of station search corridor (deg)")
    p.add_argument("--elev-threshold-m", type=float, default=0.1,
                   help="Max EPQS elevation above MSL_ref to accept a station (m)")
    p.add_argument("--output-dir", default="scripts/figures/datum_comparison")
    p.add_argument("--cache-dir", default="./tidal_datum_cache",
                   help="Cache directory for NOAA station catalogue")
    p.add_argument("--no-epqs", action="store_true",
                   help="Skip EPQS filter (accept all candidates, up to 2 closest)")
    return p


def main():
    args   = build_parser().parse_args()
    timer  = Timer()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    end_date  = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    start_2yr = end_date - timedelta(days=round(365 * args.years))
    start_1yr = end_date - timedelta(days=365)

    print(f"\n{'='*68}")
    print(f"  Datum comparison diagnostic")
    print(f"  Target  : ({args.lat:.4f}°N, {args.lon:.4f}°E)")
    print(f"  Period  : {start_2yr.date()} → {end_date.date()} ({args.years:.1f} yr)")
    print(f"  Output  : {out_dir}")
    print(f"{'='*68}\n")

    # ── 1. NOAA station catalogue ─────────────────────────────────────────────
    print("[1] NOAA CO-OPS station catalogue")
    noaa_stations = fetch_noaa_stations(
        cache_path=str(Path(args.cache_dir) / "noaa_stations.csv")
    )
    timer.lap("Station catalogue loaded")

    # ── 2. Bounding reference gauges ──────────────────────────────────────────
    print("\n[2] Finding bounding NOAA reference gauges (NAVD88) ...")
    n_gauge, s_gauge = find_bounding_gauges(args.lat, args.lon, noaa_stations)

    if n_gauge is None or s_gauge is None:
        print("ERROR: could not find bounding NOAA gauges with NAVD88.", file=sys.stderr)
        sys.exit(1)

    dn, ds = n_gauge["datums"], s_gauge["datums"]
    msl_ref = ((dn.get("MSL", float("nan")) + ds.get("MSL", float("nan"))) / 2
               if not math.isnan(dn.get("MSL", float("nan"))) else
               dn.get("MSL", 0.0))

    print(f"\n  North: {n_gauge['name']} ({n_gauge['id']})  "
          f"{n_gauge['dist_km']:.1f} km  "
          f"MHW={dn.get('MHW','?'):+.3f}  MSL={dn.get('MSL','?'):+.3f}  "
          f"MLLW={dn.get('MLLW','?'):+.3f} m NAVD88")
    print(f"  South: {s_gauge['name']} ({s_gauge['id']})  "
          f"{s_gauge['dist_km']:.1f} km  "
          f"MHW={ds.get('MHW','?'):+.3f}  MSL={ds.get('MSL','?'):+.3f}  "
          f"MLLW={ds.get('MLLW','?'):+.3f} m NAVD88")
    timer.lap("Bounding gauges identified")

    # ── 3. Candidate intermediate stations ───────────────────────────────────
    print("\n[3] Discovering candidate stations in corridor ...")
    lat_min = min(n_gauge["lat"], s_gauge["lat"])
    lat_max = max(n_gauge["lat"], s_gauge["lat"])
    lon_min = args.lon - args.corridor_lon_deg
    lon_max = args.lon + args.corridor_lon_deg

    candidates = find_candidate_stations(
        lat_min, lat_max, lon_min, lon_max, noaa_stations
    )
    ref_ids = {n_gauge["id"], s_gauge["id"]}
    candidates = [s for s in candidates if s["id"] not in ref_ids]
    print(f"  {len(candidates)} candidates before EPQS filter")
    timer.lap("Candidates discovered")

    # ── 4. EPQS filter ───────────────────────────────────────────────────────
    if not args.no_epqs and candidates:
        print(f"\n[4] EPQS queries for {len(candidates)} candidates "
              f"(threshold: EPQS ≤ {msl_ref:+.3f} + {args.elev_threshold_m} m) ...")
        epqs_all = query_epqs_parallel(candidates)

        elev_threshold = msl_ref + args.elev_threshold_m
        for c in candidates:
            val = epqs_all.get(c["id"], float("nan"))
            keep = (not math.isnan(val)) and (val <= elev_threshold)
            print(f"    [{c['source'].upper():4s}] {c['id']}  "
                  f"EPQS={val:+.3f} m  {'KEEP' if keep else 'REJECT (too high)'}")
            c["_epqs"] = val

        filtered = [c for c in candidates
                    if not math.isnan(c.get("_epqs", float("nan")))
                    and c["_epqs"] <= elev_threshold]

        if not filtered:
            print("  WARNING: no stations pass elevation filter — "
                  "relaxing to all candidates", file=sys.stderr)
            filtered = candidates
    else:
        print("\n[4] EPQS filter skipped")
        epqs_all = {c["id"]: float("nan") for c in candidates}
        for c in candidates:
            c["_epqs"] = float("nan")
        filtered = candidates

    # Sort by distance to target, take 2 closest
    for c in filtered:
        c["_dist_km"] = haversine_km(c["lon"], c["lat"], args.lon, args.lat)
    filtered.sort(key=lambda c: c["_dist_km"])
    intermediate = filtered[:2]

    print(f"\n  Selected {len(intermediate)} intermediate station(s):")
    for st in intermediate:
        epqs_str = f"  EPQS={st['_epqs']:+.3f} m" if not math.isnan(st["_epqs"]) else ""
        print(f"    [{st['source'].upper():4s}] {st['name'][:40]} ({st['id']})  "
              f"{st['_dist_km']:.1f} km{epqs_str}")

    timer.lap("EPQS filter complete")

    if not intermediate:
        print("No intermediate stations selected — exiting.")
        sys.exit(0)

    # ── 5. Download all records in one flat pool ──────────────────────────────
    print(f"\n[5] Downloading all records (flat pool, {_DL_WORKERS} workers) ...")
    ref_jobs = [
        {"id": n_gauge["id"], "datum": "NAVD"},
        {"id": s_gauge["id"], "datum": "NAVD"},
    ]
    all_ts = download_all_parallel(ref_jobs, intermediate, start_2yr, end_date)

    n_ts = all_ts.get(n_gauge["id"], pd.DataFrame(columns=["t", "v_m"]))
    s_ts = all_ts.get(s_gauge["id"], pd.DataFrame(columns=["t", "v_m"]))

    for g, ts in [(n_gauge, n_ts), (s_gauge, s_ts)]:
        print(f"  Ref {g['id']} ({g['name'][:30]}): {len(ts)} hourly records")
    for st in intermediate:
        ts = all_ts.get(st["id"], pd.DataFrame())
        print(f"  Int {st['id']} ({st['name'][:30]}): {len(ts)} hourly records")

    timer.lap("All records downloaded")

    # ── 6. Compute offsets ────────────────────────────────────────────────────
    print("\n[6] Computing datum offsets ...")
    offsets: dict[str, dict] = {}

    for st in intermediate:
        sid = st["id"]
        ts  = all_ts.get(sid, pd.DataFrame(columns=["t", "v_m"]))
        d   = {}

        off, n_hrs = compute_offset(n_ts, ts)
        d["north_2yr"] = off
        off2, n_hrs2 = compute_offset(s_ts, ts)
        d["south_2yr"] = off2
        print(f"  {sid}  north_2yr={off:+.4f} m ({n_hrs} hrs)  "
              f"south_2yr={off2:+.4f} m ({n_hrs2} hrs)")

        ts_1yr   = ts[ts["t"] >= pd.Timestamp(start_1yr)]   if not ts.empty   else ts
        n_ts_1yr = n_ts[n_ts["t"] >= pd.Timestamp(start_1yr)] if not n_ts.empty else n_ts
        off1, n_hrs1 = compute_offset(n_ts_1yr, ts_1yr)
        d["north_1yr"] = off1
        print(f"  {sid}  north_1yr={off1:+.4f} m ({n_hrs1} hrs)")

        if not math.isnan(st.get("alt_navd88_m", float("nan"))):
            d["usgs_alt"] = st["alt_navd88_m"]
            print(f"  {sid}  usgs_alt ={d['usgs_alt']:+.4f} m (site metadata)")

        epqs = epqs_all.get(sid, float("nan"))
        d["epqs_tidal"] = epqs_tidal_offset(epqs, dn, ts)
        if not math.isnan(d["epqs_tidal"]):
            print(f"  {sid}  epqs_tidal={d['epqs_tidal']:+.4f} m  "
                  f"(EPQS={epqs:.3f} m)")

        offsets[sid] = d

    timer.lap("Offsets computed")

    # ── 6a. Published datum offsets ───────────────────────────────────────────
    print("\n[6a] Fetching published datum offsets ...")
    published = fetch_published_datums_parallel(intermediate)
    for st in intermediate:
        sid = st["id"]
        poff = published.get(sid, float("nan"))
        if not math.isnan(poff):
            offsets[sid]["published_datum"] = poff
            print(f"  {sid}  published_datum={poff:+.4f} m  "
                  f"({'NOAA datums API' if st['source'] == 'noaa' else 'USGS alt_va'})")
        else:
            print(f"  {sid}  published_datum=— (not available)")
    timer.lap("Published datums fetched")

    # ── 7. Build normalised time series ───────────────────────────────────────
    def _apply(ts, offset_m):
        if ts.empty or math.isnan(offset_m):
            return pd.DataFrame(columns=["t", "v_m"])
        out = ts.copy()
        out["v_m"] = out["v_m"] + offset_m
        return out

    north_norm, south_norm, epqs_norm, published_norm = {}, {}, {}, {}
    for st in intermediate:
        sid  = st["id"]
        ts   = all_ts.get(sid, pd.DataFrame(columns=["t", "v_m"]))
        lbl  = f"{st['name'][:18]} ({sid})"
        o    = offsets.get(sid, {})
        north_norm[lbl]     = _apply(ts, o.get("north_2yr",      float("nan")))
        south_norm[lbl]     = _apply(ts, o.get("south_2yr",      float("nan")))
        epqs_norm[lbl]      = _apply(ts, o.get("epqs_tidal",     float("nan")))
        published_norm[lbl] = _apply(ts, o.get("published_datum", float("nan")))

    # ── 8. Plots ──────────────────────────────────────────────────────────────
    print("\n[7] Plotting ...")

    station_names_short = [f"{st['name'][:22]}\n({st['id']})" for st in intermediate]
    station_names_22    = [f"{st['name'][:22]} ({st['id']})"  for st in intermediate]

    methods_offsets = {
        "north_2yr":      {sn: offsets.get(st["id"], {}).get("north_2yr",       float("nan"))
                           for sn, st in zip(station_names_22, intermediate)},
        "south_2yr":      {sn: offsets.get(st["id"], {}).get("south_2yr",       float("nan"))
                           for sn, st in zip(station_names_22, intermediate)},
        "north_1yr":      {sn: offsets.get(st["id"], {}).get("north_1yr",       float("nan"))
                           for sn, st in zip(station_names_22, intermediate)},
        "usgs_alt":       {sn: offsets.get(st["id"], {}).get("usgs_alt",        float("nan"))
                           for sn, st in zip(station_names_22, intermediate)},
        "epqs_tidal":     {sn: offsets.get(st["id"], {}).get("epqs_tidal",      float("nan"))
                           for sn, st in zip(station_names_22, intermediate)},
        "published_datum":{sn: offsets.get(st["id"], {}).get("published_datum", float("nan"))
                           for sn, st in zip(station_names_22, intermediate)},
    }

    # Main 5-panel comparison
    fig, axes = plt.subplots(5, 1, figsize=(13, 20), constrained_layout=True)
    fig.suptitle(
        f"NAVD88 datum comparison  ({args.lat:.3f}°N, {args.lon:.3f}°E)\n"
        f"N-ref: {n_gauge['name']} ({n_gauge['id']})  |  "
        f"S-ref: {s_gauge['name']} ({s_gauge['id']})",
        fontsize=10,
    )
    _plot_ts_panel(axes[0], north_norm, ref_ts=n_ts,
                   ref_label=f"N ref: {n_gauge['name']}",
                   title=f"Normalised — north reference ({n_gauge['name']}, 2 yr)")
    _plot_ts_panel(axes[1], south_norm, ref_ts=s_ts,
                   ref_label=f"S ref: {s_gauge['name']}",
                   title=f"Normalised — south reference ({s_gauge['name']}, 2 yr)")
    _plot_ts_panel(axes[2], epqs_norm,
                   title="Normalised — EPQS tidal estimate (station ≈ MHW assumption)")
    _plot_ts_panel(axes[3], published_norm,
                   title="Normalised — published datum (NOAA datums API / USGS alt_va)")
    _plot_offset_comparison(axes[4], station_names_22, methods_offsets)

    path1 = out_dir / "datum_comparison.png"
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path1}")

    # 1-yr vs 2-yr stability
    off_2yr = [offsets.get(st["id"], {}).get("north_2yr", float("nan"))
               for st in intermediate]
    off_1yr = [offsets.get(st["id"], {}).get("north_1yr", float("nan"))
               for st in intermediate]

    fig2, ax2 = plt.subplots(figsize=(max(7, len(intermediate) * 2.5), 5),
                              constrained_layout=True)
    _plot_1yr_vs_2yr(ax2, station_names_22, off_2yr, off_1yr)
    path2 = out_dir / "datum_1yr_vs_2yr.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved {path2}")

    # EPQS scatter
    epqs_vals  = [epqs_all.get(st["id"], float("nan")) for st in intermediate]
    north_offs = [offsets.get(st["id"], {}).get("north_2yr", float("nan"))
                  for st in intermediate]
    fig3, ax3 = plt.subplots(figsize=(6, 5), constrained_layout=True)
    _plot_epqs_scatter(ax3, station_names_22, epqs_vals, north_offs, dn)
    path3 = out_dir / "epqs_vs_overlap.png"
    fig3.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved {path3}")

    timer.lap("Figures saved")

    # ── 9. Timing report + summary table ──────────────────────────────────────
    timer.report()

    print("\n── Datum offset summary (m, added to raw readings to get NAVD88) ──────────")
    rows = []
    for st in intermediate:
        sid = st["id"]
        o   = offsets.get(sid, {})
        n2  = o.get("north_2yr",      float("nan"))
        s2  = o.get("south_2yr",      float("nan"))
        n1  = o.get("north_1yr",      float("nan"))
        ua  = o.get("usgs_alt",       float("nan"))
        et  = o.get("epqs_tidal",     float("nan"))
        pd_ = o.get("published_datum", float("nan"))
        rows.append({
            "station":         st["name"][:35],
            "id":              sid,
            "src":             st["source"],
            "epqs_m":          f"{st['_epqs']:+.3f}" if not math.isnan(st["_epqs"]) else "—",
            "north_2yr":       f"{n2:+.4f}" if not math.isnan(n2) else "—",
            "south_2yr":       f"{s2:+.4f}" if not math.isnan(s2) else "—",
            "N_S_diff":        f"{n2-s2:+.4f}" if not (math.isnan(n2) or math.isnan(s2)) else "—",
            "north_1yr":       f"{n1:+.4f}" if not math.isnan(n1) else "—",
            "2yr_1yr_diff":    f"{n2-n1:+.4f}" if not (math.isnan(n2) or math.isnan(n1)) else "—",
            "usgs_alt":        f"{ua:+.4f}" if not math.isnan(ua) else "—",
            "epqs_tidal":      f"{et:+.4f}" if not math.isnan(et) else "—",
            "published_datum": f"{pd_:+.4f}" if not math.isnan(pd_) else "—",
        })

    summary = pd.DataFrame(rows)
    pd.set_option("display.max_colwidth", 35)
    pd.set_option("display.width", 170)
    print(summary.to_string(index=False))

    csv_path = out_dir / "datum_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")


if __name__ == "__main__":
    main()
