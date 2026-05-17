#!/usr/bin/env python3
"""
match_tidal_datums.py

For each input point (lat/lon), this script:
  1. Finds the nearest NOAA CO-OPS tide gauge that has NAVD88 datums.
  2. Finds the nearest coastal water-level station (NOAA CO-OPS or USGS).
  3. Downloads overlapping hourly time series from both stations.
  4. Matches mean water levels to derive a vertical offset that places
     the coastal station's raw readings on the NAVD88 datum.
  5. Optionally queries USGS EPQS for a land-surface NAVD88 elevation
     at the coastal station location (independent cross-check).

Matching logic
--------------
During an overlapping record:
    offset_m = mean(tide_gauge_NAVD88) - mean(coastal_raw_m)
    coastal_NAVD88(t) = coastal_raw(t) + offset_m
    coastal_MSL_NAVD88 ≈ mean_tide_navd88_m   (= mean of tide gauge in overlap)

USGS gage-height data (parameter 00065) is in feet; converted to metres.
NOAA CO-OPS data is requested in metres relative to NAVD datum for the tide
gauge, and STND (station datum) for the coastal station.

Usage
-----
# Test: first 5 North Carolina marsh cores from CCN synthesis
python3 match_tidal_datums.py --test-nc-marsh \\
    --ccn-cores ../core_data/CCN_synthesis/CCN_cores.csv \\
    --output nc_marsh_tidal_datums.csv

# Generic CSV input (must have latitude and longitude columns)
python3 match_tidal_datums.py --input points.csv --output results.csv

Options
-------
--years N         Years of water-level history to fetch (default 2)
--max-tide-dist   Max distance to tide gauge in km (default 200)
--max-wl-dist     Max distance to coastal water-level station in km (default 100)
--cache-dir DIR   Cache directory for station lists (default ./tidal_datum_cache)
--no-epqs         Skip EPQS land-surface queries
--n-test N        Number of cores in test mode (default 5)
"""

import argparse
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── API endpoints ──────────────────────────────────────────────────────────────

NOAA_MDAPI_URL  = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"
NOAA_DATA_URL   = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
NOAA_DATUMS_URL = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{sid}/datums.json"
USGS_SITE_URL   = "https://waterservices.usgs.gov/nwis/site/"
USGS_IV_URL     = "https://waterservices.usgs.gov/nwis/iv/"
EPQS_URL        = "https://epqs.nationalmap.gov/v1/json"

_DELAY        = 0.3   # seconds between API calls (used for datum/station lookups)
_TIMEOUT      = 30    # seconds per request
_FT_TO_M      = 0.3048
_NOAA_WORKERS = 4     # max concurrent NOAA data requests

STATE_FIPS_TO_ABBR = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
    "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY", "60": "AS", "66": "GU", "69": "MP",
    "72": "PR", "78": "VI",
}


# ── geometry ───────────────────────────────────────────────────────────────────

def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── NOAA CO-OPS: station list ──────────────────────────────────────────────────

def fetch_noaa_stations(cache_path=None):
    """Fetch (or load from cache) NOAA CO-OPS water-level station metadata."""
    if cache_path and Path(cache_path).exists():
        df = pd.read_csv(cache_path, dtype={"station_id": str})
        print(f"  Loaded {len(df)} NOAA stations from cache: {cache_path}")
        return df

    print("Fetching NOAA CO-OPS station list ...", flush=True)
    r = requests.get(NOAA_MDAPI_URL, params={"type": "waterlevels"}, timeout=_TIMEOUT)
    r.raise_for_status()
    stations = r.json().get("stations", [])

    rows = []
    for s in stations:
        try:
            rows.append({
                "station_id": str(s["id"]),
                "name":       s["name"],
                "state":      s.get("state", ""),
                "latitude":   float(s["lat"]),
                "longitude":  float(s["lng"]),
                "status":     s.get("status", ""),
                "tidal":      s.get("tidal"),
            })
        except (KeyError, ValueError):
            continue

    df = pd.DataFrame(rows)
    print(f"  {len(df)} NOAA CO-OPS stations")

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)

    return df


# ── NOAA CO-OPS: tidal datums ──────────────────────────────────────────────────

def fetch_noaa_datums(station_id):
    """Return dict of datum name → NAVD88 elevation (m), or {} if NAVD88 unavailable.

    NOAA datum values are relative to the station datum (STND).
    The 'NAVD88' entry is the height of STND above NAVD88, so:
        value_NAVD88 = value_STND - navd88_offset
    """
    url = NOAA_DATUMS_URL.format(sid=station_id)
    try:
        r = requests.get(url, params={"units": "metric", "datum": "NAVD"}, timeout=_TIMEOUT)
        r.raise_for_status()
        raw = {d["name"]: float(d["value"]) for d in r.json().get("datums", [])}

        navd88_offset = raw.get("NAVD88", float("nan"))
        if math.isnan(navd88_offset):
            return {}

        result = {"NAVD88_offset": navd88_offset}
        for key in ("MHW", "MHHW", "MSL", "MTL", "MLW", "MLLW"):
            if key in raw:
                result[key] = round(raw[key] - navd88_offset, 5)
        return result

    except Exception as exc:
        print(f"    WARNING: datum fetch failed {station_id}: {exc}", file=sys.stderr)
        return {}


def find_nearest_tide_gauge_navd88(lat, lon, noaa_stations, datum_cache,
                                    max_dist_km=200, max_candidates=20):
    """Return info dict for the nearest NOAA station that has NAVD88 datums."""
    stations = noaa_stations.copy()
    stations["dist_km"] = stations.apply(
        lambda r: haversine_km(r["longitude"], r["latitude"], lon, lat), axis=1
    )
    candidates = (stations[stations["dist_km"] <= max_dist_km]
                  .sort_values("dist_km")
                  .head(max_candidates))

    for _, row in candidates.iterrows():
        sid = row["station_id"]
        if sid not in datum_cache:
            time.sleep(_DELAY)
            datum_cache[sid] = fetch_noaa_datums(sid)
        if datum_cache[sid]:
            return {
                "id":       sid,
                "name":     row["name"],
                "lat":      row["latitude"],
                "lon":      row["longitude"],
                "dist_km":  float(row["dist_km"]),
                "datums":   datum_cache[sid],
            }

    return None


# ── Coastal water-level station selection ─────────────────────────────────────

def fetch_usgs_coastal_stations(bbox, parameter_cd="00065"):
    """Fetch USGS estuary (ES) and tidal-stream (ST) stations within bbox.

    Also retrieves altitude (alt_va) and datum (alt_datum_cd) from USGS site
    metadata. When datum is NAVD88, converts alt_va (feet) to metres.
    """
    xmin, ymin, xmax, ymax = bbox
    all_rows = []

    for site_type in ("ES", "ST"):
        params = {
            "format":       "rdb",
            "siteStatus":   "active",
            "parameterCd":  parameter_cd,
            "siteType":     site_type,
            "bBox":         f"{xmin:.4f},{ymin:.4f},{xmax:.4f},{ymax:.4f}",
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
            df = df.iloc[1:].copy()  # drop datatype-description row

            df["latitude"]  = pd.to_numeric(df.get("dec_lat_va",  pd.Series(dtype=str)), errors="coerce")
            df["longitude"] = pd.to_numeric(df.get("dec_long_va", pd.Series(dtype=str)), errors="coerce")
            df["alt_va"]    = pd.to_numeric(df.get("alt_va",      pd.Series(dtype=str)), errors="coerce")
            df = df.dropna(subset=["latitude", "longitude"])

            for _, row in df.iterrows():
                fips      = str(row.get("state_cd", "")).strip().zfill(2)
                state     = STATE_FIPS_TO_ABBR.get(fips, "")
                alt_datum = str(row.get("alt_datum_cd", "")).strip().upper()
                alt_navd88 = float("nan")
                if alt_datum == "NAVD88" and not pd.isna(row["alt_va"]):
                    alt_navd88 = float(row["alt_va"]) * _FT_TO_M

                all_rows.append({
                    "source":        "usgs",
                    "station_id":    str(row.get("site_no", "")).strip(),
                    "name":          str(row.get("station_nm", "")).strip(),
                    "state":         state,
                    "latitude":      float(row["latitude"]),
                    "longitude":     float(row["longitude"]),
                    "site_type":     site_type,
                    "alt_navd88_m":  alt_navd88,
                })
        except Exception as exc:
            print(f"    WARNING: USGS {site_type} station fetch failed: {exc}", file=sys.stderr)
        time.sleep(_DELAY)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["station_id"])
    return df.reset_index(drop=True)


def find_nearest_coastal_station(lat, lon, noaa_stations, exclude_id,
                                  usgs_stations, max_dist_km=100):
    """Return the nearest water-level station (NOAA or USGS), excluding exclude_id."""
    candidates = []

    # Nearest NOAA station (excluding the tide gauge already assigned)
    noaa = noaa_stations.copy()
    noaa["dist_km"] = noaa.apply(
        lambda r: haversine_km(r["longitude"], r["latitude"], lon, lat), axis=1
    )
    noaa_near = (noaa[(noaa["station_id"] != exclude_id) & (noaa["dist_km"] <= max_dist_km)]
                 .sort_values("dist_km")
                 .head(3))
    for _, row in noaa_near.iterrows():
        candidates.append({
            "source":   "noaa",
            "id":       row["station_id"],
            "name":     row["name"],
            "lat":      row["latitude"],
            "lon":      row["longitude"],
            "dist_km":  float(row["dist_km"]),
        })

    # Nearest USGS stations
    if not usgs_stations.empty:
        usgs = usgs_stations.copy()
        usgs["dist_km"] = usgs.apply(
            lambda r: haversine_km(r["longitude"], r["latitude"], lon, lat), axis=1
        )
        usgs_near = usgs[usgs["dist_km"] <= max_dist_km].sort_values("dist_km").head(3)
        for _, row in usgs_near.iterrows():
            candidates.append({
                "source":   "usgs",
                "id":       row["station_id"],
                "name":     row["name"],
                "lat":      row["latitude"],
                "lon":      row["longitude"],
                "dist_km":  float(row["dist_km"]),
            })

    if not candidates:
        return None

    candidates.sort(key=lambda x: x["dist_km"])
    return candidates[0]


# ── Time-series download ───────────────────────────────────────────────────────

def _noaa_chunks(start, end, max_days=365):
    """Yield (chunk_start, chunk_end) pairs respecting NOAA API date-range limits.

    Hourly products allow up to 365 days per request; 6-minute products allow 31.
    """
    current = start
    while current <= end:
        chunk_end = min(end, current + timedelta(days=max_days - 1))
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def download_noaa_timeseries(station_id, start, end,
                              datum="NAVD", interval="h", units="metric"):
    """Download NOAA CO-OPS hourly water levels in parallel year-sized chunks.

    datum='NAVD' returns NAVD88 metres. datum='STND' returns raw station-datum metres.
    Returns DataFrame with columns 't' (datetime) and 'v_m' (float).
    """
    max_days = 365 if interval == "h" else 31
    chunks = list(_noaa_chunks(start, end, max_days=max_days))

    def _fetch(chunk_start, chunk_end):
        params = {
            "product":      "water_level",
            "application":  "marsh_tidal_datums",
            "begin_date":   chunk_start.strftime("%Y%m%d"),
            "end_date":     chunk_end.strftime("%Y%m%d"),
            "station":      station_id,
            "datum":        datum,
            "time_zone":    "gmt",
            "units":        units,
            "interval":     interval,
            "format":       "json",
        }
        r = requests.get(NOAA_DATA_URL, params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        return chunk_start, r.json()

    rows = []
    with ThreadPoolExecutor(max_workers=_NOAA_WORKERS) as ex:
        futures = [ex.submit(_fetch, s, e) for s, e in chunks]
        for fut in futures:
            try:
                chunk_start, payload = fut.result()
                if "error" in payload:
                    msg = payload["error"]
                    if isinstance(msg, dict):
                        msg = msg.get("message", str(msg))
                    print(f"    NOAA error {station_id} {chunk_start.date()}: {msg}",
                          file=sys.stderr)
                    continue
                for item in payload.get("data", []):
                    try:
                        rows.append({"t": pd.to_datetime(item["t"]), "v_m": float(item["v"])})
                    except (KeyError, ValueError):
                        continue
            except Exception as exc:
                print(f"    WARNING: NOAA download failed {station_id}: {exc}", file=sys.stderr)

    if not rows:
        return pd.DataFrame(columns=["t", "v_m"])

    df = pd.DataFrame(rows)
    df["t"] = pd.to_datetime(df["t"])
    return df.sort_values("t").reset_index(drop=True)


def download_usgs_timeseries(site_no, start, end, parameter_cd="00065"):
    """Download USGS instantaneous gage-height data; converts feet → metres.

    Returns DataFrame with columns 't' (datetime, timezone-naive UTC) and 'v_m'.
    """
    params = {
        "format":       "json",
        "sites":        str(site_no),
        "startDT":      start.strftime("%Y-%m-%d"),
        "endDT":        end.strftime("%Y-%m-%d"),
        "parameterCd":  parameter_cd,
        "siteStatus":   "all",
    }
    try:
        r = requests.get(USGS_IV_URL, params=params, timeout=120)
        r.raise_for_status()
        payload = r.json()
    except Exception as exc:
        print(f"    WARNING: USGS download failed {site_no}: {exc}", file=sys.stderr)
        return pd.DataFrame(columns=["t", "v_m"])

    rows = []
    for ts in payload.get("value", {}).get("timeSeries", []):
        variable  = ts.get("variable", {})
        unit_code = variable.get("unit", {}).get("unitCode", "")
        is_feet   = "ft" in unit_code.lower()

        for block in ts.get("values", []):
            for item in block.get("value", []):
                try:
                    val = float(item["value"])
                    if is_feet:
                        val *= _FT_TO_M
                    rows.append({"t": pd.to_datetime(item["dateTime"]), "v_m": val})
                except (KeyError, ValueError):
                    continue

    if not rows:
        return pd.DataFrame(columns=["t", "v_m"])

    df = pd.DataFrame(rows)
    # Strip timezone offset so t columns are comparable with NOAA (both UTC)
    df["t"] = pd.to_datetime(df["t"]).dt.tz_convert("UTC").dt.tz_localize(None)
    return df.sort_values("t").reset_index(drop=True)


# ── Mean-water-level offset ────────────────────────────────────────────────────

def compute_mwl_offset(tide_ts, coastal_ts, min_overlap_hours=24 * 30):
    """Derive NAVD88 offset from overlapping hourly records.

    Both series are resampled to the hour before merging.

    offset_m:
        value to add to coastal raw (m) to express it in NAVD88.
        = mean(tide_NAVD88) − mean(coastal_raw_m)  over the overlap window.

    coastal_msl_navd88_m:
        estimated MSL at the coastal station in NAVD88 ≈ mean(tide_NAVD88)
        during the overlap.  Valid when both stations share the same tidal
        signal (no major sea-level gradient between them).

    Returns dict, or None if insufficient overlap.
    """
    if tide_ts.empty or coastal_ts.empty:
        return None

    tide  = tide_ts.copy()
    coast = coastal_ts.copy()
    tide["t"]  = tide["t"].dt.floor("h")
    coast["t"] = coast["t"].dt.floor("h")

    tide  = tide.groupby("t")["v_m"].mean().reset_index()
    coast = coast.groupby("t")["v_m"].mean().reset_index()

    merged = pd.merge(tide, coast, on="t", suffixes=("_tide", "_coast"))
    merged = merged.dropna(subset=["v_m_tide", "v_m_coast"])

    if len(merged) < min_overlap_hours:
        return None

    mean_tide  = float(merged["v_m_tide"].mean())
    mean_coast = float(merged["v_m_coast"].mean())
    offset     = mean_tide - mean_coast

    return {
        "overlap_start":         merged["t"].min().strftime("%Y-%m-%d"),
        "overlap_end":           merged["t"].max().strftime("%Y-%m-%d"),
        "n_overlap_hours":       int(len(merged)),
        "mean_tide_navd88_m":    round(mean_tide,  4),
        "mean_coastal_raw_m":    round(mean_coast, 4),
        "offset_m":              round(offset,     4),
        "coastal_msl_navd88_m":  round(mean_tide,  4),
    }


# ── EPQS land-surface elevation ────────────────────────────────────────────────

def query_epqs(lat, lon):
    """Return NAVD88 land-surface elevation (m) from USGS 3DEP, or NaN on failure.

    Note: reflects the vegetation surface (lidar first-return), not bare ground.
    For water-gauge locations this gives the land elevation at the sensor site,
    not the gage-datum elevation.
    """
    params = {
        "x":           lon,
        "y":           lat,
        "wkid":        4326,
        "units":       "Meters",
        "includeDate": "false",
    }
    try:
        r = requests.get(EPQS_URL, params=params, timeout=10)
        r.raise_for_status()
        val = r.json().get("value")
        if val is None:
            return float("nan")
        fval = float(val)
        return fval if fval > -1e5 else float("nan")
    except Exception:
        return float("nan")


# ── Per-point processing ───────────────────────────────────────────────────────

def process_point(point, noaa_stations, datum_cache, usgs_stations,
                  years=2.0, max_tide_dist=200.0, max_wl_dist=100.0, do_epqs=True):
    """Process one point; return a flat result dict."""
    lat     = float(point["latitude"])
    lon     = float(point["longitude"])
    core_id = point.get("core_id", f"pt_{lat:.4f}_{lon:.4f}")

    print(f"\n{'='*64}")
    print(f"  {core_id}  ({lat:.5f}, {lon:.5f})")

    result = {"core_id": core_id, "latitude": lat, "longitude": lon}
    for key in ("site_id", "admin_division", "habitat", "elevation", "elevation_datum"):
        if key in point:
            result[key] = point[key]

    # ── 1. Nearest tide gauge with NAVD88 ─────────────────────────────────────
    print("  [1] Nearest NOAA tide gauge with NAVD88 ...")
    tide_gauge = find_nearest_tide_gauge_navd88(
        lat, lon, noaa_stations, datum_cache,
        max_dist_km=max_tide_dist,
    )

    if tide_gauge is None:
        print(f"    No NAVD88 tide gauge within {max_tide_dist} km")
        result.update({
            "tide_gauge_id":        None,
            "tide_gauge_name":      None,
            "tide_gauge_dist_km":   None,
            "tide_gauge_mhw_navd88":  None,
            "tide_gauge_msl_navd88":  None,
            "tide_gauge_mllw_navd88": None,
        })
    else:
        d = tide_gauge["datums"]
        result.update({
            "tide_gauge_id":          tide_gauge["id"],
            "tide_gauge_name":        tide_gauge["name"],
            "tide_gauge_dist_km":     round(tide_gauge["dist_km"], 2),
            "tide_gauge_mhw_navd88":  d.get("MHW"),
            "tide_gauge_msl_navd88":  d.get("MSL"),
            "tide_gauge_mllw_navd88": d.get("MLLW"),
        })
        print(f"    → {tide_gauge['name']} ({tide_gauge['id']})  {tide_gauge['dist_km']:.1f} km")
        print(f"      MHW={d.get('MHW','?'):>+7}  MSL={d.get('MSL','?'):>+7}  MLLW={d.get('MLLW','?'):>+7} m NAVD88")

    # ── 2. Nearest coastal water-level station ────────────────────────────────
    print("  [2] Nearest coastal water-level station ...")
    coastal_st = find_nearest_coastal_station(
        lat, lon, noaa_stations,
        exclude_id=tide_gauge["id"] if tide_gauge else None,
        usgs_stations=usgs_stations,
        max_dist_km=max_wl_dist,
    )

    if coastal_st is None:
        print(f"    No coastal water-level station within {max_wl_dist} km")
        result.update({
            "coastal_source":       None,
            "coastal_station_id":   None,
            "coastal_station_name": None,
            "coastal_dist_km":      None,
            "coastal_alt_navd88_m": None,
        })
    else:
        result.update({
            "coastal_source":       coastal_st["source"],
            "coastal_station_id":   coastal_st["id"],
            "coastal_station_name": coastal_st["name"],
            "coastal_dist_km":      round(coastal_st["dist_km"], 2),
        })

        # If USGS, include benchmark altitude in NAVD88 when metadata has it
        alt_navd88 = None
        if coastal_st["source"] == "usgs" and not usgs_stations.empty:
            row = usgs_stations[usgs_stations["station_id"] == coastal_st["id"]]
            if not row.empty:
                v = float(row.iloc[0].get("alt_navd88_m", float("nan")))
                if not math.isnan(v):
                    alt_navd88 = round(v, 4)
        result["coastal_alt_navd88_m"] = alt_navd88

        print(f"    → [{coastal_st['source'].upper()}] {coastal_st['name']} ({coastal_st['id']})  "
              f"{coastal_st['dist_km']:.1f} km")
        if alt_navd88 is not None:
            print(f"      USGS benchmark altitude: {alt_navd88:.3f} m NAVD88")

    # ── 3. Download time series and compute offset ────────────────────────────
    end_date   = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=365 * years)

    result.update({
        "overlap_start":        None,
        "overlap_end":          None,
        "n_overlap_hours":      None,
        "mean_tide_navd88_m":   None,
        "mean_coastal_raw_m":   None,
        "offset_m":             None,
        "coastal_msl_navd88_m": None,
    })

    if tide_gauge and coastal_st:
        print(f"  [3] Downloading {years:.0f} yr of data "
              f"({start_date.date()} → {end_date.date()}) ...")

        print(f"      Tide gauge {tide_gauge['id']} (NOAA NAVD) ...", end=" ", flush=True)
        tide_ts = download_noaa_timeseries(
            tide_gauge["id"], start_date, end_date,
            datum="NAVD", interval="h",
        )
        print(f"{len(tide_ts)} records")

        print(f"      Coastal {coastal_st['id']} ({coastal_st['source'].upper()}) ...",
              end=" ", flush=True)
        if coastal_st["source"] == "noaa":
            coastal_ts = download_noaa_timeseries(
                coastal_st["id"], start_date, end_date,
                datum="STND", interval="h",
            )
        else:
            coastal_ts = download_usgs_timeseries(
                coastal_st["id"], start_date, end_date,
            )
        print(f"{len(coastal_ts)} records")

        print("      Computing mean water-level offset ...", end=" ", flush=True)
        offset_result = compute_mwl_offset(tide_ts, coastal_ts)
        if offset_result is None:
            print("insufficient overlap (need ≥ 30 days)")
        else:
            result.update(offset_result)
            print(
                f"{offset_result['n_overlap_hours']} hours overlap  "
                f"({offset_result['overlap_start']} to {offset_result['overlap_end']})"
            )
            print(f"      Mean tide gauge (NAVD88): {offset_result['mean_tide_navd88_m']:+.4f} m")
            print(f"      Mean coastal raw:         {offset_result['mean_coastal_raw_m']:+.4f} m")
            print(f"      Offset:                   {offset_result['offset_m']:+.4f} m")
            print(f"      Coastal MSL (NAVD88) ≈    {offset_result['coastal_msl_navd88_m']:+.4f} m")
    else:
        print("  [3] Skipping time-series download (missing tide gauge or coastal station)")

    # ── 4. EPQS at coastal station ────────────────────────────────────────────
    result["epqs_coastal_navd88_m"] = None
    if do_epqs and coastal_st:
        print(f"  [4] EPQS at coastal station ({coastal_st['lat']:.5f}, {coastal_st['lon']:.5f}) ...",
              end=" ", flush=True)
        time.sleep(_DELAY)
        epqs_val = query_epqs(coastal_st["lat"], coastal_st["lon"])
        if not math.isnan(epqs_val):
            result["epqs_coastal_navd88_m"] = round(epqs_val, 4)
            print(f"{epqs_val:.3f} m NAVD88 (land surface)")
        else:
            print("no data")

    return result


# ── Input loading ──────────────────────────────────────────────────────────────

def load_ccn_nc_marsh(ccn_path, n=5):
    """Return first n North Carolina marsh cores from CCN_cores.csv."""
    df = pd.read_csv(ccn_path, low_memory=False)
    nc_marsh = (
        df[(df["admin_division"] == "North Carolina") & (df["habitat"] == "marsh")]
        .dropna(subset=["latitude", "longitude"])
        .head(n)
    )
    keep = [c for c in
            ["core_id", "site_id", "latitude", "longitude",
             "admin_division", "habitat", "elevation", "elevation_datum"]
            if c in nc_marsh.columns]
    return nc_marsh[keep].copy()


def load_input_csv(path):
    """Load a generic input CSV; must have latitude and longitude columns."""
    df = pd.read_csv(path, low_memory=False)
    missing = {"latitude", "longitude"} - set(df.columns)
    if missing:
        raise SystemExit(f"Input CSV is missing required columns: {missing}")
    if "core_id" not in df.columns:
        df["core_id"] = [f"pt_{i}" for i in range(len(df))]
    return df


def points_bbox(lats, lons, buffer_deg=2.0):
    return (
        min(lons) - buffer_deg,
        min(lats) - buffer_deg,
        max(lons) + buffer_deg,
        max(lats) + buffer_deg,
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="Match points to NOAA tide gauges and coastal water-level "
                    "stations, then derive NAVD88 datums via mean water-level matching.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--input",
                     help="Input CSV with latitude and longitude columns")
    grp.add_argument("--test-nc-marsh", action="store_true",
                     help="Test with first N NC marsh cores from CCN synthesis")

    p.add_argument("--ccn-cores",
                   default="../core_data/CCN_synthesis/CCN_cores.csv",
                   help="Path to CCN_cores.csv (used with --test-nc-marsh)")
    p.add_argument("--output", default="tidal_datums.csv",
                   help="Output CSV path")
    p.add_argument("--years", type=float, default=2.0,
                   help="Years of water-level history to download")
    p.add_argument("--max-tide-dist", type=float, default=200.0,
                   help="Max km to NOAA tide gauge")
    p.add_argument("--max-wl-dist", type=float, default=100.0,
                   help="Max km to coastal water-level station")
    p.add_argument("--cache-dir", default="./tidal_datum_cache",
                   help="Cache directory for station lists")
    p.add_argument("--no-epqs", action="store_true",
                   help="Skip EPQS land-surface queries")
    p.add_argument("--n-test", type=int, default=5,
                   help="Number of cores to use in --test-nc-marsh mode")
    return p


def main():
    args = build_parser().parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load input points
    if args.test_nc_marsh:
        ccn_path = Path(args.ccn_cores)
        if not ccn_path.exists():
            raise SystemExit(f"CCN cores file not found: {ccn_path}")
        print(f"Loading first {args.n_test} NC marsh cores from {ccn_path.name} ...")
        points_df = load_ccn_nc_marsh(ccn_path, n=args.n_test)
    else:
        print(f"Loading input points from {args.input} ...")
        points_df = load_input_csv(args.input)

    print(f"  {len(points_df)} points to process")
    print(points_df[["core_id", "latitude", "longitude"]].to_string(index=False))

    # Bounding box for USGS query
    bbox = points_bbox(
        points_df["latitude"].tolist(),
        points_df["longitude"].tolist(),
        buffer_deg=2.0,
    )
    print(f"\nSearch bbox: {bbox}")

    # Station catalogs
    noaa_cache = str(cache_dir / "noaa_stations.csv")
    noaa_stations = fetch_noaa_stations(cache_path=noaa_cache)

    print(f"\nFetching USGS coastal stations in bbox ...")
    usgs_stations = fetch_usgs_coastal_stations(bbox)
    print(f"  {len(usgs_stations)} USGS coastal stations")

    # Process each point
    datum_cache = {}
    results = []
    for _, point in points_df.iterrows():
        result = process_point(
            point.to_dict(),
            noaa_stations=noaa_stations,
            datum_cache=datum_cache,
            usgs_stations=usgs_stations,
            years=args.years,
            max_tide_dist=args.max_tide_dist,
            max_wl_dist=args.max_wl_dist,
            do_epqs=not args.no_epqs,
        )
        results.append(result)

    # Save
    out_df = pd.DataFrame(results)
    out_path = Path(args.output)
    out_df.to_csv(out_path, index=False, float_format="%.5f")

    print(f"\n{'='*64}")
    print(f"Saved {len(out_df)} rows → {out_path}")

    summary_cols = [c for c in [
        "core_id", "tide_gauge_name", "tide_gauge_dist_km",
        "tide_gauge_msl_navd88", "coastal_station_name",
        "coastal_dist_km", "coastal_source",
        "n_overlap_hours", "offset_m",
        "coastal_msl_navd88_m", "epqs_coastal_navd88_m",
    ] if c in out_df.columns]
    print("\nSummary:")
    pd.set_option("display.max_colwidth", 35)
    pd.set_option("display.width", 160)
    print(out_df[summary_cols].to_string(index=False))


if __name__ == "__main__":
    main()
