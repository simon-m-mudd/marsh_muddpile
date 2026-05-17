#!/usr/bin/env python3
"""
build_core_gauge_index.py

For every US marsh core in the CCN synthesis database:
  1. Queries EPQS for the core's land-surface NAVD88 elevation (parallel).
  2. Finds the nearest NOAA CO-OPS gauge that has a published NAVD88 datum.
  3. Finds the 2 nearest USGS coastal/estuarine/tidal-stream stations with
     NAVD88 alt_va datum (site types ES, ST-TS, OC-CO).

Output: scripts/core_gauge_index.fgb  (FlatGeobuf, one point per core)

Caches:
  tidal_datum_cache/noaa_navd88_stations.csv  — NOAA stations with NAVD88
  tidal_datum_cache/usgs_coastal_stations.csv — USGS coastal stations with NAVD88

Usage
-----
python3 scripts/build_core_gauge_index.py \\
    [--ccn-cores   core_data/CCN_synthesis/CCN_cores.csv] \\
    [--output      scripts/core_gauge_index.fgb] \\
    [--cache-dir   tidal_datum_cache] \\
    [--epqs-workers 16] \\
    [--noaa-workers  8]
"""

import argparse
import math
import sys
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# API constants
# ---------------------------------------------------------------------------

NOAA_MDAPI_URL  = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"
NOAA_DATUMS_URL = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{sid}/datums.json"
USGS_SITE_URL   = "https://waterservices.usgs.gov/nwis/site/"
EPQS_URL        = "https://epqs.nationalmap.gov/v1/json"

_FT_TO_M       = 0.3048
_TIMEOUT_NOAA  = 30
_TIMEOUT_USGS  = 60
_TIMEOUT_EPQS  = 10
_DELAY         = 0.2   # courtesy pause between sequential API calls

# Coastal + Great-Lakes-adjacent US states to fetch USGS stations from
_COASTAL_STATES = [
    "me", "nh", "ma", "ri", "ct", "ny", "nj", "de", "md",
    "va", "nc", "sc", "ga", "fl", "al", "ms", "la", "tx",
    "ca", "or", "wa", "ak", "hi",
]

# USGS site types that indicate coastal / tidal / estuarine measurement
_COASTAL_SITE_TYPES = ("ES", "ST-TS", "OC-CO")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _haversine_matrix(lats_q, lons_q, lats_r, lons_r):
    """Return (n_query × n_ref) distance matrix in km using vectorised haversine."""
    R = 6371.0088
    lq = np.radians(lats_q)[:, None]
    rq = np.radians(lons_q)[:, None]
    lr = np.radians(lats_r)[None, :]
    rr = np.radians(lons_r)[None, :]
    dp = lr - lq
    dl = rr - rq
    a  = np.sin(dp/2)**2 + np.cos(lq)*np.cos(lr)*np.sin(dl/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ---------------------------------------------------------------------------
# NOAA: fetch catalogue + NAVD88-capable stations
# ---------------------------------------------------------------------------

def _fetch_noaa_catalogue():
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
            })
        except (KeyError, ValueError):
            continue
    return pd.DataFrame(rows)


def _check_noaa_navd88(station_id):
    """Return (station_id, has_navd88) by probing the datums API."""
    url = NOAA_DATUMS_URL.format(sid=station_id)
    try:
        r = requests.get(url, params={"units": "metric"}, timeout=_TIMEOUT_NOAA)
        if not r.ok:
            return station_id, False
        raw = {d["name"]: float(d["value"]) for d in r.json().get("datums", [])}
        offset = raw.get("NAVD88", float("nan"))
        return station_id, not math.isnan(offset)
    except Exception:
        return station_id, False


def load_noaa_navd88_stations(cache_path, noaa_workers):
    """Return DataFrame of NOAA stations that have a published NAVD88 datum."""
    if Path(cache_path).exists():
        df = pd.read_csv(cache_path, dtype={"station_id": str})
        print(f"  Loaded {len(df)} NOAA NAVD88 stations from cache", flush=True)
        return df

    all_stations = _fetch_noaa_catalogue()
    n = len(all_stations)
    print(f"  {n} NOAA stations — checking NAVD88 availability "
          f"({noaa_workers} workers) ...", flush=True)

    has_navd88: dict[str, bool] = {}
    with ThreadPoolExecutor(max_workers=noaa_workers) as ex:
        futures = {ex.submit(_check_noaa_navd88, sid): sid
                   for sid in all_stations["station_id"]}
        done = 0
        for fut in as_completed(futures):
            sid, ok = fut.result()
            has_navd88[sid] = ok
            done += 1
            print(f"    {done}/{n} NOAA datum checks\r", end="", flush=True)
    print()

    all_stations["has_navd88"] = all_stations["station_id"].map(has_navd88)
    navd88 = all_stations[all_stations["has_navd88"]].drop(columns="has_navd88").copy()
    print(f"  {len(navd88)} of {n} NOAA stations have NAVD88", flush=True)

    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    navd88.to_csv(cache_path, index=False)
    return navd88


# ---------------------------------------------------------------------------
# USGS: fetch coastal stations with NAVD88
# ---------------------------------------------------------------------------

def _fetch_usgs_state(state, site_type):
    """Return list of row dicts for one (state, site_type) combination."""
    params = {
        "format":      "rdb",
        "siteStatus":  "active",
        "parameterCd": "00065",
        "siteType":    site_type,
        "stateCd":     state,
    }
    try:
        r = requests.get(USGS_SITE_URL, params=params, timeout=_TIMEOUT_USGS)
        if not r.ok:
            return []
        lines = [ln for ln in r.text.splitlines()
                 if ln.strip() and not ln.startswith("#")]
        if len(lines) < 3:
            return []
        df = pd.read_csv(StringIO("\n".join(lines)), sep="\t", dtype=str).iloc[1:].copy()
        df["latitude"]  = pd.to_numeric(df.get("dec_lat_va",  pd.Series(dtype=str)), errors="coerce")
        df["longitude"] = pd.to_numeric(df.get("dec_long_va", pd.Series(dtype=str)), errors="coerce")
        df["alt_va"]    = pd.to_numeric(df.get("alt_va",      pd.Series(dtype=str)), errors="coerce")
        df = df.dropna(subset=["latitude", "longitude"])
        rows = []
        for _, row in df.iterrows():
            alt_datum = str(row.get("alt_datum_cd", "")).strip().upper()
            if alt_datum != "NAVD88":
                continue
            alt_va = row["alt_va"]
            alt_navd88_m = float(alt_va) * _FT_TO_M if not pd.isna(alt_va) else float("nan")
            rows.append({
                "site_no":       str(row.get("site_no", "")).strip(),
                "name":          str(row.get("station_nm", "")).strip(),
                "site_type":     str(row.get("site_tp_cd", site_type)).strip(),
                "latitude":      float(row["latitude"]),
                "longitude":     float(row["longitude"]),
                "alt_navd88_m":  alt_navd88_m,
                "state":         state.upper(),
            })
        return rows
    except Exception:
        return []


def load_usgs_coastal_stations(cache_path):
    """Return DataFrame of all US coastal USGS stations with NAVD88 datum."""
    if Path(cache_path).exists():
        df = pd.read_csv(cache_path, dtype={"site_no": str})
        print(f"  Loaded {len(df)} USGS coastal NAVD88 stations from cache", flush=True)
        return df

    jobs = [(state, stype)
            for state in _COASTAL_STATES
            for stype in _COASTAL_SITE_TYPES]

    print(f"  Fetching USGS coastal stations "
          f"({len(_COASTAL_STATES)} states × {len(_COASTAL_SITE_TYPES)} types) ...",
          flush=True)

    all_rows = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        futures = {ex.submit(_fetch_usgs_state, st, stype): (st, stype)
                   for st, stype in jobs}
        done = 0
        for fut in as_completed(futures):
            rows = fut.result()
            all_rows.extend(rows)
            done += 1
            print(f"    {done}/{len(jobs)} state/type queries\r", end="", flush=True)
    print()

    if not all_rows:
        print("  WARNING: no USGS coastal stations found", file=sys.stderr)
        return pd.DataFrame(columns=["site_no","name","site_type",
                                      "latitude","longitude","alt_navd88_m","state"])

    df = (pd.DataFrame(all_rows)
          .drop_duplicates(subset=["site_no"])
          .reset_index(drop=True))
    print(f"  {len(df)} unique USGS coastal stations with NAVD88")

    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


# ---------------------------------------------------------------------------
# EPQS
# ---------------------------------------------------------------------------

def _query_epqs(lat, lon):
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


def query_epqs_all(cores_df, workers):
    """Return array of EPQS elevations aligned to cores_df index."""
    lats = cores_df["latitude"].values
    lons = cores_df["longitude"].values
    n    = len(cores_df)
    elevs = np.full(n, float("nan"))

    def _job(i):
        return i, _query_epqs(lats[i], lons[i])

    print(f"  EPQS queries for {n} cores ({workers} workers) ...", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_job, i): i for i in range(n)}
        done = 0
        for fut in as_completed(futures):
            i, val = fut.result()
            elevs[i] = val
            done += 1
            if done % 100 == 0 or done == n:
                print(f"    {done}/{n} EPQS queries\r", end="", flush=True)
    print()
    return elevs


# ---------------------------------------------------------------------------
# Nearest-gauge matching
# ---------------------------------------------------------------------------

def match_nearest(cores_df, gauge_df, id_col, n_nearest=1):
    """
    For each core, find the n_nearest rows in gauge_df by haversine distance.

    Returns list of lists: result[i] = list of n_nearest dicts with
    {id, name, dist_km, lat, lon, ...} for core i.
    """
    if gauge_df.empty:
        return [[{}] * n_nearest for _ in range(len(cores_df))]

    clats = cores_df["latitude"].values
    clons = cores_df["longitude"].values
    glats = gauge_df["latitude"].values
    glons = gauge_df["longitude"].values

    dist = _haversine_matrix(clats, clons, glats, glons)  # (n_cores × n_gauges)

    results = []
    for i in range(len(cores_df)):
        row_dists = dist[i]
        idx_sorted = np.argsort(row_dists)[:n_nearest]
        matches = []
        for j in idx_sorted:
            g = gauge_df.iloc[j]
            m = {col: g[col] for col in gauge_df.columns}
            m["dist_km"] = float(row_dists[j])
            matches.append(m)
        # Pad if fewer gauges than requested
        while len(matches) < n_nearest:
            matches.append({})
        results.append(matches)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Build a core-to-gauge index FlatGeobuf from CCN marsh cores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ccn-cores",    default="core_data/CCN_synthesis/CCN_cores.csv")
    p.add_argument("--output",       default="scripts/core_gauge_index.fgb")
    p.add_argument("--cache-dir",    default="tidal_datum_cache")
    p.add_argument("--epqs-workers", type=int, default=16)
    p.add_argument("--noaa-workers", type=int, default=8)
    p.add_argument("--no-epqs",      action="store_true",
                   help="Skip EPQS queries (useful for fast testing)")
    return p


def main():
    args = build_parser().parse_args()
    t0   = _time.perf_counter()

    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError:
        sys.exit("geopandas and shapely are required: pip install geopandas shapely pyogrio")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load and filter CCN cores ─────────────────────────────────────────
    print("\n[1] Loading CCN cores ...", flush=True)
    all_cores = pd.read_csv(args.ccn_cores, low_memory=False)
    marsh = (all_cores[
        (all_cores["habitat"] == "marsh") &
        (all_cores["country"] == "United States")
    ].dropna(subset=["latitude", "longitude"])
     .drop_duplicates(subset=["core_id"])
     .reset_index(drop=True))
    print(f"  {len(all_cores)} total cores → {len(marsh)} US marsh cores with coordinates")
    elapsed = _time.perf_counter() - t0
    print(f"  ⏱  {elapsed:.1f} s total")

    # ── 2. NOAA NAVD88 stations ───────────────────────────────────────────────
    print("\n[2] NOAA CO-OPS stations with NAVD88 ...", flush=True)
    noaa = load_noaa_navd88_stations(
        cache_path=str(cache_dir / "noaa_navd88_stations.csv"),
        noaa_workers=args.noaa_workers,
    )
    elapsed = _time.perf_counter() - t0
    print(f"  ⏱  {elapsed:.1f} s total")

    # ── 3. USGS coastal stations ──────────────────────────────────────────────
    print("\n[3] USGS coastal stations with NAVD88 ...", flush=True)
    usgs = load_usgs_coastal_stations(
        cache_path=str(cache_dir / "usgs_coastal_stations.csv"),
    )
    elapsed = _time.perf_counter() - t0
    print(f"  ⏱  {elapsed:.1f} s total")

    # ── 4. EPQS elevations for all cores ──────────────────────────────────────
    print("\n[4] EPQS elevation queries ...", flush=True)
    if args.no_epqs:
        print("  Skipped (--no-epqs)")
        epqs_elevs = np.full(len(marsh), float("nan"))
    else:
        epqs_elevs = query_epqs_all(marsh, workers=args.epqs_workers)
        n_ok = np.sum(~np.isnan(epqs_elevs))
        print(f"  {n_ok}/{len(marsh)} cores with valid EPQS elevation")
    elapsed = _time.perf_counter() - t0
    print(f"  ⏱  {elapsed:.1f} s total")

    # ── 5. Nearest-gauge matching ─────────────────────────────────────────────
    print("\n[5] Nearest-gauge matching ...", flush=True)

    noaa_matches = match_nearest(marsh, noaa.rename(columns={"station_id": "id"}),
                                 id_col="id", n_nearest=1)

    usgs_for_match = usgs.rename(columns={"site_no": "id"}) if not usgs.empty else usgs
    usgs_matches   = match_nearest(marsh, usgs_for_match, id_col="id", n_nearest=2)

    elapsed = _time.perf_counter() - t0
    print(f"  Matched {len(marsh)} cores to NOAA + USGS gauges")
    print(f"  ⏱  {elapsed:.1f} s total")

    # ── 6. Build output GeoDataFrame ──────────────────────────────────────────
    print("\n[6] Building output ...", flush=True)
    rows = []
    for i, (_, core) in enumerate(marsh.iterrows()):
        nm  = noaa_matches[i][0]   # single nearest NOAA
        um1 = usgs_matches[i][0]   # nearest USGS
        um2 = usgs_matches[i][1]   # 2nd nearest USGS

        row = {
            "core_id":          core["core_id"],
            "study_id":         core["study_id"],
            "site_id":          core["site_id"],
            "latitude":         core["latitude"],
            "longitude":        core["longitude"],
            "habitat":          core.get("habitat", ""),
            "salinity_class":   core.get("salinity_class", ""),
            "vegetation_class": core.get("vegetation_class", ""),
            "measured_elev_m":  core.get("elevation", float("nan")),
            "elev_datum":       core.get("elevation_datum", ""),
            "epqs_elev_m":      float(epqs_elevs[i]),
            # NOAA reference gauge
            "noaa_id":          str(nm.get("id", "")),
            "noaa_name":        str(nm.get("name", ""))[:50],
            "noaa_dist_km":     round(nm.get("dist_km", float("nan")), 2),
            "noaa_lat":         nm.get("latitude", float("nan")),
            "noaa_lon":         nm.get("longitude", float("nan")),
            # USGS station 1
            "usgs1_id":         str(um1.get("id", "")),
            "usgs1_name":       str(um1.get("name", ""))[:50],
            "usgs1_dist_km":    round(um1.get("dist_km", float("nan")), 2),
            "usgs1_alt_m":      um1.get("alt_navd88_m", float("nan")),
            "usgs1_type":       um1.get("site_type", ""),
            "usgs1_lat":        um1.get("latitude", float("nan")),
            "usgs1_lon":        um1.get("longitude", float("nan")),
            # USGS station 2
            "usgs2_id":         str(um2.get("id", "")),
            "usgs2_name":       str(um2.get("name", ""))[:50],
            "usgs2_dist_km":    round(um2.get("dist_km", float("nan")), 2),
            "usgs2_alt_m":      um2.get("alt_navd88_m", float("nan")),
            "usgs2_type":       um2.get("site_type", ""),
            "usgs2_lat":        um2.get("latitude", float("nan")),
            "usgs2_lon":        um2.get("longitude", float("nan")),
        }
        rows.append(row)

    out_df = pd.DataFrame(rows)

    # Coerce numeric columns so GeoPackage/FGB can write them properly
    for col in ["noaa_dist_km", "noaa_lat", "noaa_lon",
                "usgs1_dist_km", "usgs1_alt_m", "usgs1_lat", "usgs1_lon",
                "usgs2_dist_km", "usgs2_alt_m", "usgs2_lat", "usgs2_lon",
                "measured_elev_m", "epqs_elev_m"]:
        out_df[col] = pd.to_numeric(out_df[col], errors="coerce")

    geom = [Point(r["longitude"], r["latitude"]) for _, r in out_df.iterrows()]
    gdf  = gpd.GeoDataFrame(out_df, geometry=geom, crs="EPSG:4326")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(str(out_path), driver="FlatGeobuf")
    print(f"  Saved {len(gdf)} cores → {out_path}")

    elapsed = _time.perf_counter() - t0
    print(f"\n  Total time: {elapsed:.1f} s")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n── Summary ─────────────────────────────────────────────────────────")
    print(f"  US marsh cores:            {len(gdf)}")
    print(f"  NOAA NAVD88 gauges used:   {gdf['noaa_id'].nunique()}")
    print(f"  USGS stations referenced:  {pd.concat([gdf['usgs1_id'], gdf['usgs2_id']]).nunique()}")
    print(f"  Cores with EPQS elev:      {gdf['epqs_elev_m'].notna().sum()}")
    if gdf["noaa_dist_km"].notna().any():
        print(f"  Median NOAA dist:          {gdf['noaa_dist_km'].median():.1f} km")
    if gdf["usgs1_dist_km"].notna().any():
        print(f"  Median USGS-1 dist:        {gdf['usgs1_dist_km'].median():.1f} km")

    # Top NOAA gauges by core count
    print("\n  Top 10 NOAA gauges (by core count):")
    top = (gdf.groupby(["noaa_id", "noaa_name"])
              .size()
              .sort_values(ascending=False)
              .head(10)
              .reset_index(name="n_cores"))
    for _, r in top.iterrows():
        print(f"    {r['noaa_id']}  {r['noaa_name'][:40]}  n={r['n_cores']}")


if __name__ == "__main__":
    main()
