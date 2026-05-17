#!/usr/bin/env python3
"""
match_tides_to_cores.py

Downloads water-level records for every unique gauge referenced in
core_gauge_index.fgb and computes inundation fractions for each core.

Filter applied: NOAA ≤ 28 km  OR  USGS ≤ 28 km  (either gauge within range)
Gauge selection per core:
  - Use USGS if usgs1_dist_km < noaa_dist_km, otherwise use NOAA
  - Elevation tuning always uses the nearest NOAA datum regardless

Elevation used (in priority order):
  1. measured_elev_m where elev_datum == 'NAVD88'
  2. epqs_elev_m  (lidar first-return, ~+0.15 m above bare soil)

Inundation outputs:
  inundation_frac          — fraction of time WL ≥ marsh_elev_m
  inundation_frac_corrected — fraction of time WL ≥ (marsh_elev_m − 0.15 m)
                              (lidar canopy-bias correction for EPQS elevations)

Usage
-----
python3 scripts/match_tides_to_cores.py \\
    [--index       scripts/core_gauge_index.fgb] \\
    [--output      scripts/core_inundation.fgb] \\
    [--years       2] \\
    [--dist-km     28] \\
    [--workers     16]
"""

import argparse
import math
import sys
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# API constants (shared with compare_datum_methods / build_core_gauge_index)
# ---------------------------------------------------------------------------

NOAA_DATA_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
USGS_IV_URL   = "https://waterservices.usgs.gov/nwis/iv/"

_FT_TO_M      = 0.3048
_TIMEOUT_NOAA = 30
_TIMEOUT_USGS = 120
_CANOPY_BIAS  = 0.15   # m, lidar first-return overestimate of bare-soil elevation


# ---------------------------------------------------------------------------
# Chunk helper
# ---------------------------------------------------------------------------

def _noaa_chunks(start, end, max_days=365):
    current = start
    while current <= end:
        chunk_end = min(end, current + timedelta(days=max_days - 1))
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


# ---------------------------------------------------------------------------
# Download workers
# ---------------------------------------------------------------------------

def _noaa_job(station_id, chunk_start, chunk_end):
    """Fetch one NOAA hourly_height chunk in NAVD88.
    Returns (station_id, rows | None).
    """
    params = {
        "product":     "hourly_height",
        "application": "marsh_inundation",
        "begin_date":  chunk_start.strftime("%Y%m%d"),
        "end_date":    chunk_end.strftime("%Y%m%d"),
        "station":     station_id,
        "datum":       "NAVD",
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
            return station_id, []
        rows = []
        for item in payload.get("data", []):
            try:
                rows.append({"t": item["t"], "v": float(item["v"])})
            except (KeyError, ValueError):
                pass
        return station_id, rows
    except Exception:
        return station_id, None


def _usgs_job(site_no, start, end):
    """Fetch USGS gage-height for the full period.
    Returns (site_no, rows | None).  Values are in metres above gauge datum.
    """
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
        rows = []
        for ts in r.json().get("value", {}).get("timeSeries", []):
            unit    = ts.get("variable", {}).get("unit", {}).get("unitCode", "")
            is_feet = "ft" in unit.lower()
            for block in ts.get("values", []):
                for item in block.get("value", []):
                    try:
                        val = float(item["value"])
                        if val <= -999990:
                            continue
                        if is_feet:
                            val *= _FT_TO_M
                        rows.append({"t": item["dateTime"], "v": val})
                    except (KeyError, ValueError):
                        pass
        return site_no, rows
    except Exception:
        return site_no, None


# ---------------------------------------------------------------------------
# Assemble hourly Series from raw rows
# ---------------------------------------------------------------------------

def _to_navd88_series(rows, alt_navd88_m=0.0):
    """Convert raw row list to hourly NAVD88 Series.

    For NOAA (datum=NAVD): rows are already in NAVD88; alt_navd88_m=0.
    For USGS: rows are gage height above gauge datum; add alt_navd88_m.
    """
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    df["t"] = pd.to_datetime(df["t"], errors="coerce", utc=True).dt.tz_localize(None)
    df = df.dropna().sort_values("t").set_index("t")
    # Resample to hourly mean, apply datum offset
    s = df["v"].resample("h").mean().dropna()
    if alt_navd88_m != 0.0:
        s = s + alt_navd88_m
    return s


# ---------------------------------------------------------------------------
# Flat parallel download
# ---------------------------------------------------------------------------

def download_gauges(gauge_specs, start, end, workers):
    """
    Download all gauges in one flat pool.

    gauge_specs : list of dicts with keys:
        source        'noaa' | 'usgs'
        id            station / site ID
        alt_navd88_m  offset for USGS (0.0 for NOAA)

    Returns dict: gauge_id → hourly NAVD88 pd.Series
    """
    # Build job list
    jobs = []
    for g in gauge_specs:
        if g["source"] == "noaa":
            for cs, ce in _noaa_chunks(start, end):
                jobs.append(("noaa", g["id"], g["alt_navd88_m"], cs, ce))
        else:
            jobs.append(("usgs", g["id"], g["alt_navd88_m"], start, end))

    print(f"  Submitting {len(jobs)} download jobs ({workers} workers) ...", flush=True)

    raw: dict[str, list] = {}
    alt_map: dict[str, float] = {g["id"]: g["alt_navd88_m"] for g in gauge_specs}
    errors: set[str] = set()

    def _run(job):
        if job[0] == "noaa":
            _, sid, _, cs, ce = job
            return _noaa_job(sid, cs, ce)
        else:
            _, sid, _, s, e = job
            return _usgs_job(sid, s, e)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_run, j): j for j in jobs}
        done = 0
        for fut in as_completed(futures):
            done += 1
            print(f"    {done}/{len(jobs)} jobs\r", end="", flush=True)
            try:
                sid, rows = fut.result()
            except Exception:
                continue
            if rows is None:
                errors.add(sid)
            else:
                raw.setdefault(sid, []).extend(rows)
    print()

    if errors:
        print(f"  HTTP errors for {len(errors)} gauge(s): {', '.join(sorted(errors))}",
              file=sys.stderr)

    result = {}
    for g in gauge_specs:
        sid = g["id"]
        result[sid] = _to_navd88_series(raw.get(sid, []), g["alt_navd88_m"])
        n = len(result[sid])
        print(f"  {g['source'].upper()} {sid:12s}  {n:5d} hourly NAVD88 records"
              f"{'  (no data)' if n == 0 else ''}")

    return result


# ---------------------------------------------------------------------------
# Inundation fraction
# ---------------------------------------------------------------------------

def inundation_fraction(wl_series, elev_m):
    """Fraction of hourly WL records ≥ elev_m.  Returns nan if unusable."""
    if wl_series.empty or math.isnan(elev_m):
        return float("nan")
    valid = wl_series.dropna()
    if len(valid) < 24 * 30:   # require ≥ 30 days of data
        return float("nan")
    return float((valid >= elev_m).mean())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Compute inundation fractions for CCN marsh cores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--index",   default="scripts/core_gauge_index.fgb")
    p.add_argument("--output",  default="scripts/core_inundation.fgb")
    p.add_argument("--years",   type=float, default=2.0)
    p.add_argument("--dist-km", type=float, default=28.0,
                   help="Max gauge distance — core passes if NOAA OR USGS within this range")
    p.add_argument("--workers", type=int,   default=16)
    return p


def main():
    args   = build_parser().parse_args()
    t0     = _time.perf_counter()

    try:
        import geopandas as gpd
    except ImportError:
        sys.exit("geopandas required: pip install geopandas pyogrio shapely")

    end_date  = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=round(365 * args.years))

    print(f"\n{'='*66}")
    print(f"  Tide matching: inundation fractions for CCN marsh cores")
    print(f"  Period  : {start_date.date()} → {end_date.date()} ({args.years:.1f} yr)")
    print(f"  Filter  : NOAA ≤ {args.dist_km} km  OR  USGS ≤ {args.dist_km} km")
    print(f"{'='*66}\n")

    # ── 1. Load index and apply distance filter ───────────────────────────────
    print("[1] Loading core index ...")
    index = gpd.read_file(args.index)
    passes = (index["noaa_dist_km"] <= args.dist_km) | \
             (index["usgs1_dist_km"] <= args.dist_km)
    cores = index[passes].copy().reset_index(drop=True)
    print(f"  {len(index)} total cores → {len(cores)} within {args.dist_km} km of a gauge")

    # Decide which gauge to use per core
    cores["_use_usgs"] = (
        cores["usgs1_dist_km"].notna() &
        (cores["usgs1_dist_km"] < cores["noaa_dist_km"]) &
        cores["usgs1_alt_m"].notna()
    )
    print(f"  {cores['_use_usgs'].sum()} cores will use USGS (closer);  "
          f"{(~cores['_use_usgs']).sum()} will use NOAA")

    # ── 2. Build unique gauge download list ───────────────────────────────────
    print("\n[2] Building gauge download list ...")
    gauge_specs = []
    seen: set[str] = set()

    for _, row in cores.iterrows():
        if row["_use_usgs"]:
            sid = str(row["usgs1_id"])
            if sid not in seen:
                seen.add(sid)
                gauge_specs.append({
                    "source":       "usgs",
                    "id":           sid,
                    "alt_navd88_m": float(row["usgs1_alt_m"])
                                    if not math.isnan(float(row["usgs1_alt_m"])) else 0.0,
                })
        else:
            sid = str(row["noaa_id"])
            if sid not in seen:
                seen.add(sid)
                gauge_specs.append({
                    "source":       "noaa",
                    "id":           sid,
                    "alt_navd88_m": 0.0,
                })

    noaa_count = sum(1 for g in gauge_specs if g["source"] == "noaa")
    usgs_count = sum(1 for g in gauge_specs if g["source"] == "usgs")
    print(f"  {noaa_count} unique NOAA gauges + {usgs_count} unique USGS stations "
          f"= {len(gauge_specs)} total")

    elapsed = _time.perf_counter() - t0
    print(f"  ⏱  {elapsed:.1f} s")

    # ── 3. Download all gauges in flat pool ───────────────────────────────────
    print(f"\n[3] Downloading water-level records ...")
    wl: dict[str, pd.Series] = download_gauges(
        gauge_specs, start_date, end_date, args.workers
    )

    elapsed = _time.perf_counter() - t0
    print(f"  ⏱  {elapsed:.1f} s")

    # ── 4. Compute inundation fractions ───────────────────────────────────────
    print(f"\n[4] Computing inundation fractions for {len(cores)} cores ...")

    results = []
    for _, row in cores.iterrows():
        # Gauge used for water levels
        if row["_use_usgs"]:
            gauge_src  = "usgs"
            gauge_id   = str(row["usgs1_id"])
            gauge_dist = float(row["usgs1_dist_km"])
        else:
            gauge_src  = "noaa"
            gauge_id   = str(row["noaa_id"])
            gauge_dist = float(row["noaa_dist_km"])

        wl_series = wl.get(gauge_id, pd.Series(dtype=float))

        # Marsh elevation: prefer measured NAVD88, fall back to EPQS
        meas = row["measured_elev_m"]
        epqs = row["epqs_elev_m"]
        if row["elev_datum"] == "NAVD88" and not (isinstance(meas, float) and math.isnan(meas)):
            marsh_elev = float(meas)
            elev_src   = "measured"
        elif not (isinstance(epqs, float) and math.isnan(epqs)):
            marsh_elev = float(epqs)
            elev_src   = "epqs"
        else:
            marsh_elev = float("nan")
            elev_src   = "none"

        inund      = inundation_fraction(wl_series, marsh_elev)
        inund_corr = inundation_fraction(wl_series, marsh_elev - _CANOPY_BIAS)

        results.append({
            "gauge_src":    gauge_src,
            "gauge_id":     gauge_id,
            "gauge_dist_km": round(gauge_dist, 2),
            "wl_hours":     len(wl_series.dropna()),
            "marsh_elev_m": round(marsh_elev, 4) if not math.isnan(marsh_elev) else float("nan"),
            "elev_src":     elev_src,
            "inund_frac":   round(inund,      4) if not math.isnan(inund)      else float("nan"),
            "inund_corr":   round(inund_corr, 4) if not math.isnan(inund_corr) else float("nan"),
        })

    res_df = pd.DataFrame(results)

    elapsed = _time.perf_counter() - t0
    print(f"  ⏱  {elapsed:.1f} s")

    # ── 5. Merge and write ────────────────────────────────────────────────────
    print(f"\n[5] Writing output ...")
    out = cores.copy()
    for col in ["_use_usgs"]:
        out = out.drop(columns=[col], errors="ignore")
    for col in res_df.columns:
        out[col] = res_df[col].values

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(str(out_path), driver="FlatGeobuf")
    print(f"  Saved {len(out)} cores → {out_path}")

    # ── 6. Summary ────────────────────────────────────────────────────────────
    elapsed = _time.perf_counter() - t0
    print(f"\n  Total time: {elapsed:.1f} s")

    valid_inund = out["inund_frac"].notna()
    valid_corr  = out["inund_corr"].notna()
    print(f"\n── Summary {'─'*52}")
    print(f"  Cores processed:           {len(out)}")
    print(f"  Valid inundation (raw):    {valid_inund.sum()}"
          f"  ({100*valid_inund.mean():.0f}%)")
    print(f"  Valid inundation (corrected): {valid_corr.sum()}"
          f"  ({100*valid_corr.mean():.0f}%)")
    print(f"  Elev source — measured:    {(out['elev_src']=='measured').sum()}")
    print(f"  Elev source — epqs:        {(out['elev_src']=='epqs').sum()}")
    print(f"  Elev source — none:        {(out['elev_src']=='none').sum()}")
    print(f"  Gauge — noaa:              {(out['gauge_src']=='noaa').sum()}")
    print(f"  Gauge — usgs:              {(out['gauge_src']=='usgs').sum()}")
    if valid_inund.any():
        print(f"\n  Inundation fraction (raw):")
        print(f"    Mean  {out['inund_frac'].mean():.3f}   "
              f"Median {out['inund_frac'].median():.3f}   "
              f"Std {out['inund_frac'].std():.3f}")
        print(f"    p5={out['inund_frac'].quantile(0.05):.3f}  "
              f"p25={out['inund_frac'].quantile(0.25):.3f}  "
              f"p75={out['inund_frac'].quantile(0.75):.3f}  "
              f"p95={out['inund_frac'].quantile(0.95):.3f}")
        print(f"\n  Inundation fraction (lidar corrected, −{_CANOPY_BIAS} m):")
        print(f"    Mean  {out['inund_corr'].mean():.3f}   "
              f"Median {out['inund_corr'].median():.3f}   "
              f"Std {out['inund_corr'].std():.3f}")


if __name__ == "__main__":
    main()
