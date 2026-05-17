"""
Add nearest NOAA tide gauge and inundation fraction to lter_plot_peak_biomass.csv.

For each site:
  1. Find nearest active NOAA CO-OPS water-level station (haversine distance).
  2. Fetch tidal datums (MHW, MLLW in NAVD88) for that station.
  3. Compute inundation_fraction using the sinusoidal semi-diurnal tide
     approximation — matching the model definition in
     src/processes/composite_water_level_model.cpp:
       IF = inundated_substeps / n_substeps
     Closed-form equivalent for a pure sinusoidal tide:
       IF = arccos(z_rel / A) / π   for |z_rel| <= A
       IF = 1                        for z_rel < -A  (always flooded)
       IF = 0                        for z_rel >  A  (never flooded)
     where z_rel = site_elevation_NAVD88 - MSL_NAVD88
           A     = (MHW - MLLW) / 2   (tidal amplitude, NAVD88)
           MSL   = (MHW + MLLW) / 2   (NAVD88)

Outputs: scripts/lter_plot_peak_biomass.csv  (updated in-place, backed up)

NOAA CO-OPS APIs used:
  Stations list: https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json
                   ?type=waterlevels&units=metric
  Datums:        https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{id}/datums.json
                   ?units=metric&datum=NAVD
"""

from pathlib import Path
import math
import time
import sys

import numpy as np
import pandas as pd
import requests

_THIS_DIR = Path(__file__).parent
_CSV = _THIS_DIR / "lter_plot_peak_biomass.csv"

_STATIONS_URL = (
    "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"
    "?type=waterlevels&units=metric"
)
_DATUMS_URL = (
    "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{sid}/datums.json"
    "?units=metric&datum=NAVD"
)
_TIMEOUT = 15
_DELAY = 0.3


# ── geometry ──────────────────────────────────────────────────────────────────

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(
        math.radians(lat2)
    ) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


# ── NOAA CO-OPS ───────────────────────────────────────────────────────────────

def fetch_stations():
    print("Fetching NOAA CO-OPS water-level station list …", flush=True)
    r = requests.get(_STATIONS_URL, timeout=_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    stations = data.get("stations", [])
    rows = []
    for s in stations:
        try:
            rows.append({
                "id":   s["id"],
                "name": s["name"],
                "lat":  float(s["lat"]),
                "lon":  float(s["lng"]),
            })
        except (KeyError, ValueError):
            pass
    df = pd.DataFrame(rows)
    print(f"  {len(df)} stations loaded")
    return df


def nearest_station_with_navd88(site_lat, site_lon, stations_df, datum_cache, max_candidates=15):
    """Return (gauge_id, gauge_name, distance_km) for the nearest station that has
    NAVD88 tidal datums.  Checks up to max_candidates nearest stations in order."""
    dists = stations_df.apply(
        lambda r: _haversine_km(site_lat, site_lon, r["lat"], r["lon"]), axis=1
    )
    sorted_idx = dists.sort_values().index

    for idx in sorted_idx[:max_candidates]:
        row = stations_df.loc[idx]
        gid = row["id"]
        if gid not in datum_cache:
            time.sleep(_DELAY)
            datum_cache[gid] = fetch_datums(gid)
        mhw, mllw, msl = datum_cache[gid]
        if not math.isnan(mhw):
            return row["id"], row["name"], float(dists[idx])

    # Fallback: nearest regardless
    idx = dists.idxmin()
    row = stations_df.loc[idx]
    return row["id"], row["name"], float(dists[idx])


def fetch_datums(station_id):
    """Return (mhw_navd88, mllw_navd88, msl_navd88) in metres, or (nan, nan, nan) on failure.

    NOAA returns all values relative to the local Station Datum (STND).
    The 'NAVD88' entry gives the height of STND above NAVD88, so:
        value_NAVD88 = value_relative_to_STND - navd88_offset
    """
    url = _DATUMS_URL.format(sid=station_id)
    try:
        r = requests.get(url, timeout=_TIMEOUT)
        r.raise_for_status()
        datums = {d["name"]: float(d["value"]) for d in r.json().get("datums", [])}
        navd88_offset = datums.get("NAVD88", np.nan)
        if math.isnan(navd88_offset):
            print(f"    WARNING: no NAVD88 offset for {station_id}", file=sys.stderr)
            return np.nan, np.nan, np.nan
        mhw  = datums.get("MHW",  np.nan) - navd88_offset
        mllw = datums.get("MLLW", np.nan) - navd88_offset
        msl  = datums.get("MSL",  np.nan) - navd88_offset
        return mhw, mllw, msl
    except Exception as exc:
        print(f"    WARNING: datum fetch failed for {station_id}: {exc}", file=sys.stderr)
        return np.nan, np.nan, np.nan


# ── inundation fraction ───────────────────────────────────────────────────────

def inundation_fraction(elev_navd88, mhw, mllw):
    """
    Sinusoidal semi-diurnal approximation of the model inundation_fraction.

    IF = arccos(z_rel / A) / π,  clamped to [0, 1]
    where z_rel = elev - MSL,  A = (MHW - MLLW) / 2
    """
    if any(math.isnan(v) for v in (elev_navd88, mhw, mllw)):
        return np.nan
    A   = (mhw - mllw) / 2.0
    msl = (mhw + mllw) / 2.0
    if A <= 0:
        return np.nan
    z_rel = elev_navd88 - msl
    if z_rel >= A:
        return 0.0
    if z_rel <= -A:
        return 1.0
    return math.acos(z_rel / A) / math.pi


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(_CSV)
    print(f"Loaded {len(df)} sites from {_CSV.name}")

    stations = fetch_stations()

    # ── find nearest gauge WITH NAVD88 datums per site ──
    # Pre-populate datum_cache as we probe candidates; avoids re-querying.
    datum_cache = {}
    gauge_ids, gauge_names, gauge_dists = [], [], []
    print("Finding nearest NAVD88-leveled gauge per site …")
    for _, row in df.iterrows():
        gid, gname, dist = nearest_station_with_navd88(
            row["lat"], row["lon"], stations, datum_cache
        )
        gauge_ids.append(gid)
        gauge_names.append(gname)
        gauge_dists.append(round(dist, 2))
        print(f"  {row['site_id']:20s}  →  {gname} ({gid})  {dist:.1f} km")

    df["noaa_gauge_id"]        = gauge_ids
    df["noaa_gauge_name"]      = gauge_names
    df["distance_to_gauge_km"] = gauge_dists

    # ── ensure datums for any gauges not yet fetched ──
    unique_gauges = df["noaa_gauge_id"].unique()
    print(f"\nTidal datums (NAVD88):")
    for gid in unique_gauges:
        if gid not in datum_cache:
            datum_cache[gid] = fetch_datums(gid)
            time.sleep(_DELAY)
        mhw, mllw, msl = datum_cache[gid]
        gname = df.loc[df["noaa_gauge_id"] == gid, "noaa_gauge_name"].iloc[0]
        print(f"  {gid}  {gname:40s}  MHW={mhw:+.3f} m  MLLW={mllw:+.3f} m  MSL={msl:+.3f} m")

    df["gauge_mhw_navd88"]  = df["noaa_gauge_id"].map(lambda g: datum_cache[g][0])
    df["gauge_mllw_navd88"] = df["noaa_gauge_id"].map(lambda g: datum_cache[g][1])
    df["gauge_msl_navd88"]  = df["noaa_gauge_id"].map(lambda g: datum_cache[g][2])

    # ── compute inundation fraction ──
    # Use EPQS elevation (same as model forcing default)
    df["inundation_fraction"] = df.apply(
        lambda r: inundation_fraction(
            r["epqs_elevation_m"], r["gauge_mhw_navd88"], r["gauge_mllw_navd88"]
        ),
        axis=1,
    ).round(4)

    # Also compute using RTK elevation where available
    df["inundation_fraction_rtk"] = df.apply(
        lambda r: inundation_fraction(
            r["rtk_elevation_m"], r["gauge_mhw_navd88"], r["gauge_mllw_navd88"]
        ) if not (isinstance(r["rtk_elevation_m"], float) and math.isnan(r["rtk_elevation_m"]))
        else np.nan,
        axis=1,
    ).round(4)

    # Back up original and save
    backup = _CSV.with_suffix(".csv.bak")
    import shutil
    shutil.copy(_CSV, backup)

    for col in ("gauge_mhw_navd88", "gauge_mllw_navd88", "gauge_msl_navd88"):
        df[col] = df[col].round(4)

    df.to_csv(_CSV, index=False)
    print(f"\nSaved {len(df)} rows → {_CSV}")
    print(f"(Original backed up to {backup.name})")
    print()

    cols = [
        "site_id", "epqs_elevation_m", "noaa_gauge_name", "distance_to_gauge_km",
        "gauge_mhw_navd88", "gauge_msl_navd88", "gauge_mllw_navd88",
        "inundation_fraction", "inundation_fraction_rtk",
    ]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
