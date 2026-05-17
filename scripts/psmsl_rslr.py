"""
psmsl_rslr.py

Compute PSMSL-based relative sea level rise (RSLR) for US CCN marsh cores.

RSLR from PSMSL tide gauge records already incorporates vertical land motion
(subsidence/uplift) because gauges measure water level relative to land.
This is the correct quantity for assessing whether marshes are keeping pace
with sea level rise.

Workflow
--------
1. Load core lat/lon from core_inundation.fgb (US CCN cores only).
2. Fetch the PSMSL RLR station list (single HTTP request; cached to disk).
3. For each core, find the nearest PSMSL station.
4. Download annual mean sea level data for each unique matched station
   (cached to core_data/psmsl_cache/ to avoid re-downloading).
5. Fit a weighted linear trend (weight = 1 for all years; OLS).
6. Output core_data/ccn_us_psmsl_rslr.csv.

PSMSL data format
-----------------
Station list: https://www.psmsl.org/data/obtaining/rlr.annual.data/filelist.txt
  Fields: station_id; latitude; longitude; name; coastline_code; station_code; flag

Annual data: https://www.psmsl.org/data/obtaining/rlr.annual.data/{id}.rlrdata
  Fields: year; value_mm; interpolated_flag; metric_flag
  Missing value: -99999

Quality thresholds
------------------
MIN_RECORD_YR  = 30   Minimum years of non-missing data to compute a trend.
                      Shorter records yield trends dominated by interannual
                      variability rather than the long-term signal.
MAX_DIST_KM    = 500  Stations beyond this distance are flagged low_confidence
                      but not excluded; RSLR may not represent local VLM.
"""

import sys
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from pyproj import Geod
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIN_RECORD_YR = 30
MAX_DIST_KM   = 500
_DELAY_S      = 0.25   # pause between HTTP requests (be polite to PSMSL)

_ROOT      = Path(__file__).parent.parent
_INUND_FGB = _ROOT / "scripts" / "core_inundation.fgb"
_CACHE_DIR = _ROOT / "core_data" / "psmsl_cache"
_OUT_CSV   = _ROOT / "core_data" / "ccn_us_psmsl_rslr.csv"

_FILELIST_URL = "https://www.psmsl.org/data/obtaining/rlr.annual.data/filelist.txt"
_DATA_URL     = "https://www.psmsl.org/data/obtaining/rlr.annual.data/{sid}.rlrdata"
_TIMEOUT      = 20


# ---------------------------------------------------------------------------
# PSMSL station list
# ---------------------------------------------------------------------------

def fetch_station_list() -> pd.DataFrame:
    """Return DataFrame of all PSMSL RLR stations (id, lat, lon, name)."""
    cache = _CACHE_DIR / "filelist.txt"
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if cache.exists():
        print(f"  Station list loaded from cache ({cache})")
        text = cache.read_text()
    else:
        print("  Fetching PSMSL station list …")
        r = requests.get(_FILELIST_URL, timeout=_TIMEOUT)
        r.raise_for_status()
        text = r.text
        cache.write_text(text)

    rows = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 4:
            continue
        try:
            rows.append({
                "station_id":   int(parts[0]),
                "latitude":     float(parts[1]),
                "longitude":    float(parts[2]),
                "station_name": parts[3].strip(),
            })
        except ValueError:
            pass

    df = pd.DataFrame(rows)
    print(f"  {len(df)} PSMSL RLR stations")
    return df


# ---------------------------------------------------------------------------
# Download and cache one station's annual data
# ---------------------------------------------------------------------------

def fetch_station_data(station_id: int) -> pd.DataFrame | None:
    """Return annual mean sea level DataFrame for one station, or None."""
    cache_file = _CACHE_DIR / f"{station_id}.rlrdata"

    if cache_file.exists():
        text = cache_file.read_text()
    else:
        url = _DATA_URL.format(sid=station_id)
        try:
            r = requests.get(url, timeout=_TIMEOUT)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"    WARNING: failed to fetch station {station_id}: {e}",
                  file=sys.stderr)
            return None
        text = r.text
        cache_file.write_text(text)
        time.sleep(_DELAY_S)

    rows = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 2:
            continue
        try:
            yr  = int(parts[0])
            val = float(parts[1])
            rows.append({"year": yr, "value_mm": val})
        except ValueError:
            pass

    if not rows:
        return None
    df = pd.DataFrame(rows)
    df = df[df["value_mm"] > -9999].copy()   # exclude missing (-99999)
    return df if len(df) else None


# ---------------------------------------------------------------------------
# Linear trend
# ---------------------------------------------------------------------------

def compute_trend(df: pd.DataFrame) -> dict:
    """Fit OLS linear trend to annual mean sea level; return rate in mm/yr."""
    if df is None or len(df) < MIN_RECORD_YR:
        return {
            "rslr_mmyr":    np.nan,
            "rslr_se_mmyr": np.nan,
            "record_years": 0,
            "year_start":   np.nan,
            "year_end":     np.nan,
        }
    result = stats.linregress(df["year"], df["value_mm"])
    return {
        "rslr_mmyr":    result.slope,
        "rslr_se_mmyr": result.stderr,
        "record_years": len(df),
        "year_start":   int(df["year"].min()),
        "year_end":     int(df["year"].max()),
    }


# ---------------------------------------------------------------------------
# Nearest-station matching
# ---------------------------------------------------------------------------

def match_nearest(core_lats: np.ndarray, core_lons: np.ndarray,
                  sta_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (station_indices, distances_km) for each core."""
    geod = Geod(ellps="WGS84")
    sta_lons = sta_df["longitude"].values
    sta_lats = sta_df["latitude"].values

    indices  = np.empty(len(core_lats), dtype=int)
    dists_km = np.empty(len(core_lats))

    for i, (clat, clon) in enumerate(zip(core_lats, core_lons)):
        _, _, dist_m = geod.inv(
            np.full(len(sta_lons), clon),
            np.full(len(sta_lats), clat),
            sta_lons, sta_lats,
        )
        idx = int(np.argmin(np.abs(dist_m)))
        indices[i]  = idx
        dists_km[i] = np.abs(dist_m[idx]) / 1000.0

    return indices, dists_km


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load US CCN cores
    print("Loading US CCN core locations …")
    cores = gpd.read_file(_INUND_FGB)[["core_id", "latitude", "longitude",
                                        "slr_mmyr"]].copy()
    cores = cores.dropna(subset=["latitude", "longitude"])
    print(f"  {len(cores)} US CCN cores")

    # PSMSL station list
    print("\nFetching PSMSL station list …")
    stations = fetch_station_list()

    # Match each core to nearest PSMSL station
    print("\nMatching cores to nearest PSMSL station …")
    sta_idx, dists = match_nearest(
        cores["latitude"].values,
        cores["longitude"].values,
        stations,
    )
    cores["psmsl_id"]      = stations["station_id"].values[sta_idx]
    cores["psmsl_name"]    = stations["station_name"].values[sta_idx]
    cores["psmsl_dist_km"] = dists.round(1)
    cores["psmsl_confidence"] = np.where(dists <= MAX_DIST_KM, "ok", "low_confidence")

    # Download data for each unique station
    unique_ids = cores["psmsl_id"].unique()
    print(f"\nDownloading data for {len(unique_ids)} unique PSMSL stations …")
    trends = {}
    for i, sid in enumerate(unique_ids):
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(unique_ids)} …")
        data = fetch_station_data(int(sid))
        trends[sid] = compute_trend(data)

    # Attach trends to core table
    cores["psmsl_rslr_mmyr"]    = cores["psmsl_id"].map(lambda s: trends[s]["rslr_mmyr"])
    cores["psmsl_rslr_se_mmyr"] = cores["psmsl_id"].map(lambda s: trends[s]["rslr_se_mmyr"])
    cores["psmsl_record_years"] = cores["psmsl_id"].map(lambda s: trends[s]["record_years"])
    cores["psmsl_year_start"]   = cores["psmsl_id"].map(lambda s: trends[s]["year_start"])
    cores["psmsl_year_end"]     = cores["psmsl_id"].map(lambda s: trends[s]["year_end"])

    # Summary
    n_ok = cores["psmsl_rslr_mmyr"].notna().sum()
    n_lc = (cores["psmsl_confidence"] == "low_confidence").sum()
    print(f"\nRSLR computed:     {n_ok}/{len(cores)} cores")
    print(f"Low confidence     (dist > {MAX_DIST_KM} km): {n_lc}")
    rates = cores["psmsl_rslr_mmyr"].dropna()
    print(f"PSMSL RSLR (mm/yr): median={rates.median():.2f}  "
          f"mean={rates.mean():.2f}  "
          f"range=[{rates.min():.2f}, {rates.max():.2f}]")

    # Compare with existing NOAA slr_mmyr where available
    both = cores.dropna(subset=["psmsl_rslr_mmyr", "slr_mmyr"])
    if len(both):
        from scipy.stats import pearsonr
        r, _ = pearsonr(both["psmsl_rslr_mmyr"], both["slr_mmyr"])
        diff  = both["slr_mmyr"] - both["psmsl_rslr_mmyr"]
        print(f"\nComparison with existing NOAA slr_mmyr (n={len(both)}):")
        print(f"  Pearson r = {r:.3f}")
        print(f"  Mean diff (NOAA − PSMSL) = {diff.mean():.2f} mm/yr")
        print(f"  Std diff = {diff.std():.2f} mm/yr")

    cores.drop(columns=["geometry"], errors="ignore").to_csv(_OUT_CSV, index=False)
    print(f"\nSaved: {_OUT_CSV}")


if __name__ == "__main__":
    main()
