"""
Query USGS Elevation Point Query Service (EPQS) for marsh plot locations.

API endpoint: https://epqs.nationalmap.gov/v1/json
  GET ?x=LON&y=LAT&wkid=4326&units=Meters&includeDate=false
  Returns NAVD88 elevation in metres.

Resolution note (from test_epqs_resolution.py):
  EPQS is sampling a ≤ 1 m DEM — at North Inlet, values change between
  points offset only 2 m apart, and every point in a 9-point test grid
  returned a unique value.  EPQS and the local 1 m lidar DTM agree to
  ~1–2 cm, confirming they draw from the same 3DEP 1 m dataset (or the
  same lidar acquisition).  The ~1/3 arc-second (~10 m) description in
  the EPQS documentation refers to the fallback resolution used where
  1 m coverage is unavailable.

Vegetation bias:
  EPQS (and the local lidar DTM) read ~+0.15 m above in-situ RTK-GPS
  NAVD88 surveys at NI LTER plots (range −0.07 to +0.33 m, RMSE 0.18 m).
  This is a vegetation canopy bias: the lidar first-return surface
  captures marsh grass blades rather than bare soil.  A correction of
  ~15 cm should be applied before using these elevations for inundation
  modelling against bare-ground NAVD88 benchmarks.

Sources queried:
  - lter_data/lter_sites.fgb                        (15 site centroids)
  - lter_data/edi.134.12/NILTREB_marsh_surface_elevation_init.csv  (19 NI LTER plots)
  - lter_data/knb-lter-gce.762.17/PLT-GCED-2106_coords_1_0.CSV    (47 GCE plots)

Output: scripts/lter_plot_elevations_epqs.csv
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests

_THIS_DIR = Path(__file__).parent
_LTER_DIR = _THIS_DIR.parent / "lter_data"
_OUT_CSV = _THIS_DIR / "lter_plot_elevations_epqs.csv"

_EPQS_URL      = "https://epqs.nationalmap.gov/v1/json"
_TIMEOUT_S     = 10
_EPQS_WORKERS  = 8    # USGS EPQS tolerates moderate concurrency


# ── coordinate loaders ────────────────────────────────────────────────────────

def _load_lter_sites():
    import geopandas as gpd
    gdf = gpd.read_file(_LTER_DIR / "lter_sites.fgb")
    rows = []
    for _, r in gdf.iterrows():
        rows.append({
            "source": "lter_sites",
            "site_id": r["site_code"],
            "location_type": r.get("location_type", ""),
            "lat": r.geometry.y,
            "lon": r.geometry.x,
            "measured_elevation_m": r.get("elevation_m", np.nan),
            "measured_datum": r.get("elevation_datum", ""),
        })
    return pd.DataFrame(rows)


def _load_ni_plots():
    df = pd.read_csv(
        _LTER_DIR / "edi.134.12/NILTREB_marsh_surface_elevation_init.csv"
    )
    # Mean across repeated measurements per plot
    unique = (
        df.groupby(["SITE", "LOCATION", "PLOT"])
        .agg(
            LATITUDE=("LATITUDE", "first"),
            LONGITUDE=("LONGITUDE", "first"),
            MARSH_SURFACE_ELEVATION=("MARSH_SURFACE_ELEVATION", "mean"),
        )
        .reset_index()
    )
    rows = []
    for _, r in unique.iterrows():
        rows.append({
            "source": "NI_LTER_plots",
            "site_id": f"NI-{r['SITE']}-{r['LOCATION']}-P{r['PLOT']}",
            "location_type": r["LOCATION"],
            "lat": r["LATITUDE"],
            "lon": r["LONGITUDE"],
            "measured_elevation_m": r["MARSH_SURFACE_ELEVATION"],
            "measured_datum": "NAVD88",
        })
    return pd.DataFrame(rows)


def _load_gce_plots():
    df = pd.read_csv(
        _LTER_DIR / "knb-lter-gce.762.17/PLT-GCED-2106_coords_1_0.CSV",
        skiprows=3,
        header=0,
    )
    df.columns = [
        "Site_Name", "Site_Code", "Plot", "Spartina_Zone",
        "Latitude", "Longitude", "UTM_easting", "UTM_northing",
    ]
    # Drop header/metadata rows
    df = df[~df["Latitude"].isin(["coord", "degrees", "nominal"])].copy()
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df = df.dropna(subset=["Latitude", "Longitude"])
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "source": "GCE_plots",
            "site_id": f"GCE-{r['Site_Code']}-{r['Plot']}",
            "location_type": r["Spartina_Zone"],
            "lat": r["Latitude"],
            "lon": r["Longitude"],
            "measured_elevation_m": np.nan,
            "measured_datum": "",
        })
    return pd.DataFrame(rows)


# ── EPQS query ────────────────────────────────────────────────────────────────

def _query_epqs(lat: float, lon: float) -> float | None:
    """Return NAVD88 elevation in metres from 3DEP 1 m dataset, or None on failure.

    Note: resolution is ≤ 1 m where lidar coverage exists (see module docstring).
    Values reflect the vegetation surface, not bare ground (~+0.15 m bias at NI).
    """
    params = {
        "x": lon,
        "y": lat,
        "wkid": 4326,
        "units": "Meters",
        "includeDate": "false",
    }
    try:
        r = requests.get(_EPQS_URL, params=params, timeout=_TIMEOUT_S)
        r.raise_for_status()
        data = r.json()
        val = data.get("value")
        if val is None:
            return None
        fval = float(val)
        # EPQS returns -1000000 when no data
        return fval if fval > -1e5 else None
    except Exception as exc:
        print(f"    WARNING: EPQS query failed ({exc})", file=sys.stderr)
        return None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # Load all coordinate sources
    print("Loading coordinate sources …")
    sources = [_load_lter_sites(), _load_ni_plots(), _load_gce_plots()]
    all_pts = pd.concat(sources, ignore_index=True)
    print(f"  {len(all_pts)} points total")

    # If an output file already exists, skip already-queried points
    if _OUT_CSV.exists():
        existing = pd.read_csv(_OUT_CSV)
        done_ids = set(existing["site_id"])
        pending = all_pts[~all_pts["site_id"].isin(done_ids)].copy()
        results = [existing]
        print(f"  {len(done_ids)} already cached, {len(pending)} to query")
    else:
        pending = all_pts.copy()
        results = []

    if pending.empty:
        print("All points already queried — nothing to do.")
        combined = pd.concat(results, ignore_index=True)
    else:
        n = len(pending)
        print(f"  Querying {n} points in parallel (workers={_EPQS_WORKERS}) …")
        pending_rows = list(pending.iterrows())

        def _query_row(args):
            idx, row = args
            return idx, row, _query_epqs(row["lat"], row["lon"])

        # Preserve original order for deterministic output
        idx_to_result = {}
        with ThreadPoolExecutor(max_workers=_EPQS_WORKERS) as ex:
            futures = {ex.submit(_query_row, item): item[0] for item in pending_rows}
            done = 0
            for fut in as_completed(futures):
                orig_idx, row, elev = fut.result()
                idx_to_result[orig_idx] = (row, elev)
                done += 1
                status = f"{elev:.3f} m" if elev is not None else "NO DATA"
                print(f"  [{done:3d}/{n}] {row['site_id']}  → {status}")

        queried = [
            {**row, "epqs_elevation_m": elev}
            for _, (row, elev) in sorted(idx_to_result.items())
        ]
        new_df = pd.DataFrame(queried)
        results.append(new_df)
        combined = pd.concat(results, ignore_index=True)

    # Add comparison column where measured elevation exists
    combined["elev_diff_m"] = (
        combined["epqs_elevation_m"] - combined["measured_elevation_m"]
    )

    # Save
    combined.to_csv(_OUT_CSV, index=False, float_format="%.5f")
    print(f"\nSaved {len(combined)} rows → {_OUT_CSV}")

    # Summary stats for points where we can compare
    has_both = combined.dropna(subset=["epqs_elevation_m", "measured_elevation_m"])
    if not has_both.empty:
        print(f"\nEPQS vs measured (n={len(has_both)} points with both):")
        print(f"  Mean diff (EPQS - measured): {has_both['elev_diff_m'].mean():+.3f} m")
        print(f"  Std diff:                    {has_both['elev_diff_m'].std():.3f} m")
        print(f"  Min / Max diff:              {has_both['elev_diff_m'].min():+.3f} / {has_both['elev_diff_m'].max():+.3f} m")
        print()
        print(has_both[["site_id", "measured_elevation_m", "epqs_elevation_m", "elev_diff_m"]]
              .sort_values("elev_diff_m")
              .to_string(index=False))

    return combined


if __name__ == "__main__":
    main()
