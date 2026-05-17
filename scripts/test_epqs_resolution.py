"""
Test the effective resolution of the EPQS API by querying a grid of points
offset 2 m and 4 m from several NI plot centres in each cardinal direction.

If EPQS is sampling a 1 m DTM, elevations should vary between neighbours.
If it is sampling a 10 m (or 3 m) DTM, points within one cell will return
identical values.

Anchor plots (WGS84):
  GI-HM-P5  : 33.331562, -79.198313
  GI-HM-P9  : 33.331640, -79.198197
  GI-LM-P18 : 33.331589, -79.197904
"""

from pathlib import Path
import time
import textwrap

import numpy as np
import pandas as pd
import requests
from pyproj import Transformer

_THIS_DIR = Path(__file__).parent
_DTM_PATH = _THIS_DIR.parent / "DTMs" / "north_inlet" / "clipped.tif"
_EPQS_URL = "https://epqs.nationalmap.gov/v1/json"
_DELAY    = 0.3   # seconds between API calls

# UTM17N <-> WGS84
_TO_UTM  = Transformer.from_crs("EPSG:4326", "EPSG:26917", always_xy=True)
_TO_WGS  = Transformer.from_crs("EPSG:26917", "EPSG:4326", always_xy=True)

ANCHORS = {
    "GI-HM-P5" : (33.331562, -79.198313),
    "GI-HM-P9" : (33.331640, -79.198197),
    "GI-LM-P18": (33.331589, -79.197904),
}

OFFSETS_M = [0, 2, 4]          # distances from centre
DIRECTIONS = {                  # cardinal unit vectors in (dE, dN)
    "C": ( 0,  0),
    "N": ( 0,  1),
    "S": ( 0, -1),
    "E": ( 1,  0),
    "W": (-1,  0),
}


def _epqs(lat, lon):
    r = requests.get(
        _EPQS_URL,
        params={"x": lon, "y": lat, "wkid": 4326, "units": "Meters", "includeDate": "false"},
        timeout=10,
    )
    r.raise_for_status()
    val = r.json().get("value")
    return float(val) if val is not None else np.nan


def _dtm(lat, lon):
    import rasterio
    with rasterio.open(_DTM_PATH) as src:
        x, y = _TO_UTM.transform(lon, lat)
        b = src.bounds
        if not (b.left <= x <= b.right and b.bottom <= y <= b.top):
            return np.nan
        v = float(next(src.sample([(x, y)]))[0])
        return np.nan if v == src.nodata else v


def build_grid(anchor_lat, anchor_lon):
    """Return list of (label, offset_m, direction, lat, lon)."""
    cx, cy = _TO_UTM.transform(anchor_lon, anchor_lat)
    rows = []
    for d, (de, dn) in DIRECTIONS.items():
        for dist in OFFSETS_M:
            if dist == 0 and d != "C":
                continue          # centre only once
            x = cx + de * dist
            y = cy + dn * dist
            lon, lat = _TO_WGS.transform(x, y)
            label = "C" if d == "C" else f"{d}{dist}"
            rows.append((label, dist if d != "C" else 0, d, lat, lon))
    return rows


def main():
    results = []
    for plot, (alat, alon) in ANCHORS.items():
        grid = build_grid(alat, alon)
        print(f"\n{'='*60}")
        print(f"Plot: {plot}  centre ({alat:.6f}, {alon:.6f})")
        print(f"{'Label':<6} {'Dir':<3} {'Offset(m)':<10} {'Lat':<12} {'Lon':<12} {'EPQS(m)':<10} {'DTM(m)':<10}")
        print("-"*60)
        for label, dist, dirn, lat, lon in grid:
            epqs_val = _epqs(lat, lon)
            dtm_val  = _dtm(lat, lon)
            print(f"{label:<6} {dirn:<3} {dist:<10} {lat:<12.6f} {lon:<12.6f} {epqs_val:<10.4f} {dtm_val:<10.4f}")
            results.append(dict(
                plot=plot, label=label, direction=dirn, offset_m=dist,
                lat=lat, lon=lon, epqs_m=epqs_val, dtm_m=dtm_val,
            ))
            time.sleep(_DELAY)

    df = pd.DataFrame(results)

    # ── resolution diagnosis ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Resolution diagnosis — std of EPQS across all offsets per plot:")
    for plot, grp in df.groupby("plot"):
        std_epqs = grp["epqs_m"].std()
        std_dtm  = grp["dtm_m"].std()
        # Count unique EPQS values (identical values → coarse grid)
        n_unique = grp["epqs_m"].nunique()
        print(f"  {plot}: EPQS std={std_epqs:.4f} m  ({n_unique} unique values / {len(grp)} points)  |  DTM std={std_dtm:.4f} m")

    print()
    print(textwrap.dedent("""
    Interpretation:
      std ≈ 0, all values identical   → EPQS cell > 4 m  (likely 10 m DEM)
      std > 0 at 2 m but not at 4 m  → cell ≈ 3 m
      std > 0 even at 2 m offsets     → EPQS is using ≤ 1 m DEM
    """))


if __name__ == "__main__":
    main()
