#!/usr/bin/env python3
"""
add_water_distance.py

For each marsh core in core_inundation.fgb, query the USGS 3D Hydrography
Program (3DHP) ArcGIS FeatureServer and compute:

  dist_flowline_m   — distance (m) to nearest 3DHP flowline
  dist_waterbody_m  — distance (m) to nearest 3DHP waterbody boundary
                      (0 if the core is inside a waterbody polygon)

Strategy
--------
  Cores are snapped to a 0.15° tile grid (335 unique tiles).  Each tile is
  queried once per layer (flowline + waterbody) with a 0.05° outward buffer,
  reducing 3815 individual requests to ~670 tile requests.  Geometries are
  cached per-tile and all distances are computed in EPSG:5070 (CONUS Albers
  Equal-Area) for accurate metre values.

  Concurrency: up to 8 parallel tile requests with retry on failure.

3DHP FeatureServer
------------------
  https://3dhp.nationalmap.gov/arcgis/rest/services/usgs_3dhp_all/FeatureServer
  Layer 50 — Flowline  (PolyLine)
  Layer 60 — Waterbody (Polygon)

Usage
-----
  python3 scripts/add_water_distance.py
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import requests

_THIS    = Path(__file__).parent
_ROOT    = _THIS.parent
_FGB     = _THIS / "core_inundation.fgb"

_BASE_URL  = "https://3dhp.nationalmap.gov/arcgis/rest/services/usgs_3dhp_all/FeatureServer"
_LYR_FLOW  = 50   # Flowline
_LYR_WB    = 60   # Waterbody

_TILE_DEG  = 0.15   # grid cell size in degrees
_BUF_DEG   = 0.05   # extra buffer added when querying each tile
_WORKERS   = 8
_TIMEOUT   = 30
_MAX_RETRY = 3

_CRS_IN    = "EPSG:4326"
_CRS_DIST  = "EPSG:5070"  # CONUS Albers Equal-Area — metres


# ── API helpers ───────────────────────────────────────────────────────────────

def _query_layer(layer_id: int, xmin: float, ymin: float, xmax: float, ymax: float,
                 fields: str = "*") -> gpd.GeoDataFrame | None:
    """Query a single tile bbox from a 3DHP FeatureServer layer. Returns GeoDataFrame or None."""
    url = f"{_BASE_URL}/{layer_id}/query"
    params = {
        "geometry":     f"{xmin},{ymin},{xmax},{ymax}",
        "geometryType": "esriGeometryEnvelope",
        "inSR":         4326,
        "outSR":        4326,
        "spatialRel":   "esriSpatialRelIntersects",
        "outFields":    fields,
        "returnGeometry": True,
        "f":            "geojson",
    }
    for attempt in range(_MAX_RETRY):
        try:
            r = requests.get(url, params=params, timeout=_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            if "features" not in data or not data["features"]:
                return None
            gdf = gpd.GeoDataFrame.from_features(data["features"], crs=_CRS_IN)
            return gdf if not gdf.empty else None
        except Exception as exc:
            if attempt == _MAX_RETRY - 1:
                print(f"    WARNING layer {layer_id} tile ({xmin:.2f},{ymin:.2f}): {exc}")
            else:
                time.sleep(1.5 ** attempt)
    return None


# ── tiling ────────────────────────────────────────────────────────────────────

def _tile_key(lat: float, lon: float) -> tuple[int, int]:
    return int(lat / _TILE_DEG), int(lon / _TILE_DEG)


def _tile_bbox(tk: tuple[int, int]) -> tuple[float, float, float, float]:
    trow, tcol = tk
    return (
        tcol * _TILE_DEG - _BUF_DEG,
        trow * _TILE_DEG - _BUF_DEG,
        (tcol + 1) * _TILE_DEG + _BUF_DEG,
        (trow + 1) * _TILE_DEG + _BUF_DEG,
    )


# ── distance computation ──────────────────────────────────────────────────────

def _nearest_dist_m(core_geom_proj, features_gdf_proj) -> float:
    """Distance in metres from core_geom_proj to nearest geometry in features_gdf_proj."""
    if features_gdf_proj is None or features_gdf_proj.empty:
        return np.nan
    dists = features_gdf_proj.geometry.distance(core_geom_proj)
    return float(dists.min())


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 60)
    print("  3DHP water distance → core_inundation.fgb")
    print("=" * 60)

    # Load cores
    print("\n[1] Loading cores …")
    gdf = gpd.read_file(_FGB)
    print(f"  {len(gdf)} cores")

    # Assign tile keys
    gdf["_tile"] = [_tile_key(lat, lon) for lat, lon in zip(gdf["latitude"], gdf["longitude"])]
    tiles = gdf["_tile"].unique()
    print(f"  {len(tiles)} unique 0.15° tiles")

    # Download tile geometries in parallel
    print(f"\n[2] Querying 3DHP (flowlines + waterbodies, {_WORKERS} workers) …")

    tile_flow: dict = {}
    tile_wb:   dict = {}

    jobs: list[tuple] = []
    for tk in tiles:
        bbox = _tile_bbox(tk)
        jobs.append(("flow", tk, bbox))
        jobs.append(("wb",   tk, bbox))

    print(f"  {len(jobs)} tile queries")

    def _run_job(job):
        kind, tk, bbox = job
        lid = _LYR_FLOW if kind == "flow" else _LYR_WB
        fields = "id3dhp,featuretypelabel" if kind == "flow" else "id3dhp,featuretypelabel,areasqkm"
        result = _query_layer(lid, *bbox, fields=fields)
        return kind, tk, result

    done_count = 0
    with ThreadPoolExecutor(max_workers=_WORKERS) as ex:
        futures = {ex.submit(_run_job, j): j for j in jobs}
        for fut in as_completed(futures):
            kind, tk, result = fut.result()
            if kind == "flow":
                tile_flow[tk] = result
            else:
                tile_wb[tk] = result
            done_count += 1
            if done_count % 50 == 0 or done_count == len(jobs):
                print(f"    {done_count}/{len(jobs)} queries done")

    # Report coverage
    flow_hits = sum(1 for v in tile_flow.values() if v is not None)
    wb_hits   = sum(1 for v in tile_wb.values()   if v is not None)
    print(f"  Tiles with flowlines:   {flow_hits}/{len(tiles)}")
    print(f"  Tiles with waterbodies: {wb_hits}/{len(tiles)}")

    # Project all tile GDFs to CONUS Albers once
    print("\n[3] Projecting geometries to EPSG:5070 …")
    tile_flow_proj = {
        tk: gd.to_crs(_CRS_DIST) for tk, gd in tile_flow.items() if gd is not None
    }
    tile_wb_proj = {
        tk: gd.to_crs(_CRS_DIST) for tk, gd in tile_wb.items() if gd is not None
    }

    # Compute per-core distances
    print("\n[4] Computing distances …")
    cores_proj = gdf.set_geometry(
        gpd.points_from_xy(gdf["longitude"], gdf["latitude"]),
        crs=_CRS_IN,
    ).to_crs(_CRS_DIST)

    dist_flow = np.full(len(gdf), np.nan)
    dist_wb   = np.full(len(gdf), np.nan)

    for idx, (row_i, row) in enumerate(gdf.iterrows()):
        tk = row["_tile"]
        pt = cores_proj.geometry.iloc[idx]
        dist_flow[idx] = _nearest_dist_m(pt, tile_flow_proj.get(tk))
        dist_wb[idx]   = _nearest_dist_m(pt, tile_wb_proj.get(tk))

    # Attach and save
    print("\n[5] Saving …")
    gdf["dist_flowline_m"]  = dist_flow
    gdf["dist_waterbody_m"] = dist_wb
    gdf = gdf.drop(columns=["_tile"])
    gdf.to_file(_FGB, driver="FlatGeobuf")
    print(f"  Saved {_FGB}")

    # Stats
    vf = dist_flow[~np.isnan(dist_flow)]
    vw = dist_wb[~np.isnan(dist_wb)]
    print(f"\n  dist_flowline_m  n={len(vf)}  median={np.median(vf):.0f} m  "
          f"min={vf.min():.0f} m  max={vf.max():.0f} m")
    print(f"  dist_waterbody_m n={len(vw)}  median={np.median(vw):.0f} m  "
          f"min={vw.min():.0f} m  max={vw.max():.0f} m")

    print(f"\n  Total time: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
