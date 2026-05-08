"""
query_cores_by_location.py

Search for sediment cores within a radius of a given lat/lon and report
which attributes are present in those cores.

Usage:
    python query_cores_by_location.py --lat 51.5 --lon -0.1 --radius 50
    python query_cores_by_location.py --lat 51.5 --lon -0.1 --radius 50 --fgb path/to/merged_all.fgb
    python query_cores_by_location.py --lat 51.5 --lon -0.1 --radius 50 --csv results.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Geod

# Attributes to report on, grouped by category
ATTRIBUTE_GROUPS = {
    "bulk_properties": [
        "dry_bulk_density",
        "fraction_organic_matter",
        "fraction_carbon",
        "compaction_fraction",
    ],
    "pb210": [
        "excess_pb210_activity",
        "excess_pb210_activity_se",
        "total_pb210_activity",
        "total_pb210_activity_se",
        "ra226_activity",
        "ra226_activity_se",
        "pb214_activity",
        "pb214_activity_se",
        "bi214_activity",
        "bi214_activity_se",
    ],
    "cs137": [
        "cs137_activity",
        "cs137_activity_se",
        "cs137_peak_age",
    ],
    "carbon14": [
        "c14_age",
        "c14_age_se",
        "delta_c13",
    ],
    "other_tracers": [
        "be7_activity",
        "be7_activity_se",
        "marker_date",
        "marker_type",
    ],
    "age_model": [
        "age",
        "age_min",
        "age_max",
        "age_se",
    ],
}

# Columns that describe the core location (one value per core)
CORE_META_COLS = [
    "core_id",
    "study_id_x",
    "site_id_x",
    "latitude",
    "longitude",
    "year",
    "country",
    "habitat",
    "vegetation_class",
    "salinity_class",
    "max_depth",
    "pb210_cic_accretion_rate",
    "pb210_cic_accretion_rate_se",
]


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised geodesic distance (km) using WGS-84 ellipsoid."""
    geod = Geod(ellps="WGS84")
    _, _, dist_m = geod.inv(
        np.full(len(lat2), lon1),
        np.full(len(lat2), lat1),
        lon2,
        lat2,
    )
    return np.abs(dist_m) / 1000.0


def load_data(fgb_path: str) -> gpd.GeoDataFrame:
    path = Path(fgb_path)
    if not path.exists():
        sys.exit(f"Error: file not found: {fgb_path}")
    return gpd.read_file(fgb_path)


def find_cores_in_radius(
    gdf: gpd.GeoDataFrame, lat: float, lon: float, radius_km: float
) -> gpd.GeoDataFrame:
    """Return rows whose core location falls within radius_km of (lat, lon)."""
    core_locs = (
        gdf[["core_id", "latitude", "longitude"]]
        .drop_duplicates("core_id")
        .copy()
    )
    core_locs["distance_km"] = haversine_km(
        lat, lon,
        core_locs["latitude"].values,
        core_locs["longitude"].values,
    )
    nearby_ids = core_locs.loc[
        core_locs["distance_km"] <= radius_km, ["core_id", "distance_km"]
    ]
    result = gdf[gdf["core_id"].isin(nearby_ids["core_id"])].copy()
    result = result.merge(nearby_ids, on="core_id", how="left")
    return result


def attribute_summary(core_data: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    For each core, report which attribute groups have any non-null values,
    how many depth samples exist, and the depth range.
    """
    records = []
    for core_id, grp in core_data.groupby("core_id"):
        row = {
            "core_id": core_id,
            "distance_km": grp["distance_km"].iloc[0],
            "latitude": grp["latitude"].iloc[0],
            "longitude": grp["longitude"].iloc[0],
            "n_samples": len(grp),
            "depth_min_cm": grp["depth_min"].min(),
            "depth_max_cm": grp["depth_max"].max(),
        }
        # optional core-level metadata
        for col in ["country", "habitat", "vegetation_class", "salinity_class",
                    "year", "study_id_x", "pb210_cic_accretion_rate"]:
            if col in grp.columns:
                vals = grp[col].dropna().unique()
                row[col] = vals[0] if len(vals) == 1 else (vals[0] if len(vals) > 1 else None)

        # per-group attribute presence
        for group_name, cols in ATTRIBUTE_GROUPS.items():
            present = [c for c in cols if c in grp.columns and grp[c].notna().any()]
            row[f"has_{group_name}"] = len(present) > 0
            row[f"{group_name}_attrs"] = ", ".join(present) if present else ""

        records.append(row)

    df = pd.DataFrame(records).sort_values("distance_km")
    return df


def print_summary(summary: pd.DataFrame, lat: float, lon: float, radius_km: float):
    n_cores = len(summary)
    print(f"\nSearch centre: lat={lat}, lon={lon}, radius={radius_km} km")
    print(f"Found {n_cores} core(s)\n")

    if n_cores == 0:
        return

    print(f"{'core_id':<40} {'dist_km':>8} {'n_samp':>7} {'depth_cm':>12}  attribute groups")
    print("-" * 110)
    for _, r in summary.iterrows():
        groups_present = [
            g for g in ATTRIBUTE_GROUPS if r.get(f"has_{g}", False)
        ]
        depth_str = f"{r['depth_min_cm']:.0f}–{r['depth_max_cm']:.0f}"
        print(
            f"{str(r['core_id']):<40} {r['distance_km']:>8.1f} {r['n_samples']:>7}"
            f" {depth_str:>12}  {', '.join(groups_present)}"
        )

    print()
    print("Attribute detail per core:")
    print("-" * 110)
    for _, r in summary.iterrows():
        print(f"\n  {r['core_id']}  ({r['distance_km']:.1f} km away)")
        for group_name in ATTRIBUTE_GROUPS:
            attrs = r.get(f"{group_name}_attrs", "")
            if attrs:
                print(f"    [{group_name}]  {attrs}")


def main():
    parser = argparse.ArgumentParser(
        description="Find sediment cores near a lat/lon and report available attributes."
    )
    parser.add_argument("--lat", type=float, required=True, help="Latitude (decimal degrees)")
    parser.add_argument("--lon", type=float, required=True, help="Longitude (decimal degrees)")
    parser.add_argument(
        "--radius", type=float, default=50.0,
        help="Search radius in km (default: 50)"
    )
    parser.add_argument(
        "--fgb",
        default=str(Path(__file__).parent.parent / "core_data" / "merged_all.fgb"),
        help="Path to merged_all.fgb (default: ../core_data/merged_all.fgb)"
    )
    parser.add_argument(
        "--csv", default=None,
        help="Optional path to save the summary table as CSV"
    )
    args = parser.parse_args()

    gdf = load_data(args.fgb)
    nearby = find_cores_in_radius(gdf, args.lat, args.lon, args.radius)

    if nearby.empty:
        print(f"No cores found within {args.radius} km of ({args.lat}, {args.lon}).")
        return

    summary = attribute_summary(nearby)
    print_summary(summary, args.lat, args.lon, args.radius)

    if args.csv:
        summary.to_csv(args.csv, index=False)
        print(f"\nSummary saved to {args.csv}")


if __name__ == "__main__":
    main()
