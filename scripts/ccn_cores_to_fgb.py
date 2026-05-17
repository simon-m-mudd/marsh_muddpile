"""Convert CCN_cores.csv to FlatGeobuf, writing the output alongside the source data.

Usage:
    python3 scripts/ccn_cores_to_fgb.py [--repo-root /path/to/marsh_muddpile]

The output is written to core_data/CCN_synthesis/CCN_cores.fgb.
"""

import argparse
from pathlib import Path

import pandas as pd
import geopandas as gpd


def main():
    parser = argparse.ArgumentParser(description="Convert CCN_cores.csv to FlatGeobuf.")
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Path to the marsh_muddpile repository root. Defaults to the parent of this script's directory.",
    )
    args = parser.parse_args()

    if args.repo_root is not None:
        repo_root = Path(args.repo_root)
    else:
        repo_root = Path(__file__).resolve().parent.parent

    csv_path = repo_root / "core_data" / "CCN_synthesis" / "CCN_cores.csv"
    out_path = csv_path.with_suffix(".fgb")

    print(f"Reading: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} row(s) with missing coordinates.")

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )

    gdf.to_file(out_path, driver="FlatGeobuf")
    print(f"Wrote {len(gdf)} features to: {out_path}")


if __name__ == "__main__":
    main()
