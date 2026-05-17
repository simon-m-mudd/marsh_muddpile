"""Convert CCN_depthseries.csv to FlatGeobuf by joining with CCN_cores.fgb for coordinates.

Usage:
    python3 scripts/ccn_depthseries_to_fgb.py [--repo-root /path/to/marsh_muddpile]

The output is written to core_data/CCN_synthesis/CCN_depthseries.fgb.
Rows with no matching core (and therefore no coordinates) are dropped.
"""

import argparse
from pathlib import Path

import pandas as pd
import geopandas as gpd


def main():
    parser = argparse.ArgumentParser(
        description="Convert CCN_depthseries.csv to FlatGeobuf using CCN_cores.fgb for coordinates."
    )
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

    synthesis_dir = repo_root / "core_data" / "CCN_synthesis"
    csv_path = synthesis_dir / "CCN_depthseries.csv"
    cores_path = synthesis_dir / "CCN_cores.fgb"
    out_path = synthesis_dir / "CCN_depthseries.fgb"

    print(f"Reading cores: {cores_path}")
    cores = gpd.read_file(cores_path)[["study_id", "core_id", "geometry"]]

    print(f"Reading depthseries: {csv_path}")
    ds = pd.read_csv(csv_path, low_memory=False)

    before = len(ds)
    gdf = cores.merge(ds, on=["study_id", "core_id"], how="inner")
    dropped = before - len(gdf)
    if dropped:
        print(f"Dropped {dropped} row(s) with no matching core.")

    gdf.to_file(out_path, driver="FlatGeobuf")
    print(f"Wrote {len(gdf)} features to: {out_path}")


if __name__ == "__main__":
    main()
