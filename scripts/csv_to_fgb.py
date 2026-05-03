# This converts csv data to fgb data to save some space. 
# Flatgeobuf format data is readable by geopandas
# run with
# python3 csv_to_fgb.py /path/to/your/data/core_data/my_file.csv


import argparse
from pathlib import Path
import pandas as pd
import geopandas as gpd


COMMON_LAT_NAMES = [
    "Latitude", "latitude", "LATITUDE", "lat", "Lat", "LAT", "y", "Y"
]

COMMON_LON_NAMES = [
    "Longitude", "longitude", "LONGITUDE", "lon", "Lon", "LON",
    "long", "Long", "LONG", "x", "X"
]


def clean_dataframe_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False, na=False)].copy()
    df = df.dropna(axis=1, how="all")
    return df


def resolve_column_name(requested_name, available_names):
    requested_clean = requested_name.strip().casefold()
    for actual_name in available_names:
        if str(actual_name).strip().casefold() == requested_clean:
            return actual_name
    return None


def guess_column_name(candidates, available_names):
    for cand in candidates:
        matched = resolve_column_name(cand, available_names)
        if matched is not None:
            return matched
    return None


def csv_to_geodf(csv_path, lat_col=None, lon_col=None, delimiter=","):
    df = pd.read_csv(csv_path, sep=delimiter)
    df = clean_dataframe_columns(df)

    if lat_col is not None:
        lat_name = resolve_column_name(lat_col, df.columns)
        if lat_name is None:
            raise ValueError(f"Latitude column {lat_col!r} not found in {csv_path.name}")
    else:
        lat_name = guess_column_name(COMMON_LAT_NAMES, df.columns)
        if lat_name is None:
            raise ValueError(
                f"Could not guess latitude column in {csv_path.name}. "
                f"Available columns: {list(df.columns)}"
            )

    if lon_col is not None:
        lon_name = resolve_column_name(lon_col, df.columns)
        if lon_name is None:
            raise ValueError(f"Longitude column {lon_col!r} not found in {csv_path.name}")
    else:
        lon_name = guess_column_name(COMMON_LON_NAMES, df.columns)
        if lon_name is None:
            raise ValueError(
                f"Could not guess longitude column in {csv_path.name}. "
                f"Available columns: {list(df.columns)}"
            )

    df[lat_name] = pd.to_numeric(df[lat_name], errors="coerce").astype("float64")
    df[lon_name] = pd.to_numeric(df[lon_name], errors="coerce").astype("float64")

    before = len(df)
    df = df.dropna(subset=[lat_name, lon_name]).copy()
    dropped = before - len(df)

    df["record_id"] = range(len(df))

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_name], df[lat_name]),
        crs="EPSG:4326"
    )

    return gdf, lat_name, lon_name, dropped


def convert_file(csv_file, output_file=None, lat_col=None, lon_col=None, delimiter=","):
    csv_file = Path(csv_file)

    if output_file is None:
        output_file = csv_file.with_suffix(".fgb")
    else:
        output_file = Path(output_file)

    print(f"\nProcessing: {csv_file}")
    gdf, lat_name, lon_name, dropped = csv_to_geodf(
        csv_file, lat_col=lat_col, lon_col=lon_col, delimiter=delimiter
    )

    print(f"Using latitude column:  {lat_name!r}")
    print(f"Using longitude column: {lon_name!r}")
    print(f"Rows written: {len(gdf)}")
    print(f"Rows dropped for missing coords: {dropped}")

    gdf.to_file(output_file, driver="FlatGeobuf")
    print(f"Wrote: {output_file}")


def convert_folder(folder, output_dir=None, lat_col=None, lon_col=None, delimiter=","):
    folder = Path(folder)

    if output_dir is None:
        output_dir = folder
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {folder}")

    for csv_file in csv_files:
        out_file = output_dir / f"{csv_file.stem}.fgb"
        convert_file(
            csv_file,
            output_file=out_file,
            lat_col=lat_col,
            lon_col=lon_col,
            delimiter=delimiter
        )


def main():
    parser = argparse.ArgumentParser(description="Convert CSV file(s) with lat/lon to FlatGeobuf.")
    parser.add_argument("input_path", help="CSV file or folder containing CSV files")
    parser.add_argument("--out", help="Output FGB file (for single CSV only)")
    parser.add_argument("--outdir", help="Output directory (for folder mode)")
    parser.add_argument("--lat-col", help="Latitude column name")
    parser.add_argument("--lon-col", help="Longitude column name")
    parser.add_argument("--delimiter", default=",", help="CSV delimiter, default is ','")
    args = parser.parse_args()

    input_path = Path(args.input_path)

    if input_path.is_file():
        convert_file(
            input_path,
            output_file=args.out,
            lat_col=args.lat_col,
            lon_col=args.lon_col,
            delimiter=args.delimiter
        )
    elif input_path.is_dir():
        convert_folder(
            input_path,
            output_dir=args.outdir,
            lat_col=args.lat_col,
            lon_col=args.lon_col,
            delimiter=args.delimiter
        )
    else:
        raise ValueError(f"Input path not found: {input_path}")


if __name__ == "__main__":
    main()
