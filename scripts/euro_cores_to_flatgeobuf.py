# A script to process the european carbon core database (EURO-CARBON)
# The data is here https://zenodo.org/records/14905489
# Run with:
# python3 euro_cores_to_flatgeobuf.py /path/to/your/data/core_data/Zenodo_EURO-CARBON_Dataset_FullDataset_v2.xlsx

import re
import argparse
from pathlib import Path

import pandas as pd
import geopandas as gpd


def resolve_sheet_name(requested_name, available_names):
    """
    Match worksheet name ignoring leading/trailing whitespace and case.
    """
    requested_clean = requested_name.strip().casefold()

    for actual_name in available_names:
        if actual_name.strip().casefold() == requested_clean:
            return actual_name

    raise ValueError(
        f"Worksheet named {requested_name!r} not found.\n"
        f"Available sheets: {available_names}"
    )


def clean_dataframe_columns(df):
    """
    Strip whitespace from column names, remove unnamed Excel filler columns,
    and drop completely empty columns.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False, na=False)].copy()
    df = df.dropna(axis=1, how="all")
    return df


def resolve_column_name(requested_name, available_names):
    """
    Match column name ignoring leading/trailing whitespace and case.
    """
    requested_clean = requested_name.strip().casefold()

    for actual_name in available_names:
        if str(actual_name).strip().casefold() == requested_clean:
            return actual_name

    raise ValueError(
        f"Column {requested_name!r} not found.\n"
        f"Available columns: {list(available_names)}"
    )


def make_geodf(df, lat_name="Latitude", lon_name="Longitude"):
    """
    Build a GeoDataFrame from latitude/longitude columns.
    """
    df = df.copy()

    lat_col = resolve_column_name(lat_name, df.columns)
    lon_col = resolve_column_name(lon_name, df.columns)

    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce").astype("float64")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce").astype("float64")

    before = len(df)
    df = df.dropna(subset=[lat_col, lon_col]).copy()
    dropped = before - len(df)

    df["record_id"] = range(len(df))

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326"
    )

    return gdf, dropped, lat_col, lon_col


def main():
    parser = argparse.ArgumentParser(description="Convert Excel marsh core worksheets to FlatGeobuf.")
    parser.add_argument("excel_file", help="Path to Excel workbook")
    parser.add_argument("--sediment-sheet", default="Data entry-Sediment data",
                        help="Sediment worksheet name")
    parser.add_argument("--biomass-sheet", default="Data Entry-Biomass Data",
                        help="Biomass worksheet name")
    parser.add_argument("--sediment-out", default="sediment_data.fgb",
                        help="Output FlatGeobuf for sediment data")
    parser.add_argument("--biomass-out", default="biomass_data.fgb",
                        help="Output FlatGeobuf for biomass data")
    parser.add_argument("--lat-col", default="Latitude",
                        help="Latitude column name")
    parser.add_argument("--lon-col", default="Longitude",
                        help="Longitude column name")
    args = parser.parse_args()

    excel_path = Path(args.excel_file)

    xl = pd.ExcelFile(excel_path, engine="openpyxl")

    print("Available worksheet names:")
    for i, s in enumerate(xl.sheet_names):
        print(f"  {i}: {repr(s)}")

    sediment_sheet = resolve_sheet_name(args.sediment_sheet, xl.sheet_names)
    biomass_sheet = resolve_sheet_name(args.biomass_sheet, xl.sheet_names)

    print(f"\nUsing sediment sheet: {repr(sediment_sheet)}")
    print(f"Using biomass sheet:  {repr(biomass_sheet)}")

    sediment_df = pd.read_excel(xl, sheet_name=sediment_sheet)
    biomass_df = pd.read_excel(xl, sheet_name=biomass_sheet)

    sediment_df = clean_dataframe_columns(sediment_df)
    biomass_df = clean_dataframe_columns(biomass_df)

    print("\nSediment columns after cleanup:")
    print(list(sediment_df.columns))

    print("\nBiomass columns after cleanup:")
    print(list(biomass_df.columns))

    habitat_col = resolve_column_name("Habitat", sediment_df.columns)

    sediment_df = sediment_df[
        sediment_df[habitat_col].astype("string").str.contains("Salt marsh", case=False, na=False)
    ].copy()

    sediment_gdf, sed_dropped, sed_lat_col, sed_lon_col = make_geodf(
        sediment_df, lat_name=args.lat_col, lon_name=args.lon_col
    )
    biomass_gdf, bio_dropped, bio_lat_col, bio_lon_col = make_geodf(
        biomass_df, lat_name=args.lat_col, lon_name=args.lon_col
    )

    sediment_gdf.to_file(args.sediment_out, driver="FlatGeobuf")
    biomass_gdf.to_file(args.biomass_out, driver="FlatGeobuf")

    print("\nDone.")
    print(f"Sediment output: {args.sediment_out}")
    print(f"Biomass output:  {args.biomass_out}")
    print(f"Sediment rows written: {len(sediment_gdf)}")
    print(f"Biomass rows written:  {len(biomass_gdf)}")
    print(f"Sediment rows dropped for missing coords: {sed_dropped}")
    print(f"Biomass rows dropped for missing coords:  {bio_dropped}")
    print(f"Sediment coordinate columns used: lat={sed_lat_col!r}, lon={sed_lon_col!r}")
    print(f"Biomass coordinate columns used:  lat={bio_lat_col!r}, lon={bio_lon_col!r}")


if __name__ == "__main__":
    main()
