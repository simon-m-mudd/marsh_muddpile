# Downloads some tides.
# examples:
# Get the san francisco tide gague
# python3 download_noaa_tides.py timeseries \
#  --station 9414290 \
#  --start 2024-01-01 \
#  --end 2024-03-31 \
#  --product water_level \
#  --datum MLLW \
#  --output sf_tides.csv
#
# Get locations of all gagues
# python3 download_noaa_tides.py stations \
#   --output noaa_tide_gauges.gpkg



#!/usr/bin/env python3
import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pandas as pd
import requests

DATA_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
MDAPI_URL = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"


def parse_date(s):
    return datetime.strptime(s, "%Y-%m-%d")


def _noaa_chunks(start, end, max_days=365):
    """Yield (chunk_start, chunk_end) pairs respecting NOAA API date-range limits.

    Hourly products (interval='h', product='hourly_height') allow 365 days per
    request; 6-minute products are capped at 31 days.
    """
    current = start
    while current <= end:
        chunk_end = min(end, current + timedelta(days=max_days - 1))
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def fetch_chunk(station, begin_date, end_date, product, datum, units, time_zone, interval):
    params = {
        "product": product,
        "application": "ELM_NOAA_tide_download",
        "begin_date": begin_date.strftime("%Y%m%d"),
        "end_date": end_date.strftime("%Y%m%d"),
        "station": station,
        "time_zone": time_zone,
        "units": units,
        "format": "json",
    }

    if datum:
        params["datum"] = datum
    if interval:
        params["interval"] = interval

    r = requests.get(DATA_URL, params=params, timeout=60)
    r.raise_for_status()
    payload = r.json()

    if "error" in payload:
        msg = payload["error"]
        if isinstance(msg, dict):
            msg = msg.get("message", str(msg))
        raise RuntimeError(
            f"NOAA API error for {begin_date.date()} to {end_date.date()}: {msg}"
        )

    return payload.get("data", [])


def write_csv(rows, output_file):
    if not rows:
        print("No data returned.")
        return

    fieldnames = list(rows[0].keys())
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fetch_all_noaa_stations(station_type="waterlevels"):
    """
    Download NOAA CO-OPS station metadata and return a DataFrame.

    station_type is passed to the metadata API, typically 'waterlevels'.
    If that returns nothing, the code falls back to the unfiltered station list.
    """
    candidate_params = []
    if station_type:
        candidate_params.append({"type": station_type})
    candidate_params.append({})

    stations = None
    last_payload = None

    for params in candidate_params:
        r = requests.get(MDAPI_URL, params=params, timeout=60)
        r.raise_for_status()
        payload = r.json()
        last_payload = payload

        stations = payload.get("stations") or payload.get("station") or []
        if stations:
            break

    if not stations:
        raise RuntimeError(f"Could not read station list from NOAA metadata API: {last_payload}")

    records = []
    for s in stations:
        lat = s.get("lat", s.get("latitude"))
        lon = s.get("lng", s.get("lon", s.get("longitude")))

        try:
            lat = float(lat)
            lon = float(lon)
        except (TypeError, ValueError):
            continue

        records.append({
            "station_id": s.get("id"),
            "name": s.get("name"),
            "state": s.get("state"),
            "latitude": lat,
            "longitude": lon,
            "timezone": s.get("timezone"),
            "tidal": s.get("tidal"),
            "greatlakes": s.get("greatlakes"),
            "shefcode": s.get("shefcode"),
            "deployment": s.get("deployment"),
            "status": s.get("status"),
        })

    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError("No station records with valid coordinates were found.")

    return df


def write_stations_gpkg(df, output_gpkg, layer="noaa_tide_gauges"):
    try:
        import geopandas as gpd
    except ImportError:
        raise SystemExit(
            "geopandas is required to write GeoPackage output.\n"
            "Install it with: pip install geopandas pyogrio shapely"
        )

    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )

    gdf.to_file(output_gpkg, layer=layer, driver="GPKG")
    return len(gdf)


def run_timeseries(args):
    start = parse_date(args.start)
    end = parse_date(args.end)

    if end < start:
        print("Error: end date must be on or after start date.", file=sys.stderr)
        sys.exit(1)

    is_hourly = args.interval == "h" or args.product == "hourly_height"
    max_days = 365 if is_hourly else 31
    chunks = list(_noaa_chunks(start, end, max_days=max_days))
    print(f"Downloading {len(chunks)} chunk(s) in parallel ...")

    def _fetch(chunk_start, chunk_end):
        return chunk_start, fetch_chunk(
            station=args.station,
            begin_date=chunk_start,
            end_date=chunk_end,
            product=args.product,
            datum=args.datum,
            units=args.units,
            time_zone=args.time_zone,
            interval=args.interval,
        )

    all_rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(_fetch, s, e) for s, e in chunks]
        for fut in futures:
            try:
                chunk_start, rows = fut.result()
                all_rows.extend(rows)
            except Exception as e:
                print(f"Failed for chunk: {e}", file=sys.stderr)

    write_csv(all_rows, args.output)
    print(f"Saved {len(all_rows)} records to {args.output}")

    if args.stations_gpkg:
        print("Downloading NOAA station metadata...")
        df = fetch_all_noaa_stations(station_type=args.station_type)
        n = write_stations_gpkg(df, args.stations_gpkg, layer=args.stations_layer)
        print(f"Saved {n} station locations to {args.stations_gpkg} (layer={args.stations_layer})")


def run_stations(args):
    print("Downloading NOAA station metadata...")
    df = fetch_all_noaa_stations(station_type=args.station_type)
    n = write_stations_gpkg(df, args.output, layer=args.layer)
    print(f"Saved {n} station locations to {args.output} (layer={args.layer})")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Download NOAA CO-OPS tide gauge records and/or station locations."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Time series command
    ts = subparsers.add_parser("timeseries", help="Download NOAA time series data to CSV")
    ts.add_argument("--station", required=True, help="NOAA station ID, e.g. 9414290")
    ts.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    ts.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    ts.add_argument(
        "--product",
        default="water_level",
        help="NOAA product, e.g. water_level, hourly_height, predictions",
    )
    ts.add_argument(
        "--datum",
        default="MLLW",
        help="Datum, e.g. MLLW, NAVD, MSL. Often needed for water level products.",
    )
    ts.add_argument(
        "--units",
        default="metric",
        choices=["metric", "english"],
        help="Units",
    )
    ts.add_argument(
        "--time-zone",
        default="gmt",
        choices=["gmt", "lst", "lst_ldt"],
        help="Time zone",
    )
    ts.add_argument(
        "--interval",
        default=None,
        help="Optional interval, where supported by the selected product",
    )
    ts.add_argument(
        "--output",
        default="noaa_tide_data.csv",
        help="Output CSV filename",
    )
    ts.add_argument(
        "--stations-gpkg",
        default=None,
        help="Optional GeoPackage path to also save NOAA station locations",
    )
    ts.add_argument(
        "--stations-layer",
        default="noaa_tide_gauges",
        help="Layer name for the GeoPackage written by --stations-gpkg",
    )
    ts.add_argument(
        "--station-type",
        default="waterlevels",
        help="NOAA metadata station filter, usually 'waterlevels'",
    )
    ts.set_defaults(func=run_timeseries)

    # Stations command
    st = subparsers.add_parser("stations", help="Download NOAA station locations to GeoPackage")
    st.add_argument(
        "--output",
        default="noaa_tide_gauges.gpkg",
        help="Output GeoPackage path",
    )
    st.add_argument(
        "--layer",
        default="noaa_tide_gauges",
        help="GeoPackage layer name",
    )
    st.add_argument(
        "--station-type",
        default="waterlevels",
        help="NOAA metadata station filter, usually 'waterlevels'",
    )
    st.set_defaults(func=run_stations)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
