#!/usr/bin/env python3
import argparse
import math
import os
import sys
import time
from io import StringIO
from datetime import datetime, timedelta

import pandas as pd
import requests

NOAA_DATA_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
NOAA_MDAPI_URL = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"

USGS_SITE_URL = "https://waterservices.usgs.gov/nwis/site/"
USGS_IV_URL = "https://waterservices.usgs.gov/nwis/iv/"


STATE_FIPS_TO_ABBR = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
    "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY", "60": "AS", "66": "GU", "69": "MP",
    "72": "PR", "78": "VI"
}


def parse_date(s):
    return datetime.strptime(s, "%Y-%m-%d")


def month_chunks(start, end):
    current = start
    while current <= end:
        if current.month == 12:
            next_month = datetime(current.year + 1, 1, 1)
        else:
            next_month = datetime(current.year, current.month + 1, 1)
        chunk_end = min(end, next_month - timedelta(days=1))
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def safe_bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"true", "t", "yes", "y", "1"}:
        return True
    if s in {"false", "f", "no", "n", "0"}:
        return False
    return None


def haversine_km(lon1, lat1, lon2, lat2):
    r = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def write_output(df, output_path, layer="stations"):
    if output_path.lower().endswith(".gpkg"):
        try:
            import geopandas as gpd
        except ImportError:
            raise SystemExit(
                "GeoPackage output requires geopandas.\n"
                "Install with: pip install geopandas pyogrio shapely"
            )

        if "longitude" not in df.columns or "latitude" not in df.columns:
            raise SystemExit("GeoPackage output requires longitude and latitude columns.")

        gdf = gpd.GeoDataFrame(
            df.copy(),
            geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
            crs="EPSG:4326"
        )
        gdf.to_file(output_path, layer=layer, driver="GPKG")
    else:
        df.to_csv(output_path, index=False)


# -------------------------
# NOAA CO-OPS
# -------------------------
def fetch_coops_stations(station_type="waterlevels"):
    candidates = []
    if station_type:
        candidates.append({"type": station_type})
    candidates.append({})

    stations = []
    last_payload = None

    for params in candidates:
        r = requests.get(NOAA_MDAPI_URL, params=params, timeout=60)
        r.raise_for_status()
        payload = r.json()
        last_payload = payload
        stations = payload.get("stations") or payload.get("station") or []
        if stations:
            break

    if not stations:
        raise RuntimeError(f"Could not fetch NOAA station list: {last_payload}")

    rows = []
    for s in stations:
        lat = s.get("lat", s.get("latitude"))
        lon = s.get("lng", s.get("lon", s.get("longitude")))

        try:
            lat = float(lat)
            lon = float(lon)
        except (TypeError, ValueError):
            continue

        rows.append({
            "source": "coops",
            "station_id": str(s.get("id", "")).strip(),
            "name": s.get("name"),
            "state": s.get("state"),
            "latitude": lat,
            "longitude": lon,
            "status": s.get("status"),
            "timezone": s.get("timezone"),
            "tidal": safe_bool(s.get("tidal")),
            "greatlakes": safe_bool(s.get("greatlakes")),
            "site_type": "coastal_water_level",
            "agency": "NOAA CO-OPS",
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No NOAA station records found.")
    return df


def fetch_coops_timeseries(station, start, end, product="water_level",
                           datum="MLLW", units="metric", time_zone="gmt", interval=None):
    all_rows = []

    for chunk_start, chunk_end in month_chunks(start, end):
        params = {
            "product": product,
            "application": "ELM_multi_water_levels",
            "begin_date": chunk_start.strftime("%Y%m%d"),
            "end_date": chunk_end.strftime("%Y%m%d"),
            "station": station,
            "time_zone": time_zone,
            "units": units,
            "format": "json",
        }
        if datum:
            params["datum"] = datum
        if interval:
            params["interval"] = interval

        r = requests.get(NOAA_DATA_URL, params=params, timeout=60)
        r.raise_for_status()
        payload = r.json()

        if "error" in payload:
            msg = payload["error"]
            if isinstance(msg, dict):
                msg = msg.get("message", str(msg))
            raise RuntimeError(f"NOAA error for {chunk_start.date()} to {chunk_end.date()}: {msg}")

        rows = payload.get("data", [])
        for row in rows:
            row["source"] = "coops"
            row["station_id"] = str(station)

        all_rows.extend(rows)
        time.sleep(0.2)

    return pd.DataFrame(all_rows)


# -------------------------
# USGS
# -------------------------
def fetch_usgs_stations(state=None, bbox=None, parameter_cd="00065", site_type="ST"):
    """
    parameter_cd=00065 => gage height / stage
    site_type=ST => stream
    """
    params = {
        "format": "rdb",
        "siteStatus": "active",
        "parameterCd": parameter_cd,
        "siteType": site_type,
    }

    if state:
        params["stateCd"] = state.upper()
    if bbox:
        params["bBox"] = ",".join(str(x) for x in bbox)

    r = requests.get(USGS_SITE_URL, params=params, timeout=120)

    if not r.ok:
        raise RuntimeError(
            f"USGS site query failed: {r.status_code}\n"
            f"URL: {r.url}\n"
            f"Response: {r.text[:1000]}"
        )

    text = r.text


    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    if len(lines) < 3:
        raise RuntimeError("USGS returned no site records.")

    df = pd.read_csv(StringIO("\n".join(lines)), sep="\t", dtype=str)
    df = df.iloc[1:].copy()  # drop datatype-description row

    needed = ["agency_cd", "site_no", "station_nm", "site_tp_cd", "dec_lat_va", "dec_long_va", "state_cd", "huc_cd"]
    for col in needed:
        if col not in df.columns:
            df[col] = None

    df["latitude"] = pd.to_numeric(df["dec_lat_va"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["dec_long_va"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"]).copy()

    def state_to_abbr(v):
        if v is None or pd.isna(v):
            return state.upper() if state else None
        s = str(v).strip().zfill(2)
        return STATE_FIPS_TO_ABBR.get(s, state.upper() if state else s)

    out = pd.DataFrame({
        "source": "usgs",
        "station_id": df["site_no"].astype(str).str.strip(),
        "name": df["station_nm"],
        "state": df["state_cd"].apply(state_to_abbr),
        "latitude": df["latitude"],
        "longitude": df["longitude"],
        "status": "active",
        "timezone": None,
        "tidal": None,
        "greatlakes": None,
        "site_type": df["site_tp_cd"],
        "agency": df["agency_cd"],
        "huc": df["huc_cd"],
        "parameter_cd": parameter_cd,
    })

    out = out.drop_duplicates(subset=["station_id"]).reset_index(drop=True)

    if out.empty:
        raise RuntimeError("No valid USGS station records found.")
    return out


def fetch_usgs_timeseries(site_no, start, end, parameter_cd="00065"):
    params = {
        "format": "json",
        "sites": str(site_no),
        "startDT": start.strftime("%Y-%m-%d"),
        "endDT": end.strftime("%Y-%m-%d"),
        "parameterCd": parameter_cd,
        "siteStatus": "all",
    }

    r = requests.get(USGS_IV_URL, params=params, timeout=120)
    r.raise_for_status()
    payload = r.json()

    ts_list = payload.get("value", {}).get("timeSeries", [])
    rows = []

    for ts in ts_list:
        source_info = ts.get("sourceInfo", {})
        variable = ts.get("variable", {})
        values_blocks = ts.get("values", [])

        station_id = None
        site_codes = source_info.get("siteCode", [])
        if site_codes:
            station_id = site_codes[0].get("value")

        variable_name = variable.get("variableName")
        variable_code = None
        variable_codes = variable.get("variableCode", [])
        if variable_codes:
            variable_code = variable_codes[0].get("value")

        unit = None
        unit_obj = variable.get("unit", {})
        if isinstance(unit_obj, dict):
            unit = unit_obj.get("unitCode")

        for block in values_blocks:
            for item in block.get("value", []):
                qualifiers = item.get("qualifiers", [])
                if isinstance(qualifiers, list):
                    qualifiers = ",".join(str(q) for q in qualifiers)

                rows.append({
                    "source": "usgs",
                    "station_id": station_id or str(site_no),
                    "time": item.get("dateTime"),
                    "value": item.get("value"),
                    "qualifiers": qualifiers,
                    "variable_name": variable_name,
                    "parameter_cd": variable_code or parameter_cd,
                    "unit": unit,
                })

    return pd.DataFrame(rows)


# -------------------------
# Common helpers
# -------------------------
def filter_stations(df, state=None, bbox=None, coastal_only=False):
    out = df.copy()

    if state:
        out = out[out["state"].fillna("").str.upper() == state.upper()]

    if bbox:
        xmin, ymin, xmax, ymax = bbox
        out = out[
            (out["longitude"] >= xmin) &
            (out["longitude"] <= xmax) &
            (out["latitude"] >= ymin) &
            (out["latitude"] <= ymax)
        ]

    if coastal_only:
        if "source" in out.columns:
            coops = out["source"] == "coops"
            other = ~coops

            # only apply coastal logic to CO-OPS rows
            out_coops = out[coops].copy()
            if "tidal" in out_coops.columns:
                out_coops = out_coops[(out_coops["tidal"].isna()) | (out_coops["tidal"] == True)]
            if "greatlakes" in out_coops.columns:
                out_coops = out_coops[(out_coops["greatlakes"].isna()) | (out_coops["greatlakes"] == False)]

            out_other = out[other].copy()
            out = pd.concat([out_coops, out_other], ignore_index=True)

    return out.reset_index(drop=True)


def fetch_station_catalog(source, state=None, bbox=None, coastal_only=False,
                          noaa_station_type="waterlevels", usgs_parameter_cd="00065", usgs_site_type="ST"):
    frames = []

    if source in {"coops", "all"}:
        dfc = fetch_coops_stations(station_type=noaa_station_type)
        frames.append(dfc)

    if source in {"usgs", "all"}:
        # USGS station queries should have some geographic filter
        if state is None and bbox is None:
            raise SystemExit(
                "USGS station queries require --state or --bbox.\n"
                "Example: --source usgs --state SC"
            )

        dfu = fetch_usgs_stations(
            state=state,
            bbox=bbox,
            parameter_cd=usgs_parameter_cd,
            site_type=usgs_site_type,
        )
        frames.append(dfu)

    if not frames:
        raise SystemExit("Invalid source. Choose from coops, usgs, all.")

    df = pd.concat(frames, ignore_index=True, sort=False)
    df = filter_stations(df, state=state, bbox=bbox, coastal_only=coastal_only)
    return df



def nearest_stations(df, lat, lon, n=5):
    out = df.copy()
    out["distance_km"] = out.apply(
        lambda r: haversine_km(lon, lat, r["longitude"], r["latitude"]),
        axis=1
    )
    return out.sort_values("distance_km").head(n).reset_index(drop=True)


# -------------------------
# Commands
# -------------------------
def cmd_stations(args):
    bbox = tuple(args.bbox) if args.bbox else None
    df = fetch_station_catalog(
        source=args.source,
        state=args.state,
        bbox=bbox,
        coastal_only=args.coastal_only,
        noaa_station_type=args.noaa_station_type,
        usgs_parameter_cd=args.usgs_parameter_cd,
        usgs_site_type=args.usgs_site_type,
    )
    write_output(df, args.output, layer=args.layer)
    print(f"Saved {len(df)} stations to {args.output}")


def cmd_nearest(args):
    bbox = tuple(args.bbox) if args.bbox else None
    df = fetch_station_catalog(
        source=args.source,
        state=args.state,
        bbox=bbox,
        coastal_only=args.coastal_only,
        noaa_station_type=args.noaa_station_type,
        usgs_parameter_cd=args.usgs_parameter_cd,
        usgs_site_type=args.usgs_site_type,
    )
    near = nearest_stations(df, lat=args.lat, lon=args.lon, n=args.n)
    cols = ["source", "station_id", "name", "state", "latitude", "longitude", "distance_km"]
    print(near[cols].to_string(index=False))


def cmd_timeseries(args):
    start = parse_date(args.start)
    end = parse_date(args.end)

    if end < start:
        raise SystemExit("End date must be on or after start date.")

    if args.source == "coops":
        df = fetch_coops_timeseries(
            station=args.station,
            start=start,
            end=end,
            product=args.product,
            datum=args.datum,
            units=args.units,
            time_zone=args.time_zone,
            interval=args.interval,
        )
    elif args.source == "usgs":
        df = fetch_usgs_timeseries(
            site_no=args.station,
            start=start,
            end=end,
            parameter_cd=args.usgs_parameter_cd,
        )
    else:
        raise SystemExit("timeseries supports --source coops or --source usgs")

    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Download NOAA CO-OPS and USGS water-level station metadata and time series."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # stations
    p1 = sub.add_parser("stations", help="Download station metadata to GPKG or CSV")
    p1.add_argument("--source", required=True, choices=["coops", "usgs", "all"])
    p1.add_argument("--output", required=True, help="Output .gpkg or .csv")
    p1.add_argument("--layer", default="water_level_stations", help="GeoPackage layer name")
    p1.add_argument("--state", default=None, help="Optional state filter, e.g. SC")
    p1.add_argument("--bbox", nargs=4, type=float, default=None, metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    p1.add_argument("--coastal-only", action="store_true", help="Apply coastal filter to CO-OPS rows")
    p1.add_argument("--noaa-station-type", default="waterlevels", help="NOAA metadata type filter")
    p1.add_argument("--usgs-parameter-cd", default="00065", help="USGS parameter code, default 00065 = gage height")
    p1.add_argument("--usgs-site-type", default="ST", help="USGS site type, default ST = stream")
    p1.set_defaults(func=cmd_stations)

    # nearest
    p2 = sub.add_parser("nearest", help="Find nearest stations to a point")
    p2.add_argument("--source", required=True, choices=["coops", "usgs", "all"])
    p2.add_argument("--lat", required=True, type=float)
    p2.add_argument("--lon", required=True, type=float)
    p2.add_argument("--n", type=int, default=5)
    p2.add_argument("--state", default=None, help="Optional state filter")
    p2.add_argument("--bbox", nargs=4, type=float, default=None, metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    p2.add_argument("--coastal-only", action="store_true", help="Apply coastal filter to CO-OPS rows")
    p2.add_argument("--noaa-station-type", default="waterlevels", help="NOAA metadata type filter")
    p2.add_argument("--usgs-parameter-cd", default="00065", help="USGS parameter code")
    p2.add_argument("--usgs-site-type", default="ST", help="USGS site type")
    p2.set_defaults(func=cmd_nearest)

    # timeseries
    p3 = sub.add_parser("timeseries", help="Download time-series data for one station")
    p3.add_argument("--source", required=True, choices=["coops", "usgs"])
    p3.add_argument("--station", required=True, help="Station ID: NOAA station ID or USGS site number")
    p3.add_argument("--start", required=True, help="YYYY-MM-DD")
    p3.add_argument("--end", required=True, help="YYYY-MM-DD")
    p3.add_argument("--output", required=True, help="Output CSV")
    p3.add_argument("--product", default="water_level", help="NOAA product for CO-OPS")
    p3.add_argument("--datum", default="MLLW", help="NOAA datum for CO-OPS")
    p3.add_argument("--units", default="metric", choices=["metric", "english"])
    p3.add_argument("--time-zone", default="gmt", choices=["gmt", "lst", "lst_ldt"])
    p3.add_argument("--interval", default=None, help="Optional NOAA interval")
    p3.add_argument("--usgs-parameter-cd", default="00065", help="USGS parameter code, default 00065 = gage height")
    p3.set_defaults(func=cmd_timeseries)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
