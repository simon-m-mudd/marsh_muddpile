#!/usr/bin/env python3
"""
Search for NOAA CO-OPS and USGS water-level stations near a point and compare
their records side by side.

Subcommands
-----------
search
    Given a lat/lon and radius, find all USGS stations within that radius and
    export them as a FlatGeobuf (.fgb).  Also reports the nearest NOAA CO-OPS
    tide gauge (which is used as the single tidal reference in 'plot').

plot
    Download hourly water-level data for USGS stations (from FGB or explicit
    list) plus the single nearest NOAA CO-OPS tide gauge (determined by
    --lat/--lon).  Produces:

      1. A 2-week plot centred on peak spring tides.
      2. A 2-week plot centred on minimum neap tides.
      3. A table plot (PNG + CSV) of phase lag and amplitude ratio for each
         USGS station relative to the CO-OPS tide gauge.

    Stations whose records cover less than 10 % of the longest record are
    dropped automatically.

    USGS data without an absolute datum are mean-adjusted so their annual
    mean matches the CO-OPS tide gauge.  Adjusted series are labelled
    clearly and a note is printed on each plot.

Examples
--------
# Find USGS stations within 100 km of North Inlet, SC
python3 compare_coastal_water_levels.py search \\
    --lat 33.35 --lon -79.18 --radius-km 100 \\
    --output ni_stations.fgb

# Plot all USGS stations found above
python3 compare_coastal_water_levels.py plot \\
    --stations-fgb ni_stations.fgb \\
    --lat 33.35 --lon -79.18 \\
    --output-dir ./wl_plots

# Explicit USGS station list
python3 compare_coastal_water_levels.py plot \\
    --stations 02110815,02110777 \\
    --lat 33.35 --lon -79.18 \\
    --output-dir ./wl_plots --year 2024
"""
import argparse
import math
import os
import sys
import time
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import pandas as pd
import requests


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

NOAA_DATA_URL  = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
NOAA_MDAPI_URL = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"
USGS_SITE_URL  = "https://waterservices.usgs.gov/nwis/site/"
USGS_IV_URL    = "https://waterservices.usgs.gov/nwis/iv/"

FT_TO_M = 0.3048

# USGS parameter codes that carry an absolute vertical datum
_USGS_DATUM_PARAMS = {"62620", "63160", "72279", "62619"}

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
    "72": "PR", "78": "VI",
}

_MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def haversine_km(lon1, lat1, lon2, lat2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def radius_to_bbox(lat, lon, radius_km):
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * math.cos(math.radians(lat)))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


# ---------------------------------------------------------------------------
# NOAA CO-OPS
# ---------------------------------------------------------------------------

def fetch_coops_stations():
    """Return DataFrame of all NOAA CO-OPS water-level stations."""
    for params in [{"type": "waterlevels"}, {}]:
        r = requests.get(NOAA_MDAPI_URL, params=params, timeout=60)
        r.raise_for_status()
        payload = r.json()
        stations = payload.get("stations") or payload.get("station") or []
        if stations:
            break
    else:
        raise RuntimeError("Could not retrieve NOAA CO-OPS station list.")

    rows = []
    for s in stations:
        lat = s.get("lat", s.get("latitude"))
        lon = s.get("lng", s.get("lon", s.get("longitude")))
        try:
            lat, lon = float(lat), float(lon)
        except (TypeError, ValueError):
            continue
        rows.append({
            "station_id": str(s.get("id", "")).strip(),
            "name":       s.get("name"),
            "state":      s.get("state"),
            "latitude":   lat,
            "longitude":  lon,
            "status":     s.get("status"),
        })
    return pd.DataFrame(rows)


def find_nearest_coops(lat, lon):
    """Return (station_id, name, distance_km) for the nearest CO-OPS gauge."""
    print("Fetching NOAA CO-OPS catalogue to find nearest tide gauge...")
    df = fetch_coops_stations()
    df["dist"] = df.apply(
        lambda r: haversine_km(lon, lat, r["longitude"], r["latitude"]), axis=1
    )
    row = df.loc[df["dist"].idxmin()]
    return str(row["station_id"]), str(row["name"]), float(row["dist"])


def _month_chunks(start, end):
    current = start
    while current <= end:
        if current.month == 12:
            next_month = datetime(current.year + 1, 1, 1)
        else:
            next_month = datetime(current.year, current.month + 1, 1)
        chunk_end = min(end, next_month - timedelta(days=1))
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def fetch_coops_hourly(station_id, start, end,
                       product="hourly_height", datum="MLLW", units="metric"):
    """Download CO-OPS hourly water levels; return tidy DataFrame."""
    all_rows = []
    for chunk_start, chunk_end in _month_chunks(start, end):
        params = {
            "product":     product,
            "application": "coastal_wl_compare",
            "begin_date":  chunk_start.strftime("%Y%m%d"),
            "end_date":    chunk_end.strftime("%Y%m%d"),
            "station":     station_id,
            "time_zone":   "gmt",
            "units":       units,
            "datum":       datum,
            "format":      "json",
        }
        r = requests.get(NOAA_DATA_URL, params=params, timeout=60)
        r.raise_for_status()
        payload = r.json()
        if "error" in payload:
            msg = payload["error"]
            if isinstance(msg, dict):
                msg = msg.get("message", str(msg))
            print(f"  [{station_id}] {chunk_start.date()}–{chunk_end.date()}: {msg}",
                  file=sys.stderr)
            time.sleep(0.2)
            continue
        all_rows.extend(payload.get("data", []))
        time.sleep(0.2)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["time"]        = pd.to_datetime(df["t"], errors="coerce")
    df["water_level"] = pd.to_numeric(df["v"], errors="coerce")
    df = df.dropna(subset=["time", "water_level"]).copy()
    df["station_id"]  = station_id
    df["source"]      = "coops"
    return df[["source", "station_id", "time", "water_level"]]


# ---------------------------------------------------------------------------
# USGS
# ---------------------------------------------------------------------------

def fetch_usgs_stations_bbox(bbox, parameter_cd="00065", site_types="ES,OC,ST"):
    """Return DataFrame of active USGS stations within bbox (xmin,ymin,xmax,ymax)."""
    params = {
        "format":      "rdb",
        "siteStatus":  "active",
        "parameterCd": parameter_cd,
        "siteType":    site_types,
        "bBox":        ",".join(f"{v:.6f}" for v in bbox),
    }
    r = requests.get(USGS_SITE_URL, params=params, timeout=120)
    if not r.ok:
        raise RuntimeError(f"USGS site query failed {r.status_code}: {r.text[:500]}")

    lines = [ln for ln in r.text.splitlines()
             if ln.strip() and not ln.startswith("#")]
    if len(lines) < 3:
        return pd.DataFrame()

    df = pd.read_csv(StringIO("\n".join(lines)), sep="\t", dtype=str).iloc[1:].copy()

    for col in ["site_no", "station_nm", "site_tp_cd",
                "dec_lat_va", "dec_long_va", "state_cd"]:
        if col not in df.columns:
            df[col] = None

    df["latitude"]  = pd.to_numeric(df["dec_lat_va"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["dec_long_va"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"]).copy()

    def fips_to_abbr(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return STATE_FIPS_TO_ABBR.get(str(v).strip().zfill(2))

    out = pd.DataFrame({
        "source":     "usgs",
        "station_id": df["site_no"].astype(str).str.strip(),
        "name":       df["station_nm"],
        "state":      df["state_cd"].apply(fips_to_abbr),
        "latitude":   df["latitude"],
        "longitude":  df["longitude"],
        "status":     "active",
        "site_type":  df["site_tp_cd"],
    })
    return out.drop_duplicates(subset=["station_id"]).reset_index(drop=True)


def fetch_usgs_site_coords(site_no):
    """Return (lat, lon) for a USGS site, or (None, None) on failure."""
    try:
        params = {"format": "rdb", "sites": str(site_no), "siteOutput": "expanded"}
        r = requests.get(USGS_SITE_URL, params=params, timeout=30)
        r.raise_for_status()
        lines = [ln for ln in r.text.splitlines()
                 if ln.strip() and not ln.startswith("#")]
        if len(lines) < 3:
            return None, None
        df = pd.read_csv(StringIO("\n".join(lines)), sep="\t", dtype=str).iloc[1:]
        lat = pd.to_numeric(df["dec_lat_va"].iloc[0], errors="coerce")
        lon = pd.to_numeric(df["dec_long_va"].iloc[0], errors="coerce")
        return (float(lat), float(lon)) if pd.notna(lat) and pd.notna(lon) else (None, None)
    except Exception:
        return None, None


def _usgs_has_datum(parameter_cd, variable_name=""):
    """True if the USGS parameter carries an absolute vertical datum."""
    if str(parameter_cd).strip() in _USGS_DATUM_PARAMS:
        return True
    vn = (variable_name or "").lower()
    return ("above navd" in vn or "above ngvd" in vn
            or "elevation above" in vn or "above datum" in vn)


def fetch_usgs_hourly(site_no, start, end, parameter_cd="00065", to_metres=True):
    """
    Download USGS instantaneous values at native temporal resolution.

    Returns (DataFrame, has_datum).  has_datum is True when the variable
    carries an absolute vertical reference (NAVD 88, NGVD 29, etc.).
    No resampling is applied here; call align_usgs_to_reference() afterwards
    to snap each USGS reading to the nearest CO-OPS timestamp.
    """
    params = {
        "format":      "json",
        "sites":       str(site_no),
        "startDT":     start.strftime("%Y-%m-%d"),
        "endDT":       end.strftime("%Y-%m-%d"),
        "parameterCd": parameter_cd,
        "siteStatus":  "all",
    }
    r = requests.get(USGS_IV_URL, params=params, timeout=120)
    r.raise_for_status()
    payload = r.json()

    ts_list = payload.get("value", {}).get("timeSeries", [])
    rows = []
    variable_name = ""
    has_datum = False

    for ts in ts_list:
        site_codes = ts.get("sourceInfo", {}).get("siteCode", [])
        sid = site_codes[0].get("value") if site_codes else str(site_no)

        var_obj  = ts.get("variable", {})
        vname    = var_obj.get("variableName", "")
        unit_obj = var_obj.get("unit", {})
        unit     = unit_obj.get("unitCode", "") if isinstance(unit_obj, dict) else ""

        if not variable_name and vname:
            variable_name = vname
            has_datum = _usgs_has_datum(parameter_cd, vname)

        for block in ts.get("values", []):
            for item in block.get("value", []):
                v = item.get("value")
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                if v <= -999990:
                    continue
                rows.append({"station_id": sid, "time": item.get("dateTime"),
                             "value": v, "unit": unit})

    if not rows:
        return pd.DataFrame(), has_datum

    raw = pd.DataFrame(rows)
    raw["time"] = pd.to_datetime(raw["time"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["time", "value"]).sort_values("time").copy()

    unit_sample        = str(raw["unit"].iloc[0]) if len(raw) else ""
    raw["water_level"] = raw["value"]
    raw["time"]        = raw["time"].dt.tz_localize(None)

    if to_metres and ("ft" in unit_sample.lower() or unit_sample in {"ft", "Ft", "feet"}):
        raw["water_level"] *= FT_TO_M

    raw["station_id"] = str(site_no)
    raw["source"]     = "usgs"
    return raw[["source", "station_id", "time", "water_level"]], has_datum


def align_usgs_to_reference(usgs_df, ref_df, tolerance_minutes=30):
    """
    For each CO-OPS timestamp in ref_df, select the single USGS observation
    closest in time, within tolerance_minutes.  Preserves actual measured
    values rather than averaging over a time bin.

    Returns a DataFrame with the same timestamps as ref_df (rows where no
    USGS observation falls within the tolerance are dropped).
    """
    ref_times   = ref_df[["time"]].sort_values("time").copy()
    usgs_sorted = usgs_df.sort_values("time").copy()

    aligned = pd.merge_asof(
        ref_times,
        usgs_sorted[["time", "water_level"]],
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=tolerance_minutes),
    )
    aligned = aligned.dropna(subset=["water_level"]).copy()
    aligned["station_id"] = usgs_df["station_id"].iloc[0] if not usgs_df.empty else ""
    aligned["source"]     = "usgs"
    return aligned[["source", "station_id", "time", "water_level"]]


# ---------------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------------

def filter_short_records(station_data, threshold=0.10):
    """
    Drop stations with fewer than threshold * max_records hourly values.
    Returns (filtered_dict, list_of_removed_ids).
    """
    if not station_data:
        return station_data, []
    counts    = {sid: len(df) for sid, df in station_data.items()}
    max_count = max(counts.values())
    cutoff    = max_count * threshold
    removed   = [sid for sid, n in counts.items() if n < cutoff]
    filtered  = {sid: df for sid, df in station_data.items() if sid not in removed}
    return filtered, removed


# ---------------------------------------------------------------------------
# USGS datum adjustment
# ---------------------------------------------------------------------------

def _mean_offset(usgs_df, ref_df):
    """Compute (coops_mean - usgs_mean) over overlapping hourly timestamps."""
    u = usgs_df.set_index("time")["water_level"].rename("usgs")
    c = ref_df.set_index("time")["water_level"].rename("coops")
    merged = pd.concat([u, c], axis=1).dropna()
    if merged.empty:
        return 0.0, 0
    return float(merged["coops"].mean() - merged["usgs"].mean()), len(merged)


def adjust_usgs_to_reference(station_data, sources, station_has_datum, ref_df):
    """
    Shift every USGS station that lacks an absolute datum so its annual mean
    matches the reference CO-OPS tide gauge over the overlap period.

    Returns (adjusted_station_data, set_of_adjusted_ids).
    """
    adjusted     = dict(station_data)
    adjusted_ids = set()

    for sid, df in station_data.items():
        if sources.get(sid) != "usgs" or station_has_datum.get(sid, False):
            continue
        offset, n = _mean_offset(df, ref_df)
        if n == 0:
            print(f"  Warning: no overlap for USGS {sid} with reference — skipping.",
                  file=sys.stderr)
            continue
        corrected = df.copy()
        corrected["water_level"] += offset
        adjusted[sid] = corrected
        adjusted_ids.add(sid)
        print(f"  USGS {sid}: mean-adjusted by {offset:+.3f} m "
              f"({n} overlapping hours)")

    return adjusted, adjusted_ids


# ---------------------------------------------------------------------------
# Tidal timing analysis
# ---------------------------------------------------------------------------

def detect_spring_neap_centers(ref_df, year):
    """
    Using the CO-OPS reference series, find the moment of peak spring tide
    (highest 25-hour tidal range) and minimum neap tide (lowest range) within
    the analysis year.

    Returns (spring_center, neap_center) as pandas Timestamps.
    """
    yr = ref_df[ref_df["time"].dt.year == year].copy()
    if yr.empty:
        raise RuntimeError("Reference CO-OPS data contains no records for the analysis year.")

    hourly = (yr.set_index("time")["water_level"]
                .resample("1h").mean())

    rolling_range = hourly.rolling(25, center=True, min_periods=25).agg(
        lambda x: x.max() - x.min()
    )

    spring_center = rolling_range.idxmax()
    neap_center   = rolling_range.idxmin()
    return spring_center, neap_center


# ---------------------------------------------------------------------------
# Phase / amplitude analysis
# ---------------------------------------------------------------------------

def compute_phase_amplitude(usgs_df, ref_df, max_lag_hours=12):
    """
    Cross-correlate a USGS record against the CO-OPS reference to estimate:
      - Phase lag (hours): positive = USGS tide arrives later than CO-OPS.
      - Amplitude ratio:   USGS tidal amplitude / CO-OPS tidal amplitude.
      - r²:               squared correlation at the best lag.
      - n_hours:          number of overlapping hourly records used.

    Signals are de-meaned before analysis so datum offsets do not affect the
    result.
    """
    u = usgs_df.set_index("time")["water_level"].rename("usgs")
    c = ref_df.set_index("time")["water_level"].rename("coops")
    merged = pd.concat([u, c], axis=1).dropna()

    if len(merged) < 48:
        return np.nan, np.nan, np.nan, len(merged)

    u_vals = (merged["usgs"]  - merged["usgs"].mean()).values
    c_vals = (merged["coops"] - merged["coops"].mean()).values
    n      = len(merged)

    best_r  = -np.inf
    best_lag = 0

    for lag in range(-max_lag_hours, max_lag_hours + 1):
        if lag == 0:
            u_seg, c_seg = u_vals, c_vals
        elif lag > 0:
            # USGS lags CO-OPS: align u[lag:] with c[:-lag]
            u_seg, c_seg = u_vals[lag:], c_vals[:n - lag]
        else:
            u_seg, c_seg = u_vals[:n + lag], c_vals[-lag:]

        if len(u_seg) < 24:
            continue
        r = np.corrcoef(u_seg, c_seg)[0, 1]
        if not np.isnan(r) and r > best_r:
            best_r   = r
            best_lag = lag

    c_std = np.std(c_vals)
    amp_ratio = float(np.std(u_vals) / c_std) if c_std > 0 else np.nan

    return int(best_lag), amp_ratio, float(best_r ** 2) if best_r > -np.inf else np.nan, n


# ---------------------------------------------------------------------------
# Spatial I/O
# ---------------------------------------------------------------------------

def _require_geopandas():
    try:
        import geopandas as gpd
        return gpd
    except ImportError:
        raise SystemExit(
            "geopandas is required for spatial I/O.\n"
            "Install: pip install geopandas pyogrio shapely"
        )


def write_fgb(df, output_path):
    gpd = _require_geopandas()
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    gdf.to_file(output_path, driver="FlatGeobuf")
    return len(gdf)


def read_stations_from_fgb(path):
    """Return (station_ids, names, sources, coords) from a search FGB."""
    gpd = _require_geopandas()
    gdf = gpd.read_file(path)
    if "station_id" not in gdf.columns:
        raise SystemExit(f"{path} has no 'station_id' column — was it made by 'search'?")

    station_ids = gdf["station_id"].astype(str).tolist()
    names, sources, coords = {}, {}, {}
    for _, row in gdf.iterrows():
        sid          = str(row["station_id"])
        names[sid]   = str(row.get("name") or sid)
        sources[sid] = str(row.get("source") or "usgs")
        try:
            coords[sid] = (float(row.geometry.y), float(row.geometry.x))
        except Exception:
            coords[sid] = (None, None)
    return station_ids, names, sources, coords


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _make_tidal_plot(station_data, station_names, sources, adjusted_ids,
                     t_start, t_end, title, filename, output_dir,
                     ylabel="Water level (m)"):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import matplotlib.lines as mlines
    except ImportError:
        raise SystemExit("matplotlib required. Install: pip install matplotlib")

    has_note   = bool(adjusted_ids)
    fig, ax    = plt.subplots(figsize=(14, 5))
    fig.subplots_adjust(bottom=0.17 if has_note else 0.12)

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_idx  = 0
    plotted    = 0

    for sid, df in station_data.items():
        mask   = (df["time"] >= t_start) & (df["time"] <= t_end)
        subset = df.loc[mask]
        if subset.empty:
            continue

        src       = sources.get(sid, "usgs")
        color     = prop_cycle[color_idx % len(prop_cycle)]
        linestyle = "-" if src == "coops" else "--"
        lw        = 1.4 if src == "coops" else 0.9
        label     = station_names.get(sid, sid)

        ax.plot(subset["time"], subset["water_level"],
                linestyle=linestyle, linewidth=lw,
                color=color, label=label, alpha=0.9)
        color_idx += 1
        plotted   += 1

    if plotted == 0:
        print(f"  No data to plot for: {filename}", file=sys.stderr)
        plt.close(fig)
        return

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel(ylabel)

    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate()

    # Legend with source-type keys
    handles, labels = ax.get_legend_handles_labels()
    tide_key = mlines.Line2D([], [], color="gray", linestyle="-",  lw=1.4,
                             label="— Tide gauge (NOAA CO-OPS)")
    usgs_key = mlines.Line2D([], [], color="gray", linestyle="--", lw=0.9,
                             label="-- USGS water level")
    ax.legend(handles=handles + [tide_key, usgs_key],
              labels=labels + ["— Tide gauge (NOAA CO-OPS)", "-- USGS water level"],
              fontsize=7.5, loc="upper right", framealpha=0.75, ncol=2)

    ax.grid(True, linewidth=0.4, alpha=0.5)

    if has_note:
        fig.text(
            0.5, 0.01,
            "USGS water levels adjusted to match mean tide level of nearest "
            "NOAA tide gauge",
            ha="center", va="bottom", fontsize=8, style="italic", color="dimgray",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fffbe6",
                      edgecolor="#ccbb88", alpha=0.9),
        )

    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_phase_amplitude_table(phase_amp_results, ref_name, output_dir, year):
    """
    Write a table PNG and a CSV of phase / amplitude results.

    phase_amp_results is a list of dicts with keys:
        station_id, station_name, phase_lag_h, amplitude_ratio, r_squared,
        n_overlap_h, mean_adjusted
    """
    import csv

    os.makedirs(output_dir, exist_ok=True)

    # --- CSV ---
    csv_path = os.path.join(output_dir, f"phase_amplitude_{year}.csv")
    fieldnames = ["station_id", "station_name", "phase_lag_h",
                  "amplitude_ratio", "r_squared", "n_overlap_h", "mean_adjusted"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(phase_amp_results)
    print(f"Saved: {csv_path}")

    # --- Table PNG ---
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib required. Install: pip install matplotlib")

    col_labels = ["Station ID", "Station Name", "Phase lag\n(hours)",
                  "Amplitude\nratio", "r²", "Overlap\nhours", "Mean\nadjusted"]

    cell_text = []
    for row in phase_amp_results:
        lag  = f"{row['phase_lag_h']:+d}"  if not _isnan(row["phase_lag_h"])  else "—"
        aratio = f"{row['amplitude_ratio']:.3f}" if not _isnan(row["amplitude_ratio"]) else "—"
        r2   = f"{row['r_squared']:.3f}"   if not _isnan(row["r_squared"])    else "—"
        cell_text.append([
            row["station_id"],
            row["station_name"],
            lag,
            aratio,
            r2,
            str(row["n_overlap_h"]),
            "Yes" if row["mean_adjusted"] else "No",
        ])

    n_rows = len(cell_text)
    fig_h  = max(2.0, 0.55 * n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(list(range(len(col_labels))))

    # Style header row
    for col in range(len(col_labels)):
        table[0, col].set_facecolor("#2c5f8a")
        table[0, col].set_text_props(color="white", fontweight="bold")

    # Alternate row shading
    for row_i in range(1, n_rows + 1):
        fc = "#eaf2fb" if row_i % 2 == 0 else "white"
        for col in range(len(col_labels)):
            table[row_i, col].set_facecolor(fc)

    title_text = (f"Phase lag and amplitude ratio relative to\n"
                  f"NOAA CO-OPS tide gauge: {ref_name}  ({year})\n"
                  f"Positive phase lag = USGS tide arrives later than tide gauge")
    ax.set_title(title_text, fontsize=10, pad=12)

    png_path = os.path.join(output_dir, f"phase_amplitude_{year}.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")


def _isnan(v):
    try:
        return math.isnan(v)
    except (TypeError, ValueError):
        return True


# ---------------------------------------------------------------------------
# Subcommand: search
# ---------------------------------------------------------------------------

def cmd_search(args):
    bbox = radius_to_bbox(args.lat, args.lon, args.radius_km)
    print(f"Search bbox: W={bbox[0]:.4f} S={bbox[1]:.4f} "
          f"E={bbox[2]:.4f} N={bbox[3]:.4f}\n")

    print("Fetching USGS stations...")
    try:
        df_usgs = fetch_usgs_stations_bbox(bbox,
                                           parameter_cd=args.usgs_parameter_cd,
                                           site_types=args.usgs_site_types)
        if not df_usgs.empty:
            df_usgs["distance_km"] = df_usgs.apply(
                lambda r: haversine_km(args.lon, args.lat, r["longitude"], r["latitude"]),
                axis=1,
            )
            nearby = df_usgs[df_usgs["distance_km"] <= args.radius_km].copy()
        else:
            nearby = df_usgs
        print(f"  {len(nearby)} USGS station(s) within {args.radius_km} km")
    except Exception as e:
        raise SystemExit(f"USGS fetch failed: {e}")

    ref_sid, ref_name, ref_dist = find_nearest_coops(args.lat, args.lon)
    print(f"\nNearest CO-OPS tide gauge: {ref_sid}  {ref_name}  ({ref_dist:.1f} km)")
    print(f"  (Pass --lat {args.lat} --lon {args.lon} to 'plot' to use this gauge.)\n")

    if nearby.empty:
        print(f"No USGS stations found within {args.radius_km} km.")
        return

    print(f"{'ID':>12}  {'Name':<40}  {'St':>2}  {'Type':<4}  {'Dist km':>8}")
    print(f"{'-'*12}  {'-'*40}  {'--':>2}  {'-'*4}  {'-'*8}")
    for _, row in nearby.sort_values("distance_km").iterrows():
        print(f"{row['station_id']:>12}  {str(row['name']):<40}  "
              f"{str(row.get('state') or ''):>2}  {str(row.get('site_type') or ''):>4}  "
              f"{row['distance_km']:>8.1f}")

    n = write_fgb(nearby, args.output)
    print(f"\nSaved {n} stations to {args.output}")


# ---------------------------------------------------------------------------
# Subcommand: plot
# ---------------------------------------------------------------------------

def cmd_plot(args):
    # --- Resolve USGS station list ---
    if args.stations_fgb:
        station_ids, station_names, sources, station_coords = \
            read_stations_from_fgb(args.stations_fgb)
        # Keep only USGS stations; the CO-OPS reference is added separately
        station_ids = [s for s in station_ids if sources.get(s) == "usgs"]
    else:
        station_ids    = [s.strip() for s in args.stations.split(",") if s.strip()]
        station_names  = {sid: sid for sid in station_ids}
        sources        = {sid: "usgs" for sid in station_ids}
        station_coords = {}

    year  = args.year if args.year else datetime.utcnow().year - 1
    start = datetime(year, 1, 1)
    end   = datetime(year, 12, 31)

    # --- Find and download the single CO-OPS reference gauge ---
    if args.lat is None or args.lon is None:
        # Fall back to centroid of stations if coordinates not provided
        if station_coords:
            lats = [v[0] for v in station_coords.values() if v[0] is not None]
            lons = [v[1] for v in station_coords.values() if v[1] is not None]
            ref_lat = sum(lats) / len(lats) if lats else None
            ref_lon = sum(lons) / len(lons) if lons else None
        else:
            ref_lat = ref_lon = None
        if ref_lat is None:
            raise SystemExit(
                "Cannot determine reference point. "
                "Provide --lat and --lon (same values as used in 'search')."
            )
        print(f"--lat/--lon not supplied; using station centroid "
              f"({ref_lat:.4f}, {ref_lon:.4f}) to find nearest CO-OPS gauge.")
    else:
        ref_lat, ref_lon = args.lat, args.lon

    ref_sid, ref_name, ref_dist = find_nearest_coops(ref_lat, ref_lon)
    print(f"Reference tide gauge: {ref_sid}  {ref_name}  ({ref_dist:.1f} km)\n")

    print(f"Downloading CO-OPS {ref_sid} ({ref_name})...")
    ref_df = fetch_coops_hourly(ref_sid, start, end,
                                product=args.product,
                                datum=args.datum,
                                units=args.units)
    if ref_df.empty:
        raise SystemExit(f"No data returned for reference CO-OPS station {ref_sid}.")
    print(f"  -> {len(ref_df):,} hourly records\n")

    # --- Download USGS stations ---
    to_metres = (args.units == "metric")
    print(f"Downloading USGS data for {year} ({len(station_ids)} station(s))...\n")

    station_data     = {}
    station_has_datum = {}

    for sid in station_ids:
        label = station_names.get(sid, sid)
        print(f"  [USGS] {sid}  {label}")
        try:
            df, has_datum = fetch_usgs_hourly(
                site_no=sid, start=start, end=end,
                parameter_cd=args.usgs_parameter_cd,
                to_metres=to_metres,
            )
        except Exception as e:
            print(f"  -> error: {e}", file=sys.stderr)
            continue

        if df.empty:
            print(f"  -> no data returned", file=sys.stderr)
            continue

        df = align_usgs_to_reference(df, ref_df)
        if df.empty:
            print(f"  -> no USGS observations within 30 min of CO-OPS timestamps",
                  file=sys.stderr)
            continue

        station_data[sid]      = df
        station_has_datum[sid] = has_datum
        datum_note = "datum detected" if has_datum else "no absolute datum"
        print(f"  -> {len(df):,} records matched to CO-OPS timestamps  ({datum_note})")

    if not station_data:
        raise SystemExit("No USGS data retrieved.")

    # --- Filter short records ---
    # Include the CO-OPS reference in the count so it is never removed
    all_for_filter = {ref_sid: ref_df, **station_data}
    all_for_filter, removed = filter_short_records(all_for_filter, threshold=0.10)
    removed = [s for s in removed if s != ref_sid]   # never remove the reference
    if removed:
        print(f"\nDropped (< 10 % of longest record): {', '.join(removed)}")
        for sid in removed:
            station_data.pop(sid, None)
            station_names.pop(sid, None)

    if not station_data:
        raise SystemExit("All USGS stations were removed by the short-record filter.")

    # --- Adjust USGS means to CO-OPS reference ---
    print("\nAdjusting USGS water levels to CO-OPS mean...")
    station_data, adjusted_ids = adjust_usgs_to_reference(
        station_data, sources, station_has_datum, ref_df
    )

    # --- Build combined dataset for plotting (CO-OPS + USGS) ---
    all_station_data   = {ref_sid: ref_df, **station_data}
    all_station_names  = {ref_sid: f"{ref_name} [Tide Gauge]"}
    all_sources        = {ref_sid: "coops"}

    for sid in station_data:
        label = station_names.get(sid, sid)
        if sid in adjusted_ids:
            all_station_names[sid] = f"{label} [USGS, mean adj.]"
        elif station_has_datum.get(sid):
            all_station_names[sid] = f"{label} [USGS, datum]"
        else:
            all_station_names[sid] = f"{label} [USGS]"
        all_sources[sid] = "usgs"

    # --- Detect spring / neap centres from CO-OPS reference ---
    print("\nDetecting spring / neap tide centres from CO-OPS reference...")
    spring_center, neap_center = detect_spring_neap_centers(ref_df, year)
    print(f"  Spring centre: {spring_center.date()}")
    print(f"  Neap centre:   {neap_center.date()}\n")

    spring_start = spring_center - timedelta(days=7)
    spring_end   = spring_center + timedelta(days=7)
    neap_start   = neap_center   - timedelta(days=7)
    neap_end     = neap_center   + timedelta(days=7)

    ylabel = "Water level (m)" if args.units == "metric" else "Water level (ft)"
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Spring plot ---
    spring_label = _MONTH_ABBR[spring_center.month - 1]
    _make_tidal_plot(
        station_data=all_station_data,
        station_names=all_station_names,
        sources=all_sources,
        adjusted_ids=adjusted_ids,
        t_start=spring_start,
        t_end=spring_end,
        title=(f"Spring tides — 2 weeks centred on {spring_center.strftime('%d %b %Y')}"),
        filename=f"water_levels_{year}_{spring_label}_spring.png",
        output_dir=args.output_dir,
        ylabel=ylabel,
    )

    # --- Neap plot ---
    neap_label = _MONTH_ABBR[neap_center.month - 1]
    _make_tidal_plot(
        station_data=all_station_data,
        station_names=all_station_names,
        sources=all_sources,
        adjusted_ids=adjusted_ids,
        t_start=neap_start,
        t_end=neap_end,
        title=(f"Neap tides — 2 weeks centred on {neap_center.strftime('%d %b %Y')}"),
        filename=f"water_levels_{year}_{neap_label}_neap.png",
        output_dir=args.output_dir,
        ylabel=ylabel,
    )

    # --- Phase / amplitude analysis ---
    print("\nComputing phase lag and amplitude ratio relative to reference...")
    phase_amp_results = []
    for sid, df in station_data.items():
        lag, amp, r2, n = compute_phase_amplitude(df, ref_df)
        label = station_names.get(sid, sid)
        print(f"  {sid:>12}  lag={lag:+d} h  amp={amp:.3f}  r²={r2:.3f}  n={n}")
        phase_amp_results.append({
            "station_id":      sid,
            "station_name":    label,
            "phase_lag_h":     lag,
            "amplitude_ratio": amp,
            "r_squared":       r2,
            "n_overlap_h":     n,
            "mean_adjusted":   sid in adjusted_ids,
        })

    plot_phase_amplitude_table(phase_amp_results, ref_name, args.output_dir, year)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Search for USGS water-level stations near a point and compare them "
            "against the nearest NOAA CO-OPS tide gauge."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- search ----
    ps = sub.add_parser("search",
                        help="Find nearby USGS stations and report nearest CO-OPS gauge")
    ps.add_argument("--lat", required=True, type=float,
                    help="Centre latitude (decimal degrees, WGS84)")
    ps.add_argument("--lon", required=True, type=float,
                    help="Centre longitude (decimal degrees, WGS84)")
    ps.add_argument("--radius-km", required=True, type=float,
                    help="Search radius in kilometres")
    ps.add_argument("--output", default="nearby_stations.fgb",
                    help="Output FlatGeobuf path (default: nearby_stations.fgb)")
    ps.add_argument("--usgs-parameter-cd", default="00065",
                    help="USGS parameter code (default: 00065 = gage height)")
    ps.add_argument("--usgs-site-types", default="ES,OC,ST",
                    help="USGS site types to include (default: ES,OC,ST)")
    ps.set_defaults(func=cmd_search)

    # ---- plot ----
    pp = sub.add_parser("plot",
                        help="Download data and produce tidal comparison plots + analysis table")
    src_grp = pp.add_mutually_exclusive_group(required=True)
    src_grp.add_argument("--stations",
                         help="Comma-separated USGS site numbers")
    src_grp.add_argument("--stations-fgb",
                         help="FlatGeobuf from 'search'")
    pp.add_argument("--lat", type=float, default=None,
                    help="Latitude used in 'search' (sets the CO-OPS reference gauge)")
    pp.add_argument("--lon", type=float, default=None,
                    help="Longitude used in 'search' (sets the CO-OPS reference gauge)")
    pp.add_argument("--output-dir", default=".",
                    help="Directory for output files (default: current directory)")
    pp.add_argument("--year", type=int, default=None,
                    help="Calendar year to analyse (default: most recent complete year)")
    pp.add_argument("--product", default="hourly_height",
                    help="NOAA CO-OPS product (default: hourly_height)")
    pp.add_argument("--datum", default="MLLW",
                    help="NOAA CO-OPS datum (default: MLLW)")
    pp.add_argument("--units", default="metric", choices=["metric", "english"],
                    help="Units (default: metric; USGS ft converted to m)")
    pp.add_argument("--usgs-parameter-cd", default="00065",
                    help="USGS parameter code (default: 00065 = gage height)")
    pp.set_defaults(func=cmd_plot)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
