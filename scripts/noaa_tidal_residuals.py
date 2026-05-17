"""Extract storm surge / non-tidal residuals from a NOAA CO-OPS tide gauge.

The non-tidal residual is:
    residual = observed water level - astronomical tide prediction

Positive residuals indicate storm surge (or other meteorological setup);
negative residuals indicate setdown.

Usage examples
--------------
# North Inlet proxy gauge (Springmaid Pier, SC), use cached observations:
python3 scripts/noaa_tidal_residuals.py \
    --station 8661070 \
    --start 1985 \
    --end 2024 \
    --obs-csv era5_data/noaa_8661070_hourly.csv \
    --out-csv era5_data/noaa_8661070_residuals.csv \
    --surge-threshold 0.3 \
    --plot-dir scripts/figures

# Any other station, downloading observations from scratch:
python3 scripts/noaa_tidal_residuals.py \
    --station 9414290 \
    --start 2010 \
    --end 2023 \
    --out-csv era5_data/noaa_9414290_residuals.csv \
    --surge-threshold 0.3 \
    --plot-dir scripts/figures
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# NOAA CO-OPS helpers
# ---------------------------------------------------------------------------

_DATA_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"


def _noaa_chunks(start: datetime, end: datetime, max_days: int = 365):
    """Yield (chunk_start, chunk_end) pairs. Hourly: 365 days max; 6-min: 31 days max."""
    current = start
    while current <= end:
        chunk_end = min(end, current + timedelta(days=max_days - 1))
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def _fetch_chunk(
    station: str,
    begin: datetime,
    end: datetime,
    product: str,
    datum: str,
    interval: str | None,
) -> list[dict]:
    params = {
        "product": product,
        "application": "marsh_muddpile_surge",
        "begin_date": begin.strftime("%Y%m%d"),
        "end_date": end.strftime("%Y%m%d"),
        "station": station,
        "time_zone": "gmt",
        "units": "metric",
        "format": "json",
        "datum": datum,
    }
    if interval:
        params["interval"] = interval

    r = requests.get(_DATA_URL, params=params, timeout=60)
    r.raise_for_status()
    payload = r.json()

    if "error" in payload:
        msg = payload["error"]
        if isinstance(msg, dict):
            msg = msg.get("message", str(msg))
        raise RuntimeError(f"NOAA API error {begin.date()}–{end.date()}: {msg}")

    return payload.get("data") or payload.get("predictions", [])


def _download_series(
    station: str,
    start_year: int,
    end_year: int,
    product: str,
    datum: str,
    cache_csv: Path | None,
    interval: str | None = "h",
) -> pd.DataFrame:
    """Download a NOAA CO-OPS product in parallel year-sized chunks, caching to CSV."""
    if cache_csv and cache_csv.exists():
        print(f"Loading cached {product} from {cache_csv}")
        df = pd.read_csv(cache_csv, parse_dates=["datetime_gmt"])
        df = df.set_index("datetime_gmt").sort_index()
        return df

    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    max_days = 365 if interval == "h" else 31
    chunks = list(_noaa_chunks(start, end, max_days=max_days))
    print(f"  {product}: {len(chunks)} chunk(s) for {start_year}–{end_year}", flush=True)

    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(_fetch_chunk, station, s, e, product, datum, interval)
                   for s, e in chunks]
        for fut in futures:
            try:
                rows.extend(fut.result())
            except RuntimeError as exc:
                print(f"\n  skipped chunk: {exc}")

    if not rows:
        raise RuntimeError(f"No data returned for {product} at station {station}")

    # NOAA returns {"t": "YYYY-MM-DD HH:MM", "v": "1.234", ...}
    df = pd.DataFrame(rows)
    df = df.rename(columns={"t": "datetime_gmt", "v": "value"})
    df["datetime_gmt"] = pd.to_datetime(df["datetime_gmt"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["datetime_gmt", "value"]].dropna().set_index("datetime_gmt").sort_index()

    if cache_csv:
        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        df.reset_index().to_csv(cache_csv, index=False)
        print(f"  Cached {len(df):,} records to {cache_csv}")

    return df


# ---------------------------------------------------------------------------
# Residual computation
# ---------------------------------------------------------------------------

def compute_residuals(obs: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """Merge observed and predicted on timestamp; return residual = obs - pred."""
    obs = obs.rename(columns={"value": "obs_m", "wl_m_mllw": "obs_m"})
    pred = pred.rename(columns={"value": "pred_m"})

    # normalise index name so merge works regardless of source
    obs.index.name = "datetime_gmt"
    pred.index.name = "datetime_gmt"

    merged = obs.join(pred, how="inner")
    merged.columns = pd.Index(["obs_m", "pred_m"])
    merged["residual_m"] = merged["obs_m"] - merged["pred_m"]
    return merged


# ---------------------------------------------------------------------------
# Storm surge event detection
# ---------------------------------------------------------------------------

def identify_surge_events(
    residuals: pd.DataFrame,
    threshold_m: float,
    min_gap_hours: int = 24,
) -> pd.DataFrame:
    """
    Return a table of distinct surge events where the residual exceeds
    threshold_m.  Events separated by less than min_gap_hours are merged.
    """
    above = residuals["residual_m"] >= threshold_m

    events = []
    in_event = False
    event_start = None
    last_above = None

    for ts, flag in above.items():
        if flag:
            if not in_event:
                event_start = ts
                in_event = True
            last_above = ts
        else:
            if in_event:
                gap = (ts - last_above).total_seconds() / 3600
                if gap >= min_gap_hours:
                    events.append((event_start, last_above))
                    in_event = False

    if in_event:
        events.append((event_start, last_above))

    records = []
    for start, end in events:
        window = residuals.loc[start:end, "residual_m"]
        records.append(
            {
                "start": start,
                "end": end,
                "duration_hours": (end - start).total_seconds() / 3600 + 1,
                "peak_surge_m": window.max(),
                "mean_surge_m": window.mean(),
                "year": start.year,
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_annual_max_surge(residuals: pd.DataFrame, events: pd.DataFrame, outpath: Path):
    """Annual maximum non-tidal residual + surge events summary."""
    annual_max = residuals["residual_m"].resample("YE").max()
    annual_max.index = annual_max.index.year

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)

    # --- panel 1: full residual time series (monthly envelope) ---
    monthly_max = residuals["residual_m"].resample("ME").max()
    monthly_min = residuals["residual_m"].resample("ME").min()
    ax = axes[0]
    ax.fill_between(monthly_max.index, monthly_min.values, monthly_max.values,
                    alpha=0.3, color="steelblue", label="Monthly range")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Non-tidal residual (m)")
    ax.set_title("Monthly min/max non-tidal residual")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.legend(fontsize=8)

    # --- panel 2: annual maximum residual ---
    ax = axes[1]
    ax.bar(annual_max.index, annual_max.values, color="steelblue", alpha=0.8)
    ax.set_ylabel("Annual max residual (m)")
    ax.set_title("Annual maximum non-tidal residual")
    ax.set_xlabel("Year")

    # --- panel 3: surge event peak histogram ---
    ax = axes[2]
    if not events.empty:
        ax.hist(events["peak_surge_m"], bins=20, color="firebrick", alpha=0.8, edgecolor="white")
        ax.set_xlabel("Peak surge (m)")
        ax.set_ylabel("Number of events")
        ax.set_title(
            f"Storm surge events (residual ≥ {events['peak_surge_m'].min():.2f} m)  "
            f"n = {len(events)}"
        )
    else:
        ax.text(0.5, 0.5, "No events above threshold", transform=ax.transAxes, ha="center")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved {outpath}")


def plot_surge_seasonality(events: pd.DataFrame, outpath: Path):
    """Monthly frequency of surge events."""
    if events.empty:
        return

    events = events.copy()
    events["month"] = pd.to_datetime(events["start"]).dt.month

    monthly_counts = events.groupby("month").size().reindex(range(1, 13), fill_value=0)
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.bar(monthly_counts.index, monthly_counts.values, color="steelblue", alpha=0.8, edgecolor="white")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels)
    ax.set_ylabel("Number of surge events")
    ax.set_title("Seasonal distribution of storm surge events")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved {outpath}")


def plot_largest_events(residuals: pd.DataFrame, events: pd.DataFrame, n: int, outpath: Path):
    """Time series detail for the n largest surge events."""
    if events.empty:
        return

    top = events.nlargest(n, "peak_surge_m")
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, (_, evt) in zip(axes, top.iterrows()):
        pad = timedelta(days=3)
        window = residuals.loc[evt["start"] - pad : evt["end"] + pad]
        ax.plot(window.index, window["obs_m"], lw=0.8, color="steelblue", label="Observed")
        ax.plot(window.index, window["pred_m"], lw=0.8, color="darkorange", label="Predicted")
        ax.fill_between(window.index,
                        window["pred_m"], window["obs_m"],
                        where=window["residual_m"] > 0,
                        alpha=0.3, color="firebrick", label="Surge")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_title(
            f"{evt['start'].date()} — peak surge {evt['peak_surge_m']:.3f} m, "
            f"duration {evt['duration_hours']:.0f} h"
        )
        ax.set_ylabel("Water level (m MLLW)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        if ax == axes[0]:
            ax.legend(fontsize=8, ncol=3)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved {outpath}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute non-tidal residuals (storm surge) from NOAA CO-OPS data."
    )
    p.add_argument("--station", required=True,
                   help="NOAA CO-OPS station ID, e.g. 8661070")
    p.add_argument("--start", type=int, required=True,
                   help="Start year (inclusive)")
    p.add_argument("--end", type=int, required=True,
                   help="End year (inclusive)")
    p.add_argument("--datum", default="MLLW",
                   help="Tidal datum (default: MLLW)")
    p.add_argument("--obs-csv", default=None,
                   help="Path to existing observed water level CSV (skips download "
                        "if present). Column names: datetime_gmt, wl_m_mllw  OR  "
                        "datetime_gmt, value")
    p.add_argument("--pred-csv", default=None,
                   help="Path to cache downloaded predictions CSV")
    p.add_argument("--out-csv", required=True,
                   help="Output CSV for residuals (datetime_gmt, obs_m, pred_m, residual_m)")
    p.add_argument("--surge-threshold", type=float, default=0.3,
                   help="Residual (m) above which an event is classified as a surge "
                        "(default: 0.3)")
    p.add_argument("--min-gap-hours", type=int, default=24,
                   help="Minimum quiet hours between separate surge events (default: 24)")
    p.add_argument("--plot-dir", default=None,
                   help="Directory for output figures (skipped if not given)")
    p.add_argument("--top-events", type=int, default=5,
                   help="Number of largest events to plot in detail (default: 5)")
    return p


def main():
    args = build_parser().parse_args()

    # --- observations ---
    obs_cache = Path(args.obs_csv) if args.obs_csv else None

    if obs_cache and obs_cache.exists():
        print(f"Loading observations from {obs_cache}")
        obs_df = pd.read_csv(obs_cache, parse_dates=["datetime_gmt"])
        obs_df = obs_df.set_index("datetime_gmt").sort_index()
        # normalise column name
        if "wl_m_mllw" in obs_df.columns:
            obs_df = obs_df.rename(columns={"wl_m_mllw": "value"})
        obs_df = obs_df[["value"]]
    else:
        print(f"Downloading observed water levels for station {args.station} ...")
        obs_df = _download_series(
            station=args.station,
            start_year=args.start,
            end_year=args.end,
            product="water_level",
            datum=args.datum,
            cache_csv=obs_cache,
        )

    # --- predictions ---
    pred_cache = Path(args.pred_csv) if args.pred_csv else None
    print(f"Downloading astronomical predictions for station {args.station} ...")
    pred_df = _download_series(
        station=args.station,
        start_year=args.start,
        end_year=args.end,
        product="predictions",
        datum=args.datum,
        cache_csv=pred_cache,
        interval="h",
    )

    # --- residuals ---
    print("Computing residuals ...")
    residuals = compute_residuals(obs_df, pred_df)
    print(
        f"  {len(residuals):,} matched hourly records\n"
        f"  residual range: {residuals['residual_m'].min():.3f} m "
        f"to {residuals['residual_m'].max():.3f} m\n"
        f"  mean residual:  {residuals['residual_m'].mean():.4f} m"
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    residuals.reset_index().to_csv(out_csv, index=False)
    print(f"Residuals saved to {out_csv}")

    # --- surge events ---
    events = identify_surge_events(
        residuals,
        threshold_m=args.surge_threshold,
        min_gap_hours=args.min_gap_hours,
    )
    print(f"\nSurge events (residual ≥ {args.surge_threshold} m): {len(events)}")
    if not events.empty:
        print(events.sort_values("peak_surge_m", ascending=False).head(10).to_string(index=False))

    events_csv = out_csv.with_name(out_csv.stem + "_events.csv")
    events.to_csv(events_csv, index=False)
    print(f"Events saved to {events_csv}")

    # --- plots ---
    if args.plot_dir:
        plot_dir = Path(args.plot_dir)
        stem = f"surge_{args.station}"
        plot_annual_max_surge(
            residuals, events,
            plot_dir / f"{stem}_annual.png",
        )
        plot_surge_seasonality(
            events,
            plot_dir / f"{stem}_seasonality.png",
        )
        plot_largest_events(
            residuals, events,
            n=min(args.top_events, len(events)),
            outpath=plot_dir / f"{stem}_top_events.png",
        )


if __name__ == "__main__":
    main()
