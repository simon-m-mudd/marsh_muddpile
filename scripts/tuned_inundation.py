#!/usr/bin/env python3
"""
tuned_inundation.py

1. Downloads (or loads from cache) hourly NAVD88 water-level records for every
   gauge in core_inundation.fgb and saves sorted WL arrays to
   tidal_datum_cache/wl_sorted/<gauge_id>.npy.

2. Fits a linear regression of RTK elevation ~ EPQS elevation using cores
   that have both, and applies the correction to EPQS-only cores to produce
   a bias-corrected elevation estimate.

3. Computes tuned inundation fractions:
     inund_tuned   — uses RTK elev directly, or EPQS-corrected elev
     inund_tuned_c — same but −0.15 m canopy correction applied on top
                     (for EPQS cores where the regression may not fully
                     remove canopy bias)

4. Adds tuned inundation and regression model parameters to the FGB.

5. Plots:
     a. RTK vs EPQS scatter with OLS fit  (rtk_vs_epqs_regression.png)
     b. SLR (NOAA gauge) vs Pb-210 CIC accretion rate  (slr_vs_pb210.png)

Usage
-----
python3 scripts/tuned_inundation.py \\
    [--inund-fgb scripts/core_inundation.fgb] \\
    [--ccn-cores core_data/CCN_synthesis/CCN_cores.csv] \\
    [--output    scripts/core_inundation.fgb] \\
    [--cache-dir tidal_datum_cache] \\
    [--years 2]  [--workers 16]
"""

import argparse, math, sys, time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
NOAA_DATA_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
USGS_IV_URL   = "https://waterservices.usgs.gov/nwis/iv/"
_FT_TO_M      = 0.3048
_TIMEOUT_NOAA = 30
_TIMEOUT_USGS = 120
_CANOPY_BIAS  = 0.15   # m, applied to EPQS-corrected elevations


# ---------------------------------------------------------------------------
# Download helpers (same as match_tides_to_cores.py)
# ---------------------------------------------------------------------------

def _noaa_chunks(start, end, max_days=365):
    current = start
    while current <= end:
        yield current, min(end, current + timedelta(days=max_days - 1))
        current = min(end, current + timedelta(days=max_days - 1)) + timedelta(days=1)


def _noaa_job(sid, cs, ce):
    params = {"product": "hourly_height", "application": "marsh_inundation",
              "begin_date": cs.strftime("%Y%m%d"), "end_date": ce.strftime("%Y%m%d"),
              "station": sid, "datum": "NAVD", "time_zone": "gmt",
              "units": "metric", "format": "json"}
    try:
        r = requests.get(NOAA_DATA_URL, params=params, timeout=_TIMEOUT_NOAA)
        if not r.ok: return sid, None
        payload = r.json()
        if "error" in payload: return sid, []
        rows = []
        for item in payload.get("data", []):
            try: rows.append({"t": item["t"], "v": float(item["v"])})
            except (KeyError, ValueError): pass
        return sid, rows
    except Exception:
        return sid, None


def _usgs_job(sid, start, end):
    params = {"format": "json", "sites": sid,
              "startDT": start.strftime("%Y-%m-%d"), "endDT": end.strftime("%Y-%m-%d"),
              "parameterCd": "00065", "siteStatus": "all"}
    try:
        r = requests.get(USGS_IV_URL, params=params, timeout=_TIMEOUT_USGS)
        if not r.ok: return sid, None
        rows = []
        for ts in r.json().get("value", {}).get("timeSeries", []):
            is_feet = "ft" in ts.get("variable", {}).get("unit", {}).get("unitCode", "").lower()
            for block in ts.get("values", []):
                for item in block.get("value", []):
                    try:
                        v = float(item["value"])
                        if v <= -999990: continue
                        if is_feet: v *= _FT_TO_M
                        rows.append({"t": item["dateTime"], "v": v})
                    except (KeyError, ValueError): pass
        return sid, rows
    except Exception:
        return sid, None


def _to_sorted_navd88(rows, alt_navd88_m=0.0):
    """Return sorted numpy array of valid hourly NAVD88 water levels."""
    if not rows:
        return np.array([])
    df = pd.DataFrame(rows)
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    df["t"] = pd.to_datetime(df["t"], errors="coerce", utc=True).dt.tz_localize(None)
    df = df.dropna().set_index("t")
    s = df["v"].resample("h").mean().dropna()
    if alt_navd88_m != 0.0:
        s = s + alt_navd88_m
    return np.sort(s.values)


# ---------------------------------------------------------------------------
# Cache load/save
# ---------------------------------------------------------------------------

def load_or_download_wl(gauge_specs, start, end, cache_dir, workers):
    """
    For each gauge spec, load sorted WL from cache if it exists and is
    recent enough, otherwise download and cache.

    Returns dict: gauge_id → sorted np.ndarray (NAVD88, m).
    """
    wl_dir = Path(cache_dir) / "wl_sorted"
    wl_dir.mkdir(parents=True, exist_ok=True)

    # Check which gauges need downloading
    to_download, cached = [], {}
    for g in gauge_specs:
        path = wl_dir / f"{g['id']}.npy"
        if path.exists():
            arr = np.load(str(path))
            cached[g["id"]] = arr
        else:
            to_download.append(g)

    if cached:
        print(f"  {len(cached)} gauges loaded from cache, "
              f"{len(to_download)} to download", flush=True)
    else:
        print(f"  {len(to_download)} gauges to download", flush=True)

    if not to_download:
        return cached

    # Build jobs
    jobs = []
    for g in to_download:
        if g["source"] == "noaa":
            for cs, ce in _noaa_chunks(start, end):
                jobs.append(("noaa", g["id"], g["alt_navd88_m"], cs, ce))
        else:
            jobs.append(("usgs", g["id"], g["alt_navd88_m"], start, end))

    print(f"  Submitting {len(jobs)} download jobs ({workers} workers) ...", flush=True)

    raw: dict[str, list] = {}

    def _run(job):
        if job[0] == "noaa":
            return _noaa_job(job[1], job[3], job[4])
        else:
            return _usgs_job(job[1], job[3], job[4])

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_run, j): j for j in jobs}
        done = 0
        for fut in as_completed(futures):
            done += 1
            print(f"    {done}/{len(jobs)} jobs\r", end="", flush=True)
            try:
                sid, rows = fut.result()
            except Exception:
                continue
            if rows:
                raw.setdefault(sid, []).extend(rows)
    print()

    result = dict(cached)
    for g in to_download:
        sid = g["id"]
        arr = _to_sorted_navd88(raw.get(sid, []), g["alt_navd88_m"])
        result[sid] = arr
        np.save(str(wl_dir / f"{sid}.npy"), arr)
        n = len(arr)
        print(f"  {g['source'].upper()} {sid:15s}  {n:5d} hrs"
              f"{'  (no data)' if n == 0 else ''}")

    return result


# ---------------------------------------------------------------------------
# Inundation from sorted WL array
# ---------------------------------------------------------------------------

def inund_from_sorted(sorted_wl, elev_m, min_hours=24 * 30):
    if len(sorted_wl) < min_hours or math.isnan(elev_m):
        return float("nan")
    # fraction >= elev: binary search
    idx = np.searchsorted(sorted_wl, elev_m, side="left")
    return float((len(sorted_wl) - idx) / len(sorted_wl))


# ---------------------------------------------------------------------------
# Regression RTK ~ EPQS
# ---------------------------------------------------------------------------

def fit_rtk_epqs_regression(gdf):
    """
    Fit  RTK = slope * EPQS + intercept  using cores that have both.
    Returns (slope, intercept, n, r2).
    """
    mask = (gdf["elev_src"] == "measured") & gdf["epqs_elev_m"].notna()
    sub  = gdf[mask]
    x = sub["epqs_elev_m"].values
    y = sub["marsh_elev_m"].values
    # Remove NaN
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 10:
        return 1.0, 0.0, n, float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), n, float(r2)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_rtk_vs_epqs(gdf, slope, intercept, r2, out_path):
    mask = (gdf["elev_src"] == "measured") & gdf["epqs_elev_m"].notna()
    sub  = gdf[mask]
    x    = sub["epqs_elev_m"].values
    y    = sub["marsh_elev_m"].values
    ok   = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    diff = y - x   # RTK − EPQS  (negative = EPQS overestimates)

    fig = plt.figure(figsize=(10, 9), constrained_layout=True)
    gs  = fig.add_gridspec(3, 1, height_ratios=[3.5, 1.5, 0.05])
    ax  = fig.add_subplot(gs[0])
    axh = fig.add_subplot(gs[1])
    cax = fig.add_subplot(gs[2])

    sc = ax.scatter(x, y, c=sub[ok]["latitude"], cmap="viridis_r",
                    s=7, alpha=0.55, linewidths=0, rasterized=True)
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("Latitude (°N)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    lo = min(x.min(), y.min()) - 0.1
    hi = max(x.max(), y.max()) + 0.1
    xline = np.linspace(lo, hi, 300)
    ax.plot(xline, xline,                        "k-",  lw=1.0, label="1:1")
    ax.plot(xline, xline - 0.15,                 "r--", lw=0.9,
            label="1:1 − 0.15 m (canopy bias)")
    ax.plot(xline, slope * xline + intercept, "steelblue", lw=1.3, ls="-.",
            label=f"OLS  RTK = {slope:.3f}·EPQS {intercept:+.3f} m  (R²={r2:.3f})")

    rmse = np.sqrt(np.mean(diff ** 2))
    ax.text(0.03, 0.97,
            f"n = {len(x):,}\n"
            f"Bias (RTK − EPQS): mean = {diff.mean():+.3f} m,  median = {np.median(diff):+.3f} m\n"
            f"RMSE = {rmse:.3f} m",
            transform=ax.transAxes, va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
    ax.set_xlabel("EPQS elevation  (m NAVD88)", fontsize=9)
    ax.set_ylabel("RTK elevation  (m NAVD88)", fontsize=9)
    ax.set_title("RTK-GPS vs EPQS elevation — CCN US marsh cores", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal"); ax.grid(True, lw=0.4, alpha=0.4)
    ax.tick_params(labelsize=8)

    axh.hist(diff, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
    axh.axvline(0,           color="k",      lw=1.0, label="zero")
    axh.axvline(diff.mean(), color="r",      lw=1.2, ls="--",
                label=f"mean {diff.mean():+.3f} m")
    axh.axvline(-0.15,       color="tomato", lw=1.0, ls=":",
                label="−0.15 m (canopy bias)")
    axh.set_xlabel("RTK − EPQS  (m)", fontsize=9)
    axh.set_ylabel("Count", fontsize=9)
    axh.set_title("Residual distribution", fontsize=9)
    axh.legend(fontsize=8); axh.set_xlim(-1.5, 1.5)
    axh.tick_params(labelsize=8)

    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_slr_vs_pb210(gdf, ccn_cores_path, out_path):
    """Plot NOAA gauge SLR rate vs Pb-210 CIC accretion rate."""
    cores = pd.read_csv(ccn_cores_path, low_memory=False)

    # Merge pb210 onto the inundation GDF via core_id
    pb210 = cores[["core_id", "pb210_cic_accretion_rate",
                   "pb210_cic_accretion_rate_se", "salinity_class"]].copy()
    pb210 = pb210.dropna(subset=["pb210_cic_accretion_rate"])
    pb210["pb210_mmyr"] = pb210["pb210_cic_accretion_rate"] * 10   # cm/yr → mm/yr
    pb210["pb210_se_mmyr"] = pb210["pb210_cic_accretion_rate_se"]  # already mm/yr

    merged = gdf[["core_id", "slr_mmyr", "latitude"]].merge(
        pb210[["core_id", "pb210_mmyr", "pb210_se_mmyr", "salinity_class"]],
        on="core_id", how="inner"
    ).dropna(subset=["slr_mmyr", "pb210_mmyr"])

    if merged.empty:
        print("  No cores with both SLR and Pb-210 data — skipping plot")
        return

    print(f"  {len(merged)} cores with both SLR and Pb-210 data")

    # Salinity colour map
    sal_order = ["saline", "estuarine", "brackish", "mesohaline",
                 "oligohaline", "fresh", "polyhaline", "mixoeuhaline"]
    cmap  = plt.cm.tab10
    colors = {s: cmap(i / len(sal_order)) for i, s in enumerate(sal_order)}
    default_c = "grey"

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)

    # Plot per salinity class
    plotted = set()
    for _, row in merged.iterrows():
        sal = row["salinity_class"] if pd.notna(row["salinity_class"]) else "unknown"
        c   = colors.get(sal, default_c)
        lbl = sal if sal not in plotted else "_"
        plotted.add(sal)
        kw = dict(color=c, alpha=0.75, zorder=3)
        if pd.notna(row.get("pb210_se_mmyr")) and row["pb210_se_mmyr"] > 0:
            ax.errorbar(row["slr_mmyr"], row["pb210_mmyr"],
                        yerr=row["pb210_se_mmyr"],
                        fmt="o", ms=6, lw=0.7, capsize=2, label=lbl, **kw)
        else:
            ax.scatter(row["slr_mmyr"], row["pb210_mmyr"],
                       s=40, label=lbl, **kw)

    # 1:1 line
    lo = max(0, min(merged["slr_mmyr"].min(), merged["pb210_mmyr"].min()) - 0.5)
    hi = max(merged["slr_mmyr"].max(), merged["pb210_mmyr"].max()) + 1
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="1:1  (accretion = SLR)")
    ax.fill_between([lo, hi], [lo, hi], [hi, hi], alpha=0.06, color="green",
                    label="Accretion > SLR (gaining)")
    ax.fill_between([lo, hi], [0, 0], [lo, hi], alpha=0.06, color="red",
                    label="Accretion < SLR (at risk)")

    ax.set_xlabel("NOAA gauge SLR trend (mm yr⁻¹)", fontsize=10)
    ax.set_ylabel("Pb-210 CIC accretion rate (mm yr⁻¹)", fontsize=10)
    ax.set_title("Sea-level rise vs sediment accretion rate — CCN US marsh cores",
                 fontsize=10)
    ax.set_xlim(lo, hi); ax.set_ylim(0, hi)
    ax.grid(True, lw=0.4, alpha=0.35)
    ax.tick_params(labelsize=9)

    # Deduplicated legend
    handles, labels = ax.get_legend_handles_labels()
    seen, unh, unl = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen and not l.startswith("_"):
            seen.add(l); unh.append(h); unl.append(l)
    ax.legend(unh, unl, fontsize=8, loc="upper left",
              title="Salinity class", title_fontsize=8)

    # Summary stats
    n_above = (merged["pb210_mmyr"] >= merged["slr_mmyr"]).sum()
    ax.text(0.97, 0.03,
            f"n = {len(merged)}\n"
            f"Above 1:1 (gaining): {n_above} ({100*n_above/len(merged):.0f}%)\n"
            f"Below 1:1 (at risk): {len(merged)-n_above} "
            f"({100*(len(merged)-n_above)/len(merged):.0f}%)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Add regression-corrected tuned inundation fractions to core_inundation.fgb.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--inund-fgb",  default="scripts/core_inundation.fgb")
    p.add_argument("--ccn-cores",  default="core_data/CCN_synthesis/CCN_cores.csv")
    p.add_argument("--output",     default="scripts/core_inundation.fgb")
    p.add_argument("--cache-dir",  default="tidal_datum_cache")
    p.add_argument("--years",      type=float, default=2.0)
    p.add_argument("--workers",    type=int,   default=16)
    p.add_argument("--fig-dir",    default="scripts/figures/datum_comparison")
    return p


def main():
    args = build_parser().parse_args()
    t0   = _time.perf_counter()

    try:
        import geopandas as gpd
    except ImportError:
        sys.exit("geopandas required")

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    end_date   = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=round(365 * args.years))

    print(f"\n{'='*66}")
    print(f"  Tuned inundation fractions + SLR analysis")
    print(f"  Period: {start_date.date()} → {end_date.date()}")
    print(f"{'='*66}\n")

    # ── 1. Load inundation index ──────────────────────────────────────────────
    print("[1] Loading inundation index ...")
    gdf = gpd.read_file(args.inund_fgb)
    print(f"  {len(gdf)} cores")

    # ── 2. Build gauge spec list (same logic as match_tides_to_cores.py) ──────
    print("\n[2] Building gauge list ...")
    gdf["_use_usgs"] = (
        gdf["usgs1_dist_km"].notna() &
        (gdf["usgs1_dist_km"] < gdf["noaa_dist_km"]) &
        gdf["usgs1_alt_m"].notna()
    )
    gauge_specs, seen = [], set()
    for _, row in gdf.iterrows():
        if row["_use_usgs"]:
            sid = str(row["usgs1_id"])
            if sid not in seen:
                seen.add(sid)
                gauge_specs.append({"source": "usgs", "id": sid,
                                    "alt_navd88_m": float(row["usgs1_alt_m"])})
        else:
            sid = str(row["noaa_id"])
            if sid not in seen:
                seen.add(sid)
                gauge_specs.append({"source": "noaa", "id": sid, "alt_navd88_m": 0.0})
    print(f"  {len(gauge_specs)} unique gauges")

    # ── 3. Load or download WL ────────────────────────────────────────────────
    print("\n[3] Water-level records (cache or download) ...")
    wl = load_or_download_wl(gauge_specs, start_date, end_date,
                              args.cache_dir, args.workers)
    print(f"  ⏱  {_time.perf_counter()-t0:.1f} s")

    # ── 4. RTK ~ EPQS regression ──────────────────────────────────────────────
    print("\n[4] RTK ~ EPQS regression ...")
    slope, intercept, n_reg, r2 = fit_rtk_epqs_regression(gdf)
    print(f"  RTK = {slope:.4f} × EPQS {intercept:+.4f} m   "
          f"(n={n_reg:,}  R²={r2:.4f})")
    print(f"  Interpretation: EPQS overestimates by "
          f"{-intercept/(1-slope)*1000 if slope!=1 else (intercept*-1)*1000:.0f} mm "
          f"at typical marsh elevation (~0.3 m NAVD88)")

    # ── 5. Compute tuned inundation ───────────────────────────────────────────
    print(f"\n[5] Computing tuned inundation for {len(gdf)} cores ...")

    inund_tuned   = np.full(len(gdf), float("nan"))
    inund_tuned_c = np.full(len(gdf), float("nan"))
    corrected_elev = np.full(len(gdf), float("nan"))
    elev_method    = [""] * len(gdf)

    for i, (_, row) in enumerate(gdf.iterrows()):
        gauge_id = str(row["gauge_id"]) if pd.notna(row.get("gauge_id")) else str(row["noaa_id"])
        wl_arr   = wl.get(gauge_id, np.array([]))

        if row["elev_src"] == "measured":
            elev = float(row["marsh_elev_m"])
            elev_method[i] = "rtk"
        elif row["elev_src"] == "epqs" and not math.isnan(float(row["epqs_elev_m"])):
            epqs = float(row["epqs_elev_m"])
            elev = slope * epqs + intercept
            elev_method[i] = "epqs_corrected"
        else:
            elev_method[i] = "none"
            continue

        corrected_elev[i]  = elev
        inund_tuned[i]     = inund_from_sorted(wl_arr, elev)
        inund_tuned_c[i]   = inund_from_sorted(wl_arr, elev - _CANOPY_BIAS)

    gdf["elev_tuned_m"]    = np.where(np.isfinite(corrected_elev), corrected_elev, np.nan)
    gdf["elev_tuned_src"]  = elev_method
    gdf["inund_tuned"]     = np.where(np.isfinite(inund_tuned),   inund_tuned,   np.nan)
    gdf["inund_tuned_c"]   = np.where(np.isfinite(inund_tuned_c), inund_tuned_c, np.nan)
    gdf["epqs_slope"]      = round(slope,     5)
    gdf["epqs_intercept"]  = round(intercept, 5)
    gdf["epqs_r2"]         = round(r2,        5)

    valid = np.isfinite(inund_tuned)
    print(f"  Valid tuned inundation: {valid.sum()}/{len(gdf)}")
    src_counts = pd.Series(elev_method).value_counts()
    for s, c in src_counts.items():
        print(f"    {s}: {c}")
    print(f"  Tuned inundation — mean={np.nanmean(inund_tuned):.3f}  "
          f"median={np.nanmedian(inund_tuned):.3f}")
    print(f"  Shift vs raw:   Δmean={np.nanmean(inund_tuned)-np.nanmean(gdf['inund_frac']):.3f}  "
          f"Δmedian={np.nanmedian(inund_tuned)-np.nanmedian(gdf['inund_frac']):.3f}")
    print(f"  ⏱  {_time.perf_counter()-t0:.1f} s")

    # ── 6. Save updated FGB ───────────────────────────────────────────────────
    print(f"\n[6] Saving ...")
    gdf.drop(columns=["_use_usgs"], errors="ignore").to_file(
        args.output, driver="FlatGeobuf")
    print(f"  Saved {len(gdf)} cores → {args.output}")

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    print(f"\n[7] Plots ...")
    plot_rtk_vs_epqs(gdf, slope, intercept, r2,
                     fig_dir / "rtk_vs_epqs_regression.png")
    plot_slr_vs_pb210(gdf, args.ccn_cores,
                      fig_dir / "slr_vs_pb210.png")

    print(f"\n  Total time: {_time.perf_counter()-t0:.1f} s")


if __name__ == "__main__":
    main()
