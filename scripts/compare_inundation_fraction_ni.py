"""compare_inundation_fraction_ni.py

Compare inundation fraction vs elevation at North Inlet, SC for the growing
season (April 1 – September 30) using four methods / data sources:

  a) Harmonic reconstruction using NOAA-published constituents for gauge
     8662245 (North Inlet-Winyah Bay NERR, Oyster Landing).
  b) Harmonic reconstruction using constituents fitted by least-squares to
     the observed 8662245 6-minute water level record.
  c) Direct count: fraction of 6-minute observations from 8662245 where
     water level exceeds each test elevation.
  d) Direct count from NOAA 8661070 (Springmaid Pier, Myrtle Beach) —
     the nearest active open-ocean gauge, ~30 km NNE of North Inlet.

All water levels are relative to NAVD88.

Note: NOAA 8662245 (Oyster Landing) became inactive in late March 2022.
The most recent complete growing season available is April–September 2021.

Usage
-----
    python compare_inundation_fraction_ni.py [--force]

--force re-downloads data even if the cache file already exists.
"""

from __future__ import annotations
import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
from scipy.linalg import lstsq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_STATION_NI  = "8662245"   # North Inlet-Winyah Bay NERR (Oyster Landing), SC
_STATION_SP  = "8661070"   # Springmaid Pier, Myrtle Beach, SC
_DATUM       = "NAVD"      # water levels relative to NAVD88
_START       = "20210401"  # 8662245 inactive after Mar 2022; 2021 is last full growing season
_END         = "20210930"
_CACHE_DIR   = Path(__file__).parent / "tidal_datum_cache"
_OUT_FIG     = Path(__file__).parent / "figures" / "compare_inundation_fraction_ni.png"

# Elevation range to sweep (m NAVD88) — covers the North Inlet marsh platform
_ELEV_MIN    = -0.60    # subtidal
_ELEV_MAX    =  1.10    # supratidal
_N_ELEV      = 300

# NOAA-published harmonic constituents for 8662245 (retrieved 2026-05-18)
# Amplitudes in metres; phases are Greenwich phase lags in degrees.
_NOAA_CONSTITUENTS: dict[str, tuple[float, float]] = {
    "M2": (0.6100, 32.1),
    "S2": (0.0950, 67.8),
    "N2": (0.1330, 21.9),
    "K1": (0.1020, 214.6),
    "O1": (0.0800, 217.0),
}

# Tidal angular speeds in degrees per hour (Schureman 1958)
_OMEGA_DEG_HR: dict[str, float] = {
    "M2": 28.9841042,
    "S2": 30.0000000,
    "N2": 28.4397295,
    "K1": 15.0410686,
    "O1": 13.9430356,
}

# ---------------------------------------------------------------------------
# NOAA CO-OPS data download
# ---------------------------------------------------------------------------

def _noaa_fetch_chunk(station: str, begin: str, end: str) -> list[dict]:
    """Fetch one chunk (≤31 days) from NOAA CO-OPS API."""
    url = (
        f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?"
        f"station={station}&product=water_level&datum={_DATUM}"
        f"&time_zone=GMT&units=metric&format=json"
        f"&begin_date={begin}&end_date={end}"
    )
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.loads(r.read())
    if "error" in data:
        raise RuntimeError(f"NOAA API error ({station}): {data['error']['message']}")
    return data.get("data", [])


def _month_chunks(start: str, end: str) -> list[tuple[str, str]]:
    """Split a date range into ≤31-day chunks for the NOAA API."""
    from datetime import date, timedelta
    s = date(int(start[:4]), int(start[4:6]), int(start[6:8]))
    e = date(int(end[:4]),   int(end[4:6]),   int(end[6:8]))
    chunks = []
    cur = s
    while cur <= e:
        nxt = min(cur + timedelta(days=30), e)
        chunks.append((cur.strftime("%Y%m%d"), nxt.strftime("%Y%m%d")))
        cur = nxt + timedelta(days=1)
    return chunks


def load_water_levels(station: str, force: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_hours_since_2025-01-01-UTC, wl_navd88_m) for the given station."""
    from datetime import datetime, timezone
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = _CACHE_DIR / f"wl_{station}_{_START}_{_END}.json"

    if not force and cache.exists():
        print(f"  Using cached data: {cache}")
        with open(cache) as f:
            raw = json.load(f)
    else:
        print(f"  Downloading NOAA {station} data {_START}–{_END} …")
        raw = []
        for begin, end in _month_chunks(_START, _END):
            print(f"    chunk {begin}–{end} …", end=" ", flush=True)
            chunk = _noaa_fetch_chunk(station, begin, end)
            raw.extend(chunk)
            print(f"{len(chunk)} records")
            time.sleep(0.5)
        with open(cache, "w") as f:
            json.dump(raw, f)
        print(f"  Saved to {cache}")

    epoch = datetime(2025, 1, 1, tzinfo=timezone.utc)
    t_h, wl = [], []
    for rec in raw:
        if not rec.get("v"):
            continue
        try:
            dt = datetime.strptime(rec["t"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            wl.append(float(rec["v"]))
            t_h.append((dt - epoch).total_seconds() / 3600.0)
        except (ValueError, KeyError):
            continue

    t_arr = np.asarray(t_h, dtype=float)
    wl_arr = np.asarray(wl, dtype=float)
    idx = np.argsort(t_arr)
    return t_arr[idx], wl_arr[idx]


# ---------------------------------------------------------------------------
# Harmonic reconstruction helpers
# ---------------------------------------------------------------------------

def _reconstruct(t_hours: np.ndarray,
                 constituents: dict[str, tuple[float, float]],
                 msl: float = 0.0) -> np.ndarray:
    """Reconstruct tidal signal from amplitude/phase dict.

    t_hours : hours since the same epoch used for phase reference.
              NOAA Greenwich phase lags are given relative to the equilibrium
              tide; the reconstruction is WL(t) = MSL + Σ A cos(ω t - φ),
              where ω is in rad/hr and φ is converted from degrees.
    """
    wl = np.full_like(t_hours, msl)
    for name, (amp, phase_deg) in constituents.items():
        omega = np.deg2rad(_OMEGA_DEG_HR[name])  # rad/hr
        phi   = np.deg2rad(phase_deg)
        wl   += amp * np.cos(omega * t_hours - phi)
    return wl


def fit_constituents(t_hours: np.ndarray,
                     wl: np.ndarray) -> dict[str, tuple[float, float]]:
    """Least-squares fit of M2, S2, N2, K1, O1 to observed water levels.

    Returns dict[name] -> (amplitude_m, phase_deg_greenwich_approx).
    The fitted phases are relative to t=0 of the supplied time vector;
    for an approximately consistent Greenwich phase the caller should pass
    t in hours since a UTC epoch.
    """
    names = list(_OMEGA_DEG_HR.keys())
    # Build design matrix: [1, cos(ω₁t), sin(ω₁t), cos(ω₂t), sin(ω₂t), ...]
    cols = [np.ones(len(t_hours))]
    for name in names:
        omega = np.deg2rad(_OMEGA_DEG_HR[name])
        cols.append(np.cos(omega * t_hours))
        cols.append(np.sin(omega * t_hours))
    A = np.column_stack(cols)
    coeffs, *_ = lstsq(A, wl)

    msl_fit = coeffs[0]
    result = {}
    for i, name in enumerate(names):
        c_cos = coeffs[1 + 2*i]
        c_sin = coeffs[2 + 2*i]
        amp   = np.sqrt(c_cos**2 + c_sin**2)
        # WL = amp * cos(ωt - φ) = c_cos cos(ωt) + c_sin sin(ωt)
        # → c_cos = amp cos(φ),  c_sin = amp sin(φ)  → φ = atan2(c_sin, c_cos)
        phase = np.rad2deg(np.arctan2(c_sin, c_cos)) % 360
        result[name] = (amp, phase)

    print(f"  Fitted MSL = {msl_fit:+.4f} m NAVD88")
    print(f"  {'Const':4s}  {'A_noaa':>8s}  {'ph_noaa':>9s}  {'A_fit':>8s}  {'ph_fit':>9s}")
    for name in names:
        an, pn = _NOAA_CONSTITUENTS[name]
        af, pf = result[name]
        print(f"  {name:4s}  {an:8.4f}m  {pn:8.1f}°   {af:8.4f}m  {pf:8.1f}°")

    return result, msl_fit


# ---------------------------------------------------------------------------
# Inundation fraction vs elevation
# ---------------------------------------------------------------------------

def inundation_fraction_from_record(wl: np.ndarray,
                                    elevations: np.ndarray) -> np.ndarray:
    """Fraction of observations where water level exceeds each elevation."""
    return np.array([np.mean(wl > e) for e in elevations])


def inundation_fraction_from_harmonics(t_hours: np.ndarray,
                                       constituents: dict,
                                       msl: float,
                                       elevations: np.ndarray) -> np.ndarray:
    """Reconstruct tidal signal then compute IF at each elevation."""
    wl_reconstructed = _reconstruct(t_hours, constituents, msl)
    return inundation_fraction_from_record(wl_reconstructed, elevations)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Re-download data even if cache exists")
    args = parser.parse_args()

    _OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    # --- load North Inlet (8662245) observed record ---
    print("Loading North Inlet water level record (8662245) …")
    t_h, wl_obs = load_water_levels(_STATION_NI, force=args.force)
    print(f"  {len(t_h)} obs  WL {wl_obs.min():.3f}–{wl_obs.max():.3f} m NAVD88")

    # --- load Springmaid Pier (8661070) observed record ---
    print("\nLoading Springmaid Pier water level record (8661070) …")
    t_sp, wl_sp = load_water_levels(_STATION_SP, force=args.force)
    print(f"  {len(t_sp)} obs  WL {wl_sp.min():.3f}–{wl_sp.max():.3f} m NAVD88")

    # --- fit constituents from North Inlet record ---
    print("\nFitting harmonics to North Inlet record …")
    fitted_constituents, msl_fit = fit_constituents(t_h, wl_obs)

    # --- elevation grid ---
    elevations = np.linspace(_ELEV_MIN, _ELEV_MAX, _N_ELEV)

    # --- method (a): NOAA published constituents (8662245) ---
    print("\nComputing IF — (a) NOAA published harmonics (8662245) …")
    if_noaa = inundation_fraction_from_harmonics(
        t_h, _NOAA_CONSTITUENTS, msl=0.0, elevations=elevations)

    # --- method (b): fitted harmonics from North Inlet record ---
    print("Computing IF — (b) fitted harmonics (8662245 obs) …")
    if_fit = inundation_fraction_from_harmonics(
        t_h, fitted_constituents, msl=msl_fit, elevations=elevations)

    # --- method (c): raw North Inlet record ---
    print("Computing IF — (c) direct count (8662245 raw record) …")
    if_raw = inundation_fraction_from_record(wl_obs, elevations)

    # --- method (d): raw Springmaid Pier record ---
    print("Computing IF — (d) direct count (8661070 Springmaid Pier) …")
    if_sp = inundation_fraction_from_record(wl_sp, elevations)

    # --- NOAA datums for 8662245 (NAVD88 basis) ---
    # MHW = 0.625 m, MSL ≈ -0.008 m, MLW = 1.272 - 2.040 = -0.768 m
    mhw_navd = 0.625
    msl_navd = -0.008
    mlw_navd = 1.272 - 2.040

    # --- plot ---
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.plot(if_noaa * 100, elevations,
            color="#1f77b4", lw=2.0,
            label="(a) NOAA published harmonics — 8662245 (Oyster Landing)")
    ax.plot(if_fit * 100,  elevations,
            color="#ff7f0e", lw=2.0, ls="--",
            label="(b) Fitted harmonics — 8662245 obs. record")
    ax.plot(if_raw * 100,  elevations,
            color="#2ca02c", lw=1.5, ls=":",
            label="(c) Direct count — 8662245 raw 6-min record")
    ax.plot(if_sp * 100,   elevations,
            color="#d62728", lw=2.0, ls="-.",
            label="(d) Direct count — 8661070 (Springmaid Pier)")

    # datum reference lines (8662245 NAVD88 datums)
    for elev, label, ls in [
        (mhw_navd, "MHW (NI)",  "-."),
        (msl_navd, "MSL (NI)",  "-"),
        (mlw_navd, "MLW (NI)",  "-."),
    ]:
        ax.axhline(elev, color="0.5", lw=0.8, ls=ls)
        ax.text(101.5, elev, label, va="center", ha="left",
                fontsize=8, color="0.4")

    ax.set_xlabel("Inundation fraction (%)", fontsize=11)
    ax.set_ylabel("Elevation (m NAVD88)", fontsize=11)
    ax.set_xlim(-2, 113)           # room for datum labels on right
    ax.set_ylim(_ELEV_MIN, _ELEV_MAX)   # higher elevation at top
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title(
        "North Inlet, SC — inundation fraction vs elevation\n"
        "Apr–Sep 2021  |  NOAA 8662245 (Oyster Landing) vs 8661070 (Springmaid Pier)",
        fontsize=10
    )
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(_OUT_FIG, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved → {_OUT_FIG}")


if __name__ == "__main__":
    main()
