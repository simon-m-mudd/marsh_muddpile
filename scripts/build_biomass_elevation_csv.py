"""
Build scripts/lter_plot_peak_biomass.csv — unified table of peak aboveground
biomass and elevation for:
  - NI LTER, South Carolina (edi.135.12 + edi.134.12)
  - Louisiana (LUMCON/Roberts & Hill: R1.x139.143.0054 + R4.x264.000.0007)
  - GCE LTER, Georgia (knb-lter-gce.759.27 — Altamaha River transition sites)
  - PIE LTER, Massachusetts (knb-lter-pie.624.2 — RTK biomass+elevation quadrats)

Columns:
  site_id, dataset, region, lat, lon, species,
  location_type, n_years,
  median_peak_agb_g_m2, min_peak_agb_g_m2, max_peak_agb_g_m2,
  epqs_elevation_m, rtk_elevation_m, rtk_datum
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests

_THIS_DIR = Path(__file__).parent
_LTER_DIR = _THIS_DIR.parent / "lter_data"
_OUT_CSV  = _THIS_DIR / "lter_plot_peak_biomass.csv"
_EPQS_URL = "https://epqs.nationalmap.gov/v1/json"

# Pre-queried EPQS values for Louisiana (avoid repeated API calls)
_LA_EPQS = {
    "LUM1": 0.4000, "LUM2": 0.2717, "LUM3": 0.3253,
    "TB1": -0.7000, "TB2": -0.7000, "TB3": -0.7000, "TB4": -0.7000,
}


def _epqs(lat, lon):
    r = requests.get(
        _EPQS_URL,
        params={"x": lon, "y": lat, "wkid": 4326,
                "units": "Meters", "includeDate": "false"},
        timeout=10,
    )
    r.raise_for_status()
    val = float(r.json()["value"])
    return val if val > -1e5 else np.nan


# ── NI LTER ───────────────────────────────────────────────────────────────────

def _load_ni():
    bio = pd.read_csv(
        _LTER_DIR / "edi.135.12/NILTREB_plants_aboveground_biomass_density.csv"
    )
    bio["DATE"] = pd.to_datetime(bio["DATE"])
    bio["Month"] = bio["DATE"].dt.month
    bio["Year"]  = bio["DATE"].dt.year
    bio["ABOVEGROUND_BIOMASS"] = pd.to_numeric(
        bio["ABOVEGROUND_BIOMASS"], errors="coerce"
    )
    ctrl = bio[(bio["Month"].between(7, 8)) & (bio["TREATMENT"] == "C")]

    # Annual peak per site × location × plot
    annual = (
        ctrl.groupby(["SITE", "LOCATION", "PLOT", "Year"])["ABOVEGROUND_BIOMASS"]
        .max()
        .reset_index()
    )

    # Coordinates from edi.134.12
    elev_df = pd.read_csv(
        _LTER_DIR / "edi.134.12/NILTREB_marsh_surface_elevation_init.csv"
    )
    coords = (
        elev_df.groupby(["SITE", "LOCATION", "PLOT"])
        .agg(lat=("LATITUDE", "first"), lon=("LONGITUDE", "first"),
             rtk_elevation_m=("MARSH_SURFACE_ELEVATION", "mean"))
        .reset_index()
    )

    # Site-level coords for GI/OL (plots without exact elevation match)
    site_coords = {
        ("GI", "HM"): (33.331647, -79.198201),
        ("GI", "LM"): (33.331592, -79.197927),
        ("OL", "HM"): (33.35139,  -79.19194),
        ("OL", "LM"): (33.34810,  -79.18917),
    }

    rows = []
    for (site, loc, plot), grp in annual.groupby(["SITE", "LOCATION", "PLOT"]):
        peaks = grp["ABOVEGROUND_BIOMASS"].dropna()
        if peaks.empty:
            continue

        # Elevation
        c = coords[(coords["SITE"] == site) & (coords["LOCATION"] == loc)
                   & (coords["PLOT"] == plot)]
        if not c.empty:
            lat = float(c["lat"].iloc[0])
            lon = float(c["lon"].iloc[0])
            rtk = float(c["rtk_elevation_m"].iloc[0])
        elif (site, loc) in site_coords:
            lat, lon = site_coords[(site, loc)]
            rtk = np.nan
        else:
            continue

        rows.append(dict(
            site_id=f"NI-{site}-{loc}-P{plot}",
            dataset="NI_LTER_edi135",
            region="South Carolina",
            lat=lat, lon=lon,
            species="Spartina alterniflora",
            location_type=loc,
            n_years=int(peaks.count()),
            median_peak_agb_g_m2=round(float(peaks.median()), 1),
            min_peak_agb_g_m2=round(float(peaks.min()), 1),
            max_peak_agb_g_m2=round(float(peaks.max()), 1),
            epqs_elevation_m=np.nan,   # filled below
            rtk_elevation_m=round(rtk, 4) if not np.isnan(rtk) else np.nan,
            rtk_datum="NAVD88",
        ))
    return pd.DataFrame(rows)


# ── Louisiana ─────────────────────────────────────────────────────────────────

def _parse_la_excel(path, agb_col):
    df = pd.read_excel(path, header=[0, 1])
    df.columns = [
        " ".join(str(c).strip() for c in col if "Unnamed" not in str(c)).strip()
        for col in df.columns
    ]
    df = df.rename(columns={
        "Latitude        (decimal degrees)": "Latitude",
        "Latitude        (decimal degrees) ": "Latitude",
        "Longitude              (decimal degrees)": "Longitude",
        "Longitude         (decimal degrees)": "Longitude",
        agb_col: "AGB_g_m2",
    })
    df["AGB_g_m2"] = pd.to_numeric(df["AGB_g_m2"], errors="coerce")
    return df


def _load_louisiana():
    r1 = _parse_la_excel(
        _LTER_DIR / "R1.x139.143.0054"
        / "R1.x139.143-0054_Plant data and ancillary variables_2013-2014_sub3.xlsx",
        "Plant characteristics Live aboveground            biomass           (g/m2)",
    )
    r4 = _parse_la_excel(
        _LTER_DIR / "R4.x264.000.0007"
        / "R4.x264.000-0007_Roberts and Hill plant biomass 2015_sub2.xlsx",
        "Plant characteristics Live aboveground biomass (g/m2)",
    )
    combined = pd.concat([r1, r4], ignore_index=True)

    # Coords per site
    coords = (
        combined.groupby("Site")[["Latitude", "Longitude"]]
        .first()
        .reset_index()
    )

    # Annual peak = max monthly site mean (avg A/B/C plots) per year
    site_month = (
        combined.groupby(["Site", "Year", "Month-number"])["AGB_g_m2"]
        .mean()
        .reset_index()
    )
    annual_peak = (
        site_month.groupby(["Site", "Year"])["AGB_g_m2"]
        .max()
        .reset_index()
        .rename(columns={"AGB_g_m2": "annual_peak"})
    )

    rows = []
    for site, grp in annual_peak.groupby("Site"):
        peaks = grp["annual_peak"].dropna()
        coord = coords[coords["Site"] == site].iloc[0]
        lat = float(coord["Latitude"])
        lon = float(coord["Longitude"])
        rows.append(dict(
            site_id=f"LA-{site}",
            dataset="LUMCON_Roberts_Hill",
            region="Louisiana",
            lat=lat, lon=lon,
            species="Spartina alterniflora",
            location_type="LM",      # coastal Louisiana = low marsh S. alt.
            n_years=int(peaks.count()),
            median_peak_agb_g_m2=round(float(peaks.median()), 1),
            min_peak_agb_g_m2=round(float(peaks.min()), 1),
            max_peak_agb_g_m2=round(float(peaks.max()), 1),
            epqs_elevation_m=_LA_EPQS.get(site, np.nan),
            rtk_elevation_m=np.nan,
            rtk_datum="",
        ))
    return pd.DataFrame(rows)


# ── GCE LTER (Georgia) ───────────────────────────────────────────────────────

# Approximate centroids for Altamaha River transition sites (from plot script)
_GCE_COORDS = {
    "SCSA": (31.323, -81.374),
    "ZSC1": (31.328, -81.451),
    "ZSC2": (31.341, -81.454),
}


def _load_gce():
    """Load annual aboveground biomass from GCE Altamaha River transition sites.

    Source: PLT-GCES-1609c — annual biomass stats sampled in October.
    Total_Plant_Biomass_m2 is the sum over all plants within each 0.25 m²
    quadrat, scaled to g m⁻².  October is post-peak for Georgia; treat as
    annual biomass proxy (no July–August filter possible here).
    Species: Spartina alterniflora (zone 1/SCSA) and S. cynosuroides / mixed
    (ZSC1/ZSC2 transition sites).
    """
    df = pd.read_csv(
        _LTER_DIR / "knb-lter-gce.759.27/PLT-GCES-1609c_Biomass_Stats_5_0.CSV",
        skiprows=2, header=0,
    )
    df = df.iloc[2:].reset_index(drop=True)
    df.columns = [
        "Year", "Site", "Zone", "Plot", "Quadrat_Area", "Plot_Disturbance",
        "Pct_Calc", "Num_plants", "Min_biomass", "Max_biomass", "Total_biomass",
        "Mean_biomass", "SD_biomass", "SE_biomass",
    ]
    df["Total_biomass"] = pd.to_numeric(df["Total_biomass"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    # Annual peak per site × year (max quadrat value in that year)
    annual = (
        df.dropna(subset=["Total_biomass", "Year"])
        .groupby(["Site", "Year"])["Total_biomass"]
        .max()
        .reset_index()
    )

    rows = []
    for site, grp in annual.groupby("Site"):
        peaks = grp["Total_biomass"].dropna()
        if peaks.empty or site not in _GCE_COORDS:
            continue
        lat, lon = _GCE_COORDS[site]
        species = "Spartina alterniflora" if site == "SCSA" else "Spartina alterniflora / cynosuroides"
        rows.append(dict(
            site_id=f"GCE-{site}",
            dataset="GCE_LTER_gce759",
            region="Georgia",
            lat=lat, lon=lon,
            species=species,
            location_type="LM",        # creek-bank / low marsh dominant
            n_years=int(peaks.count()),
            median_peak_agb_g_m2=round(float(peaks.median()), 1),
            min_peak_agb_g_m2=round(float(peaks.min()), 1),
            max_peak_agb_g_m2=round(float(peaks.max()), 1),
            epqs_elevation_m=np.nan,
            rtk_elevation_m=np.nan,
            rtk_datum="",
        ))
    return pd.DataFrame(rows)


# ── PIE LTER (Massachusetts) ──────────────────────────────────────────────────

def _load_pie():
    """Load peak-summer aboveground biomass from PIE LTER Rowley marshes.

    Source: knb-lter-pie.624.2 — July quadrat harvests at 7 sites, 2018–2024.
    Elevation column is RTK GPS NAVD88 (m).  Each row is one 0.053 m² quadrat.
    Aggregate to site × year means, then summarise across years.
    """
    df = pd.read_csv(
        _LTER_DIR / "knb-lter-pie.624.2/MAR-RO-ST-Biomass_V.03.csv"
    )
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Biomass"] = pd.to_numeric(df["Biomass"], errors="coerce")
    df["Elevation"] = pd.to_numeric(df["Elevation"], errors="coerce")

    rows = []
    for site, grp in df.groupby("Site"):
        # Annual peak = max quadrat biomass within a site × year
        annual = grp.groupby("Year")["Biomass"].max().dropna()
        if annual.empty:
            continue
        lat = float(grp["Latitude"].mean())
        lon = float(grp["Longitude"].mean())
        rtk_elev = float(grp["Elevation"].mean())
        rows.append(dict(
            site_id=f"PIE-{site.replace(' ', '_')}",
            dataset="PIE_LTER_pie624",
            region="Massachusetts",
            lat=lat, lon=lon,
            species="Spartina alterniflora",
            location_type="LM",
            n_years=int(annual.count()),
            median_peak_agb_g_m2=round(float(annual.median()), 1),
            min_peak_agb_g_m2=round(float(annual.min()), 1),
            max_peak_agb_g_m2=round(float(annual.max()), 1),
            epqs_elevation_m=np.nan,
            rtk_elevation_m=round(rtk_elev, 4),
            rtk_datum="NAVD88",
        ))
    return pd.DataFrame(rows)


# ── fill EPQS for NI rows ─────────────────────────────────────────────────────

def _fill_epqs(df):
    # Load existing EPQS cache
    cache_path = _THIS_DIR / "lter_plot_elevations_epqs.csv"
    if cache_path.exists():
        cache = pd.read_csv(cache_path).set_index("site_id")["epqs_elevation_m"].to_dict()
    else:
        cache = {}

    queried = {}
    for i, row in df.iterrows():
        sid = row["site_id"]
        if not np.isnan(row["epqs_elevation_m"]):
            continue                          # already set (Louisiana)
        if sid in cache:
            df.at[i, "epqs_elevation_m"] = cache[sid]
        elif sid in queried:
            df.at[i, "epqs_elevation_m"] = queried[sid]
        else:
            print(f"  Querying EPQS for {sid} …", end=" ", flush=True)
            val = _epqs(row["lat"], row["lon"])
            print(f"{val:.3f} m")
            queried[sid] = val
            df.at[i, "epqs_elevation_m"] = val
            time.sleep(0.25)
    return df


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading NI LTER data …")
    ni  = _load_ni()
    print(f"  {len(ni)} NI plots")

    print("Loading Louisiana data …")
    la  = _load_louisiana()
    print(f"  {len(la)} Louisiana sites")

    print("Loading GCE LTER data …")
    gce = _load_gce()
    print(f"  {len(gce)} GCE sites")

    print("Loading PIE LTER data …")
    pie = _load_pie()
    print(f"  {len(pie)} PIE sites")

    df = pd.concat([ni, la, gce, pie], ignore_index=True)

    print("Filling EPQS elevations …")
    df = _fill_epqs(df)

    df = df.round({"epqs_elevation_m": 4, "rtk_elevation_m": 4})
    df.to_csv(_OUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows → {_OUT_CSV}")
    print()
    print(df[["site_id", "region", "n_years",
              "median_peak_agb_g_m2", "min_peak_agb_g_m2", "max_peak_agb_g_m2",
              "epqs_elevation_m", "rtk_elevation_m"]].to_string(index=False))
    return df


if __name__ == "__main__":
    main()
