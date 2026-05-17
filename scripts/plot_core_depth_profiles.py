"""
plot_core_depth_profiles.py

Find sediment cores near a lat/lon and plot depth profiles for requested
attributes. Depth 0 is at the surface (top of core).

By default loads from the CCN synthesis CSV files (CCN_cores.csv +
CCN_depthseries.csv).  Pass --fgb to use a pre-merged FlatGeobuf instead.

Usage:
    # North Inlet (default)
    python3 plot_core_depth_profiles.py

    # Custom location with CCN data
    python3 plot_core_depth_profiles.py --lat 31.342025 --lon -81.371261 \
        --attrs fraction_organic_matter excess_pb210_activity --outdir figures

    # Use a pre-merged FGB instead of CCN CSVs
    python3 plot_core_depth_profiles.py --fgb core_data/merged_all.fgb \
        --lat 33.33 --lon -79.20 --attrs fraction_organic_matter

Available attributes (depth-varying):
    dry_bulk_density, fraction_organic_matter, fraction_carbon,
    excess_pb210_activity, total_pb210_activity, cs137_activity,
    ra226_activity, pb214_activity, bi214_activity,
    be7_activity, c14_age, delta_c13, age
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
from pyproj import Geod

matplotlib.rcParams.update({"font.size": 9})

# Map attribute name -> (axis label, whether to draw error bars, error column)
ATTR_CONFIG = {
    "dry_bulk_density":        ("Dry bulk density (g cm⁻³)",         False, None),
    "fraction_organic_matter": ("Fraction organic matter",            False, None),
    "fraction_carbon":         ("Fraction carbon",                    False, None),
    "compaction_fraction":     ("Compaction fraction",                False, None),
    "excess_pb210_activity":   ("Excess ²¹⁰Pb activity",             True,  "excess_pb210_activity_se"),
    "total_pb210_activity":    ("Total ²¹⁰Pb activity",              True,  "total_pb210_activity_se"),
    "ra226_activity":          ("²²⁶Ra activity",                    True,  "ra226_activity_se"),
    "pb214_activity":          ("²¹⁴Pb activity",                    True,  "pb214_activity_se"),
    "bi214_activity":          ("²¹⁴Bi activity",                    True,  "bi214_activity_se"),
    "cs137_activity":          ("¹³⁷Cs activity",                    True,  "cs137_activity_se"),
    "be7_activity":            ("⁷Be activity",                      True,  "be7_activity_se"),
    "c14_age":                 ("¹⁴C age (yr BP)",                   True,  "c14_age_se"),
    "delta_c13":               ("δ¹³C (‰)",                         False, None),
    "age":                     ("Age (yr)",                           True,  "age_se"),
}

# Attributes whose units are stored in a companion column
UNIT_COLS = {
    "excess_pb210_activity": "pb210_unit",
    "total_pb210_activity":  "pb210_unit",
    "ra226_activity":        "ra226_unit",
    "pb214_activity":        "pb214_unit",
    "bi214_activity":        "bi214_unit",
    "cs137_activity":        "cs137_unit",
    "be7_activity":          "be7_unit",
}


def haversine_km(lat1, lon1, lat2, lon2):
    geod = Geod(ellps="WGS84")
    _, _, dist_m = geod.inv(
        np.full(len(lat2), lon1),
        np.full(len(lat2), lat1),
        lon2,
        lat2,
    )
    return np.abs(dist_m) / 1000.0


def load_fgb(fgb_path: str) -> gpd.GeoDataFrame:
    path = Path(fgb_path)
    if not path.exists():
        sys.exit(f"Error: file not found: {fgb_path}")
    return gpd.read_file(fgb_path)


_MARSH_VEGETATION_CLASSES  = {"emergent", "emergent marsh", "marsh"}
_EXCLUDED_SALINITY_CLASSES = {"fresh", "oligohaline"}


def load_ccn_data(cores_csv: str, depthseries_csv: str,
                  species_csv: str | None = None) -> gpd.GeoDataFrame:
    """Join CCN cores metadata + species with depthseries; return GeoDataFrame."""
    for p in (cores_csv, depthseries_csv):
        if not Path(p).exists():
            sys.exit(f"Error: CCN file not found: {p}")

    cores = pd.read_csv(cores_csv, low_memory=False)
    ds    = pd.read_csv(depthseries_csv, low_memory=False)

    # Aggregate per-core species into a JSON list string matching _parse_species format
    if species_csv and Path(species_csv).exists():
        import json
        sp = pd.read_csv(species_csv, low_memory=False)
        sp_agg = (
            sp.groupby("core_id")["species_code"]
            .apply(lambda x: json.dumps(sorted(x.dropna().unique().tolist())))
            .reset_index()
            .rename(columns={"species_code": "species_code_list"})
        )
        cores = cores.merge(sp_agg, on="core_id", how="left")

    core_cols = [
        "study_id", "site_id", "core_id",
        "latitude", "longitude",
        "elevation", "elevation_datum",
        "vegetation_class", "salinity_class",
        "habitat", "country", "admin_division",
        "year", "month", "species_code_list",
    ]
    core_cols = [c for c in core_cols if c in cores.columns]
    merged = ds.merge(cores[core_cols], on="core_id", how="left")

    gdf = gpd.GeoDataFrame(
        merged,
        geometry=gpd.points_from_xy(merged["longitude"], merged["latitude"]),
        crs="EPSG:4326",
    )
    return gdf


def find_cores_in_radius(
    gdf: gpd.GeoDataFrame, lat: float, lon: float, radius_km: float,
    marsh_only: bool = True,
) -> gpd.GeoDataFrame:
    core_locs = (
        gdf[["core_id", "latitude", "longitude"]]
        .drop_duplicates("core_id")
        .copy()
    )
    core_locs["distance_km"] = haversine_km(
        lat, lon,
        core_locs["latitude"].values,
        core_locs["longitude"].values,
    )
    nearby_ids = core_locs.loc[
        core_locs["distance_km"] <= radius_km, ["core_id", "distance_km"]
    ]
    result = gdf[gdf["core_id"].isin(nearby_ids["core_id"])].copy()
    result = result.merge(nearby_ids, on="core_id", how="left")

    if marsh_only and "vegetation_class" in result.columns:
        veg_ok = (
            result["vegetation_class"]
            .str.strip()
            .str.lower()
            .isin(_MARSH_VEGETATION_CLASSES)
        )
        result = result[veg_ok].copy()

    if marsh_only and "salinity_class" in result.columns:
        sal_col = result["salinity_class"].str.strip().str.lower()
        sal_ok  = sal_col.isna() | (~sal_col.isin(_EXCLUDED_SALINITY_CLASSES))
        result  = result[sal_ok].copy()

    return result


def get_depth_midpoint(grp: pd.DataFrame) -> pd.Series:
    """Midpoint depth for each sample row (cm from surface).

    If depth_min/depth_max are absent or NaN (e.g. bulk surface samples),
    falls back to 0 so the point is plotted at the surface rather than dropped.
    """
    d_min = grp["depth_min"].fillna(0) if "depth_min" in grp.columns else pd.Series(0, index=grp.index)
    d_max = grp["depth_max"].fillna(d_min) if "depth_max" in grp.columns else d_min
    return (d_min + d_max) / 2.0


def _parse_species(raw) -> str:
    """Turn '["Spartina alterniflora", "Distichlis spicata"]' into abbreviated names."""
    if pd.isna(raw) or str(raw).strip() == "":
        return ""
    import json
    try:
        names = json.loads(str(raw))
    except (json.JSONDecodeError, ValueError):
        # fallback: strip brackets and quotes
        names = [s.strip().strip('"') for s in str(raw).strip("[]").split(",")]
    abbrevs = []
    for n in names:
        n = n.strip()
        parts = n.split()
        if len(parts) >= 2:
            abbrevs.append(f"{parts[0][0]}. {parts[1]}")
        elif parts:
            abbrevs.append(parts[0])
    return " / ".join(abbrevs)


def _fmt_2sig(value) -> str:
    """Format a number to 2 significant figures."""
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.2g}"


def _is_spartina(core_id: str, core_data: gpd.GeoDataFrame) -> bool:
    """True if any species entry for this core contains 'Spartina'."""
    import json
    row = core_data[core_data["core_id"] == core_id].iloc[0]
    raw = row.get("species_code_list", None)
    if pd.isna(raw) or str(raw).strip() in ("", "[]"):
        return False
    try:
        names = json.loads(str(raw))
    except (json.JSONDecodeError, ValueError):
        names = [s.strip().strip('"') for s in str(raw).strip("[]").split(",")]
    return any("Spartina" in str(n) for n in names)


def core_label(core_id: str, grp: pd.DataFrame, dist_km: float) -> str:
    """Build a legend label: core_id (dist km) / species | elev"""
    row = grp.iloc[0]
    species = _parse_species(row.get("species_code_list"))
    elev = row.get("elevation")
    datum = row.get("elevation_datum")
    if pd.notna(elev):
        elev_str = f"{float(elev):.2f} m"
        if pd.notna(datum) and str(datum).strip():
            elev_str += f" {datum}"
    else:
        elev_str = "elev n/a"
    meta = " | ".join(filter(None, [species, elev_str]))
    return f"{core_id} ({dist_km:.1f} km)\n{meta}" if meta else f"{core_id} ({dist_km:.1f} km)"


def get_axis_label(attr: str, units_present: set) -> str:
    label, _, _ = ATTR_CONFIG.get(attr, (attr, False, None))
    unit_col = UNIT_COLS.get(attr)
    if unit_col and units_present:
        unit_str = " / ".join(sorted(units_present))
        label = f"{label} ({unit_str})"
    return label


_GREY_NON_SPARTINA = "#000000"
_SPARTINA_CMAP     = matplotlib.colormaps.get_cmap("tab10")


def plot_profiles_for_attribute(
    attr: str,
    core_data: gpd.GeoDataFrame,
    ax: plt.Axes,
    max_cores: int = 20,
):
    """Plot depth profiles for one attribute.

    Spartina alterniflora / Spartina cores are drawn in distinct tab10 colours.
    All other cores are drawn in light grey behind them.
    """
    _, has_err, err_col = ATTR_CONFIG.get(attr, (attr, False, None))
    if err_col and err_col not in core_data.columns:
        has_err = False

    cores_with_data = [
        cid for cid, grp in core_data.groupby("core_id")
        if grp[attr].notna().any()
    ]

    if not cores_with_data:
        ax.text(0.5, 0.5, f"No data for\n{attr}", ha="center", va="center",
                transform=ax.transAxes, color="grey")
        ax.set_title(attr)
        return

    dist_map = (
        core_data[["core_id", "distance_km"]]
        .drop_duplicates("core_id")
        .set_index("core_id")["distance_km"]
    )
    cores_with_data = sorted(cores_with_data, key=lambda c: dist_map.get(c, 9999))
    if len(cores_with_data) > max_cores:
        cores_with_data = cores_with_data[:max_cores]

    # Split into Spartina (coloured) and other (grey); grey drawn first
    spartina_ids = [c for c in cores_with_data if _is_spartina(c, core_data)]
    other_ids    = [c for c in cores_with_data if c not in spartina_ids]

    units_seen: set = set()
    unit_col = UNIT_COLS.get(attr)

    def _draw(core_id, color, zorder, alpha=1.0):
        grp = core_data[core_data["core_id"] == core_id].copy()
        grp = grp.sort_values("depth_min")
        dist = dist_map.get(core_id, 0)
        label = core_label(core_id, grp, dist)
        grp = grp.dropna(subset=[attr])
        depth = get_depth_midpoint(grp)
        values = grp[attr]

        if unit_col and unit_col in grp.columns:
            units_seen.update(grp[unit_col].dropna().unique())

        kw = dict(color=color, zorder=zorder, alpha=alpha)
        if has_err and err_col:
            errs = grp[err_col].fillna(0)
            ax.errorbar(values, depth, xerr=errs, fmt="o-", label=label,
                        markersize=3, linewidth=0.9, elinewidth=0.6, capsize=2, **kw)
        else:
            ax.plot(values, depth, "o-", label=label,
                    markersize=3, linewidth=0.9, **kw)

    # Grey non-Spartina behind
    for cid in other_ids:
        _draw(cid, color=_GREY_NON_SPARTINA, zorder=1, alpha=0.7)

    # Coloured Spartina on top
    for i, cid in enumerate(spartina_ids):
        _draw(cid, color=_SPARTINA_CMAP(i % 10), zorder=2)

    xlabel = get_axis_label(attr, units_seen)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Depth (cm)")
    ax.invert_yaxis()
    ax.set_title(attr.replace("_", " "))
    ax.axhline(0, color="k", linewidth=0.4, linestyle="--")

    ncol = 1 if len(cores_with_data) <= 6 else 2
    ax.legend(fontsize=6, loc="best", ncol=ncol, framealpha=0.5, labelspacing=0.5)


def plot_all_attributes(
    core_data: gpd.GeoDataFrame,
    attrs: list,
    lat: float,
    lon: float,
    radius_km: float,
    outdir: Path,
    max_cores: int,
):
    # One figure per attribute keeps things readable; also produce a combined figure
    outdir.mkdir(parents=True, exist_ok=True)
    n = len(attrs)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 5 * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for i, attr in enumerate(attrs):
        if attr not in core_data.columns:
            print(f"Warning: '{attr}' not found in data — skipping.")
            axes_flat[i].set_visible(False)
            continue
        plot_profiles_for_attribute(attr, core_data, axes_flat[i], max_cores=max_cores)

    # hide unused axes
    for j in range(len(attrs), len(axes_flat)):
        axes_flat[j].set_visible(False)

    n_cores = core_data["core_id"].nunique()
    fig.suptitle(
        f"Core depth profiles — centre ({lat}, {lon}), radius {radius_km} km\n"
        f"{n_cores} core(s) found",
        fontsize=10,
    )
    fig.tight_layout()

    out_path = outdir / f"depth_profiles_lat{lat}_lon{lon}_r{radius_km:.0f}km.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved combined figure: {out_path}")
    plt.close(fig)

    # Per-attribute figures
    for attr in attrs:
        if attr not in core_data.columns:
            continue
        fig_single, ax_single = plt.subplots(figsize=(6, 7))
        plot_profiles_for_attribute(attr, core_data, ax_single, max_cores=max_cores)
        ax_single.set_title(
            f"{attr.replace('_', ' ')}\n"
            f"centre ({lat}, {lon}), radius {radius_km} km",
            fontsize=9,
        )
        fig_single.tight_layout()
        per_path = outdir / f"depth_profile_{attr}_lat{lat}_lon{lon}_r{radius_km:.0f}km.png"
        fig_single.savefig(per_path, dpi=150, bbox_inches="tight")
        print(f"Saved {per_path}")
        plt.close(fig_single)


def list_available_attributes():
    print("Plottable attributes:")
    for attr, (label, has_err, err_col) in ATTR_CONFIG.items():
        err_info = f"  [±{err_col}]" if has_err and err_col else ""
        print(f"  {attr:<35} {label}{err_info}")


_CCN_DIR = Path(__file__).parent.parent / "core_data" / "CCN_synthesis"

# North Inlet, SC — NOAA gauge 8661070 lat/lon
_NI_LAT = 33.331659
_NI_LON = -79.198208


def main():
    parser = argparse.ArgumentParser(
        description="Plot sediment core depth profiles near a lat/lon."
    )
    parser.add_argument("--lat", type=float, default=_NI_LAT,
                        help=f"Centre latitude (default: {_NI_LAT}, North Inlet)")
    parser.add_argument("--lon", type=float, default=_NI_LON,
                        help=f"Centre longitude (default: {_NI_LON}, North Inlet)")
    parser.add_argument(
        "--radius", type=float, default=16.0,
        help="Search radius in km (default: 16)",
    )
    parser.add_argument(
        "--attrs", nargs="+",
        default=["fraction_organic_matter", "excess_pb210_activity",
                 "fraction_carbon", "dry_bulk_density"],
        help="Attribute(s) to plot. Run --list-attrs to see options.",
    )
    parser.add_argument(
        "--fgb", default=None,
        help="Use a pre-merged FlatGeobuf instead of CCN CSVs.",
    )
    parser.add_argument(
        "--ccn-cores",
        default=str(_CCN_DIR / "CCN_cores.csv"),
        help="CCN cores CSV (default: core_data/CCN_synthesis/CCN_cores.csv)",
    )
    parser.add_argument(
        "--ccn-depthseries",
        default=str(_CCN_DIR / "CCN_depthseries.csv"),
        help="CCN depthseries CSV (default: core_data/CCN_synthesis/CCN_depthseries.csv)",
    )
    parser.add_argument(
        "--ccn-species",
        default=str(_CCN_DIR / "CCN_species.csv"),
        help="CCN species CSV (default: core_data/CCN_synthesis/CCN_species.csv)",
    )
    parser.add_argument(
        "--outdir",
        default=str(Path(__file__).parent.parent / "viz" / "core_profiles"),
        help="Directory for output figures (default: viz/core_profiles/)",
    )
    parser.add_argument(
        "--max-cores", type=int, default=20,
        help="Maximum cores per attribute panel (default: 20)",
    )
    parser.add_argument(
        "--list-attrs", action="store_true",
        help="Print all plottable attribute names and exit",
    )
    args = parser.parse_args()

    if args.list_attrs:
        list_available_attributes()
        return

    if args.fgb:
        gdf = load_fgb(args.fgb)
    else:
        gdf = load_ccn_data(args.ccn_cores, args.ccn_depthseries, args.ccn_species)

    nearby = find_cores_in_radius(gdf, args.lat, args.lon, args.radius, marsh_only=True)

    if nearby.empty:
        print(f"No cores found within {args.radius} km of ({args.lat}, {args.lon}).")
        return

    n_cores = nearby["core_id"].nunique()
    print(f"Found {n_cores} core(s) within {args.radius} km of ({args.lat}, {args.lon}).")

    for attr in args.attrs:
        if attr not in ATTR_CONFIG:
            print(f"Warning: '{attr}' is not in the known attribute list — will attempt anyway.")

    plot_all_attributes(
        nearby,
        args.attrs,
        args.lat,
        args.lon,
        args.radius,
        Path(args.outdir),
        args.max_cores,
    )


if __name__ == "__main__":
    main()
