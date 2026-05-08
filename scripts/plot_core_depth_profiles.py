"""
plot_core_depth_profiles.py

Find sediment cores near a lat/lon and plot depth profiles for requested
attributes. Depth 0 is at the surface (top of core).

Usage:
    python plot_core_depth_profiles.py --lat 51.5 --lon -0.1 --radius 50 \
        --attrs dry_bulk_density fraction_organic_matter excess_pb210_activity

    python plot_core_depth_profiles.py --lat 51.5 --lon -0.1 --radius 50 \
        --attrs cs137_activity total_pb210_activity --outdir figures/

Available attributes (depth-varying):
    dry_bulk_density, fraction_organic_matter, fraction_carbon,
    excess_pb210_activity, total_pb210_activity, cs137_activity,
    ra226_activity, pb214_activity, bi214_activity,
    be7_activity, c14_age, delta_c13, age

For north inlet:
    python3 query_cores_by_location.py --lat 33.331659 --lon -79.198208
    python3 plot_core_depth_profiles.py --lat 33.331659 --lon -79.198208 \ 
        --attrs fraction_organic_matter excess_pb210_activity --outdir figures

for GIA
    python3 query_cores_by_location.py --lat 31.342025 --lon -81.371261
    python3 plot_core_depth_profiles.py --lat 31.342025 --lon -81.371261 --attrs fraction_organic_matter --outdir figures
                   
               
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


def load_data(fgb_path: str) -> gpd.GeoDataFrame:
    path = Path(fgb_path)
    if not path.exists():
        sys.exit(f"Error: file not found: {fgb_path}")
    return gpd.read_file(fgb_path)


def find_cores_in_radius(
    gdf: gpd.GeoDataFrame, lat: float, lon: float, radius_km: float
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
    return result


def get_depth_midpoint(grp: pd.DataFrame) -> pd.Series:
    """Midpoint depth for each sample row (cm from surface)."""
    return (grp["depth_min"] + grp["depth_max"]) / 2.0


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


def core_label(core_id: str, grp: pd.DataFrame, dist_km: float) -> str:
    """
    Build a multi-line legend label:
      core_id (dist km)
      species | elev X.XX m DATUM | SLA X.X
    """
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

    sla = row.get("SLA Trend")
    sla_str = f"SLA {_fmt_2sig(sla)} mm/yr"

    meta = " | ".join(filter(None, [species, elev_str, sla_str]))
    return f"{core_id} ({dist_km:.0f} km)\n{meta}" if meta else f"{core_id} ({dist_km:.0f} km)"


def get_axis_label(attr: str, units_present: set) -> str:
    label, _, _ = ATTR_CONFIG.get(attr, (attr, False, None))
    unit_col = UNIT_COLS.get(attr)
    if unit_col and units_present:
        unit_str = " / ".join(sorted(units_present))
        label = f"{label} ({unit_str})"
    return label


def plot_profiles_for_attribute(
    attr: str,
    core_data: gpd.GeoDataFrame,
    ax: plt.Axes,
    max_cores: int = 20,
):
    """Plot depth profiles for one attribute across all nearby cores."""
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

    # Sort by distance so closest cores get the clearest colours
    dist_map = (
        core_data[["core_id", "distance_km"]]
        .drop_duplicates("core_id")
        .set_index("core_id")["distance_km"]
    )
    cores_with_data = sorted(cores_with_data, key=lambda c: dist_map.get(c, 9999))
    if len(cores_with_data) > max_cores:
        cores_with_data = cores_with_data[:max_cores]

    cmap = matplotlib.colormaps.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(cores_with_data))]

    units_seen: set = set()
    unit_col = UNIT_COLS.get(attr)

    for color, core_id in zip(colors, cores_with_data):
        grp = core_data[core_data["core_id"] == core_id].copy()
        grp = grp.sort_values("depth_min")
        dist = dist_map.get(core_id, 0)
        label = core_label(core_id, grp, dist)
        grp = grp.dropna(subset=[attr])
        depth = get_depth_midpoint(grp)
        values = grp[attr]

        if unit_col and unit_col in grp.columns:
            u = grp[unit_col].dropna().unique()
            units_seen.update(u)

        if has_err and err_col:
            errs = grp[err_col].fillna(0)
            ax.errorbar(
                values, depth,
                xerr=errs,
                fmt="o-", color=color, label=label,
                markersize=3, linewidth=0.8, elinewidth=0.6, capsize=2,
            )
        else:
            ax.plot(values, depth, "o-", color=color, label=label,
                    markersize=3, linewidth=0.8)

    xlabel = get_axis_label(attr, units_seen)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Depth (cm)")
    ax.invert_yaxis()
    ax.set_title(attr.replace("_", " "))
    ax.axhline(0, color="k", linewidth=0.4, linestyle="--")

    n_cores_shown = len(cores_with_data)
    ncol = 1 if n_cores_shown <= 6 else 2
    ax.legend(fontsize=5, loc="best", ncol=ncol, framealpha=0.5, labelspacing=0.6)


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


def main():
    parser = argparse.ArgumentParser(
        description="Plot sediment core depth profiles near a lat/lon."
    )
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument(
        "--radius", type=float, default=50.0,
        help="Search radius in km (default: 50)"
    )
    parser.add_argument(
        "--attrs", nargs="+",
        default=["dry_bulk_density", "fraction_organic_matter",
                 "excess_pb210_activity", "cs137_activity"],
        help="Attribute(s) to plot (space-separated). Run --list-attrs to see options.",
    )
    parser.add_argument(
        "--fgb",
        default=str(Path(__file__).parent.parent / "core_data" / "merged_all.fgb"),
        help="Path to merged_all.fgb",
    )
    parser.add_argument(
        "--outdir",
        default=str(Path(__file__).parent.parent / "viz" / "core_profiles"),
        help="Directory for output figures (default: ../viz/core_profiles/)",
    )
    parser.add_argument(
        "--max-cores", type=int, default=20,
        help="Maximum number of cores to plot per attribute (closest first, default: 20)",
    )
    parser.add_argument(
        "--list-attrs", action="store_true",
        help="Print all plottable attribute names and exit",
    )
    args = parser.parse_args()

    if args.list_attrs:
        list_available_attributes()
        return

    gdf = load_data(args.fgb)
    nearby = find_cores_in_radius(gdf, args.lat, args.lon, args.radius)

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
