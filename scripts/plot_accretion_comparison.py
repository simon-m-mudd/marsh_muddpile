"""
plot_accretion_comparison.py

Combined figure comparing marsh sediment accretion rates derived from four
geochronological methods:

  ²¹⁰Pb CIC   — Constant Initial Concentration, filtered to R² ≥ R2_MIN
  ²¹⁰Pb CFCS  — Constant Flux Constant Sedimentation, filtered to R² ≥ R2_MIN
  ²¹⁰Pb CRS   — Constant Rate of Supply (no R² filter; CRS r² is always high)
  ¹³⁷Cs       — 1963 weapons-test peak (distinct-peak cores only)

Surface Elevation Table (SET) measurements from Saintilan et al. 2022 (Science,
doi:10.1126/science.abo7872) are overlaid in two ways:
  (a) as a separate distribution using the marker-horizon (MH) accretion rate
      column ("accretion"), which measures the same physical quantity as the
      core methods;
  (b) as a point-by-point comparison for cores whose matched SET instrument
      is within MATCH_RADIUS_M metres.

Note on SET vs accretion:
  The "elevation.rate" column in the SET file records the net surface elevation
  change from SET readings (elevation gain minus shallow compaction and
  subsidence). This is systematically lower than the sediment accretion rate
  and is NOT used in the primary comparison.  The "accretion" column records
  the surface sediment accretion rate from a feldspar marker horizon deployed
  alongside the SET, which directly parallels what core-based methods measure.
  For sites where no MH accretion is available, elevation.rate is shown in a
  separate track labelled "SET (elev.)".

Data availability note (2025):
  The Saintilan et al. 2022 dataset (data_S1) was the most comprehensive
  global SET compilation available at time of writing.  Osland et al. 2022
  (Environ. Challenges, doi:10.1016/j.envc.2022.100593) presents a more
  recent global synthesis but does not release a single downloadable table.
  Updated US Pacific Coast SET-MH data are archived on USGS ScienceBase
  (sciencebase.gov/catalog/item/56ec2b7ee4b0f59b85da144b).

R² threshold:
  R2_MIN = 0.70 is applied to CIC and CFCS.  Profiles with a shallower slope
  than the ²¹⁰Pb decay length will have poor fits regardless of data quality,
  so a goodness-of-fit filter is essential for these exponential models.
  CRS always yields a high r² (median 0.95 in this dataset) because it
  computes a model-consistent age at every layer; no R² filter is applied.

Usage:
    python3 plot_accretion_comparison.py
    python3 plot_accretion_comparison.py --r2-min 0.8 --match-radius 100
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pyproj import Geod
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

R2_MIN         = 0.70   # R² threshold for CIC and CFCS; documented in module docstring
MATCH_RADIUS_M = 100.0  # metres; SET instruments within this distance are compared
RATE_CLIP_MM   = 15.0   # clip for violin/box plots (mm/yr); extreme outliers omitted

# Colour palette (colourblind-friendly)
COLOURS = {
    "CIC":        "#1b7837",
    "CFCS":       "#762a83",
    "CRS":        "#1f78b4",
    "Cs137":      "#e08214",
    "SET_acc":    "#d73027",
    "SET_elev":   "#fc8d59",
}

_CCN_DIR  = Path(__file__).parent.parent / "core_data" / "CCN_synthesis"
_FIG_DIR  = Path(__file__).parent.parent / "scripts" / "figures"
_SET_FILE = Path(__file__).parent.parent / "core_data" / "saintilan2022_SET.csv"
_PB_CSV   = _FIG_DIR / "pb210_accretion_rates.csv"
_CS_CSV   = _FIG_DIR / "cs137_accretion_rates.csv"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_rates(r2_min: float) -> dict[str, pd.DataFrame]:
    """Return dict of tidy DataFrames, each with columns rate_mm_yr, rate_se_mm_yr,
    core_id, latitude, longitude (and source column)."""

    pb = pd.read_csv(_PB_CSV)
    cs = pd.read_csv(_CS_CSV)

    def _pb_method(model: str, r2_filter: bool) -> pd.DataFrame:
        flag_ok = pb[f"{model}_flag"].str.startswith("ok").fillna(False)
        r2_ok   = (pb[f"{model}_r2"] >= r2_min) if r2_filter else pd.Series(True, index=pb.index)
        sel     = pb[flag_ok & r2_ok].copy()
        sel     = sel.rename(columns={
            f"{model}_r_mm_yr":    "rate_mm_yr",
            f"{model}_r_se_mm_yr": "rate_se_mm_yr",
            f"{model}_r2":         "r2",
        })
        sel["source"] = model.upper()
        return sel[["core_id","latitude","longitude","rate_mm_yr","rate_se_mm_yr","r2","source"]].dropna(subset=["rate_mm_yr"])

    cic  = _pb_method("cic",  r2_filter=True)
    cfcs = _pb_method("cfcs", r2_filter=True)
    crs  = _pb_method("crs",  r2_filter=False)

    cs_ok = cs[cs["distinct_peak"] == True].copy()
    cs_ok = cs_ok.rename(columns={"accretion_mm_yr": "rate_mm_yr"})
    cs_ok["rate_se_mm_yr"] = np.nan
    cs_ok["r2"]            = np.nan
    cs_ok["source"]        = "Cs137"
    cs_ok = cs_ok[["core_id","latitude","longitude","rate_mm_yr","rate_se_mm_yr","r2","source"]].dropna(subset=["rate_mm_yr"])

    return {"CIC": cic, "CFCS": cfcs, "CRS": crs, "Cs137": cs_ok}


def load_set() -> pd.DataFrame:
    """Load SET/MH data from Saintilan et al. 2022 supplementary table."""
    df = pd.read_csv(_SET_FILE)
    df = df.dropna(subset=["latitude","longitude"])
    return df


def match_set_to_cores(core_df: pd.DataFrame,
                       set_df:  pd.DataFrame,
                       radius_m: float) -> pd.DataFrame:
    """Find SET measurements within radius_m of each core.

    Returns DataFrame with core_id, SET identifier, distance_m,
    set_elevation_rate_mm_yr, set_accretion_mm_yr.
    """
    geod   = Geod(ellps="WGS84")
    pairs  = []

    for _, core in core_df.iterrows():
        clat, clon = core["latitude"], core["longitude"]
        _, _, dist = geod.inv(
            [clon] * len(set_df), [clat] * len(set_df),
            set_df["longitude"].values,
            set_df["latitude"].values,
        )
        dist_m = np.abs(dist)
        near   = np.where(dist_m <= radius_m)[0]
        for idx in near:
            row = set_df.iloc[idx]
            pairs.append({
                "core_id":              core["core_id"],
                "core_source":          core["source"],
                "core_rate_mm_yr":      core["rate_mm_yr"],
                "core_rate_se_mm_yr":   core.get("rate_se_mm_yr", np.nan),
                "set_id":               row.get("site.SET.identifier", ""),
                "distance_m":           dist_m[idx],
                "set_elevation_mm_yr":  row.get("elevation.rate", np.nan),
                "set_accretion_mm_yr":  row.get("accretion", np.nan),
                "set_years":            row.get("yearsNew", np.nan),
            })

    return pd.DataFrame(pairs) if pairs else pd.DataFrame(columns=[
        "core_id","core_source","core_rate_mm_yr","core_rate_se_mm_yr",
        "set_id","distance_m","set_elevation_mm_yr","set_accretion_mm_yr","set_years"])


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _violin_data(rates: dict[str, pd.DataFrame],
                 set_df: pd.DataFrame,
                 clip: float) -> tuple[list, list, list]:
    """Return (labels, data_arrays, n_counts) for violin plot."""
    labels = []
    arrays = []
    counts = []

    for key in ("CIC", "CFCS", "CRS", "Cs137"):
        vals = rates[key]["rate_mm_yr"].clip(0, clip).dropna().values
        labels.append(f"{key}\n(n={len(vals)})")
        arrays.append(vals)
        counts.append(len(vals))

    # SET marker-horizon accretion
    set_acc = set_df["accretion"].dropna().clip(0, clip).values
    labels.append(f"SET (MH)\n(n={len(set_acc)})")
    arrays.append(set_acc)
    counts.append(len(set_acc))

    # SET surface elevation change
    set_elev = set_df["elevation.rate"].clip(0, clip).dropna().values
    labels.append(f"SET (elev.)\n(n={len(set_elev)})")
    arrays.append(set_elev)
    counts.append(len(set_elev))

    return labels, arrays, counts


def _stats_line(ax, arr, colour, x_pos, clip):
    med = np.median(arr)
    p25, p75 = np.percentile(arr, [25, 75])
    ax.hlines(med, x_pos - 0.3, x_pos + 0.3, colors=colour, lw=1.8, zorder=5)
    ax.vlines(x_pos, p25, p75, colors=colour, lw=4.0, alpha=0.6, zorder=4)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def make_figure(rates: dict[str, pd.DataFrame],
                set_df: pd.DataFrame,
                matches: pd.DataFrame,
                r2_min: float,
                outdir: Path,
                rate_clip: float = RATE_CLIP_MM):

    outdir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 9))
    # Layout: left 55% violin, right 45% scatter panels (2 rows)
    gs  = fig.add_gridspec(2, 2, width_ratios=[1.1, 0.9], hspace=0.45, wspace=0.35)

    ax_violin = fig.add_subplot(gs[:, 0])     # full-height left panel
    ax_sc1    = fig.add_subplot(gs[0, 1])     # top-right: core vs SET accretion
    ax_sc2    = fig.add_subplot(gs[1, 1])     # bottom-right: core vs SET elev.rate

    method_colour_list = [
        COLOURS["CIC"], COLOURS["CFCS"], COLOURS["CRS"],
        COLOURS["Cs137"], COLOURS["SET_acc"], COLOURS["SET_elev"],
    ]

    # ── Violin / box plot ──────────────────────────────────────────────────
    labels, arrays, counts = _violin_data(rates, set_df, rate_clip)

    parts = ax_violin.violinplot(
        arrays, positions=range(len(arrays)),
        showmedians=False, showextrema=False,
        widths=0.65,
    )
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(method_colour_list[i])
        pc.set_alpha(0.55)
        pc.set_edgecolor(method_colour_list[i])
        pc.set_linewidth(0.8)

    for i, (arr, col) in enumerate(zip(arrays, method_colour_list)):
        if len(arr) == 0:
            continue
        med = np.median(arr)
        p25, p75 = np.percentile(arr, [25, 75])
        p5,  p95 = np.percentile(arr, [5,  95])
        # IQR bar
        ax_violin.vlines(i, p25, p75, colors=col, lw=5, alpha=0.75, zorder=4)
        # whisker (5–95th)
        ax_violin.vlines(i, p5,  p95, colors=col, lw=1.5, alpha=0.45, zorder=3)
        # median tick
        ax_violin.hlines(med, i - 0.28, i + 0.28, colors="white", lw=2.5, zorder=6)
        ax_violin.hlines(med, i - 0.28, i + 0.28, colors=col,     lw=1.5, zorder=7)
        # median text
        ax_violin.text(i, med + 0.15, f"{med:.1f}", ha="center", va="bottom",
                       fontsize=7, color=col, fontweight="bold")

    ax_violin.set_xticks(range(len(labels)))
    ax_violin.set_xticklabels(labels, fontsize=8)
    ax_violin.set_ylabel("Accretion / elevation rate  (mm yr⁻¹)", fontsize=9)
    ax_violin.set_ylim(0, rate_clip * 1.05)
    ax_violin.set_title(
        f"Marsh accretion rate distributions\n"
        f"CIC/CFCS filtered R² ≥ {r2_min}  |  ²¹⁰Pb CRS unfiltered  |  "
        f"¹³⁷Cs distinct-peak only\n"
        f"SET: Saintilan et al. 2022 (Science, doi:10.1126/science.abo7872)",
        fontsize=8.5, loc="left",
    )
    ax_violin.axhline(0, color="k", lw=0.5, ls="--", alpha=0.3)
    ax_violin.axvline(3.5, color="k", lw=0.6, ls=":", alpha=0.4)  # divide core vs SET
    ax_violin.text(4.5, rate_clip * 0.97, "← core methods    SET →",
                   ha="center", va="top", fontsize=7, color="#555555")

    # ── Scatter: core methods vs SET accretion (within MATCH_RADIUS_M m) ──
    _scatter_panel(ax_sc1, matches,
                   y_col="set_accretion_mm_yr",
                   y_label="SET marker-horizon accretion (mm yr⁻¹)",
                   title=f"Co-located cores vs SET accretion\n(within {MATCH_RADIUS_M:.0f} m, Saintilan 2022)",
                   clip=rate_clip)

    _scatter_panel(ax_sc2, matches,
                   y_col="set_elevation_mm_yr",
                   y_label="SET surface elevation change (mm yr⁻¹)",
                   title=f"Co-located cores vs SET elevation rate\n(within {MATCH_RADIUS_M:.0f} m)",
                   clip=rate_clip)

    # Legend for scatter panels
    handles = [
        mpatches.Patch(color=COLOURS[s], label=s)
        for s in ("CIC","CFCS","CRS","Cs137")
        if not matches[matches["core_source"]==s].empty
    ]
    ax_sc1.legend(handles=handles, fontsize=7, loc="upper left", framealpha=0.6)

    fig.text(0.02, 0.01,
             "Note: SET elevation.rate = surface elevation change (SET reading) — "
             "lower than accretion due to shallow compaction.\n"
             "SET accretion = co-located marker-horizon measurement, directly comparable to core methods.",
             fontsize=6.5, color="#555555")

    out_path = outdir / f"accretion_comparison_r2min{r2_min:.2f}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _scatter_panel(ax, matches: pd.DataFrame, y_col: str,
                   y_label: str, title: str, clip: float):
    """Scatter plot: x = core rate, y = SET rate, one colour per method."""
    if matches.empty:
        ax.text(0.5, 0.5, "No co-located pairs", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title, fontsize=8)
        return

    plotted = False
    for method in ("CIC", "CFCS", "CRS", "Cs137"):
        sub = matches[matches["core_source"] == method].copy()
        sub = sub.dropna(subset=["core_rate_mm_yr", y_col])
        sub["core_rate_mm_yr"] = sub["core_rate_mm_yr"].clip(0, clip)
        sub[y_col] = sub[y_col].clip(0, clip)
        if sub.empty:
            continue
        xe = sub["core_rate_se_mm_yr"].fillna(0).clip(0, clip / 2)
        ax.errorbar(
            sub["core_rate_mm_yr"], sub[y_col],
            xerr=xe,
            fmt="o", ms=5, lw=0, elinewidth=0.7, capsize=2,
            color=COLOURS[method], label=method, alpha=0.85, zorder=3,
        )
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, f"No data for {y_col}", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title, fontsize=8)
        return

    lim = clip * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, zorder=1, label="1:1")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Core-derived accretion (mm yr⁻¹)", fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    ax.set_title(title, fontsize=8)

    # Pearson r (all methods pooled)
    all_x = matches["core_rate_mm_yr"].clip(0, clip).dropna()
    all_y = matches[y_col].clip(0, clip).dropna()
    common = matches[["core_rate_mm_yr", y_col]].dropna()
    common["core_rate_mm_yr"] = common["core_rate_mm_yr"].clip(0, clip)
    common[y_col] = common[y_col].clip(0, clip)
    if len(common) >= 3:
        r, p = stats.pearsonr(common["core_rate_mm_yr"], common[y_col])
        ax.text(0.97, 0.05, f"r = {r:.2f}  n = {len(common)}",
                ha="right", va="bottom", transform=ax.transAxes, fontsize=7)


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def save_summary(rates: dict, matches: pd.DataFrame, outdir: Path):
    """Write a tidy CSV of all core rates plus matched SET values."""
    all_rates = pd.concat(list(rates.values()), ignore_index=True)

    # Attach best SET accretion match per core (smallest distance)
    if not matches.empty:
        best_set = (
            matches.sort_values("distance_m")
            .groupby("core_id")
            .first()
            .reset_index()[["core_id","set_id","distance_m",
                            "set_accretion_mm_yr","set_elevation_mm_yr"]]
        )
        all_rates = all_rates.merge(best_set, on="core_id", how="left")

    out = outdir / "accretion_all_methods.csv"
    all_rates.to_csv(out, index=False)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot combined accretion rate comparison (²¹⁰Pb, ¹³⁷Cs, SET)."
    )
    parser.add_argument("--r2-min",       type=float, default=R2_MIN,
                        help=f"R² threshold for CIC/CFCS (default: {R2_MIN})")
    parser.add_argument("--match-radius", type=float, default=MATCH_RADIUS_M,
                        help=f"SET match radius in metres (default: {MATCH_RADIUS_M})")
    parser.add_argument("--rate-clip",    type=float, default=RATE_CLIP_MM,
                        help=f"Clip rates at this value for plots (default: {RATE_CLIP_MM})")
    parser.add_argument("--outdir", default=str(_FIG_DIR))
    args = parser.parse_args()

    rates   = load_rates(args.r2_min)
    set_df  = load_set()

    # Build one combined core table for matching
    all_cores = pd.concat(list(rates.values()), ignore_index=True).dropna(subset=["latitude","longitude"])
    matches   = match_set_to_cores(all_cores, set_df, args.match_radius)

    print(f"\nMethod counts after filters (R² ≥ {args.r2_min} for CIC/CFCS):")
    for k, v in rates.items():
        print(f"  {k:6s}: {len(v)} cores")
    print(f"\nSET records: {len(set_df)}")
    print(f"Core-SET pairs within {args.match_radius:.0f} m: {len(matches)}  "
          f"({matches['core_id'].nunique() if not matches.empty else 0} unique cores)")

    make_figure(rates, set_df, matches, args.r2_min, Path(args.outdir),
                rate_clip=args.rate_clip)
    save_summary(rates, matches, Path(args.outdir))


if __name__ == "__main__":
    main()
