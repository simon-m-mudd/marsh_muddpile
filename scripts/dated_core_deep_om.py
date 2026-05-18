"""
dated_core_deep_om.py

For east-coast US marsh cores with chronologically constrained accretion rates
(¹³⁷Cs distinct peak or well-constrained CRS ²¹⁰Pb), check whether the core
also has a well-fit OM depth profile and a measurable deep organic fraction,
then plot deep_om_frac as a function of accretion rate.

Dating sources
--------------
¹³⁷Cs   scripts/figures/cs137_accretion_rates.csv
  Produced by cs137_accretion.py.  Cores with distinct_peak == True and
  peak_background_ratio ≥ MIN_CS_PBR.

CRS ²¹⁰Pb   scripts/figures/pb210_accretion_rates.csv
  Produced by pb210_accretion.py.  CRS model used (not CIC/CFCS) because it
  does not assume a constant flux or sedimentation rate.
  Quality filter: crs_flag == 'ok' and crs_r2 ≥ CRS_R2_MIN.

OM profile source
-----------------
scripts/core_inundation.fgb
  deep_om_frac  — OMinf from best-fit model; NaN if no valid fit or no data
                  below 30 cm
  best_model    — NaN if R² < 0.75 or < 6 layers

Definitions
-----------
well-fit     best_model is not NaN  (R² ≥ 0.75, ≥ 6 depth layers)
deep OM      deep_om_frac > DEEP_OM_THRESHOLD (default 0.02, i.e. 2 wt%)
east coast   lat 24–47 °N, lon -85–-60 °E

Output
------
  scripts/figures/dated_core_deep_om.png      — combined figure
  scripts/figures/dated_core_deep_om.csv      — merged table used for plotting
"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEEP_OM_THRESHOLD = 0.02   # minimum OMinf to count as deep organic
MIN_CS_PBR        = 5.0    # minimum peak-to-background ratio for ¹³⁷Cs
CRS_R2_MIN        = 0.75   # minimum R² for CRS ²¹⁰Pb model

LAT_LO, LAT_HI   = 24.0, 47.0
LON_LO, LON_HI   = -85.0, -60.0

_ROOT    = Path(__file__).parent.parent
_FGB     = _ROOT / "scripts" / "core_inundation.fgb"
_CS_CSV  = _ROOT / "scripts" / "figures" / "cs137_accretion_rates.csv"
_PB_CSV  = _ROOT / "scripts" / "figures" / "pb210_accretion_rates.csv"
_FIG_DIR = _ROOT / "scripts" / "figures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _east_coast(df, lat_col="latitude", lon_col="longitude"):
    return df[
        (df[lat_col] >= LAT_LO) & (df[lat_col] <= LAT_HI) &
        (df[lon_col] >= LON_LO) & (df[lon_col] <= LON_HI)
    ].copy()


def _load_om_index() -> pd.DataFrame:
    gdf = gpd.read_file(str(_FGB))
    ec  = _east_coast(gdf)
    return ec[["core_id", "best_model", "deep_om_frac",
               "doubling_depth_cm", "mean_om_above_dbl",
               "inund_frac", "inund_tuned",
               "latitude", "longitude"]].copy()


def _load_cs137() -> pd.DataFrame:
    df  = pd.read_csv(_CS_CSV)
    ec  = _east_coast(df)
    ec  = ec[ec["distinct_peak"] == True].copy()
    ec  = ec[ec["peak_background_ratio"] >= MIN_CS_PBR].copy()
    ec  = ec[ec["accretion_mm_yr"].notna() & (ec["accretion_mm_yr"] > 0)].copy()
    ec["dating_method"] = "137Cs"
    ec["accretion_mm_yr_se"] = np.nan    # cs137 gives single-point estimate
    return ec[["core_id", "dating_method", "accretion_mm_yr",
               "accretion_mm_yr_se", "latitude", "longitude"]]


def _load_pb210_crs() -> pd.DataFrame:
    df = pd.read_csv(_PB_CSV)
    ec = _east_coast(df)
    ec = ec[ec["crs_flag"] == "ok"].copy()
    ec = ec[ec["crs_r2"] >= CRS_R2_MIN].copy()
    ec = ec[ec["crs_r_mm_yr"].notna() & (ec["crs_r_mm_yr"] > 0)].copy()
    ec["dating_method"]      = "210Pb CRS"
    ec = ec.rename(columns={
        "crs_r_mm_yr":    "accretion_mm_yr",
        "crs_r_se_mm_yr": "accretion_mm_yr_se",
    })
    return ec[["core_id", "dating_method", "accretion_mm_yr",
               "accretion_mm_yr_se", "latitude", "longitude"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _FIG_DIR.mkdir(parents=True, exist_ok=True)

    om  = _load_om_index()
    cs  = _load_cs137()
    pb  = _load_pb210_crs()

    dated = pd.concat([cs, pb], ignore_index=True)

    # merge on core_id
    merged = dated.merge(
        om[["core_id", "best_model", "deep_om_frac",
            "doubling_depth_cm", "mean_om_above_dbl"]],
        on="core_id", how="left",
    )

    print(f"Dated cores (east coast): {len(dated)}")
    print(f"  137Cs distinct peak:   {(dated.dating_method == '137Cs').sum()}")
    print(f"  210Pb CRS ok:          {(dated.dating_method == '210Pb CRS').sum()}")
    print(f"After joining OM profiles:")
    print(f"  have best_model fit:   {merged['best_model'].notna().sum()}")
    print(f"  have deep_om_frac:     {merged['deep_om_frac'].notna().sum()}")
    deep_mask = merged["best_model"].notna() & \
                merged["deep_om_frac"].notna() & \
                (merged["deep_om_frac"] > DEEP_OM_THRESHOLD)
    print(f"  well-fit + deep OM > {DEEP_OM_THRESHOLD:.0%}: {deep_mask.sum()}")

    # save merged table
    csv_out = _FIG_DIR / "dated_core_deep_om.csv"
    merged.to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}")

    # --- plotting ---
    plot_df = merged[deep_mask].copy()

    cs_mask  = plot_df["dating_method"] == "137Cs"
    pb_mask  = plot_df["dating_method"] == "210Pb CRS"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax_all, ax_cs, ax_pb = axes

    marker_cs = "o"
    marker_pb = "s"
    col_cs    = "#1f77b4"
    col_pb    = "#ff7f0e"

    def _plot_panel(ax, sub_cs, sub_pb, title):
        if len(sub_cs):
            ax.scatter(sub_cs["accretion_mm_yr"], sub_cs["deep_om_frac"],
                       c=col_cs, s=30, alpha=0.75, marker=marker_cs,
                       edgecolors="none", label=f"¹³⁷Cs (n={len(sub_cs)})")
            # error bars where available
            se = sub_cs["accretion_mm_yr_se"].dropna()
            if len(se):
                idx = sub_cs["accretion_mm_yr_se"].notna()
                ax.errorbar(sub_cs.loc[idx, "accretion_mm_yr"],
                            sub_cs.loc[idx, "deep_om_frac"],
                            xerr=sub_cs.loc[idx, "accretion_mm_yr_se"],
                            fmt="none", ecolor=col_cs, alpha=0.35, lw=0.7)

        if len(sub_pb):
            ax.errorbar(
                sub_pb["accretion_mm_yr"], sub_pb["deep_om_frac"],
                xerr=sub_pb["accretion_mm_yr_se"].fillna(0),
                fmt=marker_pb, color=col_pb, ms=5, alpha=0.75,
                ecolor=col_pb, elinewidth=0.7, capsize=2,
                label=f"²¹⁰Pb CRS (n={len(sub_pb)})")

        ax.set_xlabel("Accretion rate (mm yr⁻¹)", fontsize=10)
        ax.set_ylabel("Deep OM fraction (OMinf)", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
        ax.grid(True, lw=0.3, alpha=0.4)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=8)

    _plot_panel(ax_all,
                plot_df[cs_mask], plot_df[pb_mask],
                "Both dating methods")
    _plot_panel(ax_cs,
                plot_df[cs_mask], pd.DataFrame(),
                "¹³⁷Cs only")
    _plot_panel(ax_pb,
                pd.DataFrame(), plot_df[pb_mask],
                "²¹⁰Pb CRS only")

    fig.suptitle(
        f"East-coast US marsh cores  |  well-fit OM profile (R² ≥ 0.75)  |  "
        f"deep OM > {DEEP_OM_THRESHOLD:.0%}  |  "
        f"¹³⁷Cs peak/bg ≥ {MIN_CS_PBR:.0f}  |  CRS R² ≥ {CRS_R2_MIN}",
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig_out = _FIG_DIR / "dated_core_deep_om.png"
    fig.savefig(fig_out, dpi=150, bbox_inches="tight")
    print(f"Saved: {fig_out}")
    plt.close(fig)

    # --- supplementary: latitude colour ---
    if len(plot_df):
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        lat_min = plot_df["latitude"].min()
        lat_max = plot_df["latitude"].max()
        norm = plt.Normalize(vmin=lat_min, vmax=lat_max)
        cmap = plt.get_cmap("RdYlBu")
        sc = None
        for row in plot_df.itertuples():
            mk = marker_cs if row.dating_method == "137Cs" else marker_pb
            sc = ax2.scatter(row.accretion_mm_yr, row.deep_om_frac,
                             c=[row.latitude], cmap="RdYlBu",
                             vmin=lat_min, vmax=lat_max,
                             s=35, alpha=0.8, edgecolors="none", marker=mk)
        cb = plt.colorbar(sc, ax=ax2)
        cb.set_label("Latitude (°N)", fontsize=9)
        ax2.set_xlabel("Accretion rate (mm yr⁻¹)", fontsize=10)
        ax2.set_ylabel("Deep OM fraction (OMinf)", fontsize=10)
        ax2.set_title("Deep OM vs accretion rate — coloured by latitude", fontsize=10)
        ax2.set_ylim(bottom=0)
        ax2.set_xlim(left=0)
        ax2.grid(True, lw=0.3, alpha=0.4)
        # custom legend for marker shape
        ax2.scatter([], [], marker=marker_cs, c="grey", s=30, label="¹³⁷Cs")
        ax2.scatter([], [], marker=marker_pb, c="grey", s=30, label="²¹⁰Pb CRS")
        ax2.legend(fontsize=8)
        plt.tight_layout()
        fig2_out = _FIG_DIR / "dated_core_deep_om_latitude.png"
        fig2.savefig(fig2_out, dpi=150, bbox_inches="tight")
        print(f"Saved: {fig2_out}")
        plt.close(fig2)


if __name__ == "__main__":
    main()
