"""
Two-panel figure for North Inlet and Louisiana salt marsh sites.

Panel (a) — Elevation bias (EPQS/DTM − RTK) vs peak biomass.
  North Inlet only (RTK data available).
  Biomass assigned by: exact plot match, same-zone mean, site mean, or
  inferred from location type using GI means as representative NI marsh.

Panel (b) — EPQS elevation vs median peak biomass.
  All sites: NI LTER (SC) + Louisiana (LUMCON/Roberts & Hill).
  Error bars = min/max annual peak across years.
  Louisiana sites have no RTK so appear only in panel (b).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

_THIS_DIR = Path(__file__).parent
_LTER_DIR = _THIS_DIR.parent / "lter_data"
_OUT = _THIS_DIR / "figures" / "elevation_bias_vs_biomass_NI.png"

_REGION_COLOR = {
    "South Carolina": "#2a9d8f",
    "Louisiana":      "#e07b39",
}
_LOC_COLOR = {
    "HM": "#e07b39",
    "LM": "#2a9d8f",
    "MM": "#8338ec",
    "MF": "#aaaaaa",
    "plot": "#457b9d",
}


# ── Panel (a) data ────────────────────────────────────────────────────────────

def _ni_biomass_means():
    bio = pd.read_csv(
        _LTER_DIR / "edi.135.12/NILTREB_plants_aboveground_biomass_density.csv"
    )
    bio["DATE"] = pd.to_datetime(bio["DATE"])
    bio["Month"] = bio["DATE"].dt.month
    bio["ABOVEGROUND_BIOMASS"] = pd.to_numeric(
        bio["ABOVEGROUND_BIOMASS"], errors="coerce"
    )
    ctrl = bio[(bio["Month"].between(7, 8)) & (bio["TREATMENT"] == "C")]
    plot_means = (
        ctrl.groupby(["SITE", "LOCATION", "PLOT"])["ABOVEGROUND_BIOMASS"]
        .mean().reset_index().rename(columns={"ABOVEGROUND_BIOMASS": "biomass"})
    )
    loc_means = (
        ctrl.groupby(["SITE", "LOCATION"])["ABOVEGROUND_BIOMASS"]
        .agg(mean="mean", std="std", count="count").reset_index()
    )
    loc_means["se"] = loc_means["std"] / np.sqrt(loc_means["count"])
    return plot_means, loc_means


def _loc_bio(loc_means, site, loc):
    row = loc_means[(loc_means["SITE"] == site) & (loc_means["LOCATION"] == loc)]
    if row.empty:
        return np.nan, np.nan
    return float(row["mean"].iloc[0]), float(row["se"].iloc[0])


def build_bias_data():
    epqs = pd.read_csv(_THIS_DIR / "lter_plot_elevations_epqs.csv")
    ni = epqs[epqs["site_id"].str.startswith("NI")].dropna(
        subset=["measured_elevation_m"]
    ).copy()

    plot_means, loc_means = _ni_biomass_means()

    _INFER = {
        "HM": ("GI", "HM"),
        "LM": ("GI", "LM"),
        "MM": ("GI", "HM"),
        "MF": None,
        "plot": None,
    }

    rows = []
    for _, r in ni.iterrows():
        sid, loc_type = r["site_id"], r["location_type"]
        biomass, biomass_se, quality = np.nan, np.nan, "inferred"

        if sid in ("NI-GI-HM-P5", "NI-GI-HM-P9", "NI-GI-HM-P12"):
            p = int(sid.split("-P")[1])
            row_b = plot_means[
                (plot_means["SITE"] == "GI") & (plot_means["LOCATION"] == "HM")
                & (plot_means["PLOT"] == p)
            ]
            if not row_b.empty:
                biomass, quality = float(row_b["biomass"].iloc[0]), "exact"
        elif sid in ("NI-GI-LM-P16", "NI-GI-LM-P17", "NI-GI-LM-P18"):
            biomass, biomass_se = _loc_bio(loc_means, "GI", "LM")
            quality = "same-zone"
        elif sid == "NI-GIHM":
            biomass, biomass_se = _loc_bio(loc_means, "GI", "HM")
            quality = "site-mean"
        elif sid == "NI-GILM":
            biomass, biomass_se = _loc_bio(loc_means, "GI", "LM")
            quality = "site-mean"
        elif sid == "NI-OLHM":
            biomass, biomass_se = _loc_bio(loc_means, "OL", "HM")
            quality = "site-mean"
        elif sid == "NI-OLLM":
            biomass, biomass_se = _loc_bio(loc_means, "OL", "LM")
            quality = "site-mean"
        elif loc_type in _INFER and _INFER[loc_type] is not None:
            biomass, biomass_se = _loc_bio(loc_means, *_INFER[loc_type])
        elif loc_type == "MF":
            biomass, biomass_se = 0.0, 0.0

        rows.append({
            **r,
            "biomass_g_m2": biomass,
            "biomass_se": biomass_se,
            "match_quality": quality,
            "epqs_bias_m": r["epqs_elevation_m"] - r["measured_elevation_m"],
            "dtm_bias_m":  r["dtm_elevation_m"]  - r["measured_elevation_m"],
        })
    return pd.DataFrame(rows)


# ── figure ────────────────────────────────────────────────────────────────────

def make_figure():
    bias_df = build_bias_data()
    elev_df = pd.read_csv(_THIS_DIR / "lter_plot_peak_biomass.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Salt marsh elevation and lidar bias vs aboveground biomass\n"
        "S. alterniflora — North Inlet SC and Louisiana",
        fontsize=12,
    )

    # ── panel (a): bias vs biomass (NI only) ─────────────────────────────────
    ax1 = axes[0]

    _QUALITY_STYLE = {
        "exact":     dict(marker="o", ms=10, zorder=5),
        "same-zone": dict(marker="s", ms=9,  zorder=4),
        "site-mean": dict(marker="^", ms=9,  zorder=4),
        "inferred":  dict(marker="D", ms=7,  zorder=3),
    }

    valid_bias = bias_df.dropna(subset=["biomass_g_m2"])
    for _, row in valid_bias.iterrows():
        style = _QUALITY_STYLE.get(row["match_quality"], _QUALITY_STYLE["inferred"])
        color = _LOC_COLOR.get(row["location_type"], "#888888")
        for bias_val, marker_ec in [
            (row["epqs_bias_m"], "k"),
            (row["dtm_bias_m"],  "gray"),
        ]:
            if np.isnan(bias_val):
                continue
            ax1.errorbar(
                row["biomass_g_m2"], bias_val,
                xerr=row["biomass_se"] if not np.isnan(row["biomass_se"]) else 0,
                fmt=style["marker"], color=color,
                markersize=style["ms"], zorder=style["zorder"],
                capsize=3, elinewidth=0.8,
                markeredgecolor=marker_ec, markeredgewidth=0.8,
            )

    ax1.axhline(0, color="k", lw=0.8, ls="--", zorder=1)
    ax1.set_xlabel("Peak aboveground biomass (g m⁻²)", fontsize=11)
    ax1.set_ylabel("Elevation bias (remote sensing − RTK, m)", fontsize=11)
    ax1.set_title("(a) Elevation bias vs biomass — North Inlet, SC", fontsize=10)
    ax1.set_xlim(-50, 950)
    ax1.grid(True, alpha=0.3)

    # bias legend
    bias_handles = [
        mlines.Line2D([], [], color="k",    marker="o", linestyle="None",
                      markersize=7, markeredgecolor="k", label="EPQS bias"),
        mlines.Line2D([], [], color="#888", marker="o", linestyle="None",
                      markersize=7, markeredgecolor="gray", label="Lidar DTM bias"),
    ]
    loc_handles = [
        mlines.Line2D([], [], color=c, marker="o", linestyle="None",
                      markersize=7, markeredgecolor="k", label=lbl)
        for lbl, c in [("HM", _LOC_COLOR["HM"]), ("LM", _LOC_COLOR["LM"]),
                        ("MM", _LOC_COLOR["MM"]), ("MF", _LOC_COLOR["MF"])]
    ]
    quality_handles = [
        mlines.Line2D([], [], color="k", marker=s["marker"], linestyle="None",
                      markersize=s["ms"], label=lbl)
        for lbl, s in [
            ("Exact plot match",        _QUALITY_STYLE["exact"]),
            ("Same zone, diff. plot",   _QUALITY_STYLE["same-zone"]),
            ("Site-level mean biomass", _QUALITY_STYLE["site-mean"]),
            ("Inferred from loc. type", _QUALITY_STYLE["inferred"]),
        ]
    ]
    ax1.legend(handles=bias_handles + loc_handles + quality_handles,
               fontsize=7, loc="upper left", ncol=2)

    # ── panel (b): EPQS elevation vs median peak biomass (all sites) ──────────
    ax2 = axes[1]

    for region, grp in elev_df.groupby("region"):
        color = _REGION_COLOR.get(region, "#888888")
        for _, row in grp.iterrows():
            xerr_lo = row["median_peak_agb_g_m2"] - row["min_peak_agb_g_m2"]
            xerr_hi = row["max_peak_agb_g_m2"]    - row["median_peak_agb_g_m2"]
            ax2.errorbar(
                row["median_peak_agb_g_m2"], row["epqs_elevation_m"],
                xerr=[[xerr_lo], [xerr_hi]],
                fmt="o", color=color, markersize=8,
                capsize=3, elinewidth=0.9,
                markeredgecolor="k", markeredgewidth=0.5, zorder=3,
            )
            ax2.annotate(
                row["site_id"].replace("NI-", "").replace("LA-", ""),
                (row["median_peak_agb_g_m2"], row["epqs_elevation_m"]),
                xytext=(4, 3), textcoords="offset points", fontsize=6,
            )

    ax2.axhline(0, color="k", lw=0.8, ls="--", zorder=1, label="NAVD88 MSL")
    ax2.set_xlabel("Median annual peak biomass (g m⁻²)", fontsize=11)
    ax2.set_ylabel("EPQS elevation (m NAVD88)", fontsize=11)
    ax2.set_title(
        "(b) EPQS elevation vs biomass — NI SC + Louisiana\n"
        "(error bars = min/max annual peak across years)",
        fontsize=10,
    )
    region_handles = [
        mlines.Line2D([], [], color=c, marker="o", linestyle="None",
                      markersize=8, markeredgecolor="k", label=lbl)
        for lbl, c in [
            ("South Carolina (NI LTER)", _REGION_COLOR["South Carolina"]),
            ("Louisiana (LUMCON/Roberts & Hill)", _REGION_COLOR["Louisiana"]),
        ]
    ] + [mlines.Line2D([], [], color="k", lw=0.8, ls="--", label="NAVD88 ≈ 0 m")]
    ax2.legend(handles=region_handles, fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    _OUT.parent.mkdir(exist_ok=True)
    fig.savefig(_OUT, dpi=150, bbox_inches="tight")
    print(f"Saved → {_OUT}")


if __name__ == "__main__":
    make_figure()
