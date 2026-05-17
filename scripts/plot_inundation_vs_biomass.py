"""
Inundation fraction vs peak aboveground biomass for all LTER/field sites.

Layout: main panel (all sites) + inset zoom on the North Inlet cluster.
Points coloured by latitude; marker shape by region.
Error bars show min/max annual peak across years.

Output: scripts/figures/inundation_vs_biomass.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

_THIS_DIR = Path(__file__).parent
_CSV = _THIS_DIR / "lter_plot_peak_biomass.csv"
_OUT = _THIS_DIR / "figures" / "inundation_vs_biomass.png"

_SHORT = {
    "NI-GI-HM-P5":  "GI-HM-P5",
    "NI-GI-HM-P9":  "GI-HM-P9",
    "NI-GI-HM-P12": "GI-HM-P12",
    "NI-GI-HM-P15": "GI-HM-P15",
    "NI-GI-LM-P7":  "GI-LM-P7",
    "NI-GI-LM-P8":  "GI-LM-P8",
    "NI-GI-LM-P9":  "GI-LM-P9",
    "NI-OL-HM-P1":  "OL-HM-P1",
    "NI-OL-HM-P2":  "OL-HM-P2",
    "NI-OL-HM-P3":  "OL-HM-P3",
    "NI-OL-LM-P4":  "OL-LM-P4",
    "NI-OL-LM-P5":  "OL-LM-P5",
    "NI-OL-LM-P6":  "OL-LM-P6",
    "LA-LUM1":       "LUM1",
    "LA-LUM2":       "LUM2",
    "LA-LUM3":       "LUM3",
    "LA-TB1":        "TB1",
    "LA-TB2":        "TB2",
    "LA-TB3":        "TB3",
    "LA-TB4":        "TB4",
    "GCE-SCSA":      "SCSA",
    "GCE-ZSC1":      "ZSC1*",
    "GCE-ZSC2":      "ZSC2",
    "PIE-Club_Head":    "Club Head",
    "PIE-LM2":          "LM2",
    "PIE-LM3":          "LM3",
    "PIE-Nelson":       "Nelson",
    "PIE-Patmos":       "Patmos",
    "PIE-Rowley_House": "Rowley Hse",
    "PIE-West":         "West",
}

_MARKER = {
    "South Carolina": "o",
    "Louisiana":      "s",
    "Georgia":        "^",
    "Massachusetts":  "D",
}

# Offsets for the outer (non-NI) labels: (dx, dy) in data coords
# NI sites are handled in the inset only
_OUTER_OFFSET = {
    "LA-LUM1":  (  30,  0.025),
    "LA-LUM2":  (  30, -0.050),
    "LA-LUM3":  (-250, -0.050),
    "LA-TB1":   ( -80,  0.040),
    "LA-TB2":   (  30,  0.040),
    "LA-TB3":   ( -80, -0.050),
    "LA-TB4":   (  30, -0.050),
    "GCE-SCSA": (  30,  0.025),
    "GCE-ZSC1": (  30,  0.025),
    "GCE-ZSC2": (  30,  0.025),
    "PIE-Club_Head":    (-280,  0.040),
    "PIE-LM2":          ( -250, -0.050),
    "PIE-LM3":          (  30,  0.025),
    "PIE-Nelson":       (  30,  0.025),
    "PIE-Patmos":       (  30, -0.050),
    "PIE-Rowley_House": (-300, -0.050),
    "PIE-West":         (  30, -0.050),
}

# Offsets for the NI inset labels: (dx, dy)
_NI_OFFSET = {
    "NI-GI-HM-P5":  ( -10,  0.010),
    "NI-GI-HM-P9":  (  10,  0.010),
    "NI-GI-HM-P12": (  10, -0.014),
    "NI-GI-HM-P15": ( -10, -0.014),
    "NI-GI-LM-P7":  ( -10,  0.010),
    "NI-GI-LM-P8":  (  10,  0.010),
    "NI-GI-LM-P9":  (  10, -0.014),
    "NI-OL-HM-P1":  ( -10,  0.010),
    "NI-OL-HM-P2":  ( -10, -0.014),
    "NI-OL-HM-P3":  (  10,  0.010),
    "NI-OL-LM-P4":  (  10,  0.010),
    "NI-OL-LM-P5":  (  10, -0.014),
    "NI-OL-LM-P6":  ( -10, -0.014),
}


def _plot_points(ax, df, norm, cmap, label_offsets, label_fontsize=7,
                 ms=9, lw_err=0.7, show_lat=True, unreliable=None):
    if unreliable is None:
        unreliable = set()
    for _, row in df.iterrows():
        sid = row["site_id"]
        bio = row["median_peak_agb_g_m2"]
        ifrac = row["inundation_fraction"]
        lat = row["lat"]
        xerr_lo = bio - row["min_peak_agb_g_m2"]
        xerr_hi = row["max_peak_agb_g_m2"] - bio

        color = cmap(norm(lat))
        marker = _MARKER.get(row["region"], "o")
        bad = sid in unreliable

        ax.errorbar(
            bio, ifrac,
            xerr=[[xerr_lo], [xerr_hi]],
            fmt=marker, color=color,
            markersize=ms,
            markeredgecolor="red" if bad else "k",
            markeredgewidth=1.5 if bad else 0.7,
            capsize=3, elinewidth=lw_err, ecolor=color,
            zorder=3, linestyle="none",
        )

        if sid not in label_offsets:
            continue
        dx, dy = label_offsets[sid]
        short = _SHORT.get(sid, sid)
        txt = f"{short}\n{lat:.1f}°N" if show_lat else short
        ax.annotate(
            txt,
            xy=(bio, ifrac),
            xytext=(bio + dx, ifrac + dy),
            fontsize=label_fontsize,
            color="0.15",
            va="center",
            ha="left" if dx >= 0 else "right",
            arrowprops=dict(arrowstyle="-", color="0.6", lw=0.5)
            if abs(dx) > 50 or abs(dy) > 0.018 else None,
        )


def make_figure():
    df = pd.read_csv(_CSV)

    lats = df["lat"].values
    cmap = cm.plasma_r
    norm = mcolors.Normalize(vmin=lats.min(), vmax=lats.max())

    unreliable = {"GCE-ZSC1"}
    ni_df = df[df["site_id"].str.startswith("NI")]

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2.2], hspace=0.35)
    ax_ni  = fig.add_subplot(gs[0])   # NI zoom
    ax_all = fig.add_subplot(gs[1])   # all sites

    # ── top panel: North Inlet zoom ───────────────────────────────────────────
    _plot_points(ax_ni, ni_df, norm, cmap, _NI_OFFSET,
                 label_fontsize=6.5, ms=9, lw_err=0.7, show_lat=True)

    ax_ni.set_xlim(50, 1350)
    ax_ni.set_ylim(0.08, 0.38)
    ax_ni.set_title("(a) North Inlet, SC — zoom", fontsize=10, loc="left")
    ax_ni.set_xlabel("Median annual peak aboveground biomass (g m⁻²)", fontsize=9)
    ax_ni.set_ylabel("Inundation fraction", fontsize=9)
    ax_ni.grid(True, alpha=0.25)
    ax_ni.tick_params(labelsize=8)

    # ── bottom panel: all sites ───────────────────────────────────────────────
    _plot_points(ax_all, df, norm, cmap, _OUTER_OFFSET,
                 label_fontsize=7.5, ms=10, unreliable=unreliable)

    ax_all.set_xlim(-150, 3700)
    ax_all.set_ylim(-0.13, 1.18)
    ax_all.axhline(0, color="0.7", lw=0.6, ls="--", zorder=1)
    ax_all.axhline(1, color="0.7", lw=0.6, ls="--", zorder=1)
    ax_all.set_xlabel("Median annual peak aboveground biomass (g m⁻²)", fontsize=11)
    ax_all.set_ylabel("Inundation fraction (sinusoidal approximation)", fontsize=11)
    ax_all.set_title(
        "(b) All sites — US east coast salt marshes\n"
        r"$\it{Spartina\ alterniflora}$ (and $\it{S.\ cynosuroides}$ at GCE transition sites) · "
        "EPQS elevation + nearest NAVD88-leveled NOAA tide gauge",
        fontsize=9, loc="left",
    )
    ax_all.grid(True, alpha=0.2)

    # Latitude colorbar (shared)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_ni, ax_all], pad=0.02, shrink=0.9,
                        fraction=0.025)
    cbar.set_label("Latitude (°N)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # Region + quality legend
    handles = [
        Line2D([0], [0], marker=m, color="0.5", markeredgecolor="k",
               linestyle="None", markersize=8, label=r)
        for r, m in _MARKER.items()
    ] + [
        Line2D([0], [0], marker="o", color="w", markeredgecolor="red",
               markeredgewidth=1.5, linestyle="None", markersize=8,
               label="* unreliable EPQS (coords land in water)")
    ]
    ax_all.legend(handles=handles, fontsize=8, loc="upper right",
                  framealpha=0.88, title="Region", title_fontsize=8)

    _OUT.parent.mkdir(exist_ok=True)
    fig.savefig(_OUT, dpi=150, bbox_inches="tight")
    print(f"Saved → {_OUT}")


if __name__ == "__main__":
    make_figure()
