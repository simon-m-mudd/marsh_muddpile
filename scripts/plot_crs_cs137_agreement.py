"""
plot_crs_cs137_agreement.py

Four-panel figure examining agreement between ²¹⁰Pb CRS and ¹³⁷Cs accretion
rates for the 126 marsh cores that have both measurements, plus a triple-method
confidence assessment using CIC and CFCS.

Panels
------
A  Scatter: CRS vs ¹³⁷Cs for all 126 cores
B  Bland-Altman: mean(CRS, ¹³⁷Cs) vs (¹³⁷Cs − CRS)
C  Scatter: same 126 cores, but points are sized/coloured by ²¹⁰Pb
   internal consistency (CV across CIC, CFCS, CRS) where available — shows
   that high-CV cores (models disagree) are also the scattered outliers
D  Scatter: CRS vs ¹³⁷Cs restricted to cores where all three ²¹⁰Pb models
   agree within CV < CV_THRESHOLD — a "high-confidence" subset

Why CIC/CFCS outliers are artefacts, not real
---------------------------------------------
CIC and CFCS fit an exponential slope to the excess ²¹⁰Pb profile.
When the profile has a bioturbated or physically mixed surface layer, the
apparent slope over the fitted depth window is steeper than the true
sedimentation gradient, yielding an inflated rate.  CRS integrates the full
inventory so it is insensitive to this shape distortion.  For the 11 cores
with CIC > 8 mm/yr that also have a CRS estimate, CIC averages 2× CRS
(e.g. RB_W: CIC=20.9, CRS=10.2; FORT_W: CIC=16.7, CRS=7.1).
R² ≥ 0.70 removes the worst fits but does not remove profiles where a mixed
surface layer produces a genuinely good — but biased — exponential fit.

The CV filter (panel D) is more powerful: when CIC, CFCS, and CRS all
return similar rates (CV < 0.3), the mixed-layer problem is absent and all
three models are sampling the same underlying sedimentation signal.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
R2_MIN       = 0.70   # applied to CIC and CFCS
CV_THRESHOLD = 0.30   # coefficient of variation across CIC/CFCS/CRS
CLIP         = 14.0   # mm/yr — display clip (extreme outliers omitted from axes)

_FIG_DIR = Path(__file__).parent.parent / "scripts" / "figures"


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (dual, triple) DataFrames."""
    pb = pd.read_csv(_FIG_DIR / "pb210_accretion_rates.csv")
    cs = pd.read_csv(_FIG_DIR / "cs137_accretion_rates.csv")

    cic_ok  = pb[pb["cic_flag"].str.startswith("ok").fillna(False)
                 & (pb["cic_r2"] >= R2_MIN)][
                 ["core_id","cic_r_mm_yr","cic_r_se_mm_yr","cic_r2"]]
    cfcs_ok = pb[pb["cfcs_flag"].str.startswith("ok").fillna(False)
                 & (pb["cfcs_r2"] >= R2_MIN)][
                 ["core_id","cfcs_r_mm_yr","cfcs_r_se_mm_yr","cfcs_r2"]]
    crs_ok  = pb[pb["crs_flag"].str.startswith("ok").fillna(False)][
                 ["core_id","crs_r_mm_yr","crs_r_se_mm_yr","crs_r2"]]
    cs_ok   = cs[cs["distinct_peak"] == True][
                 ["core_id","accretion_mm_yr","collection_year"]]

    # 126-core dual table
    dual = crs_ok.merge(cs_ok, on="core_id").copy()
    dual.rename(columns={"accretion_mm_yr": "cs_mm_yr"}, inplace=True)

    # Triple-method agreement table
    triple = (cic_ok
              .merge(cfcs_ok, on="core_id")
              .merge(crs_ok,  on="core_id"))
    triple["mean3"] = triple[["cic_r_mm_yr","cfcs_r_mm_yr","crs_r_mm_yr"]].mean(axis=1)
    triple["cv3"]   = (triple[["cic_r_mm_yr","cfcs_r_mm_yr","crs_r_mm_yr"]].std(axis=1)
                       / triple["mean3"])

    # Attach CV to the dual table where available
    dual = dual.merge(triple[["core_id","cv3"]], on="core_id", how="left")

    return dual, triple


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _unity_lim(ax, lim):
    ax.plot([0, lim], [0, lim], "k--", lw=0.9, zorder=1, label="1:1")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_aspect("equal")


def _pearson_text(ax, x, y, loc="upper left"):
    if len(x) < 3:
        return
    r, p = stats.pearsonr(x, y)
    slope, intercept, *_ = stats.linregress(x, y)
    sign = "+" if intercept >= 0 else ""
    ax.text(0.03 if "left" in loc else 0.97, 0.96,
            f"r = {r:.2f},  OLS: y = {slope:.2f}x {sign}{intercept:.2f}\nn = {len(x)}",
            transform=ax.transAxes, fontsize=7.5,
            ha="left" if "left" in loc else "right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))


def _ols_line(ax, x, y, color="#d73027", lw=1.4):
    slope, intercept, *_ = stats.linregress(x, y)
    xs = np.array([0, ax.get_xlim()[1]])
    ax.plot(xs, slope * xs + intercept, color=color, lw=lw, zorder=3)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def panel_scatter_all(ax, dual):
    x = dual["crs_r_mm_yr"].clip(0, CLIP)
    y = dual["cs_mm_yr"].clip(0, CLIP)
    xe = dual["crs_r_se_mm_yr"].fillna(0).clip(0, CLIP / 2)

    sc = ax.scatter(x, y, c=dual["collection_year"],
                    cmap="plasma", s=22, alpha=0.8, zorder=3, linewidths=0)
    ax.errorbar(x, y, xerr=xe, fmt="none", elinewidth=0.5,
                color="#aaaaaa", alpha=0.45, zorder=2, capsize=1.5)
    _unity_lim(ax, CLIP * 1.02)
    _ols_line(ax, x, y)
    _pearson_text(ax, x, y)

    cb = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label("Collection year", fontsize=7)
    ax.set_xlabel("²¹⁰Pb CRS  (mm yr⁻¹)", fontsize=9)
    ax.set_ylabel("¹³⁷Cs  (mm yr⁻¹)", fontsize=9)
    ax.set_title("A  All 126 cores", fontsize=9, loc="left", fontweight="bold")


def panel_bland_altman(ax, dual):
    x = dual["crs_r_mm_yr"]
    y = dual["cs_mm_yr"]
    mean_xy = (x + y) / 2.0
    diff_yx  = y - x          # positive = Cs137 > CRS

    bias = diff_yx.mean()
    sd   = diff_yx.std()
    loa_hi = bias + 1.96 * sd
    loa_lo = bias - 1.96 * sd

    # Colour by collection year
    sc = ax.scatter(mean_xy.clip(0, CLIP), diff_yx.clip(-CLIP, CLIP),
                    c=dual["collection_year"], cmap="plasma",
                    s=22, alpha=0.8, zorder=3, linewidths=0)

    xlim = (0, CLIP * 1.02)
    for val, ls, lbl in [(bias, "-", f"Bias {bias:+.2f}"),
                          (loa_hi, "--", f"+1.96 SD {loa_hi:+.2f}"),
                          (loa_lo, "--", f"−1.96 SD {loa_lo:+.2f}")]:
        ax.axhline(val, color="#d73027", lw=1.2, ls=ls, zorder=2)
        ax.text(xlim[1] * 0.98, val + 0.15, lbl,
                ha="right", va="bottom", fontsize=6.5, color="#d73027")

    ax.axhline(0, color="k", lw=0.7, ls=":", alpha=0.5)
    ax.set_xlim(*xlim)
    ax.set_xlabel("Mean of CRS and ¹³⁷Cs  (mm yr⁻¹)", fontsize=9)
    ax.set_ylabel("¹³⁷Cs − CRS  (mm yr⁻¹)", fontsize=9)
    ax.set_title("B  Bland–Altman", fontsize=9, loc="left", fontweight="bold")

    cb = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label("Collection year", fontsize=7)

    # Percentage within LoA
    pct = ((diff_yx >= loa_lo) & (diff_yx <= loa_hi)).mean() * 100
    ax.text(0.03, 0.04,
            f"{pct:.0f}% within LoA\nSD = {sd:.2f} mm/yr",
            transform=ax.transAxes, fontsize=7.5, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))


def panel_cv_scatter(ax, dual):
    """Same as panel A but coloured by CV across the three ²¹⁰Pb models
    (where available).  Grey = no triple data."""
    has_cv  = dual["cv3"].notna()
    no_cv   = ~has_cv

    x = dual["crs_r_mm_yr"].clip(0, CLIP)
    y = dual["cs_mm_yr"].clip(0, CLIP)

    # Grey for cores without triple data
    ax.scatter(x[no_cv], y[no_cv], c="#cccccc", s=18, alpha=0.6,
               zorder=2, linewidths=0, label="CIC/CFCS unavailable")

    # Coloured by CV
    sc = ax.scatter(x[has_cv], y[has_cv],
                    c=dual.loc[has_cv, "cv3"], cmap="RdYlGn_r",
                    vmin=0, vmax=0.6,
                    s=28, alpha=0.85, zorder=3, linewidths=0.3,
                    edgecolors="#444444")

    _unity_lim(ax, CLIP * 1.02)
    _ols_line(ax, x, y)
    _pearson_text(ax, x, y)

    cb = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label("CV(CIC,CFCS,CRS)", fontsize=7)
    ax.set_xlabel("²¹⁰Pb CRS  (mm yr⁻¹)", fontsize=9)
    ax.set_ylabel("¹³⁷Cs  (mm yr⁻¹)", fontsize=9)
    ax.set_title(f"C  Coloured by ²¹⁰Pb internal CV\n"
                 f"    (green=consistent, red=models disagree)",
                 fontsize=9, loc="left", fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")


def panel_high_confidence(ax, dual, triple):
    """Restrict to cores where all three ²¹⁰Pb methods agree (CV < threshold)."""
    hc_ids = triple.loc[triple["cv3"] < CV_THRESHOLD, "core_id"]
    sub    = dual[dual["core_id"].isin(hc_ids)].copy()

    x  = sub["crs_r_mm_yr"].clip(0, CLIP)
    y  = sub["cs_mm_yr"].clip(0, CLIP)
    xe = sub["crs_r_se_mm_yr"].fillna(0).clip(0, CLIP / 2)

    sc = ax.scatter(x, y, c=sub["cv3"], cmap="RdYlGn_r",
                    vmin=0, vmax=0.6,
                    s=40, alpha=0.9, zorder=3, linewidths=0.3,
                    edgecolors="#333333")
    ax.errorbar(x, y, xerr=xe, fmt="none", elinewidth=0.6,
                color="#aaaaaa", alpha=0.5, zorder=2, capsize=1.5)

    _unity_lim(ax, CLIP * 1.02)
    _ols_line(ax, x, y)
    _pearson_text(ax, x, y)

    cb = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label("CV(CIC,CFCS,CRS)", fontsize=7)
    ax.set_xlabel("²¹⁰Pb CRS  (mm yr⁻¹)", fontsize=9)
    ax.set_ylabel("¹³⁷Cs  (mm yr⁻¹)", fontsize=9)
    ax.set_title(f"D  High-confidence subset: CV < {CV_THRESHOLD}\n"
                 f"    (CIC, CFCS, CRS all agree)",
                 fontsize=9, loc="left", fontweight="bold")


# ---------------------------------------------------------------------------
# Assemble figure
# ---------------------------------------------------------------------------

def make_figure(outdir: Path):
    dual, triple = load()

    fig = plt.figure(figsize=(13, 11))
    gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.38)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_scatter_all(ax_a, dual)
    panel_bland_altman(ax_b, dual)
    panel_cv_scatter(ax_c, dual)
    panel_high_confidence(ax_d, dual, triple)

    # Summary stats in figure footer
    diff = dual["cs_mm_yr"] - dual["crs_r_mm_yr"]
    n_hc = len(triple[triple["cv3"] < CV_THRESHOLD].merge(
        dual[["core_id"]], on="core_id"))
    fig.text(0.01, 0.005,
             f"CRS R² filter: none   |   CIC/CFCS R² ≥ {R2_MIN}   |   "
             f"Triple-agreement CV threshold: {CV_THRESHOLD}   |   "
             f"Bland–Altman bias: {diff.mean():+.2f} mm/yr  "
             f"(¹³⁷Cs − CRS)   |   "
             f"Cores in high-confidence subset (D): {n_hc}",
             fontsize=6.5, color="#555555")

    out = outdir / f"crs_cs137_agreement_r2{R2_MIN}_cv{CV_THRESHOLD}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # Print Bland-Altman numbers
    print(f"\nBland-Altman summary (¹³⁷Cs − CRS, mm/yr):")
    print(f"  Bias (mean diff):  {diff.mean():+.3f}")
    print(f"  SD of diff:        {diff.std():.3f}")
    print(f"  95% LoA:           [{diff.mean()-1.96*diff.std():.2f},  {diff.mean()+1.96*diff.std():.2f}]")
    pct = ((diff >= diff.mean()-1.96*diff.std()) & (diff <= diff.mean()+1.96*diff.std())).mean()*100
    print(f"  % within LoA:      {pct:.0f}%")
    print(f"\nHigh-confidence subset (CV < {CV_THRESHOLD}, with Cs137): {n_hc} cores")
    hc = dual[dual["core_id"].isin(triple.loc[triple["cv3"] < CV_THRESHOLD, "core_id"])]
    if len(hc):
        r, _ = stats.pearsonr(hc["crs_r_mm_yr"], hc["cs_mm_yr"])
        print(f"  Pearson r (CRS vs Cs137): {r:.3f}")


if __name__ == "__main__":
    make_figure(_FIG_DIR)
