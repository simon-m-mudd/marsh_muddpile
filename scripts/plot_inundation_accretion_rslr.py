"""
plot_inundation_accretion_rslr.py

Two-panel figure:

  A  Inundation fraction vs RSLR
       CCN US cores (core_inundation.fgb + PSMSL RSLR)
       Saintilan 2022 SET stations (inund_frac + RSLR.contemporaneous)

  B  Accretion rate vs RSLR
       ¹³⁷Cs rates (most robust; distinct-peak cores only)
       High-confidence ²¹⁰Pb CRS (CV < CV_THRESHOLD across CIC/CFCS/CRS)
       SET marker-horizon accretion (Saintilan 2022)
       1:1 line: accretion = RSLR (marsh in elevation equilibrium)

RSLR sources
------------
CCN cores:  PSMSL tide gauge trends (core_data/ccn_us_psmsl_rslr.csv)
            Falls back to NOAA slr_mmyr in core_inundation.fgb where
            PSMSL trend unavailable (record < 30 yr) — both are RSLR.
SET data:   RSLR.contemporaneous from Saintilan et al. 2022 (tide-gauge
            based, already incorporates VLM).

Accretion / inundation sources
-------------------------------
¹³⁷Cs:     scripts/figures/cs137_accretion_rates.csv
²¹⁰Pb CRS: scripts/figures/pb210_accretion_rates.csv
            filtered: R² ≥ 0.70 for CIC/CFCS, CV < CV_THRESHOLD across
            all three ²¹⁰Pb models
SET:        core_data/saintilan2022_SET_inundation.csv
CCN inund:  scripts/core_inundation.fgb
"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

R2_MIN       = 0.70
CV_THRESHOLD = 0.30
RSLR_CLIP    = (0.0, 12.0)   # mm/yr display range
ACC_CLIP     = (0.0, 15.0)
INUND_CLIP   = (0.0, 1.0)

# Colours
C_CS137   = "#e08214"
C_PB210   = "#1f78b4"
C_SET_ACC = "#d73027"
C_CCN_INF = "#4dac26"
C_SET_INF = "#d01c8b"

_ROOT    = Path(__file__).parent.parent
_FIG_DIR = _ROOT / "scripts" / "figures"

_INUND_FGB   = _ROOT / "scripts" / "core_inundation.fgb"
_PSMSL_CSV   = _ROOT / "core_data" / "ccn_us_psmsl_rslr.csv"
_SET_CSV     = _ROOT / "core_data" / "saintilan2022_SET_inundation.csv"
_CS_CSV      = _FIG_DIR / "cs137_accretion_rates.csv"
_PB_CSV      = _FIG_DIR / "pb210_accretion_rates.csv"
_SPECIES_CSV = _ROOT / "core_data" / "CCN_synthesis" / "CCN_species.csv"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _spartina_core_ids() -> set:
    """Return set of core_ids that have any Spartina species recorded."""
    sp = pd.read_csv(_SPECIES_CSV)
    return set(sp.loc[sp["species_code"].str.startswith("Spartina", na=False), "core_id"])


def _best_rslr(psmsl: pd.DataFrame) -> pd.DataFrame:
    """Prefer PSMSL rslr; fall back to NOAA slr_mmyr."""
    p = psmsl.copy()
    p["rslr_mmyr"] = np.where(
        p["psmsl_rslr_mmyr"].notna(),
        p["psmsl_rslr_mmyr"],
        p["slr_mmyr"],
    )
    p["rslr_source"] = np.where(
        p["psmsl_rslr_mmyr"].notna(), "PSMSL", "NOAA"
    )
    return p


def load_inundation(spartina_only: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ccn_inund, set_inund) DataFrames with inund_frac + rslr_mmyr.

    If spartina_only=True, restrict CCN cores to those with a Spartina species
    and return an empty set_inund DataFrame.
    """

    # CCN US cores
    inund  = gpd.read_file(_INUND_FGB)[
        ["core_id", "inund_frac", "slr_mmyr"]].copy()
    psmsl  = pd.read_csv(_PSMSL_CSV)[
        ["core_id", "psmsl_rslr_mmyr"]].copy()
    ccn    = inund.merge(psmsl, on="core_id", how="left")
    ccn    = _best_rslr(ccn)
    ccn    = ccn.dropna(subset=["inund_frac", "rslr_mmyr"])

    if spartina_only:
        sids = _spartina_core_ids()
        ccn  = ccn[ccn["core_id"].isin(sids)].copy()
        set_df = pd.DataFrame(columns=["inund_frac", "rslr_mmyr"])
    else:
        # SET stations — USA only
        set_df = pd.read_csv(_SET_CSV)
        set_df = set_df[set_df["country"] == "USA"].copy()
        set_df = set_df.rename(columns={"RSLR.contemporaneous": "rslr_mmyr"})
        set_df = set_df[["inund_frac", "rslr_mmyr"]].dropna()

    return ccn, set_df


def load_accretion(spartina_only: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (cs137, pb210_hc, set_acc) DataFrames with rslr_mmyr + rate_mm_yr.

    If spartina_only=True, restrict CCN-derived rates to Spartina cores and
    return an empty set_acc DataFrame.
    """

    psmsl = pd.read_csv(_PSMSL_CSV)[["core_id", "psmsl_rslr_mmyr", "slr_mmyr"]]
    psmsl = _best_rslr(psmsl)[["core_id", "rslr_mmyr"]]

    sids = _spartina_core_ids() if spartina_only else None

    # ¹³⁷Cs
    cs = pd.read_csv(_CS_CSV)
    cs = cs[cs["distinct_peak"] == True].copy()
    if sids is not None:
        cs = cs[cs["core_id"].isin(sids)].copy()
    cs = cs.rename(columns={"accretion_mm_yr": "rate_mm_yr"})
    cs = cs.merge(psmsl, on="core_id", how="inner")
    cs = cs.dropna(subset=["rate_mm_yr", "rslr_mmyr"])

    # High-confidence ²¹⁰Pb
    pb = pd.read_csv(_PB_CSV)
    if sids is not None:
        pb = pb[pb["core_id"].isin(sids)].copy()
    cic_ok  = pb["cic_flag"].str.startswith("ok").fillna(False) & (pb["cic_r2"] >= R2_MIN)
    cfcs_ok = pb["cfcs_flag"].str.startswith("ok").fillna(False) & (pb["cfcs_r2"] >= R2_MIN)
    crs_ok  = pb["crs_flag"].str.startswith("ok").fillna(False)
    triple  = pb[cic_ok & cfcs_ok & crs_ok].copy()
    triple["mean3"] = triple[["cic_r_mm_yr","cfcs_r_mm_yr","crs_r_mm_yr"]].mean(axis=1)
    triple["cv3"]   = (triple[["cic_r_mm_yr","cfcs_r_mm_yr","crs_r_mm_yr"]].std(axis=1)
                       / triple["mean3"])
    hc = triple[triple["cv3"] < CV_THRESHOLD].copy()
    hc = hc.rename(columns={"crs_r_mm_yr": "rate_mm_yr"})
    hc = hc.merge(psmsl, on="core_id", how="inner")
    hc = hc.dropna(subset=["rate_mm_yr", "rslr_mmyr"])

    if spartina_only:
        set_df = pd.DataFrame(columns=["rate_mm_yr", "rslr_mmyr"])
    else:
        # SET marker-horizon accretion — USA only
        set_df = pd.read_csv(_SET_CSV)
        set_df = set_df[set_df["country"] == "USA"].copy()
        set_df = set_df.rename(columns={
            "accretion":             "rate_mm_yr",
            "RSLR.contemporaneous":  "rslr_mmyr",
        })
        set_df = set_df[["rate_mm_yr", "rslr_mmyr"]].dropna()

    return cs, hc, set_df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def panel_inundation(ax, ccn, set_df, ccn_label="CCN US cores"):
    xlo, xhi = RSLR_CLIP

    # CCN cores
    x_ccn = ccn["rslr_mmyr"].clip(*RSLR_CLIP)
    y_ccn = ccn["inund_frac"].clip(*INUND_CLIP)
    ax.scatter(x_ccn, y_ccn, color=C_CCN_INF, s=12, alpha=0.45,
               linewidths=0, label=f"{ccn_label} (n={len(ccn)})", zorder=2)

    pairs = [(x_ccn, y_ccn, C_CCN_INF)]

    # SET stations (only when data provided)
    if len(set_df):
        x_set = set_df["rslr_mmyr"].clip(*RSLR_CLIP)
        y_set = set_df["inund_frac"].clip(*INUND_CLIP)
        ax.scatter(x_set, y_set, color=C_SET_INF, s=18, alpha=0.65,
                   linewidths=0, marker="^", label=f"SET stations (n={len(set_df)})", zorder=3)
        pairs.append((x_set, y_set, C_SET_INF))

    # Running median for each dataset
    for x_vals, y_vals, color in pairs:
        df_tmp = pd.DataFrame({"x": x_vals, "y": y_vals}).sort_values("x")
        if len(df_tmp) >= 20:
            df_tmp["bin"] = pd.cut(df_tmp["x"], bins=10)
            med = df_tmp.groupby("bin", observed=True)["y"].median()
            centers = [b.mid for b in med.index]
            ax.plot(centers, med.values, color=color, lw=2, ls="-",
                    alpha=0.85, zorder=4)

    ax.set_xlim(*RSLR_CLIP)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("RSLR (mm yr⁻¹)", fontsize=10)
    ax.set_ylabel("Inundation fraction", fontsize=10)
    ax.set_title("A  Inundation fraction vs RSLR", fontsize=10,
                 loc="left", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.axhline(0.5, color="k", lw=0.5, ls=":", alpha=0.4)


def panel_accretion(ax, cs, hc, set_acc):
    xlo, xhi = RSLR_CLIP

    # Scatter each dataset
    def _scatter(xdata, ydata, color, marker, label, size=18, alpha=0.65):
        x = xdata.clip(*RSLR_CLIP)
        y = ydata.clip(*ACC_CLIP)
        ax.scatter(x, y, color=color, s=size, alpha=alpha,
                   linewidths=0, marker=marker, label=label, zorder=3)

    if len(set_acc):
        _scatter(set_acc["rslr_mmyr"], set_acc["rate_mm_yr"],
                 C_SET_ACC, "^", f"SET MH accretion (n={len(set_acc)})", size=22)
    _scatter(cs["rslr_mmyr"], cs["rate_mm_yr"],
             C_CS137, "o", f"¹³⁷Cs (n={len(cs)})")
    _scatter(hc["rslr_mmyr"], hc["rate_mm_yr"],
             C_PB210, "s", f"²¹⁰Pb CRS high-conf. CV<{CV_THRESHOLD} (n={len(hc)})",
             size=18, alpha=0.75)

    # 1:1 line
    lim = xhi
    ax.plot([0, lim], [0, lim], "k--", lw=1.1, zorder=1, label="1:1 (equilibrium)")

    # OLS fits
    ols_datasets = [(cs["rslr_mmyr"].clip(*RSLR_CLIP), cs["rate_mm_yr"].clip(*ACC_CLIP), C_CS137)]
    if len(set_acc):
        ols_datasets.append(
            (set_acc["rslr_mmyr"].clip(*RSLR_CLIP), set_acc["rate_mm_yr"].clip(*ACC_CLIP), C_SET_ACC)
        )
    for xdata, ydata, color in ols_datasets:
        if len(xdata) >= 5:
            sl, ic, *_ = stats.linregress(xdata, ydata)
            xs = np.array([0, lim])
            ax.plot(xs, sl * xs + ic, color=color, lw=1.3, ls="--",
                    alpha=0.7, zorder=2)

    ax.set_xlim(*RSLR_CLIP)
    ax.set_ylim(*ACC_CLIP)
    ax.set_xlabel("RSLR (mm yr⁻¹)", fontsize=10)
    ax.set_ylabel("Accretion rate (mm yr⁻¹)", fontsize=10)
    ax.set_title("B  Accretion rate vs RSLR", fontsize=10,
                 loc="left", fontweight="bold")

    # Shade "keeping pace" region (accretion ≥ RSLR)
    xs = np.linspace(0, lim, 200)
    ax.fill_between(xs, xs, ACC_CLIP[1], alpha=0.06, color="green",
                    label="keeping pace (accretion ≥ RSLR)")
    ax.fill_between(xs, 0, xs, alpha=0.06, color="red",
                    label="losing elevation (accretion < RSLR)")

    ax.legend(fontsize=8, loc="upper left")

    # Fraction above 1:1 for Cs137
    if len(cs):
        frac = (cs["rate_mm_yr"] >= cs["rslr_mmyr"]).mean()
        ax.text(0.97, 0.04,
                f"¹³⁷Cs above 1:1: {frac*100:.0f}%",
                transform=ax.transAxes, fontsize=8, ha="right",
                color=C_CS137,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def _build_figure(ccn_inund, set_inund, cs, hc, set_acc,
                  suptitle: str, out: Path, ccn_label: str = "CCN US cores"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    panel_inundation(axes[0], ccn_inund, set_inund, ccn_label=ccn_label)
    panel_accretion(axes[1], cs, hc, set_acc)
    fig.suptitle(suptitle, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── All-species figure ──────────────────────────────────────────────────
    print("Loading inundation data (all species) …")
    ccn_inund, set_inund = load_inundation()
    print(f"  CCN inundation: {len(ccn_inund)} cores,  SET: {len(set_inund)} stations")

    print("Loading accretion data (all species) …")
    cs, hc, set_acc = load_accretion()
    print(f"  ¹³⁷Cs: {len(cs)},  ²¹⁰Pb HC: {len(hc)},  SET: {len(set_acc)}")

    _build_figure(
        ccn_inund, set_inund, cs, hc, set_acc,
        suptitle=(
            "RSLR from PSMSL tide gauge trends (CCN) and Saintilan 2022 (SET)\n"
            "RSLR incorporates vertical land motion; all CCN US marsh cores + USA SET stations"
        ),
        out=_FIG_DIR / "inundation_accretion_vs_rslr.png",
    )

    # ── Spartina-only figure ────────────────────────────────────────────────
    print("\nLoading inundation data (Spartina only) …")
    ccn_sp, _  = load_inundation(spartina_only=True)
    print(f"  Spartina CCN inundation: {len(ccn_sp)} cores")

    print("Loading accretion data (Spartina only) …")
    cs_sp, hc_sp, _ = load_accretion(spartina_only=True)
    print(f"  ¹³⁷Cs: {len(cs_sp)},  ²¹⁰Pb HC: {len(hc_sp)}")

    _build_figure(
        ccn_sp, pd.DataFrame(columns=["inund_frac", "rslr_mmyr"]),
        cs_sp, hc_sp, pd.DataFrame(columns=["rate_mm_yr", "rslr_mmyr"]),
        suptitle=(
            "Spartina cores only — RSLR from PSMSL tide gauge trends\n"
            "No SET data; CCN US marsh cores with Spartina spp. recorded"
        ),
        out=_FIG_DIR / "inundation_accretion_vs_rslr_spartina.png",
        ccn_label="CCN Spartina cores",
    )


if __name__ == "__main__":
    main()
