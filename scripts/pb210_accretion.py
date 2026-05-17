"""
pb210_accretion.py

Compute ²¹⁰Pb-based sediment accretion rates from raw excess ²¹⁰Pb depth
profiles in the CCN synthesis dataset, using three standard geochronological
models.  Results are written to a CSV and a set of comparison figures.

Physics background
------------------
²¹⁰Pb (half-life 22.3 yr, decay constant λ = 0.03114 yr⁻¹) is supplied to
surface sediments from the atmosphere at a roughly constant rate. Once buried,
excess (unsupported) ²¹⁰Pb decays in place. The depth distribution of excess
activity encodes the age-depth history of the sediment column.

Three models
------------

CIC — Constant Initial Concentration (Robbins & Edgington 1975)
  Assumes the ²¹⁰Pb activity at the sediment–water interface has been
  constant over time.  Excess activity decays exponentially with linear depth:
      A_ex(z) = A₀ · exp(−λz / r)
  → ln A_ex = ln A₀ − (λ/r) · z
  Fit by weighted linear regression of ln(A_ex) vs z [cm].
  Rate r (cm yr⁻¹) = λ / |slope|.
  Suitable when: bulk density is uniform, sediment supply is steady,
  and no mixing or hiatus.  The simplest and most widely used model.

CFCS — Constant Flux Constant Sedimentation (Robbins 1978)
  Same conceptual model as CIC but uses mass depth m [g cm⁻²] instead of
  linear depth, accounting for compaction and porosity changes:
      m(z) = ∫₀ᶻ ρ_b(z′) dz′   (computed from dry bulk density)
  Fit: ln A_ex vs m [g cm⁻²]; slope = −λ / F.
  Mass accumulation rate F (g cm⁻² yr⁻¹) = λ / |slope|.
  Linear accretion rate r (cm yr⁻¹) = F / mean(ρ_b) over the fitted section.
  Requires dry_bulk_density in every fitted layer; skipped otherwise.
  CFCS ≡ CIC when bulk density is constant.

CRS — Constant Rate of Supply (Appleby & Oldfield 1978)
  Assumes the total ²¹⁰Pb flux to the sediment surface is constant, but the
  initial concentration may vary (e.g. due to changing sediment supply).
  Age at depth z:
      t(z) = (1/λ) · ln[ I(0,∞) / I(z,∞) ]
  where I(z,∞) = ∫ᶻ^∞ A_ex(z′) ρ_b(z′) dz′  is the cumulative inventory
  below z (Bq cm⁻²).  If bulk density is missing, the unscaled inventory
  I(z,∞) = ∫ᶻ^∞ A_ex(z′) dz′ is used instead (ratios cancel the unit).
  A mean accretion rate is extracted by linear regression of depth vs age.
  CRS is the most general model and is preferred when initial concentration
  varies, but it requires the profile to extend to background activity.

Uncertainty treatment
---------------------
When excess_pb210_activity_se is available, weighted least squares is used
for CIC and CFCS (weights = 1/σ² in log-space, i.e. w_i = (A_i/σ_i)²).
Uncertainties in slope are propagated analytically through the WLS normal
equations (Bevington & Robinson 2003, eq. 6.12–6.18).

For CRS, uncertainty in each cumulative inventory is propagated in
quadrature from the per-layer activity uncertainties:
    σ_I(z) = √[ Σ_{z′≥z} (σ_Ai · ρ_i · Δz_i)² ]
Age uncertainty σ_t is then:
    σ_t(z) = (1/λ) · √[ (σ_I₀/I₀)² + (σ_I(z)/I(z))² ]
The mean accretion rate uncertainty is the regression slope SE propagated
through these age uncertainties.

When SE is absent, OLS is used and the rate SE is derived from residuals.

Quality flags
-------------
cic_flag / cfcs_flag / crs_flag: one of
  'ok'               — fit converged
  'too_few_layers'   — fewer than MIN_LAYERS positive-activity layers
  'non_monotonic'    — positive slope (activity increases with depth) — rejected
  'insufficient_depth_range' — fitted depth span < MIN_DEPTH_RANGE_CM
  'no_bulk_density'  — BD missing (CFCS only)
  'did_not_reach_background' — deepest layer has >10 % of total inventory
                               (CRS only; rate may be underestimated)
  'no_se'            — activity SE absent; OLS used instead of WLS

References
----------
Appleby, P.G. & Oldfield, F. 1978. The calculation of lead-210 dates
  assuming a constant rate of supply of unsupported 210Pb to the sediment.
  Catena 5, 1–8.
Robbins, J.A. & Edgington, D.N. 1975. Determination of recent sedimentation
  rates in Lake Michigan using Pb-210 and Cs-137. Geochim. Cosmochim. Acta
  39, 285–304.
Robbins, J.A. 1978. Geochemical and geophysical applications of radioactive
  lead. In: The Biogeochemistry of Lead in the Environment, ed. Nriagu,
  Elsevier, pp. 285–393.
Sanchez-Cabeza, J.A. & Ruiz-Fernández, A.C. 2012. 210Pb sediment
  radiochronology: an integrated formulation and classification of dating
  models. Geochim. Cosmochim. Acta 82, 183–200.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

PB210_HALF_LIFE_YR = 22.3
LAMBDA = np.log(2.0) / PB210_HALF_LIFE_YR   # 0.03107 yr⁻¹

# ---------------------------------------------------------------------------
# Model thresholds (documented in module docstring)
# ---------------------------------------------------------------------------

MIN_LAYERS          = 5      # minimum layers with positive excess activity
MIN_DEPTH_RANGE_CM  = 3.0    # minimum depth span (cm) of fitted layers
CRS_BG_THRESHOLD    = 0.10   # flag if deepest layer holds > this fraction of I_total

# ---------------------------------------------------------------------------
# Marsh filter (same as cs137_accretion.py and plot_core_depth_profiles.py)
# ---------------------------------------------------------------------------

_MARSH_VEG  = {"emergent", "emergent marsh", "marsh"}
_EXCL_SAL   = {"fresh", "oligohaline"}

# ---------------------------------------------------------------------------
# Weighted linear regression helper
# ---------------------------------------------------------------------------

def _wls(x: np.ndarray, y: np.ndarray, sigma_y: np.ndarray | None):
    """Weighted least-squares fit y = a + b*x.

    Returns (slope, intercept, slope_se, intercept_se, r2, used_weights).
    Uses analytical WLS when sigma_y is supplied and all values are finite
    and positive; falls back to OLS otherwise.
    """
    n = len(x)
    use_w = (
        sigma_y is not None
        and np.all(np.isfinite(sigma_y))
        and np.all(sigma_y > 0)
    )

    if use_w:
        w = 1.0 / sigma_y**2
        S   = w.sum()
        Sx  = (w * x).sum()
        Sy  = (w * y).sum()
        Sxx = (w * x**2).sum()
        Sxy = (w * x * y).sum()
        D   = S * Sxx - Sx**2
        if D == 0:
            use_w = False  # degenerate; fall through to OLS
        else:
            b  = (S * Sxy - Sx * Sy) / D
            a  = (Sxx * Sy - Sx * Sxy) / D
            sb = np.sqrt(S / D)
            sa = np.sqrt(Sxx / D)
            y_mean_w = Sy / S
            ss_res = (w * (y - a - b * x)**2).sum()
            ss_tot = (w * (y - y_mean_w)**2).sum()
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            return b, a, sb, sa, r2, True

    # OLS fallback
    result = stats.linregress(x, y)
    return result.slope, result.intercept, result.stderr, result.intercept_stderr, result.rvalue**2, False


# ---------------------------------------------------------------------------
# CIC
# ---------------------------------------------------------------------------

def fit_cic(profile: pd.DataFrame) -> dict:
    """Fit Constant Initial Concentration model to one core profile."""

    p = profile.copy()
    p["z_mid"] = (p["depth_min"] + p["depth_max"]) / 2.0

    # Only positive excess activities can be log-transformed
    pos = p[p["excess_pb210_activity"] > 0].sort_values("z_mid")

    if len(pos) < MIN_LAYERS:
        return {"flag": "too_few_layers"}
    if pos["z_mid"].max() - pos["z_mid"].min() < MIN_DEPTH_RANGE_CM:
        return {"flag": "insufficient_depth_range"}

    z   = pos["z_mid"].values
    lnA = np.log(pos["excess_pb210_activity"].values)

    # Uncertainty in log-space: σ_lnA ≈ σ_A / A
    has_se = (
        "excess_pb210_activity_se" in pos.columns
        and pos["excess_pb210_activity_se"].notna().all()
        and (pos["excess_pb210_activity_se"] > 0).all()
    )
    sigma_lnA = (pos["excess_pb210_activity_se"].values /
                 pos["excess_pb210_activity"].values) if has_se else None

    slope, intercept, slope_se, _, r2, weighted = _wls(z, lnA, sigma_lnA)

    if slope >= 0:
        return {"flag": "non_monotonic"}

    r     = LAMBDA / (-slope)           # cm yr⁻¹
    r_se  = r * slope_se / abs(slope)   # error propagation: σ_r/r = σ_slope/|slope|

    return {
        "flag":            "ok" if has_se else "no_se",
        "r_cm_yr":         r,
        "r_se_cm_yr":      r_se,
        "r_mm_yr":         r * 10.0,
        "r_se_mm_yr":      r_se * 10.0,
        "A0":              np.exp(intercept),
        "r2":              r2,
        "n_layers":        len(pos),
        "depth_min_cm":    pos["depth_min"].min(),
        "depth_max_cm":    pos["depth_max"].max(),
        "weighted":        weighted,
    }


# ---------------------------------------------------------------------------
# CFCS
# ---------------------------------------------------------------------------

def fit_cfcs(profile: pd.DataFrame) -> dict:
    """Fit Constant Flux Constant Sedimentation model (requires bulk density)."""

    p = profile.copy()
    p["z_mid"]    = (p["depth_min"] + p["depth_max"]) / 2.0
    p["thickness"] = p["depth_max"] - p["depth_min"]

    # Need bulk density for every row we fit
    if "dry_bulk_density" not in p.columns or p["dry_bulk_density"].isna().any():
        return {"flag": "no_bulk_density"}

    # Mass-depth midpoint for each layer (g cm⁻²)
    # Cumulative mass from the surface to the *bottom* of layer i:
    #   M_bottom_i = Σ_{j≤i} ρ_j Δz_j
    # Midpoint mass depth = M_bottom_i − ρ_i Δz_i / 2
    p = p.sort_values("z_mid").copy()
    p["m_layer"]  = p["dry_bulk_density"] * p["thickness"]
    p["M_bottom"] = p["m_layer"].cumsum()
    p["m_mid"]    = p["M_bottom"] - p["m_layer"] / 2.0

    pos = p[p["excess_pb210_activity"] > 0].copy()

    if len(pos) < MIN_LAYERS:
        return {"flag": "too_few_layers"}
    if pos["m_mid"].max() - pos["m_mid"].min() < 0.01:  # g cm⁻²
        return {"flag": "insufficient_depth_range"}

    m   = pos["m_mid"].values
    lnA = np.log(pos["excess_pb210_activity"].values)

    has_se = (
        "excess_pb210_activity_se" in pos.columns
        and pos["excess_pb210_activity_se"].notna().all()
        and (pos["excess_pb210_activity_se"] > 0).all()
    )
    sigma_lnA = (pos["excess_pb210_activity_se"].values /
                 pos["excess_pb210_activity"].values) if has_se else None

    slope, intercept, slope_se, _, r2, weighted = _wls(m, lnA, sigma_lnA)

    if slope >= 0:
        return {"flag": "non_monotonic"}

    F      = LAMBDA / (-slope)               # mass accumulation rate g cm⁻² yr⁻¹
    F_se   = F * slope_se / abs(slope)

    rho_mean = pos["dry_bulk_density"].mean()   # g cm⁻³
    r        = F / rho_mean                     # linear rate cm yr⁻¹
    # σ_r ≈ σ_F / ρ_mean  (ignoring uncertainty in ρ_mean itself)
    r_se     = F_se / rho_mean

    return {
        "flag":            "ok" if has_se else "no_se",
        "r_cm_yr":         r,
        "r_se_cm_yr":      r_se,
        "r_mm_yr":         r * 10.0,
        "r_se_mm_yr":      r_se * 10.0,
        "F_g_cm2_yr":      F,
        "F_se_g_cm2_yr":   F_se,
        "A0":              np.exp(intercept),
        "r2":              r2,
        "n_layers":        len(pos),
        "mean_BD_g_cm3":   rho_mean,
        "weighted":        weighted,
    }


# ---------------------------------------------------------------------------
# CRS
# ---------------------------------------------------------------------------

def fit_crs(profile: pd.DataFrame) -> dict:
    """Fit Constant Rate of Supply model."""

    p = profile.copy()
    p["z_mid"]     = (p["depth_min"] + p["depth_max"]) / 2.0
    p["thickness"] = p["depth_max"] - p["depth_min"]
    p = p.sort_values("z_mid").reset_index(drop=True)

    has_bd = (
        "dry_bulk_density" in p.columns
        and p["dry_bulk_density"].notna().all()
    )
    has_se = (
        "excess_pb210_activity_se" in p.columns
        and p["excess_pb210_activity_se"].notna().all()
    )

    # Clamp activities to ≥0 for inventory integration (negatives = at background)
    A = np.maximum(p["excess_pb210_activity"].values, 0.0)
    dz = p["thickness"].values
    bd = p["dry_bulk_density"].values if has_bd else np.ones(len(p))

    # Per-layer inventory (Bq cm⁻² if using BD, or activity·cm without BD)
    I_layer = A * bd * dz

    if I_layer.sum() <= 0:
        return {"flag": "too_few_layers"}

    # Cumulative inventory from the surface downward and from below
    I_above = np.cumsum(I_layer)                       # inventory above + including layer i
    I_total = I_layer.sum()
    # I_below[i] = inventory from layer i to the bottom (inclusive)
    I_below = I_total - np.concatenate([[0], I_above[:-1]])   # I_total − I_above_prev

    # Only layers where I_below > 0 yield a valid age
    valid = I_below > 0
    if valid.sum() < MIN_LAYERS:
        return {"flag": "too_few_layers"}

    t = np.full(len(p), np.nan)
    t[valid] = (1.0 / LAMBDA) * np.log(I_total / I_below[valid])

    # Uncertainty in ages
    if has_se:
        se_A = np.maximum(p["excess_pb210_activity_se"].values, 0.0)
        # Uncertainty in I_total
        sigma_I_total = np.sqrt(np.sum((se_A * bd * dz) ** 2))

        # Uncertainty in I_below[i]: sqrt(Σ_{j≥i} (σ_Aj bd_j dz_j)²)
        # Build from the bottom up: reverse cumsum of squared terms
        sq_terms = (se_A * bd * dz) ** 2
        # cumsum from the right: element i = sum of sq_terms[i:]
        sigma_I_below = np.sqrt(np.cumsum(sq_terms[::-1])[::-1])

        sigma_t = np.full(len(p), np.nan)
        sigma_t[valid] = (1.0 / LAMBDA) * np.sqrt(
            (sigma_I_total / I_total) ** 2
            + (sigma_I_below[valid] / I_below[valid]) ** 2
        )
    else:
        sigma_t = None

    # Mean accretion rate from linear regression of z_mid vs t
    z_v = p["z_mid"].values[valid]
    t_v = t[valid]
    sig_v = sigma_t[valid] if sigma_t is not None else None

    if z_v.max() - z_v.min() < MIN_DEPTH_RANGE_CM:
        return {"flag": "insufficient_depth_range"}

    # Regress z (depth) on t (age): slope = dz/dt = accretion rate
    slope, intercept, slope_se, _, r2, weighted = _wls(t_v, z_v, sig_v)

    if slope <= 0:
        return {"flag": "non_monotonic"}

    # Did the profile reach background?
    reached_bg = (I_below[-1] / I_total) < CRS_BG_THRESHOLD

    base_flag = "ok" if has_se else "no_se"
    flag = base_flag if reached_bg else f"{base_flag},did_not_reach_background"

    return {
        "flag":              flag,
        "r_cm_yr":           slope,
        "r_se_cm_yr":        slope_se,
        "r_mm_yr":           slope * 10.0,
        "r_se_mm_yr":        slope_se * 10.0,
        "r2":                r2,
        "n_layers":          int(valid.sum()),
        "I_total":           I_total,
        "reached_background": reached_bg,
        "used_bulk_density":  has_bd,
        "weighted":          weighted,
    }


# ---------------------------------------------------------------------------
# Data loading and filtering
# ---------------------------------------------------------------------------

def _load(ccn_dir: Path):
    cores = pd.read_csv(ccn_dir / "CCN_cores.csv", low_memory=False)
    ds    = pd.read_csv(ccn_dir / "CCN_depthseries.csv", low_memory=False)
    return cores, ds


def _marsh_ids(cores: pd.DataFrame) -> set:
    veg = cores["vegetation_class"].str.strip().str.lower()
    sal = cores["salinity_class"].str.strip().str.lower()
    return set(cores[(veg.isin(_MARSH_VEG)) & (sal.isna() | ~sal.isin(_EXCL_SAL))]["core_id"])


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyse(ccn_dir: Path, outdir: Path,
            min_layers: int = MIN_LAYERS) -> pd.DataFrame:
    outdir.mkdir(parents=True, exist_ok=True)

    cores, ds = _load(ccn_dir)
    marsh_ids = _marsh_ids(cores)

    pb = ds[
        ds["core_id"].isin(marsh_ids)
        & ds["excess_pb210_activity"].notna()
        & (ds["depth_max"] > ds["depth_min"])
    ].copy()

    global MIN_LAYERS
    MIN_LAYERS = min_layers

    print(f"Marsh cores with excess ²¹⁰Pb data: {pb['core_id'].nunique()}")

    meta_cols = [c for c in ["core_id","study_id","latitude","longitude","year",
                              "vegetation_class","salinity_class"] if c in cores.columns]
    meta = cores[meta_cols].drop_duplicates("core_id")

    rows = []
    for core_id, grp in pb.groupby("core_id"):
        grp = grp.sort_values("depth_min").reset_index(drop=True)

        cic  = fit_cic(grp)
        cfcs = fit_cfcs(grp)
        crs  = fit_crs(grp)

        row = {"core_id": core_id}

        for prefix, result in [("cic", cic), ("cfcs", cfcs), ("crs", crs)]:
            for k, v in result.items():
                row[f"{prefix}_{k}"] = v

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.merge(meta, on="core_id", how="left")

    # Summary
    for model in ("cic", "cfcs", "crs"):
        ok = df[f"{model}_flag"].str.startswith("ok").fillna(False)
        rates = df.loc[ok, f"{model}_r_mm_yr"].dropna()
        print(f"\n{model.upper()}: {ok.sum()} cores converged")
        if len(rates):
            print(f"  Accretion rate (mm yr⁻¹): "
                  f"median {rates.median():.1f}  mean {rates.mean():.1f}  "
                  f"std {rates.std():.1f}  "
                  f"5–95th pct [{rates.quantile(0.05):.1f}, {rates.quantile(0.95):.1f}]")

    csv_path = outdir / "pb210_accretion_rates.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    _plot(df, outdir)
    return df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot(df: pd.DataFrame, outdir: Path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    clip_hi = 10.0  # mm/yr — clip display for legibility

    # Row 0: histograms
    for ax, model, colour in zip(axes[0], ("cic", "cfcs", "crs"),
                                  ("#2a6496", "#6a3d9a", "#1a7a4a")):
        ok   = df[f"{model}_flag"].str.startswith("ok").fillna(False)
        vals = df.loc[ok, f"{model}_r_mm_yr"].dropna().clip(0, clip_hi)
        if vals.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
        else:
            ax.hist(vals, bins=30, color=colour, edgecolor="white", linewidth=0.4)
            ax.axvline(vals.median(), color="k", lw=1.2, ls="--",
                       label=f"median {vals.median():.1f}")
            ax.legend(fontsize=7)
        ax.set_xlabel("Accretion rate (mm yr⁻¹)")
        ax.set_ylabel("Number of cores")
        ax.set_title(f"{model.upper()}  (n={len(vals)})")

    # Row 1: pairwise comparisons
    def _pairplot(ax, df, m1, m2, col1, col2):
        ok1 = df[f"{m1}_flag"].str.startswith("ok").fillna(False)
        ok2 = df[f"{m2}_flag"].str.startswith("ok").fillna(False)
        both = df[ok1 & ok2].copy()
        x = both[f"{m1}_r_mm_yr"].clip(0, clip_hi)
        y = both[f"{m2}_r_mm_yr"].clip(0, clip_hi)
        if len(x) < 3:
            ax.text(0.5, 0.5, "too few pairs", ha="center", va="center",
                    transform=ax.transAxes)
            return
        # Error bars if available
        xe = both.get(f"{m1}_r_se_mm_yr", pd.Series(np.nan, index=both.index)).fillna(0).clip(0, 5)
        ye = both.get(f"{m2}_r_se_mm_yr", pd.Series(np.nan, index=both.index)).fillna(0).clip(0, 5)
        ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="o", ms=3, lw=0, elinewidth=0.5,
                    color="#444444", alpha=0.6, capsize=1.5, zorder=2)
        lim = max(clip_hi, max(x.max(), y.max()) * 1.05)
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, zorder=1)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel(f"{m1.upper()} (mm yr⁻¹)")
        ax.set_ylabel(f"{m2.upper()} (mm yr⁻¹)")
        ax.set_title(f"{m1.upper()} vs {m2.upper()}  (n={len(x)})")
        # Pearson r
        if len(x) >= 3:
            r, p = stats.pearsonr(x, y)
            ax.text(0.05, 0.92, f"r = {r:.2f}", transform=ax.transAxes, fontsize=7)

    _pairplot(axes[1, 0], df, "cic",  "crs",  "#2a6496", "#1a7a4a")
    _pairplot(axes[1, 1], df, "cfcs", "crs",  "#6a3d9a", "#1a7a4a")
    _pairplot(axes[1, 2], df, "cic",  "cfcs", "#2a6496", "#6a3d9a")

    fig.suptitle("²¹⁰Pb accretion rates — CCN marsh cores (non-fresh)", fontsize=11)
    fig.tight_layout()
    fig_path = outdir / "pb210_accretion_rates.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")

    # Example profiles (up to 6 CIC-ok cores)
    _plot_example_profiles(outdir)


def _plot_example_profiles(outdir: Path, n_examples: int = 6):
    """Re-fit and plot a few profiles showing the exponential fits."""
    ccn_dir = Path(__file__).parent.parent / "core_data" / "CCN_synthesis"
    if not (ccn_dir / "CCN_depthseries.csv").exists():
        return
    cores, ds = _load(ccn_dir)
    marsh_ids = _marsh_ids(cores)

    pb = ds[
        ds["core_id"].isin(marsh_ids)
        & ds["excess_pb210_activity"].notna()
        & (ds["depth_max"] > ds["depth_min"])
    ].copy()

    # Pick cores where CIC ok and enough positive layers
    candidates = []
    for cid, grp in pb.groupby("core_id"):
        grp = grp.sort_values("depth_min").reset_index(drop=True)
        res = fit_cic(grp)
        if res.get("flag", "").startswith("ok"):
            candidates.append((cid, grp, res))
        if len(candidates) >= n_examples:
            break

    if not candidates:
        return

    nc = len(candidates)
    fig, axes = plt.subplots(1, nc, figsize=(3.5 * nc, 6), sharey=False)
    if nc == 1:
        axes = [axes]

    for ax, (cid, grp, cic_res) in zip(axes, candidates):
        pos = grp[grp["excess_pb210_activity"] > 0].copy()
        pos["z_mid"] = (pos["depth_min"] + pos["depth_max"]) / 2.0
        z = pos["z_mid"].values

        # Plot data
        has_se = "excess_pb210_activity_se" in pos.columns and pos["excess_pb210_activity_se"].notna().all()
        if has_se:
            ax.errorbar(pos["excess_pb210_activity"], z,
                        xerr=pos["excess_pb210_activity_se"],
                        fmt="ko", ms=3, elinewidth=0.7, capsize=2, label="data")
        else:
            ax.plot(pos["excess_pb210_activity"], z, "ko", ms=3, label="data")

        # CIC fit line
        z_fit = np.linspace(z.min(), z.max(), 200)
        A0 = cic_res["A0"]
        r  = cic_res["r_cm_yr"]
        A_fit = A0 * np.exp(-LAMBDA * z_fit / r)
        ax.plot(A_fit, z_fit, "r-", lw=1.2,
                label=f"CIC  r={r*10:.1f}±{cic_res['r_se_cm_yr']*10:.1f} mm/yr")

        ax.invert_yaxis()
        ax.set_xlabel("Excess ²¹⁰Pb activity")
        ax.set_ylabel("Depth (cm)")
        ax.set_title(cid, fontsize=7)
        ax.legend(fontsize=6)

    fig.suptitle("Example CIC fits", fontsize=10)
    fig.tight_layout()
    ex_path = outdir / "pb210_example_cic_fits.png"
    fig.savefig(ex_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {ex_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_CCN_DIR = Path(__file__).parent.parent / "core_data" / "CCN_synthesis"


def main():
    parser = argparse.ArgumentParser(
        description="Compute CIC / CFCS / CRS ²¹⁰Pb accretion rates for CCN marsh cores."
    )
    parser.add_argument("--ccn-dir", default=str(_CCN_DIR))
    parser.add_argument(
        "--outdir",
        default=str(Path(__file__).parent.parent / "scripts" / "figures"),
    )
    parser.add_argument(
        "--min-layers", type=int, default=MIN_LAYERS,
        help=f"Minimum positive-activity layers required (default: {MIN_LAYERS})",
    )
    args = parser.parse_args()

    analyse(Path(args.ccn_dir), Path(args.outdir), min_layers=args.min_layers)


if __name__ == "__main__":
    main()
