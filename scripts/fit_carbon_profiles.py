"""
fit_carbon_profiles.py

Fit depth-organic-matter profiles for CCN marsh cores using three models:

  1. Exponential + offset   OM(z) = (OM0 - OMinf)*exp(-k*z) + OMinf
  2. Gaussian + offset      OM(z) = A*exp(-(z-mu)²/(2*sigma²)) + OMinf
  3. Asymmetric + offset    OM(z) = A*logistic(z,zp,s)*exp(-k*(z-zp)) + OMinf

The gamma-shape model is excluded because it forces OM→0 at the surface,
which is physically unrealistic for marsh soils.

All profiles are in fraction_organic_matter units.  Where only fraction_carbon
is available, it is converted via OM = OC / 0.43.

Quality filter
--------------
MIN_LAYERS = 6   minimum depth layers per core
R2_MIN     = 0.75 best-fit R² threshold; cores below this get NaN in all
                  derived columns (they are not dropped from the FGB)

Derived quantities (all in fraction_organic_matter unless noted)
----------------------------------------------------------------
best_model          name of best-fitting model (lowest AIC, R²≥R2_MIN), or NaN
deep_om_frac        OMinf (asymptote of best fit); NaN if no data below 30 cm
doubling_depth_cm   deepest depth where fitted OM(z) > 2 × OMinf; NaN if
                    OM never exceeds 2×OMinf or OMinf ≤ 0
mean_om_above_dbl   mean fitted OM from z=0 to doubling_depth_cm

Outputs
-------
  scripts/core_inundation.fgb              — updated with new columns
  scripts/figures/carbon_profile_fits.png  — summary figure
  scripts/figures/carbon_profile_fit_results.csv
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIN_LAYERS   = 6
R2_MIN       = 0.75
MAX_DEPTH_CM = 200
OC_TO_OM     = 1.0 / 0.43     # OM = OC / 0.43

_ROOT    = Path(__file__).parent.parent
_FIG_DIR = _ROOT / "scripts" / "figures"
_FGB     = _ROOT / "scripts" / "core_inundation.fgb"
_DS_CSV  = _ROOT / "core_data" / "CCN_synthesis" / "CCN_depthseries.csv"
_CO_CSV  = _ROOT / "core_data" / "CCN_synthesis" / "CCN_cores.csv"


# ---------------------------------------------------------------------------
# Model definitions  (all in OM units)
# ---------------------------------------------------------------------------

def model_exp(z, OM0, OMinf, k):
    return (OM0 - OMinf) * np.exp(-k * z) + OMinf


def model_gaussian(z, A, mu, sigma, OMinf):
    return A * np.exp(-0.5 * ((z - mu) / sigma) ** 2) + OMinf


def model_asym(z, A, zp, k, s, OMinf):
    # clip exponent to avoid overflow
    arg = np.clip(-s * (z - zp), -500, 500)
    rise  = 1.0 / (1.0 + np.exp(arg))
    decay = np.exp(-k * np.maximum(z - zp, 0))
    return A * rise * decay + OMinf


_MODELS = {
    "exp":   (model_exp,      3),
    "gauss": (model_gaussian, 4),
    "asym":  (model_asym,     5),
}

# Cinf is the last parameter in every model
def _cinf(tag, params):
    return params[-1]


# ---------------------------------------------------------------------------
# AIC / R²
# ---------------------------------------------------------------------------

def aic(y, y_pred, n_params):
    n   = len(y)
    rss = max(np.sum((y - y_pred) ** 2), 1e-30)
    return n * np.log(rss / n) + 2 * n_params


def r2_score(y, y_pred):
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


# ---------------------------------------------------------------------------
# Fit one core
# ---------------------------------------------------------------------------

def _deep_om_mean(z: np.ndarray, om: np.ndarray) -> float:
    """Mean OM in the deepest 25% of layers (at least 2 points)."""
    n_deep = max(2, int(np.ceil(len(z) * 0.25)))
    idx    = np.argsort(z)[-n_deep:]
    return float(om[idx].mean())


def _validate_fit(tag: str, params: np.ndarray,
                  z: np.ndarray, om: np.ndarray) -> bool:
    """Return False if the fit is physically implausible.

    Checks:
    1. OMinf must be ≤ deep_om_mean × 1.5  (asymptote cannot exceed actual deep OM)
    2. OMinf must be ≤ min(om) × 1.5       (cannot sit above nearly all the data)
    3. exp model: OM0 (params[0]) must be ≥ OMinf (profile must decay, not rise)
    """
    om_inf    = float(params[-1])
    deep_mean = _deep_om_mean(z, om)

    if om_inf > deep_mean * 1.5:
        return False
    if om_inf > om.min() * 1.5 + 0.01:   # small additive tolerance for near-zero data
        return False
    if tag == "exp" and params[0] < params[1]:
        return False   # OM0 < OMinf → profile increases with depth → wrong shape
    return True


def fit_core(z: np.ndarray, om: np.ndarray, has_deep: bool) -> dict:
    """Fit all models; return diagnostics dict."""
    om_lo   = om.min()
    om_hi   = om.max()
    om_rng  = max(om_hi - om_lo, 1e-4)
    z_mid   = np.median(z)
    k_guess = 1.0 / max(z_mid, 5.0)

    # Dynamic upper bound on OMinf: no higher than deep data mean
    deep_mean  = _deep_om_mean(z, om)
    ominf_ceil = max(deep_mean * 1.5, om_lo + 0.005)  # small floor for flat profiles

    p0_bounds = {
        "exp":   (
            [om_hi, om_lo, k_guess],
            ([0, 0, 1e-5], [2.0, ominf_ceil, 10.0]),
        ),
        "gauss": (
            [om_rng, z_mid, z_mid, om_lo],
            ([0, -50, 0.5, 0], [2.0, 200, 200, ominf_ceil]),
        ),
        "asym":  (
            [om_rng, z_mid, k_guess, 0.2, om_lo],
            ([0, 0, 1e-5, 0.01, 0], [5.0, 200, 5.0, 5.0, ominf_ceil]),
        ),
    }

    fits = {}
    for tag, (func, n_p) in _MODELS.items():
        p0, (lb, ub) = p0_bounds[tag]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(func, z, om, p0=p0,
                                    bounds=(lb, ub), maxfev=10000)
            pred  = func(z, *popt)
            valid = _validate_fit(tag, popt, z, om)
            fits[tag] = {
                "params": popt,
                "r2":     r2_score(om, pred) if valid else np.nan,
                "aic":    aic(om, pred, n_p) if valid else np.nan,
                "ok":     valid,
            }
        except Exception:
            fits[tag] = {"params": None, "r2": np.nan, "aic": np.nan, "ok": False}

    # Best model: lowest AIC among those with R² ≥ R2_MIN
    qualified = {t: v for t, v in fits.items() if v["ok"] and v["r2"] >= R2_MIN}
    best_tag  = min(qualified, key=lambda t: qualified[t]["aic"]) if qualified else None

    out = {"n": len(z), "z_max": z.max(), "best_model": best_tag}

    for tag, v in fits.items():
        out[f"{tag}_r2"]  = v["r2"]
        out[f"{tag}_aic"] = v["aic"]
        out[f"{tag}_ok"]  = v["ok"]
        if v["params"] is not None:
            for j, val in enumerate(v["params"]):
                out[f"{tag}_p{j}"] = val

    if best_tag is None:
        out.update({"deep_om_frac": np.nan,
                    "doubling_depth_cm": np.nan,
                    "mean_om_above_dbl": np.nan})
        return out

    # --- Derived quantities from best fit ---
    params = fits[best_tag]["params"]
    func   = _MODELS[best_tag][0]
    om_inf = _cinf(best_tag, params)

    # deep_om_frac: only if data exists below 30 cm AND deep data is
    # consistent with asymptote (within factor 2)
    if has_deep and om_inf > 0 and om_inf <= deep_mean * 2.0:
        out["deep_om_frac"] = float(om_inf)
    else:
        out["deep_om_frac"] = np.nan

    # Evaluate fitted curve on fine grid (only to last data point)
    z_fine = np.linspace(0, z.max(), 2000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        om_fine = func(z_fine, *params)

    # Doubling depth: last depth where fitted OM > 2 × om_inf
    if om_inf > 0 and np.any(om_fine > 2.0 * om_inf):
        last_idx  = np.where(om_fine > 2.0 * om_inf)[0][-1]
        dbl_depth = float(z_fine[last_idx])
    else:
        dbl_depth = np.nan

    out["doubling_depth_cm"] = dbl_depth

    # Mean OM above doubling depth
    if not np.isnan(dbl_depth) and dbl_depth > 0:
        mask = z_fine <= dbl_depth
        out["mean_om_above_dbl"] = float(om_fine[mask].mean())
    else:
        out["mean_om_above_dbl"] = np.nan

    return out


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    cores     = pd.read_csv(_CO_CSV, usecols=["core_id", "habitat"])
    marsh_ids = set(cores.loc[cores["habitat"] == "marsh", "core_id"])

    ds = pd.read_csv(_DS_CSV, usecols=[
        "core_id", "depth_min", "depth_max",
        "fraction_organic_matter", "fraction_carbon"])
    ds = ds[ds["core_id"].isin(marsh_ids)].copy()

    ds["depth_mid"] = (ds["depth_min"].fillna(0) +
                       ds["depth_max"].fillna(ds["depth_min"].fillna(0))) / 2.0

    # OM: use fraction_organic_matter directly; convert OC if missing
    ds["om"] = ds["fraction_organic_matter"].where(
        ds["fraction_organic_matter"].notna(),
        ds["fraction_carbon"] * OC_TO_OM,
    )
    ds = ds.dropna(subset=["om", "depth_mid"])
    ds = ds[ds["depth_mid"] <= MAX_DEPTH_CM]
    ds = ds[ds["om"].between(0, 1.5)]   # allow slightly above 1 for peat
    return ds


# ---------------------------------------------------------------------------
# Main fitting loop
# ---------------------------------------------------------------------------

def run_fits(ds: pd.DataFrame) -> pd.DataFrame:
    grouped  = ds.groupby("core_id")
    valid    = [cid for cid, g in grouped if len(g) >= MIN_LAYERS]
    print(f"Fitting {len(valid)} cores (≥{MIN_LAYERS} layers, depth ≤ {MAX_DEPTH_CM} cm) …")

    records = []
    for i, cid in enumerate(valid):
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(valid)}")
        g        = grouped.get_group(cid).sort_values("depth_mid")
        z        = g["depth_mid"].values.astype(float)
        om       = g["om"].values.astype(float)
        has_deep = bool(np.any(z > 30.0))
        r        = fit_core(z, om, has_deep)
        r["core_id"] = cid
        records.append(r)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Update FGB
# ---------------------------------------------------------------------------

def update_fgb(results: pd.DataFrame):
    print(f"Updating {_FGB} …")
    gdf = gpd.read_file(_FGB)

    new_cols = ["best_model", "deep_om_frac", "doubling_depth_cm", "mean_om_above_dbl"]

    # Drop old versions of these columns if they exist
    gdf = gdf.drop(columns=[c for c in new_cols if c in gdf.columns], errors="ignore")

    merge = results[["core_id"] + new_cols].copy()
    gdf   = gdf.merge(merge, on="core_id", how="left")

    gdf.to_file(_FGB, driver="FlatGeobuf")
    n_filled = gdf["best_model"].notna().sum()
    print(f"  {n_filled}/{len(gdf)} FGB cores have a valid best_model")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(df: pd.DataFrame, ds: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    tags   = ["exp", "gauss", "asym"]
    labels = ["Exp+offset", "Gaussian+offset", "Asymmetric+offset"]
    colors = ["#1f78b4", "#e08214", "#d73027"]

    fig = plt.figure(figsize=(14, 14))
    gs  = plt.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

    # ── A: R² distributions ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :2])
    for t, lbl, col in zip(tags, labels, colors):
        vals = df[f"{t}_r2"].dropna()
        vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=40, alpha=0.5, color=col,
                label=f"{lbl} (med={vals.median():.2f})")
    ax.axvline(R2_MIN, color="k", lw=1, ls="--", alpha=0.6, label=f"R²={R2_MIN}")
    ax.set_xlabel("R²", fontsize=9)
    ax.set_ylabel("Number of cores", fontsize=9)
    ax.set_title("A  R² distribution", fontsize=9, loc="left", fontweight="bold")
    ax.legend(fontsize=7)

    # ── B: % cores with R² ≥ R2_MIN ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2:])
    fracs = [(df[f"{t}_r2"] >= R2_MIN).mean() * 100 for t in tags]
    bars  = ax2.bar(labels, fracs, color=colors, alpha=0.8)
    ax2.bar_label(bars, fmt="%.0f%%", fontsize=8)
    ax2.set_ylabel(f"% cores with R² ≥ {R2_MIN}", fontsize=9)
    ax2.set_title(f"B  Good fits (R² ≥ {R2_MIN})", fontsize=9, loc="left", fontweight="bold")
    ax2.tick_params(axis="x", labelsize=7)
    ax2.set_ylim(0, 100)

    # ── C: ΔAIC ──────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    best_counts = df["best_model"].value_counts()
    for t, lbl, col in zip(tags, labels, colors):
        vals = df[f"{t}_daic"].dropna() if f"{t}_daic" in df else pd.Series(dtype=float)
        # compute on-the-fly
        best_aic = df[[f"{tt}_aic" for tt in tags]].min(axis=1)
        vals = (df[f"{t}_aic"] - best_aic).dropna()
        vals = vals[vals < 20]
        ax3.hist(vals, bins=40, alpha=0.5, color=col,
                 label=f"{lbl} (best: {best_counts.get(t, 0)})")
    ax3.axvline(2, color="k", lw=0.7, ls="--", alpha=0.5, label="ΔAIC=2")
    ax3.set_xlabel("ΔAIC vs best model", fontsize=9)
    ax3.set_ylabel("Number of cores", fontsize=9)
    ax3.set_title("C  ΔAIC (0 = best for that core)", fontsize=9,
                  loc="left", fontweight="bold")
    ax3.legend(fontsize=7)

    # ── D: doubling depth distribution ───────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2:])
    dd = df["doubling_depth_cm"].dropna()
    ax4.hist(dd.clip(0, 100), bins=40, color="#7b2d8b", alpha=0.8)
    ax4.axvline(dd.median(), color="k", lw=1.2, ls="--",
                label=f"Median {dd.median():.1f} cm")
    ax4.set_xlabel("Doubling depth (cm)", fontsize=9)
    ax4.set_ylabel("Number of cores", fontsize=9)
    ax4.set_title("D  Depth where OM = 2 × deep OM", fontsize=9,
                  loc="left", fontweight="bold")
    ax4.legend(fontsize=8)

    # ── Row 2: Example profiles (best R² per winning model) ──────────────
    grouped = ds.groupby("core_id")
    ex_axes = [fig.add_subplot(gs[2, i]) for i in range(3)]

    for ax_e, t, lbl, col in zip(ex_axes, tags, labels, colors):
        sub = df[(df["best_model"] == t) & df[f"{t}_r2"].notna()]
        if len(sub) == 0:
            sub = df[df[f"{t}_r2"].notna()]
        sub = sub.nlargest(1, f"{t}_r2")
        if len(sub) == 0:
            ax_e.set_visible(False)
            continue
        row = sub.iloc[0]
        cid = row["core_id"]
        if cid not in grouped.groups:
            continue
        g  = grouped.get_group(cid).sort_values("depth_mid")
        z  = g["depth_mid"].values.astype(float)
        om = g["om"].values.astype(float)

        n_p    = _MODELS[t][1]
        params = [row.get(f"{t}_p{j}", np.nan) for j in range(n_p)]
        if any(np.isnan(params)):
            continue

        ax_e.scatter(om, z, color="#555555", s=20, zorder=3, label="data")
        z_fine  = np.linspace(0, min(z.max() * 1.05, MAX_DEPTH_CM), 500)
        func    = _MODELS[t][0]
        try:
            pred = func(z_fine, *params)
            ax_e.plot(pred, z_fine, color=col, lw=2,
                      label=f"fit R²={row[f'{t}_r2']:.2f}", zorder=4)
        except Exception:
            pass

        # Mark deep OM and doubling depth
        om_inf = params[-1]
        if om_inf > 0:
            ax_e.axvline(om_inf, color=col, lw=0.8, ls=":", alpha=0.7,
                         label=f"OMinf={om_inf:.3f}")
            ax_e.axvline(2 * om_inf, color=col, lw=0.8, ls="--", alpha=0.7,
                         label=f"2×OMinf={2*om_inf:.3f}")
        if not np.isnan(row.get("doubling_depth_cm", np.nan)):
            ax_e.axhline(row["doubling_depth_cm"], color="grey",
                         lw=0.8, ls="--", alpha=0.7)

        ax_e.invert_yaxis()
        ax_e.set_xlabel("Fraction organic matter", fontsize=8)
        ax_e.set_ylabel("Depth (cm)", fontsize=8)
        roman = ["i", "ii", "iii", "iv"][tags.index(t)]
        ax_e.set_title(f"E{roman}  {lbl}\n{cid[:28]}", fontsize=7.5,
                       loc="left", fontweight="bold")
        ax_e.legend(fontsize=6.5)

    fig.suptitle(
        f"Organic matter depth profile fits — CCN marsh cores\n"
        f"n = {len(df)} cores fitted (≥{MIN_LAYERS} layers); "
        f"R² ≥ {R2_MIN} required for derived quantities; "
        f"OM = fraction_organic_matter or OC/0.43",
        fontsize=9,
    )

    out = outdir / "carbon_profile_fits.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    ds = load_data()
    print(f"  {ds['core_id'].nunique()} marsh cores, {len(ds)} depth layers")

    results = run_fits(ds)

    # ── Summary ────────────────────────────────────────────────────────
    tags   = ["exp", "gauss", "asym"]
    labels = ["Exp+offset", "Gaussian+offset", "Asymmetric+offset"]
    best_counts = results["best_model"].value_counts()

    print(f"\n── Model fit summary (R²≥{R2_MIN} to qualify as best) ─────────────")
    print(f"{'Model':<26}  {'n_ok':>5}  {f'R²≥{R2_MIN}':>7}  {'median R²':>9}  {'best (AIC)':>10}")
    for t, lbl in zip(tags, labels):
        ok  = results[f"{t}_r2"].notna()
        r2  = results.loc[ok, f"{t}_r2"]
        frac = (r2 >= R2_MIN).mean() * 100
        print(f"  {lbl:<24}  {ok.sum():>5}  {frac:>6.0f}%  {r2.median():>9.3f}  "
              f"{best_counts.get(t, 0):>10}")

    n_best = results["best_model"].notna().sum()
    print(f"\n  Cores with a qualifying best model: {n_best}/{len(results)}")

    dd = results["doubling_depth_cm"].dropna()
    print(f"  Doubling depth: n={len(dd)}, median={dd.median():.1f} cm, "
          f"mean={dd.mean():.1f} cm")

    dom = results["deep_om_frac"].dropna()
    print(f"  Deep OM fraction (OMinf): n={len(dom)}, median={dom.median():.3f}, "
          f"mean={dom.mean():.3f}")

    # Save CSV — params included so plot scripts don't need to refit
    out_csv = _FIG_DIR / "carbon_profile_fit_results.csv"
    results.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    make_figure(results, ds, _FIG_DIR)
    update_fgb(results)


if __name__ == "__main__":
    main()
