"""
plot_dbd_loi_by_depth.py

Plot dry bulk density (DBD) vs LOI for CCN marsh cores, stratified into
20 cm depth bins, to test whether the Morris et al. (2016) mixing model
has any systematic dependence on overburden depth.

For each depth bin:
  - Show a hexbin scatter of the CCN data
  - Fit the mixing model  DBD = 1 / (LOI/k1 + (1-LOI)/k2)
  - Overlay the best-fit curve and the Morris published curve

Two output figures:
  scripts/figures/dbd_loi_by_depth_usa.png   (USA cores only)
  scripts/figures/dbd_loi_by_depth_uk.png    (UK cores only)

A summary table of fitted k1, k2 vs depth mid-point is printed to stdout
and saved as scripts/figures/dbd_loi_depth_params.csv.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT    = Path(__file__).parent.parent
_FIG_DIR = _ROOT / "scripts" / "figures"
_DS_CSV  = _ROOT / "core_data" / "CCN_synthesis" / "CCN_depthseries.csv"
_CO_CSV  = _ROOT / "core_data" / "CCN_synthesis" / "CCN_cores.csv"

# Morris et al. (2016) published constants [g/cm³]
K1_MORRIS = 0.085
K2_MORRIS = 1.99

# Depth bins (cm)
DEPTH_EDGES = [0, 20, 40, 60, 80, 100, 150, 9999]

# ---------------------------------------------------------------------------
# Mixing model
# ---------------------------------------------------------------------------

def mixing_model(loi, k1, k2):
    return 1.0 / (loi / k1 + (1.0 - loi) / k2)


def fit_mixing(loi, dbd):
    try:
        popt, pcov = curve_fit(
            mixing_model, loi, dbd,
            p0=[K1_MORRIS, K2_MORRIS],
            bounds=([0.005, 0.3], [0.8, 3.0]),
            maxfev=20000,
        )
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except (RuntimeError, ValueError):
        return None, None


def r_squared(loi, dbd, k1, k2):
    pred   = mixing_model(loi, k1, k2)
    ss_res = np.sum((dbd - pred) ** 2)
    ss_tot = np.sum((dbd - dbd.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


# ---------------------------------------------------------------------------
# Load and clean data
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    print("Loading depthseries …")
    ds = pd.read_csv(_DS_CSV, low_memory=False,
                     usecols=["core_id", "dry_bulk_density",
                               "fraction_organic_matter", "fraction_carbon",
                               "depth_min", "depth_max"])
    co = pd.read_csv(_CO_CSV, low_memory=False,
                     usecols=["core_id", "country"])

    ds["loi"] = ds["fraction_organic_matter"].where(
        ds["fraction_organic_matter"].notna(),
        ds["fraction_carbon"] / 0.43,
    )
    ds["depth_mid"] = (ds["depth_min"] + ds["depth_max"]) / 2.0

    df = ds[ds["dry_bulk_density"].notna() & ds["loi"].notna() &
            ds["depth_mid"].notna()].copy()
    df = df[(df["dry_bulk_density"] > 0) & (df["dry_bulk_density"] < 2.5) &
            (df["loi"] >= 0) & (df["loi"] <= 1.0) &
            (df["depth_mid"] >= 0)]

    df = df.merge(co.drop_duplicates("core_id"), on="core_id", how="left")
    print(f"  {len(df):,} rows after quality filter")
    return df


# ---------------------------------------------------------------------------
# Make depth-bin label
# ---------------------------------------------------------------------------

def _bin_label(lo, hi):
    if hi >= 9000:
        return f"{lo}+ cm"
    return f"{lo}–{hi} cm"


# ---------------------------------------------------------------------------
# Plot one country
# ---------------------------------------------------------------------------

def plot_country(df: pd.DataFrame, country_label: str, out_path: Path,
                 params_rows: list) -> None:
    n_bins = len(DEPTH_EDGES) - 1
    ncols  = 3
    nrows  = (n_bins + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4),
                             sharex=True, sharey=True)
    axes_flat = axes.flatten()

    loi_line = np.linspace(0.001, 1.0, 500)

    for idx, (lo, hi) in enumerate(zip(DEPTH_EDGES[:-1], DEPTH_EDGES[1:])):
        ax = axes_flat[idx]
        label = _bin_label(lo, hi)

        sub = df[(df["depth_mid"] >= lo) & (df["depth_mid"] < hi)]
        n   = len(sub)

        if n < 20:
            ax.text(0.5, 0.5, f"{label}\nn={n} (too few)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_visible(True)
            continue

        loi = sub["loi"].values
        dbd = sub["dry_bulk_density"].values

        # Background hexbin
        if n > 500:
            ax.hexbin(loi, dbd, gridsize=50, cmap="YlOrRd",
                      mincnt=1, linewidths=0.1, alpha=0.85)
        else:
            ax.scatter(loi, dbd, s=6, alpha=0.35, color="#2c7bb6", rasterized=True)

        # Morris model
        r2_morris = r_squared(loi, dbd, K1_MORRIS, K2_MORRIS)
        ax.plot(loi_line, mixing_model(loi_line, K1_MORRIS, K2_MORRIS),
                "k--", lw=1.8,
                label=f"Morris  k₁={K1_MORRIS}, k₂={K2_MORRIS}  R²={r2_morris:.3f}")

        # Best-fit model
        popt, perr = fit_mixing(loi, dbd)
        r2_fit = np.nan
        if popt is not None:
            k1f, k2f = popt
            k1e, k2e = perr
            r2_fit = r_squared(loi, dbd, k1f, k2f)
            ax.plot(loi_line, mixing_model(loi_line, k1f, k2f),
                    "r-", lw=1.8,
                    label=f"Fit  k₁={k1f:.3f}±{k1e:.3f}\n"
                          f"     k₂={k2f:.3f}±{k2e:.3f}  R²={r2_fit:.3f}")
            depth_mid_cm = (lo + min(hi, 150)) / 2.0
            params_rows.append({
                "country":       country_label,
                "depth_bin":     label,
                "depth_lo_cm":   lo,
                "depth_hi_cm":   min(hi, 150),
                "depth_mid_cm":  depth_mid_cm,
                "n":             n,
                "k1":            round(k1f, 5),
                "k1_se":         round(k1e, 5),
                "k2":            round(k2f, 5),
                "k2_se":         round(k2e, 5),
                "r2_fit":        round(r2_fit, 4),
                "r2_morris":     round(r2_morris, 4),
            })
            print(f"  {country_label:5s} {label:12s}  n={n:6,}  "
                  f"k1={k1f:.4f}±{k1e:.4f}  k2={k2f:.4f}±{k2e:.4f}  "
                  f"R²_fit={r2_fit:.3f}  R²_Morris={r2_morris:.3f}")

        ax.set_xlim(-0.02, 0.80)
        ax.set_ylim(-0.05, 2.05)
        ax.set_title(f"{label}  (n={n:,})  R²: fit={r2_fit:.3f}  Morris={r2_morris:.3f}",
                     fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.25)
        if idx % ncols == 0:
            ax.set_ylabel("DBD (g cm⁻³)", fontsize=9)
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("LOI (fraction OM)", fontsize=9)

    # Hide unused axes
    for i in range(n_bins, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle(f"{country_label} marsh cores — DBD vs LOI by depth bin", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Polynomial fit helper
# ---------------------------------------------------------------------------

def fit_poly2(x, y):
    """Fit a 2nd-order polynomial; return (coeffs, r2).  x, y are arrays."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    coeffs = np.polyfit(x, y, 2)          # [a2, a1, a0]
    pred   = np.polyval(coeffs, x)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return coeffs, r2


# ---------------------------------------------------------------------------
# Summary k1/k2 vs depth plot
# ---------------------------------------------------------------------------

def plot_param_trends(params_df: pd.DataFrame, out_path: Path) -> None:
    countries  = params_df["country"].unique()
    colors     = {"USA": "#1b7837", "UK": "#762a83", "All": "#d95f02"}
    poly_ls    = {"USA": "--",      "UK": "-.",      "All": "-"}
    depth_line = np.linspace(0, 200, 300)   # cm

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for param, ax, ylabel, morris_val in [
        ("k1", axes[0], "k₁ (organic end-member density, g cm⁻³)", K1_MORRIS),
        ("k2", axes[1], "k₂ (mineral end-member density, g cm⁻³)", K2_MORRIS),
    ]:
        ax.axhline(morris_val, color="k", ls=":", lw=1.5,
                   label=f"Morris 2016 ({morris_val})")

        for country in countries:
            sub = params_df[params_df["country"] == country].sort_values("depth_mid_cm")
            if len(sub) < 3:
                continue
            col = colors.get(country, "grey")

            # Binned data points with error bars
            ax.errorbar(sub["depth_mid_cm"], sub[param],
                        yerr=sub[f"{param}_se"],
                        fmt="o", color=col, capsize=4, lw=1.5, ms=6,
                        label=f"{country} bins")

            # 2nd-order polynomial fit
            coeffs, r2 = fit_poly2(sub["depth_mid_cm"].values, sub[param].values)
            fit_vals = np.polyval(coeffs, depth_line)
            ax.plot(depth_line, fit_vals,
                    color=col, ls=poly_ls.get(country, "-"), lw=2.0,
                    label=f"{country} poly fit  R²={r2:.3f}\n"
                          f"  a₀={coeffs[2]:.5f}  a₁={coeffs[1]:.5f}  a₂={coeffs[0]:.5f}")

            # Print coefficients to stdout (depth in cm and metres)
            coeffs_m, _ = fit_poly2([d / 100 for d in sub["depth_mid_cm"].values],
                                    sub[param].values)
            print(f"  {country:5s} {param}: a0={coeffs_m[2]:.6f}  "
                  f"a1={coeffs_m[1]:.6f}  a2={coeffs_m[0]:.6f}  R²={r2:.4f}  "
                  f"(depth in m)")

        ax.set_xlabel("Depth bin mid-point (cm)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"Best-fit {param} vs depth — 2nd-order polynomial", fontsize=10)
        ax.set_xlim(0, 175)
        ax.legend(fontsize=7, loc="upper right" if param == "k2" else "lower right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Mixing-model parameters vs depth (CCN synthesis database)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_data()
    params_rows = []

    for country, label, fname in [
        ("United States", "USA", "dbd_loi_by_depth_usa.png"),
        ("United Kingdom", "UK",  "dbd_loi_by_depth_uk.png"),
        (None,             "All", "dbd_loi_by_depth_all.png"),
    ]:
        sub = df if country is None else df[df["country"] == country]
        print(f"\n{label}: {len(sub):,} rows with depth data")
        plot_country(sub, label, _FIG_DIR / fname, params_rows)

    if params_rows:
        params_df = pd.DataFrame(params_rows)
        csv_out = _FIG_DIR / "dbd_loi_depth_params.csv"
        params_df.to_csv(csv_out, index=False)
        print(f"\nParameter table saved: {csv_out}")
        print(params_df[["country","depth_bin","n","k1","k1_se","k2","k2_se"]].to_string(index=False))

        plot_param_trends(params_df, _FIG_DIR / "dbd_loi_depth_trends.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
