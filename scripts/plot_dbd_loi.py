"""
plot_dbd_loi.py

Plot dry bulk density (DBD) vs loss-on-ignition (LOI / fraction organic matter)
for CCN marsh cores, overlaid with:
  1. Morris et al. (2016) mixing model:  DBD = 1 / (LOI/k1 + (1-LOI)/k2)
     published values k1=0.085 g/cm³, k2=1.99 g/cm³
  2. Best-fit mixing model using least-squares optimisation on the CCN data
     (fitting k1, k2 independently for each panel dataset).

Three panels / output files:
  - USA data only
  - UK data only
  - All data

Outputs: scripts/figures/dbd_loi_usa.png
         scripts/figures/dbd_loi_uk.png
         scripts/figures/dbd_loi_all.png
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

# Morris et al. (2016) published constants  [g/cm³]
K1_MORRIS = 0.085   # organic end-member density
K2_MORRIS = 1.99    # mineral end-member density

# ---------------------------------------------------------------------------
# Mixing model
# ---------------------------------------------------------------------------

def mixing_model(loi, k1, k2):
    """DBD = 1 / (LOI/k1 + (1-LOI)/k2)  — Morris et al. 2016."""
    return 1.0 / (loi / k1 + (1.0 - loi) / k2)


# ---------------------------------------------------------------------------
# Load and clean data
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    print("Loading depthseries …")
    ds = pd.read_csv(_DS_CSV, low_memory=False,
                     usecols=["core_id", "dry_bulk_density",
                               "fraction_organic_matter", "fraction_carbon"])
    print("Loading cores …")
    co = pd.read_csv(_CO_CSV, low_memory=False,
                     usecols=["core_id", "country"])

    # LOI: prefer fraction_organic_matter; fall back on fraction_carbon / 0.43
    ds["loi"] = ds["fraction_organic_matter"].where(
        ds["fraction_organic_matter"].notna(),
        ds["fraction_carbon"] / 0.43,
    )

    df = ds[ds["dry_bulk_density"].notna() & ds["loi"].notna()].copy()

    # Quality filter: physical limits
    df = df[(df["dry_bulk_density"] > 0) & (df["dry_bulk_density"] < 2.5) &
            (df["loi"] >= 0) & (df["loi"] <= 1.0)]

    df = df.merge(co.drop_duplicates("core_id"), on="core_id", how="left")
    print(f"  {len(df):,} rows after quality filter")
    return df


# ---------------------------------------------------------------------------
# Fit mixing model to data
# ---------------------------------------------------------------------------

def fit_mixing_model(loi: np.ndarray, dbd: np.ndarray):
    """Return best-fit (k1, k2) and their standard errors."""
    try:
        popt, pcov = curve_fit(
            mixing_model, loi, dbd,
            p0=[K1_MORRIS, K2_MORRIS],
            bounds=([0.01, 0.5], [0.5, 3.0]),
            maxfev=10000,
        )
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except RuntimeError:
        return None, None


# ---------------------------------------------------------------------------
# Plot one panel
# ---------------------------------------------------------------------------

def plot_panel(df: pd.DataFrame, title: str, out_path: Path) -> None:
    loi = df["loi"].values
    dbd = df["dry_bulk_density"].values

    # Best fit
    popt, perr = fit_mixing_model(loi, dbd)

    # Model curves
    loi_line = np.linspace(0, 1, 500)
    dbd_morris   = mixing_model(loi_line, K1_MORRIS, K2_MORRIS)
    dbd_bestfit  = mixing_model(loi_line, *popt) if popt is not None else None

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter — use density-style hexbin for large datasets
    n = len(df)
    if n > 5000:
        hb = ax.hexbin(loi, dbd, gridsize=80, cmap="YlOrRd",
                       mincnt=1, linewidths=0.1)
        cb = fig.colorbar(hb, ax=ax, label="Layer count")
        scatter_label = f"CCN data (n={n:,})"
    else:
        ax.scatter(loi, dbd, s=6, alpha=0.35, color="#2c7bb6",
                   rasterized=True, label=f"CCN data (n={n:,})")

    # Morris et al. (2016) mixing model
    ax.plot(loi_line, dbd_morris, "k--", lw=2.0,
            label=f"Morris et al. 2016  (k₁={K1_MORRIS}, k₂={K2_MORRIS})")

    # Best-fit mixing model
    if popt is not None:
        k1f, k2f = popt
        k1e, k2e = perr
        ax.plot(loi_line, dbd_bestfit, "r-", lw=2.0,
                label=f"Best fit  k₁={k1f:.3f}±{k1e:.3f}, k₂={k2f:.3f}±{k2e:.3f} g cm⁻³")
        print(f"  {title}: k1={k1f:.4f}±{k1e:.4f}  k2={k2f:.4f}±{k2e:.4f}")

    # Formatting
    ax.set_xlabel("Loss on ignition (fraction organic matter)", fontsize=12)
    ax.set_ylabel("Dry bulk density (g cm⁻³)", fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 2.0)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.25)

    if n > 5000:
        ax.text(0.02, 0.97, scatter_label, transform=ax.transAxes,
                fontsize=9, va="top", ha="left")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_data()

    datasets = [
        (df[df["country"] == "United States"], "USA marsh cores",      "dbd_loi_usa.png"),
        (df[df["country"] == "United Kingdom"], "UK marsh cores",       "dbd_loi_uk.png"),
        (df,                                    "All CCN marsh cores",  "dbd_loi_all.png"),
    ]

    for subset, title, fname in datasets:
        n = len(subset)
        print(f"\n{title}: {n:,} rows")
        if n < 10:
            print("  Too few points — skipping")
            continue
        plot_panel(subset, f"{title} — DBD vs LOI\nn={n:,}", _FIG_DIR / fname)

    print("\nDone.")


if __name__ == "__main__":
    main()
