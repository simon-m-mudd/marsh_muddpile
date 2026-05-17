"""
plot_spartina_profiles_by_state.py

One figure per east-coast US state showing OM depth profiles for all
Spartina marsh cores with a qualifying fit (R² ≥ 0.75, ≥ 6 layers).

Within each figure the depth (y) axis is identical across all panels so
profiles are directly comparable.  Panels are sorted by latitude (S → N).

Outputs
-------
  scripts/figures/spartina_profiles_<state>.png  for each state
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT    = Path(__file__).parent.parent
_FIG_DIR = _ROOT / "scripts" / "figures"
_DS_CSV  = _ROOT / "core_data" / "CCN_synthesis" / "CCN_depthseries.csv"
_CO_CSV  = _ROOT / "core_data" / "CCN_synthesis" / "CCN_cores.csv"
_SP_CSV  = _ROOT / "core_data" / "CCN_synthesis" / "CCN_species.csv"
_FIT_CSV = _FIG_DIR / "carbon_profile_fit_results.csv"

OC_TO_OM  = 1.0 / 0.43
MAX_DEPTH  = 200.0
NCOLS      = 5        # panels per row

# ---------------------------------------------------------------------------
# State bounding boxes  (name, lat_lo, lat_hi, lon_lo, lon_hi)  N → S
# lon_lo is the more-negative (western) bound
# ---------------------------------------------------------------------------
STATE_BOXES = [
    ("Maine",          43.0, 47.5, -71.2, -66.9),
    ("New Hampshire",  42.7, 43.3, -71.4, -70.5),
    ("Massachusetts",  41.2, 42.9, -73.6, -69.9),
    ("Rhode Island",   41.0, 42.1, -72.0, -71.0),
    ("Connecticut",    40.9, 42.1, -73.8, -71.6),
    ("New York",       40.3, 41.5, -74.4, -71.6),
    ("New Jersey",     38.9, 41.4, -75.7, -73.8),
    ("Delaware",       38.3, 39.9, -75.9, -74.8),
    ("Maryland",       37.8, 39.7, -77.4, -74.9),
    ("Virginia",       36.4, 38.7, -77.6, -75.1),
    ("North Carolina", 33.7, 36.7, -79.0, -75.3),
    ("South Carolina", 31.9, 34.0, -81.6, -78.3),
    ("Georgia",        30.2, 32.2, -82.3, -80.3),
    ("Florida",        24.3, 30.5, -83.0, -79.8),
]

# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------

def model_exp(z, OM0, OMinf, k):
    return (OM0 - OMinf) * np.exp(-k * z) + OMinf

def model_gaussian(z, A, mu, sigma, OMinf):
    return A * np.exp(-0.5 * ((z - mu) / sigma) ** 2) + OMinf

def model_asym(z, A, zp, k, s, OMinf):
    arg   = np.clip(-s * (z - zp), -500, 500)
    rise  = 1.0 / (1.0 + np.exp(arg))
    decay = np.exp(-k * np.maximum(z - zp, 0))
    return A * rise * decay + OMinf

_MODELS = {
    "exp":   (model_exp,      3),
    "gauss": (model_gaussian, 4),
    "asym":  (model_asym,     5),
}
_COLORS = {"exp": "#1f78b4", "gauss": "#e08214", "asym": "#d73027"}

def get_params(row: pd.Series, tag: str) -> np.ndarray | None:
    """Read fitted parameters from results row (saved by fit_carbon_profiles.py)."""
    n_p = _MODELS[tag][1]
    params = [row.get(f"{tag}_p{j}", np.nan) for j in range(n_p)]
    if any(pd.isna(p) for p in params):
        return None
    return np.array(params, dtype=float)

# ---------------------------------------------------------------------------
# Assign state
# ---------------------------------------------------------------------------

def assign_state(lat: float, lon: float) -> str | None:
    for name, lat_lo, lat_hi, lon_lo, lon_hi in STATE_BOXES:
        if lat_lo <= lat <= lat_hi and lon_lo <= lon <= lon_hi:
            return name
    return None

# ---------------------------------------------------------------------------
# Plot one state
# ---------------------------------------------------------------------------

def plot_state(state: str, core_rows: pd.DataFrame,
               ds: pd.DataFrame, results: pd.DataFrame,
               cores_meta: pd.DataFrame, sp: pd.DataFrame):

    # Sort S → N
    core_rows = core_rows.sort_values("latitude")
    ids = core_rows["core_id"].tolist()
    n   = len(ids)

    ncols = min(NCOLS, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(3.5 * ncols, 7 * nrows),
                              sharey=True)
    axes_flat = np.array(axes).flatten()

    # Hide unused panels
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    # Determine shared y limit: deepest data point across all cores in state
    max_z = 0.0
    grouped = ds.groupby("core_id")
    for cid in ids:
        if cid in grouped.groups:
            g = grouped.get_group(cid)
            g_z = (g["depth_min"].fillna(0) +
                   g["depth_max"].fillna(g["depth_min"].fillna(0))) / 2.0
            g_z = g_z.dropna()
            g_z = g_z[g_z <= MAX_DEPTH]
            if len(g_z):
                max_z = max(max_z, g_z.max())
    max_z = min(max_z * 1.08, MAX_DEPTH)

    for ax, cid in zip(axes_flat[:n], ids):
        row  = results[results["core_id"] == cid].iloc[0]
        tag  = row["best_model"]
        col  = _COLORS[tag]

        # Load OM data
        if cid not in grouped.groups:
            ax.set_visible(False)
            continue
        g = grouped.get_group(cid).copy()
        g["depth_mid"] = (g["depth_min"].fillna(0) +
                          g["depth_max"].fillna(g["depth_min"].fillna(0))) / 2.0
        g["om"] = g["fraction_organic_matter"].where(
            g["fraction_organic_matter"].notna(),
            g["fraction_carbon"] * OC_TO_OM)
        g = g.dropna(subset=["om", "depth_mid"])
        g = g[g["depth_mid"] <= MAX_DEPTH].sort_values("depth_mid")

        z  = g["depth_mid"].values.astype(float)
        om = g["om"].values.astype(float)

        # Load params from CSV (same fit used for reported stats)
        params = get_params(row, tag)
        if params is None:
            ax.set_visible(False)
            continue

        # Fitted curve — only to the deepest data point
        z_fine = np.linspace(0, z.max(), 500)
        func   = _MODELS[tag][0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            om_fit = func(z_fine, *params)

        ax.scatter(om, z, color="#444444", s=18, zorder=3)
        ax.plot(om_fit, z_fine, color=col, lw=2.0, zorder=4,
                label=f"{tag}  R²={row[f'{tag}_r2']:.2f}")

        # Deep OM asymptote and doubling depth
        om_inf = float(params[-1])
        if om_inf > 0:
            ax.axvline(om_inf, color=col, lw=0.8, ls=":", alpha=0.65,
                       label=f"OMinf={om_inf:.3f}")
            ax.axvline(2 * om_inf, color=col, lw=0.8, ls="--", alpha=0.65,
                       label=f"2×={2*om_inf:.3f}")
        dbl = row.get("doubling_depth_cm", np.nan)
        if not np.isnan(dbl):
            ax.axhline(dbl, color="grey", lw=0.9, ls="--", alpha=0.7)

        # Stats box
        lines = [
            f"deep OM={om_inf:.3f}",
            f"dbl={dbl:.0f} cm" if not np.isnan(dbl) else "dbl=n/a",
        ]
        if not np.isnan(row.get("mean_om_above_dbl", np.nan)):
            lines.append(f"mean↑={row['mean_om_above_dbl']:.3f}")
        ax.text(0.97, 0.97, "\n".join(lines),
                transform=ax.transAxes, fontsize=6.5, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

        # Title: core ID, study, species, lat
        spp = sp[sp["core_id"] == cid]["species_code"].tolist()
        spp_short = [s.replace("Spartina ", "S.") for s in spp]
        study = cores_meta[cores_meta["core_id"] == cid]["study_id"].iloc[0]
        study_short = study.replace("_et_al_", " ").replace("_", " ")
        lat   = cores_meta[cores_meta["core_id"] == cid]["latitude"].iloc[0]
        ax.set_title(
            f"{cid}\n{study_short}\n{', '.join(spp_short)}  {lat:.2f}°N",
            fontsize=7, loc="left")

        ax.invert_yaxis()
        ax.set_ylim(max_z, 0)
        ax.set_xlabel("Fraction OM", fontsize=8)
        ax.legend(fontsize=6, loc="lower right")

    # y-axis label on leftmost panels only
    for r in range(nrows):
        axes_flat[r * ncols].set_ylabel("Depth (cm)", fontsize=9)

    fig.suptitle(
        f"{state} — Spartina marsh cores, OM depth profiles\n"
        f"n={n} cores with qualifying fit (R² ≥ 0.75); "
        f"sorted S → N; depth axis identical across panels",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()

    safe = state.lower().replace(" ", "_")
    out  = _FIG_DIR / f"spartina_profiles_{safe}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _FIG_DIR.mkdir(parents=True, exist_ok=True)

    cores_meta = pd.read_csv(_CO_CSV, low_memory=False)
    sp         = pd.read_csv(_SP_CSV)
    results    = pd.read_csv(_FIT_CSV)
    ds         = pd.read_csv(_DS_CSV, low_memory=False,
                              usecols=["core_id", "depth_min", "depth_max",
                                       "fraction_organic_matter", "fraction_carbon"])

    spart_ids = set(sp[sp["species_code"].str.startswith("Spartina", na=False)]["core_id"])
    good_ids  = set(results[results["best_model"].notna()]["core_id"])

    # East coast cores
    ec = cores_meta[
        cores_meta["latitude"].between(24, 48) &
        cores_meta["longitude"].between(-83, -65) &
        cores_meta["core_id"].isin(spart_ids) &
        cores_meta["core_id"].isin(good_ids)
    ].copy()

    ec["state"] = ec.apply(
        lambda r: assign_state(r["latitude"], r["longitude"]), axis=1)

    n_unassigned = ec["state"].isna().sum()
    if n_unassigned:
        print(f"Warning: {n_unassigned} cores unassigned to a state — skipped")

    ec = ec.dropna(subset=["state"])

    for state, *_ in STATE_BOXES:
        state_cores = ec[ec["state"] == state]
        if len(state_cores) == 0:
            continue
        print(f"{state}: {len(state_cores)} cores")
        plot_state(state, state_cores, ds, results, cores_meta, sp)

    print("\nDone.")


if __name__ == "__main__":
    main()
