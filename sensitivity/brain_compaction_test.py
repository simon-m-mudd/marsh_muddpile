#!/usr/bin/env python3
"""
brain_compaction_test.py

Test the Brain (2012) two-stage compaction model across a range of LOI
values and depth profiles, then compare modelled DBD vs LOI against the
CCN core database for USA and UK.

For each combination of mean_loi (0.00–0.70, step 0.02) and profile_type
(uniform / top_heavy / bottom_heavy), a 100-layer, 1-cm-thick column is
initialised with sand + refractory_organic and run for 1 year with no
vegetation growth, no deposition, and no decay.  Only the Brain compaction
model is active.

Outputs
-------
    sensitivity/figures/brain_compaction_dbd_loi.png

Usage
-----
    python sensitivity/brain_compaction_test.py --binary ./build/marsh_cli
    python sensitivity/brain_compaction_test.py --plot-only
    python sensitivity/brain_compaction_test.py --force
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

_THIS_DIR  = Path(__file__).parent.resolve()
_CALIB_DIR = _THIS_DIR.parent / "calibration"
sys.path.insert(0, str(_CALIB_DIR))

from yaml_writer import _DEFAULT_PARAMETERS, _DEFAULT_MATERIALS
from model_runner import run_model

try:
    import netCDF4 as nc4
except ImportError as exc:
    raise ImportError("netCDF4 is required: pip install netCDF4") from exc

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError("matplotlib is required") from exc

try:
    import pandas as pd
except ImportError as exc:
    raise ImportError("pandas is required") from exc

try:
    from scipy.optimize import curve_fit
except ImportError as exc:
    raise ImportError("scipy is required") from exc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_LAYERS         = 100
LAYER_THICKNESS  = 0.01       # m (1 cm)
INITIAL_POROSITY = 0.90
SURFACE_ELEV_M   = 2.0        # m MSL — always above tidal influence
RHO_ORG          = 1400.0     # refractory_organic (kg/m³)
RHO_MIN          = 2650.0     # sand (kg/m³)
_DT_DAYS         = 365.25 / 12.0

LOI_VALUES    = np.round(np.arange(0.0, 0.72, 0.02), 2)   # 36 values (0.00 – 0.70)
PROFILE_TYPES = ["uniform", "top_heavy", "bottom_heavy"]

OUT_BASE = _THIS_DIR / "runs" / "brain_compaction_test"
OUT_DIR  = OUT_BASE / "mixing"     # default (kept for back-compat with _plot)
FIG_DIR  = _THIS_DIR / "figures"
FIG_OUT  = FIG_DIR / "brain_compaction_dbd_loi.png"
FIG_CMP  = FIG_DIR / "brain_vs_mixing_compaction_dbd_loi.png"

# CCN data
_ROOT   = _THIS_DIR.parent
_DS_CSV = _ROOT / "core_data" / "CCN_synthesis" / "CCN_depthseries.csv"
_CO_CSV = _ROOT / "core_data" / "CCN_synthesis" / "CCN_cores.csv"

# Published and best-fit mixing-model constants [g/cm³]
K1_MORRIS = 0.085;  K2_MORRIS = 1.99      # Morris et al. 2016
K1_USA    = 0.0999; K2_USA    = 1.5402    # CCN USA best fit
K1_UK     = 0.1128; K2_UK     = 1.2035    # CCN UK best fit

# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def loi_to_vol_frac(loi: float) -> float:
    """Organic volume fraction of the solid matrix for a given mass LOI."""
    return loi * RHO_MIN / (loi * RHO_MIN + (1.0 - loi) * RHO_ORG)


def make_loi_profile(mean_loi: float, profile_type: str) -> np.ndarray:
    """Return LOI array (N_LAYERS) from surface (index 0) to base (index 99)."""
    depth_frac = (np.arange(N_LAYERS) + 0.5) / N_LAYERS   # 0 = near surface, 1 = deep base
    if profile_type == "uniform":
        loi = np.full(N_LAYERS, mean_loi)
    elif profile_type == "top_heavy":
        loi = 2.0 * mean_loi * (1.0 - depth_frac)
    elif profile_type == "bottom_heavy":
        loi = 2.0 * mean_loi * depth_frac
    else:
        raise ValueError(f"Unknown profile_type: {profile_type}")
    return np.clip(loi, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Initial column builder
# ---------------------------------------------------------------------------

def _build_layers(loi_profile: np.ndarray) -> List[dict]:
    """100 × 1-cm layers, returned in bottom-up order (as required by the YAML)."""
    solid_vol = (1.0 - INITIAL_POROSITY) * LAYER_THICKNESS   # m³ solid per m² per layer
    layers = []
    for i, loi in enumerate(loi_profile):           # i=0 is the surface layer
        top_elev = SURFACE_ELEV_M - i * LAYER_THICKNESS
        vf       = loi_to_vol_frac(float(loi))
        m_org    = float(RHO_ORG * vf * solid_vol)
        m_min    = float(RHO_MIN * (1.0 - vf) * solid_vol)
        layers.append({
            "top_elevation_m": round(top_elev, 4),
            "thickness_m":     LAYER_THICKNESS,
            "porosity":        INITIAL_POROSITY,
            "age_days":        0.0,
            "mass_kg_m2": {
                "sand":               round(m_min, 6),
                "refractory_organic": round(m_org, 6),
                "labile_organic":     0.0,
                "roots":              0.0,
            },
        })
    return list(reversed(layers))   # model expects deepest layer first


# ---------------------------------------------------------------------------
# Materials: disable decay for all organic materials
# ---------------------------------------------------------------------------

def _compaction_materials() -> list:
    mats = copy.deepcopy(_DEFAULT_MATERIALS)
    for mat in mats:
        if "decay" in mat:
            mat["decay"]["k_0"] = 0.0
    return mats


# ---------------------------------------------------------------------------
# Forcing: 12 monthly steps, no tidal inundation, no SSC
# ---------------------------------------------------------------------------

def _build_forcing() -> List[dict]:
    steps = []
    for i in range(12):
        steps.append({
            "model_time_days":                  round((i + 1) * _DT_DAYS, 4),
            "dt_days":                          round(_DT_DAYS, 4),
            "mean_sea_level":                   0.0,
            "mean_high_tide":                   0.001,
            "tidal_amplitude":                  0.001,
            "tidal_period_hours":               12.42,
            "temperature":                      20.0,
            "precipitation_mm_d":               0.0,
            "par_umol_m2_d":                    0.0,
            "creek_salinity_ppt":               0.0,
            "freshwater_input_mm_d":            0.0,
            "storm_surge_residual_m":           0.0,
            "suspended_sediment_concentration": 0.0,
            "fine_sediment_concentration":      0.0,
            "external_pb210_supply":            0.0,
        })
    return steps


# ---------------------------------------------------------------------------
# Run ID
# ---------------------------------------------------------------------------

def _run_id(mean_loi: float, profile_type: str) -> str:
    return f"loi{mean_loi:.2f}_{profile_type}".replace(".", "_")


def _model_out_dir(compaction_model: str) -> Path:
    return OUT_BASE / compaction_model.replace("_compaction", "")


# ---------------------------------------------------------------------------
# YAML builder
# ---------------------------------------------------------------------------

def _build_doc(mean_loi: float, profile_type: str, nc_path: Path,
               compaction_model: str = "mixing_compaction") -> dict:
    parameters = dict(_DEFAULT_PARAMETERS)

    # Zero out all vegetation effects
    parameters["vegetation_lue_gC_per_umol"]             = 0.0
    parameters["vegetation_aboveground_capacity_kg_m2"]  = 0.001
    parameters["vegetation_belowground_capacity_kg_m2"]  = 0.001

    # No deposition
    parameters["deposition_distance_from_edge_m"]        = 1000.0
    parameters["deposition_basin_length_m"]              = 1000.0

    # Minimal tidal constituents (column at 2 m MSL — never flooded)
    for constituent in ("M2", "S2", "N2", "K1", "O1"):
        parameters[f"water_level_constituent_{constituent}_amplitude_m"] = 0.001
        parameters[f"water_level_constituent_{constituent}_phase_deg"]   = 0.0
    parameters["water_level_constituent_S2_amplitude_m"] = 0.0
    parameters["water_level_constituent_N2_amplitude_m"] = 0.0
    parameters["water_level_constituent_K1_amplitude_m"] = 0.0
    parameters["water_level_constituent_O1_amplitude_m"] = 0.0

    loi_profile = make_loi_profile(mean_loi, profile_type)
    layers = _build_layers(loi_profile)

    return {
        "simulation": {
            "start_year":                      0,
            "end_year":                        1,
            "dt_days":                         _DT_DAYS,
            "water_level_model_name":          "composite_water_level",
            "salinity_model_name":             "distance_flushing_salinity",
            "evapotranspiration_model_name":   "simple_canopy_et",
            "vegetation_model_name":           "marsh_gpp_biomass",
            "deposition_model_name":           "edge_distance_deposition",
            "root_allocation_model_name":      "exponential_root_allocation",
            "decay_model_name":                "marsh_decay",
            "compaction_model_name":           compaction_model,
            "porewater_chemistry_model_name":  "none",
            "methane_model_name":              "none",
        },
        "site": {
            "distance_from_creek_m":  1000.0,
            "creek_bank_elevation_m": 0.0,
            "local_tidal_offset_m":   0.0,
        },
        "parameters": parameters,
        "materials": _compaction_materials(),
        "forcing":   {"steps": _build_forcing()},
        "initial_state": {"layers": layers},
        "initial_ecohydrology_state": {
            "root_zone_salinity_ppt":    0.0,
            "aboveground_biomass_kg_m2": 0.0,
            "belowground_biomass_kg_m2": 0.0,
            "lai":                       0.0,
            "litter_kg_m2":              0.0,
        },
        "output": {
            "file":                   str(nc_path),
            "write_time_series":      True,
            "write_column_snapshots": True,
            "snapshot_times_days":    [365.25],
        },
    }


# ---------------------------------------------------------------------------
# Run one simulation
# ---------------------------------------------------------------------------

def _run_one(mean_loi: float, profile_type: str,
             cli_binary: str, force: bool,
             compaction_model: str = "mixing_compaction") -> Optional[Path]:
    out_dir  = _model_out_dir(compaction_model)
    rid      = _run_id(mean_loi, profile_type)
    nc_path  = out_dir / f"{rid}.nc"
    if not force and nc_path.exists():
        return nc_path

    out_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = out_dir / f"{rid}.yaml"
    doc = _build_doc(mean_loi, profile_type, nc_path, compaction_model)
    with open(yaml_path, "w") as fh:
        yaml.dump(doc, fh, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)
    try:
        run_model(str(yaml_path), cli_binary=cli_binary, silent=True)
        return nc_path
    except Exception as exc:
        print(f"    FAILED: {exc}")
        return None


def _run_model_set(compaction_model: str, cli_binary: str, force: bool,
                   ) -> Dict[Tuple[float, str], Optional[Path]]:
    combos = [(loi, pt) for pt in PROFILE_TYPES for loi in LOI_VALUES]
    n = len(combos)
    label = compaction_model.replace("_compaction", "").replace("_", " ").title()
    print(f"\n{label} compaction model: {n} runs")
    results: Dict[Tuple[float, str], Optional[Path]] = {}
    for i, (mean_loi, ptype) in enumerate(combos, 1):
        print(f"  [{i:3d}/{n}] LOI={mean_loi:.2f} {ptype:<12s} …",
              end=" ", flush=True)
        nc = _run_one(mean_loi, ptype, cli_binary, force, compaction_model)
        print("ok" if nc else "FAILED")
        results[(mean_loi, ptype)] = nc
    return results


# ---------------------------------------------------------------------------
# Harvest DBD and LOI from final snapshot
# ---------------------------------------------------------------------------

def _harvest(nc_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (loi_arr, dbd_g_cm3) for all layers in the final snapshot."""
    with nc4.Dataset(str(nc_path), "r") as ds:
        mat_names = [b"".join(r.compressed()).decode()
                     for r in ds.variables["material_name"][:]]
        snap_n = int(ds.variables["snapshot_n_layers"][-1])
        mass   = np.array(ds.variables["layer_mass"][-1, :snap_n, :])    # (n_layers, n_mats)
        thick  = np.array(ds.variables["layer_thickness"][-1, :snap_n])  # (n_layers,)

    i_refrac = mat_names.index("refractory_organic")
    total    = mass.sum(axis=1)
    refrac   = mass[:, i_refrac]

    with np.errstate(invalid="ignore", divide="ignore"):
        loi_arr  = np.where(total > 0, refrac / total, 0.0)
        # kg/m² / m = kg/m³; divide by 1000 → g/cm³
        dbd_gcm3 = np.where(thick > 0, total / thick / 1000.0, np.nan)

    return loi_arr, dbd_gcm3


# ---------------------------------------------------------------------------
# CCN data loader
# ---------------------------------------------------------------------------

def _load_ccn() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Return {"usa": (loi, dbd), "uk": (loi, dbd)} arrays from CCN synthesis."""
    print("  Loading CCN depthseries …")
    ds = pd.read_csv(_DS_CSV, low_memory=False,
                     usecols=["core_id", "dry_bulk_density",
                               "fraction_organic_matter", "fraction_carbon"])
    co = pd.read_csv(_CO_CSV, low_memory=False,
                     usecols=["core_id", "country"])

    ds["loi"] = ds["fraction_organic_matter"].where(
        ds["fraction_organic_matter"].notna(),
        ds["fraction_carbon"] / 0.43,
    )
    df = ds[ds["dry_bulk_density"].notna() & ds["loi"].notna()].copy()
    df = df[(df["dry_bulk_density"] > 0) & (df["dry_bulk_density"] < 2.5) &
            (df["loi"] >= 0) & (df["loi"] <= 1.0)]
    df = df.merge(co.drop_duplicates("core_id"), on="core_id", how="left")
    print(f"  {len(df):,} CCN rows after quality filter")

    result = {}
    for country, key in [("United States", "usa"), ("United Kingdom", "uk")]:
        sub = df[df["country"] == country]
        result[key] = (sub["loi"].values, sub["dry_bulk_density"].values)
        print(f"  {key.upper()}: {len(sub):,} rows")
    return result


# ---------------------------------------------------------------------------
# Mixing model
# ---------------------------------------------------------------------------

def mixing_model(loi, k1, k2):
    return 1.0 / (loi / k1 + (1.0 - loi) / k2)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

_PROFILE_COLOR  = {"uniform": "#1b7837", "top_heavy": "#762a83", "bottom_heavy": "#e66101"}
_PROFILE_MARKER = {"uniform": "o",       "top_heavy": "^",       "bottom_heavy": "s"}
_PROFILE_LABEL  = {"uniform": "uniform",
                   "top_heavy": "top-heavy",
                   "bottom_heavy": "bottom-heavy"}


def _plot(all_results: Dict[Tuple[float, str], Optional[Path]]) -> None:
    print("\nLoading CCN data …")
    ccn = _load_ccn()

    # Harvest model data, grouped by profile type
    per_profile: Dict[str, Tuple[List[float], List[float]]] = {
        pt: ([], []) for pt in PROFILE_TYPES
    }
    n_ok = 0
    for (mean_loi, ptype), nc_path in all_results.items():
        if nc_path is None or not nc_path.exists():
            continue
        try:
            loi_arr, dbd_arr = _harvest(nc_path)
        except Exception as exc:
            print(f"  Error reading {nc_path.name}: {exc}")
            continue
        per_profile[ptype][0].extend(loi_arr.tolist())
        per_profile[ptype][1].extend(dbd_arr.tolist())
        n_ok += 1
    print(f"  Harvested {n_ok} model runs")

    loi_line = np.linspace(0.001, 1.0, 500)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    panels = [
        ("USA marsh cores", "usa", K1_USA, K2_USA),
        ("UK marsh cores",  "uk",  K1_UK,  K2_UK),
    ]

    for ax, (title, ckey, k1f, k2f) in zip(axes, panels):
        loi_ccn, dbd_ccn = ccn[ckey]
        n_ccn = len(loi_ccn)

        # CCN data as hexbin density background
        hb = ax.hexbin(loi_ccn, dbd_ccn, gridsize=60, cmap="YlOrRd",
                       mincnt=1, linewidths=0.1, alpha=0.85, zorder=1)
        fig.colorbar(hb, ax=ax, label="CCN layer count", pad=0.02)

        # Brain model scatter by profile type
        for ptype in PROFILE_TYPES:
            loi_arr = np.array(per_profile[ptype][0])
            dbd_arr = np.array(per_profile[ptype][1])
            valid   = np.isfinite(dbd_arr)
            ax.scatter(loi_arr[valid], dbd_arr[valid],
                       s=5, alpha=0.55,
                       color=_PROFILE_COLOR[ptype],
                       marker=_PROFILE_MARKER[ptype],
                       label=_PROFILE_LABEL[ptype],
                       rasterized=True, zorder=4)

        # Mixing model curves
        ax.plot(loi_line, mixing_model(loi_line, K1_MORRIS, K2_MORRIS),
                "k--", lw=2.0,
                label=f"Morris 2016  k₁={K1_MORRIS}, k₂={K2_MORRIS}",
                zorder=5)
        ax.plot(loi_line, mixing_model(loi_line, k1f, k2f),
                "b-", lw=2.0,
                label=f"CCN best fit  k₁={k1f:.4f}, k₂={k2f:.4f}",
                zorder=5)

        ax.set_xlabel("LOI (fraction organic matter)", fontsize=11)
        ax.set_ylabel("Dry bulk density (g cm⁻³)", fontsize=11)
        ax.set_xlim(-0.02, 0.75)
        ax.set_ylim(-0.05, 2.05)
        ax.set_title(f"{title}  (n={n_ccn:,})", fontsize=11)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Brain (2012) two-stage compaction model vs CCN core data\n"
        "100-layer columns, 1 cm layers, no vegetation/deposition/decay",
        fontsize=12,
    )
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {FIG_OUT}")


# ---------------------------------------------------------------------------
# Comparison plot: Brain vs Mixing vs CCN (all data)
# ---------------------------------------------------------------------------

def _collect_layers(results: Dict[Tuple[float, str], Optional[Path]]
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Harvest all (loi, dbd) layers across all runs."""
    all_loi, all_dbd = [], []
    for nc_path in results.values():
        if nc_path is None or not nc_path.exists():
            continue
        try:
            loi_arr, dbd_arr = _harvest(nc_path)
            all_loi.extend(loi_arr.tolist())
            all_dbd.extend(dbd_arr.tolist())
        except Exception:
            pass
    return np.array(all_loi), np.array(all_dbd)


def _plot_comparison(
    brain_results:  Dict[Tuple[float, str], Optional[Path]],
    mixing_results: Dict[Tuple[float, str], Optional[Path]],
) -> None:
    print("\nBuilding comparison figure …")
    ccn = _load_ccn()

    # Pool all CCN data
    loi_all = np.concatenate([ccn["usa"][0], ccn["uk"][0]])
    dbd_all = np.concatenate([ccn["usa"][1], ccn["uk"][1]])

    brain_loi,  brain_dbd  = _collect_layers(brain_results)
    mixing_loi, mixing_dbd = _collect_layers(mixing_results)

    loi_line = np.linspace(0.001, 0.75, 400)

    # Depth-dependent CCN mixing model at surface (d=0) and 50 cm
    K1_0  = 0.105944;  K2_0  = 1.399719   # d=0 m
    K1_50 = 0.105944 + (-0.012864)*0.5 + 0.027568*0.25   # d=0.5 m
    K2_50 = 1.399719 + 0.170120*0.5    + (-0.313556)*0.25

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    panels = [
        ("Brain (two-stage)\nvs All CCN",   brain_loi,  brain_dbd,  "#c2523c"),
        ("Mixing (depth-dep.)\nvs All CCN", mixing_loi, mixing_dbd, "#2166ac"),
    ]

    for ax, (title, mod_loi, mod_dbd, col) in zip(axes[:2], panels):
        hb = ax.hexbin(loi_all, dbd_all, gridsize=60, cmap="YlOrRd",
                       mincnt=1, linewidths=0.1, alpha=0.85, zorder=1)
        fig.colorbar(hb, ax=ax, label="CCN layer count", pad=0.02)

        valid = np.isfinite(mod_dbd)
        ax.scatter(mod_loi[valid], mod_dbd[valid],
                   s=4, alpha=0.5, color=col, rasterized=True, zorder=4,
                   label=f"Model output (n={valid.sum():,})")

        ax.plot(loi_line, mixing_model(loi_line, K1_MORRIS, K2_MORRIS),
                "k--", lw=1.8, zorder=5,
                label=f"Morris 2016  k₁={K1_MORRIS}, k₂={K2_MORRIS}")
        ax.plot(loi_line, mixing_model(loi_line, K1_0, K2_0),
                "b-",  lw=1.8, zorder=5,
                label=f"CCN fit d=0 m  k₁={K1_0:.3f}, k₂={K2_0:.3f}")
        ax.plot(loi_line, mixing_model(loi_line, K1_50, K2_50),
                "b:",  lw=1.8, zorder=5,
                label=f"CCN fit d=0.5 m  k₁={K1_50:.3f}, k₂={K2_50:.3f}")

        ax.set_xlabel("LOI (fraction organic matter)", fontsize=11)
        ax.set_ylabel("Dry bulk density (g cm⁻³)", fontsize=11)
        ax.set_xlim(-0.02, 0.77)
        ax.set_ylim(-0.05, 2.05)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(True, alpha=0.25)

    # Right panel: direct overlay, no hexbin
    ax3 = axes[2]
    ax3.hexbin(loi_all, dbd_all, gridsize=60, cmap="Greys",
               mincnt=1, linewidths=0.0, alpha=0.4, zorder=1)

    b_valid = np.isfinite(brain_dbd)
    m_valid = np.isfinite(mixing_dbd)
    ax3.scatter(brain_loi[b_valid],  brain_dbd[b_valid],
                s=4, alpha=0.45, color="#c2523c", rasterized=True, zorder=3,
                label=f"Brain  (n={b_valid.sum():,})")
    ax3.scatter(mixing_loi[m_valid], mixing_dbd[m_valid],
                s=4, alpha=0.45, color="#2166ac", rasterized=True, zorder=4,
                label=f"Mixing  (n={m_valid.sum():,})")

    ax3.plot(loi_line, mixing_model(loi_line, K1_MORRIS, K2_MORRIS),
             "k--", lw=1.8, zorder=5, label="Morris 2016")
    ax3.plot(loi_line, mixing_model(loi_line, K1_0, K2_0),
             "b-",  lw=1.8, zorder=5, label="CCN fit d=0 m")

    ax3.set_xlabel("LOI (fraction organic matter)", fontsize=11)
    ax3.set_xlim(-0.02, 0.77)
    ax3.set_ylim(-0.05, 2.05)
    ax3.set_title("Brain vs Mixing — direct comparison", fontsize=11)
    ax3.legend(fontsize=7.5, loc="upper right")
    ax3.grid(True, alpha=0.25)

    fig.suptitle(
        "Brain (two-stage) vs Mixing (depth-dependent) compaction models\n"
        "100-layer, 1 cm columns — all CCN cores in background",
        fontsize=12,
    )
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_CMP, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {FIG_CMP}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary",    default="marsh_cli",
                    help="path to marsh_cli binary (default: marsh_cli)")
    ap.add_argument("--plot-only", action="store_true",
                    help="skip model runs, plot existing outputs")
    ap.add_argument("--force",     action="store_true",
                    help="re-run even if output NetCDF already exists")
    return ap.parse_args()


def main():
    args = _cli()
    combos = [(loi, pt) for pt in PROFILE_TYPES for loi in LOI_VALUES]
    n = len(combos)
    print(f"Brain vs Mixing compaction test: {n} runs × 2 models\n"
          f"Output base: {OUT_BASE}")

    models = [
        ("two_stage_compaction", "Brain (two-stage)"),
        ("mixing_compaction",    "Mixing (depth-dep.)"),
    ]

    all_results: Dict[str, Dict[Tuple[float, str], Optional[Path]]] = {}

    for model_name, label in models:
        out_dir = _model_out_dir(model_name)
        if not args.plot_only:
            all_results[model_name] = _run_model_set(model_name, args.binary, args.force)
        else:
            r: Dict[Tuple[float, str], Optional[Path]] = {}
            for mean_loi, ptype in combos:
                nc = out_dir / f"{_run_id(mean_loi, ptype)}.nc"
                r[(mean_loi, ptype)] = nc if nc.exists() else None
            completed = sum(1 for v in r.values() if v is not None)
            print(f"  {label}: found {completed}/{n} runs in {out_dir.name}/")
            all_results[model_name] = r

    # Individual plots (mixing model shown as "default")
    global OUT_DIR
    OUT_DIR = _model_out_dir("mixing_compaction")
    _plot(all_results["mixing_compaction"])

    # Comparison plot
    _plot_comparison(all_results["two_stage_compaction"],
                     all_results["mixing_compaction"])
    print("Done.")


if __name__ == "__main__":
    main()
