#!/usr/bin/env python3
"""
ni_long_profile.py

500-year marsh column simulations starting from a sand pile at MHW elevation.
Four SLR scenarios: 2, 3, 4, 5 mm/yr.  SSC = 10 mg/L.  Seed biomass = 0.1 kg/m².

Marker horizon analysis: the surface elevation at year 470 is the emplacement
elevation.  At year 500 the marker's absolute position is identified from
layer ages in the final column snapshot (layers deposited before year 470 have
age > 30 yr in the final snapshot).  Subsidence rate = (emplacement elevation
− marker elevation at year 500) / 30 yr.

Usage
-----
    python sensitivity/ni_long_profile.py --binary ./build/marsh_cli
    python sensitivity/ni_long_profile.py --plot-only
    python sensitivity/ni_long_profile.py --force

Output
------
    sensitivity/runs/ni_long_profile/
    sensitivity/figures/ni_long_profile.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

_THIS_DIR  = Path(__file__).parent.resolve()
_CALIB_DIR = _THIS_DIR.parent / "calibration"
sys.path.insert(0, str(_CALIB_DIR))

from site_config import PlotConfig, north_inlet_default_met, north_inlet_default_tides
from yaml_writer import _DEFAULT_PARAMETERS, _DEFAULT_MATERIALS
from forcing_builder import build_generated_forcing_block
from model_runner import run_model

try:
    import netCDF4 as nc4
except ImportError as exc:
    raise ImportError("netCDF4 required: pip install netCDF4") from exc

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError("matplotlib required: pip install matplotlib") from exc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SLR_RATES_M_YR  = [0.002, 0.003, 0.004, 0.005]
RUN_YEARS       = 500
MARKER_YEAR     = 470              # year marker is emplaced
INITIAL_ELEV_M  = 0.625            # MHW at North Inlet (m NAVD88)
SSC_KG_M3       = 0.010            # 10 mg/L
SEED_BIOMASS    = 0.1              # kg/m²
NI_DISTANCE_M   = 30.0

SIGMA_H_BEST    = 0.25
SIGMA_R_BEST    = 0.22
CAPACITY_KG_M2  = 0.95
LUE             = 1.6e-6

# Age of marker layers in the final snapshot (days)
_MARKER_AGE_DAYS = (RUN_YEARS - MARKER_YEAR) * 365.25

# Initial column uses build_initial_column_layers (20 layers × 0.05 m, organic/silt
# profile calibrated for mixing_compaction).  This is the standard starting
# condition for North Inlet; the 500-year run builds the full profile above it.

OUTPUT_DIR = _THIS_DIR / "runs" / "ni_long_profile"
FIGURE_DIR = _THIS_DIR / "figures"

COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]


def _build_sand_column(surface_elevation_m: float,
                       n_layers: int = 20,
                       layer_thickness_m: float = 0.05) -> List[dict]:
    """Pure-sand initial column in equilibrium with mixing_compaction.

    mixing_compaction assigns DBD = k2(d) for zero-LOI material, where
    k2(0) ≈ 1400 kg/m³.  The equilibrium porosity for sand (rho=2650) is
    1 - 1400/2650 = 0.472.  Using this porosity at startup gives < 2 cm
    surface displacement when the model first applies mixing_compaction.
    """
    SAND_POROSITY = 0.472
    RHO_SAND = 2650.0
    solid_vol = (1.0 - SAND_POROSITY) * layer_thickness_m
    sand_mass = round(RHO_SAND * solid_vol, 4)

    layers = []
    for i in range(n_layers):
        top_elev = surface_elevation_m - i * layer_thickness_m
        layers.append({
            "top_elevation_m": round(top_elev, 4),
            "thickness_m":     layer_thickness_m,
            "porosity":        SAND_POROSITY,
            "age_days":        0.0,
            "mass_kg_m2":      {"sand": sand_mass},
        })
    return layers


# ---------------------------------------------------------------------------
# YAML construction
# ---------------------------------------------------------------------------

def _run_id(slr_m_yr: float) -> str:
    return "ni_lp_slr%d" % round(slr_m_yr * 1000)


def _tidal_params(tides) -> dict:
    return {
        "water_level_constituent_M2_amplitude_m": tides.M2_amplitude_m,
        "water_level_constituent_M2_phase_deg":   tides.M2_phase_deg,
        "water_level_constituent_S2_amplitude_m": tides.S2_amplitude_m,
        "water_level_constituent_S2_phase_deg":   tides.S2_phase_deg,
        "water_level_constituent_N2_amplitude_m": tides.N2_amplitude_m,
        "water_level_constituent_N2_phase_deg":   tides.N2_phase_deg,
        "water_level_constituent_K1_amplitude_m": tides.K1_amplitude_m,
        "water_level_constituent_K1_phase_deg":   tides.K1_phase_deg,
        "water_level_constituent_O1_amplitude_m": tides.O1_amplitude_m,
        "water_level_constituent_O1_phase_deg":   tides.O1_phase_deg,
    }


def _write_yaml(slr_m_yr: float, yaml_path: Path) -> None:
    tides = north_inlet_default_tides()
    tides.suspended_sediment_concentration_kg_m3 = SSC_KG_M3
    tides.fine_sediment_concentration_kg_m3      = SSC_KG_M3 * 0.25

    config = PlotConfig(
        site_name="north_inlet",
        plot_id=_run_id(slr_m_yr),
        surface_elevation_m=INITIAL_ELEV_M,
        distance_from_creek_m=NI_DISTANCE_M,
        tides=tides,
        met=north_inlet_default_met(),
        n_years=RUN_YEARS,
        sea_level_rise_m_yr=slr_m_yr,
        output_dir=str(OUTPUT_DIR),
    )
    config.initial_salinity_ppt = tides.creek_salinity_ppt

    parameters = dict(_DEFAULT_PARAMETERS)
    parameters.update(_tidal_params(tides))
    parameters["deposition_distance_from_edge_m"]      = NI_DISTANCE_M
    parameters.setdefault("deposition_basin_length_m",    50.0)
    parameters.setdefault("deposition_length_scale_beta", 3.0)
    parameters["vegetation_lue_gC_per_umol"]            = LUE
    parameters["vegetation_aboveground_capacity_kg_m2"] = CAPACITY_KG_M2
    parameters["vegetation_belowground_capacity_kg_m2"] = CAPACITY_KG_M2 * 2.0
    parameters["vegetation_hydroperiod_sigma_fraction"] = SIGMA_H_BEST
    parameters["vegetation_inundation_sigma_fraction"]  = SIGMA_R_BEST

    doc = {
        "simulation": {
            "start_year": 0,
            "end_year":   RUN_YEARS,
            "dt_days":    365.0,
            "water_level_model_name":         "composite_water_level",
            "salinity_model_name":            "distance_flushing_salinity",
            "evapotranspiration_model_name":  "simple_canopy_et",
            "vegetation_model_name":          "marsh_gpp_biomass",
            "deposition_model_name":          "edge_distance_deposition",
            "root_allocation_model_name":     "exponential_root_allocation",
            "decay_model_name":               "marsh_decay",
            "compaction_model_name":          "mixing_compaction",
            "porewater_chemistry_model_name": "none",
            "methane_model_name":             "none",
        },
        "site": {
            "distance_from_creek_m":  NI_DISTANCE_M,
            "creek_bank_elevation_m": tides.mean_sea_level_m,
            "local_tidal_offset_m":   0.0,
        },
        "parameters": parameters,
        "materials":  _DEFAULT_MATERIALS,
        "forcing":    {"generated": build_generated_forcing_block(config)},
        "initial_state": {"layers": _build_sand_column(INITIAL_ELEV_M)},
        "initial_ecohydrology_state": {
            "root_zone_salinity_ppt":    config.initial_salinity_ppt,
            "aboveground_biomass_kg_m2": SEED_BIOMASS,
            "belowground_biomass_kg_m2": SEED_BIOMASS,
            "lai": round(SEED_BIOMASS * parameters["vegetation_lai_per_kg_m2"], 4),
            "litter_kg_m2": 0.0,
        },
        "output": {
            "file":                    str(yaml_path).replace(".yaml", ".nc"),
            "write_time_series":       True,
            "write_column_snapshots":  True,
            "snapshot_every_n_steps":  60,   # every 5 years
        },
    }

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as fh:
        yaml.dump(doc, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Run management
# ---------------------------------------------------------------------------

def _run_all(cli_binary: str, force: bool) -> Dict[float, Path]:
    results: Dict[float, Path] = {}
    n = len(SLR_RATES_M_YR)
    for i, slr in enumerate(SLR_RATES_M_YR, 1):
        rid       = _run_id(slr)
        yaml_path = OUTPUT_DIR / ("%s.yaml" % rid)
        nc_path   = OUTPUT_DIR / ("%s.nc"   % rid)
        mm        = round(slr * 1000)
        if not force and nc_path.exists():
            print("  [%d/%d] SLR=%d mm/yr: cached." % (i, n, mm))
        else:
            print("  [%d/%d] SLR=%d mm/yr ..." % (i, n, mm))
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            _write_yaml(slr, yaml_path)
            try:
                run_model(str(yaml_path), cli_binary=cli_binary, silent=False)
            except RuntimeError as exc:
                print("    ERROR: %s" % exc)
                continue
        if nc_path.exists():
            results[slr] = nc_path
    return results


def _collect_existing() -> Dict[float, Path]:
    results: Dict[float, Path] = {}
    for slr in SLR_RATES_M_YR:
        nc_path = OUTPUT_DIR / ("%s.nc" % _run_id(slr))
        if nc_path.exists():
            results[slr] = nc_path
    return results


# ---------------------------------------------------------------------------
# Profile extraction
# ---------------------------------------------------------------------------

def _decode_mat_names(raw) -> List[str]:
    names = []
    for row in raw:
        chars = []
        for c in row:
            if hasattr(c, "decode"):
                ch = c.decode()
            elif isinstance(c, str):
                ch = c
            else:
                break   # masked or None
            if ch == "\x00":
                break
            chars.append(ch)
        names.append("".join(chars).strip())
    return names


def _read_profile(nc_path: Path, slr_m_yr: float) -> Optional[dict]:
    with nc4.Dataset(str(nc_path)) as ds:
        t       = np.array(ds["model_time_days"][:],           dtype=np.float64)
        surf_ts = np.array(ds["surface_elevation"][:],         dtype=np.float64)
        abg_ts  = np.array(ds["aboveground_biomass_kg_m2"][:], dtype=np.float64)

        mat_names = _decode_mat_names(ds["material_name"][:])
        sand_i  = mat_names.index("sand")
        lab_i   = mat_names.index("labile_organic")
        ref_i   = mat_names.index("refractory_organic")
        root_i  = mat_names.index("roots")

        # surface elevation at marker emplacement (year 470)
        idx_470       = int(np.argmin(np.abs(t - MARKER_YEAR * 365.25)))
        elev_at_marker = float(surf_ts[idx_470])

        # final column snapshot
        snap_t  = np.array(ds["snapshot_time_days"][:], dtype=np.float64)
        fi      = int(np.argmax(snap_t))
        n_lay   = int(ds["snapshot_n_layers"][fi])

        thick    = np.array(ds["layer_thickness"][fi,    :n_lay],    dtype=np.float64)
        top_elev = np.array(ds["layer_top_elevation"][fi, :n_lay],   dtype=np.float64)
        age      = np.array(ds["layer_age"][fi,           :n_lay],   dtype=np.float64)
        mass     = np.array(ds["layer_mass"][fi,          :n_lay, :], dtype=np.float64)

    # sort layers top → bottom
    order    = np.argsort(top_elev)[::-1]
    thick    = thick[order]
    top_elev = top_elev[order]
    age      = age[order]
    mass     = mass[order, :]

    surf_final = float(top_elev[0])   # top of highest layer = column surface

    total_mass   = mass.sum(axis=1)
    organic_mass = mass[:, lab_i] + mass[:, ref_i] + mass[:, root_i]

    # marker horizon: first layer (top→bottom) whose age exceeds _MARKER_AGE_DAYS
    marker_elev_final = None
    for k in range(n_lay):
        if age[k] > _MARKER_AGE_DAYS:
            marker_elev_final = float(top_elev[k])
            break
    if marker_elev_final is None:
        marker_elev_final = surf_final

    yrs_since_marker = RUN_YEARS - MARKER_YEAR   # 30
    subsidence_m_yr  = (elev_at_marker - marker_elev_final) / yrs_since_marker
    accretion_m_yr   = (surf_final     - elev_at_marker)    / yrs_since_marker

    # Aggregate into 1 cm depth bins (avoids noise from thousands of thin layers)
    bin_cm   = 1.0
    max_cm   = 250.0
    bins     = np.arange(0.0, max_cm + bin_cm, bin_cm)
    n_bins   = len(bins) - 1
    bin_mass_total   = np.zeros(n_bins)
    bin_mass_organic = np.zeros(n_bins)
    bin_thick        = np.zeros(n_bins)

    for k in range(n_lay):
        if thick[k] <= 0:
            continue
        depth_top = surf_final - top_elev[k]          # depth to top of layer
        depth_bot = depth_top  + thick[k]             # depth to bottom of layer
        depth_top_cm = depth_top * 100.0
        depth_bot_cm = depth_bot * 100.0

        for b in range(n_bins):
            lo = bins[b]
            hi = bins[b + 1]
            # overlap between [depth_top_cm, depth_bot_cm] and [lo, hi]
            overlap = max(0.0, min(depth_bot_cm, hi) - max(depth_top_cm, lo))
            if overlap <= 0:
                continue
            frac = overlap / (depth_bot_cm - depth_top_cm)
            bin_mass_total[b]   += total_mass[k]   * frac
            bin_mass_organic[b] += organic_mass[k] * frac
            bin_thick[b]        += thick[k] * frac

    bin_centres = (bins[:-1] + bins[1:]) / 2.0
    valid = bin_thick > 1e-6
    bin_dbd = np.where(valid, bin_mass_total   / bin_thick,       0.0)
    bin_loi = np.where(valid, bin_mass_organic / np.where(bin_mass_total > 1e-9,
                                                           bin_mass_total, 1.0) * 100.0, 0.0)
    bin_loi[~valid] = np.nan
    bin_dbd[~valid] = np.nan

    return {
        "slr_m_yr":           slr_m_yr,
        "depth_cm":           bin_centres,
        "loi":                bin_loi,
        "dbd":                bin_dbd,
        "valid":              valid,
        "surf_final":         surf_final,
        "elev_at_marker":     elev_at_marker,
        "marker_elev_final":  marker_elev_final,
        "subsidence_m_yr":    subsidence_m_yr,
        "accretion_m_yr":     accretion_m_yr,
        "final_abg":          float(abg_ts[-1]),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(profiles: List[dict], save: bool) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 8))
    ax_loi, ax_dbd, ax_sc = axes

    max_depth_cm = 200.0

    for prof, col in zip(profiles, COLORS):
        mm    = round(prof["slr_m_yr"] * 1000)
        label = "SLR %d mm/yr" % mm
        depth_cm = prof["depth_cm"]
        mask  = prof["valid"] & (depth_cm <= max_depth_cm)

        ax_loi.plot(prof["loi"][mask], depth_cm[mask], color=col, lw=1.8, label=label)
        ax_dbd.plot(prof["dbd"][mask], depth_cm[mask], color=col, lw=1.8, label=label)
        ax_sc.scatter(prof["loi"][mask], prof["dbd"][mask],
                      color=col, s=12, alpha=0.55, label=label)

    for ax in (ax_loi, ax_dbd):
        ax.invert_yaxis()
        ax.set_ylim(max_depth_cm, 0)
        ax.grid(True, lw=0.3, alpha=0.4)
        ax.tick_params(labelsize=9)
        ax.set_ylabel("Depth (cm)", fontsize=10)

    ax_loi.set_xlabel("LOI (%)", fontsize=10)
    ax_loi.set_xlim(left=0)
    ax_loi.legend(fontsize=8)
    ax_loi.set_title("Loss on ignition vs depth", fontsize=10)

    ax_dbd.set_xlabel("Dry bulk density (kg m⁻³)", fontsize=10)
    ax_dbd.legend(fontsize=8)
    ax_dbd.set_title("Dry bulk density vs depth", fontsize=10)

    ax_sc.set_xlabel("LOI (%)", fontsize=10)
    ax_sc.set_ylabel("Dry bulk density (kg m⁻³)", fontsize=10)
    ax_sc.grid(True, lw=0.3, alpha=0.4)
    ax_sc.legend(fontsize=8)
    ax_sc.set_title("DBD vs LOI", fontsize=10)

    sub_str = "  ".join(
        "SLR%d → %.1f mm/yr" % (round(p["slr_m_yr"]*1000), p["subsidence_m_yr"]*1000)
        for p in profiles
    )
    fig.suptitle(
        "500-yr NI  |  start MHW %.2f m NAVD88  |  SSC=10 mg/L  |  30 m from creek\n"
        "Marker subsidence (yr 470–500):  %s" % (INITIAL_ELEV_M, sub_str),
        fontsize=9,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        out = FIGURE_DIR / "ni_long_profile.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print("Saved: %s" % out)
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(profiles: List[dict]) -> None:
    print()
    hdr = "%-10s  %-10s  %-12s  %-14s  %-14s  %-14s" % (
        "SLR(mm/yr)", "surf(m)", "elev@470(m)", "marker@500(m)",
        "subsid(mm/yr)", "accret(mm/yr)")
    print(hdr)
    print("-" * len(hdr))
    for p in profiles:
        print("%-10d  %-10.3f  %-12.3f  %-14.3f  %-14.2f  %-14.2f" % (
            round(p["slr_m_yr"] * 1000),
            p["surf_final"],
            p["elev_at_marker"],
            p["marker_elev_final"],
            p["subsidence_m_yr"] * 1000,
            p["accretion_m_yr"]  * 1000,
        ))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--binary",    default="marsh_cli")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--plot-only", action="store_true")
    g.add_argument("--force",     action="store_true")
    p.add_argument("--no-save",   action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.plot_only:
        results = _collect_existing()
        if not results:
            print("No existing outputs — run without --plot-only first.")
            return
        print("Found %d existing outputs." % len(results))
    else:
        print("Running %d long-profile simulations (%d yr each) ..." % (
            len(SLR_RATES_M_YR), RUN_YEARS))
        results = _run_all(args.binary, force=args.force)

    if not results:
        print("No outputs available.")
        return

    print("Reading profiles ...")
    profiles = []
    for slr in sorted(results):
        p = _read_profile(results[slr], slr)
        if p:
            profiles.append(p)
        else:
            print("  Warning: could not read %s" % results[slr].name)

    if not profiles:
        print("No valid profiles.")
        return

    _print_summary(profiles)
    print("Plotting ...")
    _plot(profiles, save=not args.no_save)


if __name__ == "__main__":
    main()
