#!/usr/bin/env python3
"""
calibrate_ch4_ni.py

Two-stage CH4 model calibration against North Inlet observations.

Stage 1 — SMTZ depth
    Optimise ch4_km_so4_umol_L, ch4_flushing_depth_scale_m,
    and ch4_tidal_flushing_rate_per_d so that the modelled sulfate-methane
    transition zone (SMTZ) matches the depth inferred from NILTREB S2- profiles.

Stage 2 — Surface CH4 flux
    Fixing Stage-1 parameters, optimise ch4_oxidation_rate_per_d,
    ch4_oxidation_depth_scale_m, ch4_ebullition_threshold_umol_L, and
    ch4_plant_transport_factor to minimise RMSE against BICEFS monthly CH4 flux.

GPP parameters are loaded from the existing calibration/ best-fit JSON and
held fixed throughout.  Only CH4-module parameters are varied.

Usage
-----
  # Full two-stage run (default site: ol_hm_c):
  python calibrate_ch4_ni.py

  # Stage 1 only (SMTZ):
  python calibrate_ch4_ni.py --stage 1

  # Specify which GPP calibration to load:
  python calibrate_ch4_ni.py --gpp-site gi_hm_c

  # Differential evolution instead of Nelder-Mead:
  python calibrate_ch4_ni.py --method de

  # Custom number of Nelder-Mead restarts:
  python calibrate_ch4_ni.py --n-starts 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import differential_evolution, minimize

# Allow imports from the parent calibration/ directory
_CALIB_DIR = Path(__file__).parent.parent / "calibration"
sys.path.insert(0, str(_CALIB_DIR))
sys.path.insert(0, str(Path(__file__).parent))

from model_runner import run_model
from output_reader import read_time_series
from site_config import north_inlet_default_met, north_inlet_default_tides, PlotConfig
from yaml_writer import write_config
from ch4_output_reader import (
    mean_ch4_flux_monthly,
    smtz_depth_m,
    summarise_ch4_run,
)
from extract_ni_ch4 import load_bicefs, aggregate_monthly
from extract_ni_porewater import load_porewater, depth_profiles, smtz_depths

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

CLI_BINARY  = "marsh_cli"
RUN_YEARS   = 30
SKIP_SPINUP = 5    # longer spinup so porewater reaches steady depth profile
OUT_DIR     = Path(__file__).parent / "calibration_runs"
OPT_DIR     = OUT_DIR / "opt"
CKPT_DIR    = OUT_DIR / "checkpoints"
DATA_DIR    = Path(__file__).parent / "data"

# GPP calibration results from calibration/ directory
GPP_RUNS_DIR = _CALIB_DIR / "calibration_runs"

# SMTZ calibration target defaults (cm); overridden if extract_ni_porewater
# data is available.
_SMTZ_TARGET_CM_DEFAULT = 25.0   # ~ OL/HM mean from NILTREB

# ---------------------------------------------------------------------------
# Parameter search spaces
# ---------------------------------------------------------------------------

STAGE1_PARAMS = [
    "ch4_km_so4_umol_L",
    "ch4_flushing_depth_scale_m",
    "ch4_tidal_flushing_rate_per_d",
]
STAGE1_BOUNDS = [
    (200.0,  5000.0),    # ch4_km_so4_umol_L
    (0.02,   0.25),      # ch4_flushing_depth_scale_m
    (0.05,   1.0),       # ch4_tidal_flushing_rate_per_d
]
STAGE1_DEFAULTS = [1000.0, 0.10, 0.30]

STAGE2_PARAMS = [
    "ch4_oxidation_rate_per_d",
    "ch4_oxidation_depth_scale_m",
    "ch4_ebullition_threshold_umol_L",
    "ch4_plant_transport_factor",
]
STAGE2_BOUNDS = [
    (0.01,  0.25),       # ch4_oxidation_rate_per_d
    (0.02,  0.20),       # ch4_oxidation_depth_scale_m
    (100.0, 2000.0),     # ch4_ebullition_threshold_umol_L
    (0.5,   5.0),        # ch4_plant_transport_factor
]
STAGE2_DEFAULTS = [0.05, 0.05, 500.0, 2.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_gpp_params(gpp_site: str) -> Dict[str, float]:
    """Load best-fit vegetation parameters from the GPP calibration."""
    json_path = GPP_RUNS_DIR / f"ni_{gpp_site}_best_params.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"GPP calibration result not found: {json_path}\n"
            f"Run calibration/calibrate_ni.py --site {gpp_site} first."
        )
    with open(json_path) as fh:
        data = json.load(fh)
    # Extract only the actual parameter keys (not _metadata fields)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _load_obs_monthly_ch4() -> Optional[np.ndarray]:
    """Load BICEFS monthly mean CH4 flux (μmol m-2 s-1), shape (12,)."""
    monthly_file = DATA_DIR / "monthly_ch4_ni.csv"
    if not monthly_file.exists():
        print(f"  [warn] {monthly_file} not found; run extract_ni_ch4.py first.")
        return None
    import pandas as pd
    df = pd.read_csv(monthly_file)
    # Use root+shoot (total) flux as the calibration target
    rs = df[df["chamber_type"] == "root_shoot"].set_index("month")
    obs = np.full(12, np.nan)
    for m in range(1, 13):
        if m in rs.index:
            obs[m - 1] = rs.loc[m, "ch4_mean"]
    return obs


def _load_obs_smtz_depth() -> Optional[float]:
    """Load observed SMTZ depth (cm) from extracted porewater data."""
    smtz_file = DATA_DIR / "smtz_depth_ni.csv"
    if not smtz_file.exists():
        print(f"  [warn] {smtz_file} not found; run extract_ni_porewater.py first.")
        return None
    import pandas as pd
    df = pd.read_csv(smtz_file)
    # Use OL high marsh as the primary target
    row = df[(df["site"] == "OL") & (df["location"] == "HM")]
    if row.empty:
        row = df[df["location"] == "HM"]
    if row.empty or row["smtz_depth_cm"].isna().all():
        return None
    return float(row["smtz_depth_cm"].dropna().mean())


def _checkpoint_path(tag: str) -> Path:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    return CKPT_DIR / f"ch4_{tag}_checkpoint.json"


def _save_checkpoint(tag: str, method: str, n_starts: int,
                     starts_done: int, best_x: np.ndarray,
                     best_loss: float, n_evals: int) -> None:
    rec = {
        "tag": tag, "method": method, "n_starts": n_starts,
        "starts_done": starts_done, "best_x": best_x.tolist(),
        "best_loss": best_loss, "n_evals": n_evals,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    p = _checkpoint_path(tag)
    tmp = str(p) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=2)
    os.replace(tmp, p)


def _load_checkpoint(tag: str, method: str, n_starts: int) -> Optional[dict]:
    p = _checkpoint_path(tag)
    if not p.exists():
        return None
    try:
        with open(p) as fh:
            ckpt = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
    if ckpt.get("method") != method or ckpt.get("n_starts") != n_starts:
        return None
    return ckpt


# ---------------------------------------------------------------------------
# Model runner helper
# ---------------------------------------------------------------------------

def _run_with_params(
    run_id: str,
    extra_params: Dict[str, float],
    gpp_params: Dict[str, float],
    elevation_m: float,
    distance_m: float,
    output_dir: Path,
    cli_binary: str = CLI_BINARY,
) -> str:
    """Write YAML, run model, return nc_path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = str(output_dir / f"{run_id}.yaml")

    config = PlotConfig(
        site_name="north_inlet_ch4",
        plot_id=run_id,
        surface_elevation_m=elevation_m,
        distance_from_creek_m=distance_m,
        tides=north_inlet_default_tides(),
        met=north_inlet_default_met(),
        n_years=RUN_YEARS,
        sea_level_rise_m_yr=0.006,
        mean_aboveground_biomass_kg_m2=0.5,
        output_dir=str(output_dir),
    )

    all_extra = {**gpp_params, **extra_params}
    write_config(config, yaml_path, extra_parameters=all_extra)
    run_model(yaml_path, cli_binary=cli_binary, silent=True)
    return yaml_path.replace(".yaml", ".nc")


# ---------------------------------------------------------------------------
# Stage 1 — SMTZ calibration
# ---------------------------------------------------------------------------

class SmtzCalibrator:
    """Calibrate SO4/SMTZ-controlling parameters against observed SMTZ depth."""

    def __init__(
        self,
        gpp_params: Dict[str, float],
        smtz_target_cm: float,
        elevation_m: float = 0.45,
        distance_m: float = 33.0,
        cli_binary: str = CLI_BINARY,
        verbose: bool = True,
    ) -> None:
        self.gpp_params = gpp_params
        self.smtz_target_m = smtz_target_cm / 100.0
        self.elevation_m = elevation_m
        self.distance_m = distance_m
        self.cli_binary = cli_binary
        self.verbose = verbose
        self._n_evals = 0
        self._best_loss = np.inf
        self._t0 = time.time()

    def objective(self, log_params: np.ndarray) -> float:
        params = np.exp(log_params)
        extra = {name: float(v) for name, v in zip(STAGE1_PARAMS, params)}
        # Stage 1 needs methane enabled; fix Stage 2 params at defaults
        for name, val in zip(STAGE2_PARAMS, STAGE2_DEFAULTS):
            extra[name] = val

        self._n_evals += 1
        run_id = f"s1_e{self._n_evals:04d}"
        try:
            nc_path = _run_with_params(
                run_id, extra, self.gpp_params,
                self.elevation_m, self.distance_m,
                OPT_DIR, self.cli_binary,
            )
            mod_smtz = smtz_depth_m(nc_path, SKIP_SPINUP)
            if np.isnan(mod_smtz):
                loss = 10.0   # severe penalty if SMTZ undefined
            else:
                # Squared relative error on SMTZ depth
                loss = ((mod_smtz - self.smtz_target_m) / self.smtz_target_m) ** 2
        except Exception as exc:
            loss = 1e6
            if self.verbose:
                print(f"  [s1 eval {self._n_evals:4d}] ERROR: {exc}")
            return float(loss)
        finally:
            for ext in (".yaml", ".nc"):
                p = str(OPT_DIR / f"{run_id}{ext}")
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass

        if loss < self._best_loss:
            self._best_loss = loss

        if self.verbose:
            elapsed = time.time() - self._t0
            print(
                f"  [s1 {self._n_evals:4d}] "
                f"km={params[0]:.0f}  depth_scale={params[1]:.3f}  "
                f"flush_rate={params[2]:.3f}"
                f"  → SMTZ={mod_smtz*100:.1f} cm  "
                f"target={self.smtz_target_m*100:.1f} cm  "
                f"loss={loss:.4f}  best={self._best_loss:.4f}  ({elapsed:.0f}s)"
            )
        return float(loss)

    def run(self, method: str = "nelder-mead", n_starts: int = 3,
            tag: str = "stage1") -> Tuple[Dict[str, float], float]:
        log_bounds = [(np.log(lo), np.log(hi)) for lo, hi in STAGE1_BOUNDS]
        log_defaults = np.log(STAGE1_DEFAULTS)

        ckpt = _load_checkpoint(tag, method, n_starts)
        starts_done = 0
        best_result = None
        if ckpt:
            starts_done = ckpt["starts_done"]
            self._n_evals = ckpt["n_evals"]
            self._best_loss = ckpt["best_loss"]
            best_result = type("_R", (), {"x": np.array(ckpt["best_x"]),
                                          "fun": ckpt["best_loss"]})()
            if self.verbose:
                print(f"  Resuming Stage 1 from checkpoint "
                      f"({starts_done}/{n_starts} starts done)")

        if self.verbose:
            print(f"\n{'='*55}")
            print(f"Stage 1 — SMTZ depth  (target: {self.smtz_target_m*100:.1f} cm)")
            print(f"{'='*55}")

        if method == "de":
            if starts_done == 0:
                res = differential_evolution(
                    self.objective, log_bounds, maxiter=60, popsize=8,
                    tol=1e-3, seed=42, polish=True)
                best_result = res
                _save_checkpoint(tag, method, n_starts, 1,
                                 best_result.x, float(best_result.fun), self._n_evals)
        else:
            rng = np.random.default_rng(42)
            for _ in range(max(0, starts_done - 1)):
                [rng.uniform(lo, hi) for lo, hi in log_bounds]
            for i in range(starts_done, n_starts):
                x0 = log_defaults if i == 0 else np.array(
                    [rng.uniform(lo, hi) for lo, hi in log_bounds])
                res = minimize(self.objective, x0, method="Nelder-Mead",
                               options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-5})
                if best_result is None or res.fun < best_result.fun:
                    best_result = res
                _save_checkpoint(tag, method, n_starts, i + 1,
                                 best_result.x, float(best_result.fun), self._n_evals)

        best_params = {name: float(np.exp(v))
                       for name, v in zip(STAGE1_PARAMS, best_result.x)}
        if self.verbose:
            print(f"\n  Stage 1 best parameters:")
            for k, v in best_params.items():
                print(f"    {k}: {v:.5g}")
            print(f"  Final loss: {best_result.fun:.4f}")

        try:
            os.remove(_checkpoint_path(tag))
        except FileNotFoundError:
            pass

        return best_params, float(best_result.fun)


# ---------------------------------------------------------------------------
# Stage 2 — CH4 flux calibration
# ---------------------------------------------------------------------------

class Ch4FluxCalibrator:
    """Calibrate CH4 surface flux parameters against BICEFS monthly observations."""

    def __init__(
        self,
        gpp_params: Dict[str, float],
        stage1_params: Dict[str, float],
        obs_monthly: np.ndarray,
        elevation_m: float = 0.45,
        distance_m: float = 33.0,
        cli_binary: str = CLI_BINARY,
        verbose: bool = True,
    ) -> None:
        self.gpp_params = gpp_params
        self.stage1_params = stage1_params
        self.obs_monthly = obs_monthly          # shape (12,), nan where no obs
        self.obs_valid = ~np.isnan(obs_monthly)
        self.obs_mean = float(np.nanmean(obs_monthly))
        self.elevation_m = elevation_m
        self.distance_m = distance_m
        self.cli_binary = cli_binary
        self.verbose = verbose
        self._n_evals = 0
        self._best_loss = np.inf
        self._t0 = time.time()

        if self.obs_mean <= 0:
            raise ValueError("Observed monthly CH4 flux mean is ≤ 0; check data.")

    def objective(self, log_params: np.ndarray) -> float:
        params = np.exp(log_params)
        extra = {**self.stage1_params,
                 **{name: float(v) for name, v in zip(STAGE2_PARAMS, params)}}

        self._n_evals += 1
        run_id = f"s2_e{self._n_evals:04d}"
        try:
            nc_path = _run_with_params(
                run_id, extra, self.gpp_params,
                self.elevation_m, self.distance_m,
                OPT_DIR, self.cli_binary,
            )
            mod_monthly, _, _ = mean_ch4_flux_monthly(nc_path, SKIP_SPINUP)
            valid = self.obs_valid & ~np.isnan(mod_monthly)
            if not valid.any():
                loss = 1e6
            else:
                # Normalised RMSE on months with observations
                residuals = (mod_monthly[valid] - self.obs_monthly[valid]) / self.obs_mean
                rmse_loss = float(np.sqrt(np.mean(residuals ** 2)))
                # Penalise log-scale bias to prevent trivial zero solutions
                bias = np.log(max(np.nanmean(mod_monthly[valid]), 1e-6) /
                              self.obs_mean)
                loss = rmse_loss + 0.3 * bias ** 2

        except Exception as exc:
            loss = 1e6
            if self.verbose:
                print(f"  [s2 eval {self._n_evals:4d}] ERROR: {exc}")
            return float(loss)
        finally:
            for ext in (".yaml", ".nc"):
                p = str(OPT_DIR / f"{run_id}{ext}")
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass

        if loss < self._best_loss:
            self._best_loss = loss

        if self.verbose:
            elapsed = time.time() - self._t0
            mod_mean = float(np.nanmean(mod_monthly[self.obs_valid]))
            print(
                f"  [s2 {self._n_evals:4d}] "
                f"ox_rate={params[0]:.3f}  ox_depth={params[1]:.3f}  "
                f"ebull={params[2]:.0f}  beta={params[3]:.2f}"
                f"  → mod_flux={mod_mean:.3f}  obs={self.obs_mean:.3f}"
                f"  loss={loss:.4f}  best={self._best_loss:.4f}  ({elapsed:.0f}s)"
            )
        return float(loss)

    def run(self, method: str = "nelder-mead", n_starts: int = 5,
            tag: str = "stage2") -> Tuple[Dict[str, float], float]:
        log_bounds = [(np.log(lo), np.log(hi)) for lo, hi in STAGE2_BOUNDS]
        log_defaults = np.log(STAGE2_DEFAULTS)

        ckpt = _load_checkpoint(tag, method, n_starts)
        starts_done = 0
        best_result = None
        if ckpt:
            starts_done = ckpt["starts_done"]
            self._n_evals = ckpt["n_evals"]
            self._best_loss = ckpt["best_loss"]
            best_result = type("_R", (), {"x": np.array(ckpt["best_x"]),
                                          "fun": ckpt["best_loss"]})()
            if self.verbose:
                print(f"  Resuming Stage 2 from checkpoint "
                      f"({starts_done}/{n_starts} starts done)")

        n_obs_months = int(self.obs_valid.sum())
        if self.verbose:
            print(f"\n{'='*55}")
            print(f"Stage 2 — CH4 flux  "
                  f"(obs mean={self.obs_mean:.3f} μmol m-2 s-1, "
                  f"n={n_obs_months} months)")
            print(f"{'='*55}")

        if method == "de":
            if starts_done == 0:
                res = differential_evolution(
                    self.objective, log_bounds, maxiter=80, popsize=10,
                    tol=1e-3, seed=42, polish=True)
                best_result = res
                _save_checkpoint(tag, method, n_starts, 1,
                                 best_result.x, float(best_result.fun), self._n_evals)
        else:
            rng = np.random.default_rng(99)
            for _ in range(max(0, starts_done - 1)):
                [rng.uniform(lo, hi) for lo, hi in log_bounds]
            for i in range(starts_done, n_starts):
                x0 = log_defaults if i == 0 else np.array(
                    [rng.uniform(lo, hi) for lo, hi in log_bounds])
                res = minimize(self.objective, x0, method="Nelder-Mead",
                               options={"maxiter": 600, "xatol": 1e-4, "fatol": 1e-6})
                if best_result is None or res.fun < best_result.fun:
                    best_result = res
                _save_checkpoint(tag, method, n_starts, i + 1,
                                 best_result.x, float(best_result.fun), self._n_evals)

        best_params = {name: float(np.exp(v))
                       for name, v in zip(STAGE2_PARAMS, best_result.x)}
        if self.verbose:
            print(f"\n  Stage 2 best parameters:")
            for k, v in best_params.items():
                print(f"    {k}: {v:.5g}")
            print(f"  Final loss: {best_result.fun:.4f}")

        try:
            os.remove(_checkpoint_path(tag))
        except FileNotFoundError:
            pass

        return best_params, float(best_result.fun)


# ---------------------------------------------------------------------------
# Final run and output
# ---------------------------------------------------------------------------

def run_final(
    all_params: Dict[str, float],
    gpp_params: Dict[str, float],
    elevation_m: float,
    distance_m: float,
    cli_binary: str,
    site_key: str,
) -> str:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = f"ni_{site_key}_ch4_best"
    nc_path = _run_with_params(
        run_id, all_params, gpp_params,
        elevation_m, distance_m,
        OUT_DIR, cli_binary,
    )
    diag = summarise_ch4_run(nc_path, SKIP_SPINUP)

    record = {**all_params, **gpp_params,
              "_site_key": site_key,
              "_elevation_m": elevation_m,
              **{f"_diag_{k}": v for k, v in diag.items()}}
    json_path = OUT_DIR / f"ni_{site_key}_ch4_best_params.json"
    with open(json_path, "w") as fh:
        json.dump(record, fh, indent=2)

    print(f"\nFinal run saved: {nc_path}")
    print(f"Parameters saved: {json_path}")
    print("\nDiagnostics:")
    for k, v in diag.items():
        print(f"  {k}: {v}")
    return nc_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpp-site", default="ol_hm_c",
                   help="Site key for GPP calibration to load (default: ol_hm_c)")
    p.add_argument("--stage", type=int, choices=[1, 2, 12], default=12,
                   help="Run stage 1 only (1), stage 2 only (2), or both (12, default)")
    p.add_argument("--method", choices=["nelder-mead", "de"], default="nelder-mead")
    p.add_argument("--n-starts", type=int, default=3,
                   help="Nelder-Mead restarts per stage (default: 3)")
    p.add_argument("--elevation", type=float, default=0.45,
                   help="Plot elevation m NAVD88 (default: 0.45)")
    p.add_argument("--distance", type=float, default=33.0,
                   help="Distance from creek m (default: 33.0)")
    p.add_argument("--cli", default=CLI_BINARY)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    verbose = not args.quiet
    site_key = args.gpp_site

    print(f"Loading GPP parameters from site: {site_key}")
    gpp_params = _load_gpp_params(site_key)

    # Load observations
    obs_smtz_cm = _load_obs_smtz_depth() or _SMTZ_TARGET_CM_DEFAULT
    obs_monthly  = _load_obs_monthly_ch4()

    if obs_monthly is None and args.stage in (2, 12):
        print("No monthly CH4 observations found. Run extract_ni_ch4.py first.")
        print("Continuing with Stage 1 only.")
        args.stage = 1

    print(f"SMTZ calibration target: {obs_smtz_cm:.1f} cm")
    if obs_monthly is not None:
        print(f"CH4 flux obs mean (non-NaN months): "
              f"{np.nanmean(obs_monthly):.3f} μmol m-2 s-1")

    stage1_params: Dict[str, float] = {}
    stage2_params: Dict[str, float] = {}

    # ---- Stage 1 ----
    if args.stage in (1, 12):
        cal1 = SmtzCalibrator(
            gpp_params=gpp_params,
            smtz_target_cm=obs_smtz_cm,
            elevation_m=args.elevation,
            distance_m=args.distance,
            cli_binary=args.cli,
            verbose=verbose,
        )
        stage1_params, s1_loss = cal1.run(method=args.method, n_starts=args.n_starts)
    else:
        # Load from previous Stage 1 result if available
        s1_json = OUT_DIR / f"ni_{site_key}_ch4_best_params.json"
        if s1_json.exists():
            with open(s1_json) as fh:
                prev = json.load(fh)
            stage1_params = {k: v for k in STAGE1_PARAMS
                             if (v := prev.get(k)) is not None}
            print(f"Loaded Stage 1 params from {s1_json}")
        else:
            stage1_params = {name: val for name, val in zip(STAGE1_PARAMS, STAGE1_DEFAULTS)}
            print("Using Stage 1 default parameters.")

    # ---- Stage 2 ----
    if args.stage in (2, 12) and obs_monthly is not None:
        cal2 = Ch4FluxCalibrator(
            gpp_params=gpp_params,
            stage1_params=stage1_params,
            obs_monthly=obs_monthly,
            elevation_m=args.elevation,
            distance_m=args.distance,
            cli_binary=args.cli,
            verbose=verbose,
        )
        stage2_params, s2_loss = cal2.run(method=args.method, n_starts=args.n_starts)
    else:
        stage2_params = {name: val for name, val in zip(STAGE2_PARAMS, STAGE2_DEFAULTS)}

    # ---- Final run ----
    all_ch4_params = {**stage1_params, **stage2_params}
    nc_path = run_final(
        all_ch4_params, gpp_params,
        args.elevation, args.distance,
        args.cli, site_key,
    )

    print(f"\nCalibration complete. Final output: {nc_path}")
    print(f"Plot with: python plot_ch4_calibration.py --nc {nc_path} --gpp-site {site_key}")


if __name__ == "__main__":
    main()
