#!/usr/bin/env python3
"""
calibrate_ni.py

Calibrate marsh_muddpile vegetation parameters against North Inlet LTER
observations.  Optimises three parameters to minimise a combined loss of:

  1. Relative squared error in mean annual ANPP (post-spinup).
  2. Normalised RMSE of the mean seasonal aboveground biomass cycle
     (model post-spinup monthly means vs observed monthly means averaged
     over all plots, subplots, and years).

Both terms are dimensionless and equally weighted by default.  The weight
on the biomass term is controlled by ``--biomass-weight`` (default 1.0).
Set it to 0 to revert to ANPP-only calibration.

Sites
-----
  gi_hm_c   GI high-marsh control         elev 0.6 m, 30 m from channel
  gi_hm_np  GI high-marsh N+P treatment   elev 0.6 m, 30 m from channel
  gi_lm_c   GI low-marsh control          elev 0.3 m, 30 m from channel
  ol_hm_c   OL high-marsh control         elev 0.6 m, 30 m from channel
  ol_lm_c   OL low-marsh control          elev 0.3 m, 30 m from channel

Note: gi_hm_np uses N+P-fertilised observations.  The model has no nutrient
limitation, so calibrated parameters for that site reflect a high-productivity
regime rather than a true nutrient response.

Optimised parameters
--------------------
  vegetation_aboveground_capacity_kg_m2            [0.10, 3.0]
  vegetation_aboveground_mortality_base_per_day    [1e-4, 5e-3]
  vegetation_cold_mortality_slope_per_day_per_c    [1e-4, 1e-2]
  vegetation_lue_gC_per_umol                       [1e-5, 5e-4]

LUE (light-use efficiency) sets the overall productivity level.  The
cold-mortality slope drives mechanistic autumn die-back by raising
mortality per degree below the cold threshold (default 20 °C).

Usage
-----
  # All sites, Nelder-Mead (default):
  python calibrate_ni.py

  # One site:
  python calibrate_ni.py --site gi_hm_c

  # Differential evolution (more thorough, slower):
  python calibrate_ni.py --method de

  # Custom number of Nelder-Mead restarts:
  python calibrate_ni.py --n-starts 5 --site ol_lm_c
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import differential_evolution, minimize

sys.path.insert(0, os.path.dirname(__file__))

from lter_reader import read_ni_annual_productivity, read_ni_monthly_biomass
from model_runner import run_model
from output_reader import mean_annual_anpp_g_m2_yr, read_time_series
from site_config import PlotConfig, north_inlet_default_met, north_inlet_default_tides
from yaml_writer import write_config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CLI_BINARY = "marsh_cli"
RUN_YEARS = 30
SKIP_SPINUP = 3
OPT_RUN_DIR = "calibration_runs/opt"
OUT_DIR = "calibration_runs"
CHECKPOINT_DIR = "calibration_runs/checkpoints"

# Each entry: elevation_m and distance_m are the plot-level inputs.
# SITE / LOCATION / TREATMENT identify which LTER rows to use as calibration target.
NI_SITES: Dict[str, dict] = {
    "gi_hm_c":  dict(site="GI", location="HM", treatment="C",  elevation_m=0.50, distance_m=62.0),
    "gi_hm_np": dict(site="GI", location="HM", treatment="NP", elevation_m=0.50, distance_m=62.0),
    "gi_lm_c":  dict(site="GI", location="LM", treatment="C",  elevation_m=0.20, distance_m=42.0),
    "ol_hm_c":  dict(site="OL", location="HM", treatment="C",  elevation_m=0.45, distance_m=33.0),
    "ol_lm_c":  dict(site="OL", location="LM", treatment="C",  elevation_m=0.25, distance_m=33.0),
}

PARAM_NAMES: List[str] = [
    "vegetation_aboveground_capacity_kg_m2",
    "vegetation_aboveground_mortality_base_per_day",
    "vegetation_cold_mortality_slope_per_day_per_c",
    "vegetation_lue_gC_per_umol",
]

# Search bounds (physical units, not log-transformed)
PARAM_BOUNDS: List[Tuple[float, float]] = [
    (0.10, 3.0),
    (1e-4, 5e-3),
    (1e-4, 1e-2),
    (1e-7, 1e-4),   # LUE: extended lower bound to cover calibrated 1.6e-6
]

# Default starting point for Nelder-Mead (biomass-curve calibration, Morris 2013).
PARAM_DEFAULTS = [0.95, 0.001, 0.002, 1.6e-6]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _checkpoint_path(site_key: str) -> str:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, f"ni_{site_key}_checkpoint.json")


def _save_checkpoint(
    site_key: str,
    method: str,
    n_starts_requested: int,
    starts_completed: int,
    best_x: np.ndarray,
    best_loss: float,
    n_evals: int,
) -> None:
    record = {
        "site_key": site_key,
        "method": method,
        "n_starts_requested": n_starts_requested,
        "starts_completed": starts_completed,
        "best_x": best_x.tolist(),
        "best_loss": float(best_loss),
        "n_evals_total": n_evals,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = _checkpoint_path(site_key)
    # Write to a temp file first so a partial write never corrupts the checkpoint.
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(record, fh, indent=2)
    os.replace(tmp, path)


def _load_checkpoint(site_key: str, method: str, n_starts: int) -> Optional[dict]:
    """Return checkpoint dict if it exists and is compatible, else None."""
    path = _checkpoint_path(site_key)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            ckpt = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
    # Invalidate if the method or start count changed.
    if ckpt.get("method") != method or ckpt.get("n_starts_requested") != n_starts:
        return None
    return ckpt


def _model_monthly_biomass(nc_path: str, skip_spinup_years: int) -> np.ndarray:
    """Return mean post-spinup biomass (g m-2) for each calendar month (length 12)."""
    series = read_time_series(nc_path)
    time_days = series["model_time_days"]
    biomass_g_m2 = series["aboveground_biomass_kg_m2"] * 1000.0
    year_idx = (time_days / 365.25).astype(int)
    month_idx = ((time_days % 365.25) / (365.25 / 12.0)).astype(int) % 12 + 1  # 1-12
    mask = year_idx >= skip_spinup_years
    return np.array([
        biomass_g_m2[mask & (month_idx == m)].mean()
        for m in range(1, 13)
    ])


# ---------------------------------------------------------------------------
# Calibrator class
# ---------------------------------------------------------------------------

class SiteCalibrator:
    """Runs the model repeatedly to minimise combined ANPP + biomass residuals."""

    def __init__(
        self,
        site_key: str,
        cli_binary: str = CLI_BINARY,
        biomass_weight: float = 1.0,
        verbose: bool = True,
    ) -> None:
        if site_key not in NI_SITES:
            raise ValueError(f"Unknown site '{site_key}'. Options: {list(NI_SITES)}")

        self.site_key = site_key
        self.site_info = NI_SITES[site_key]
        self.cli_binary = cli_binary
        self.biomass_weight = biomass_weight
        self.verbose = verbose
        self._eval_count = 0
        self._best_loss = np.inf
        self._t0 = time.time()

        # Annual ANPP target
        obs = read_ni_annual_productivity(treatment=self.site_info["treatment"])
        obs = obs[
            (obs["SITE"] == self.site_info["site"])
            & (obs["LOCATION"] == self.site_info["location"])
        ]
        if obs.empty:
            raise RuntimeError(f"No LTER observations found for site '{site_key}'")
        self.obs = obs
        self.target_anpp = float(obs["PRODUCTIVITY"].mean())

        # Monthly biomass target: mean over all plots, subplots, and years by month.
        # Averaging across subplots is correct here because the annual productivity
        # data already represents a site-level mean; we want a consistent target.
        bio = read_ni_monthly_biomass(
            treatment=self.site_info["treatment"],
            location=self.site_info["location"],
        )
        bio = bio[bio["SITE"] == self.site_info["site"]].copy()
        bio["_month"] = bio["DATE"].dt.month
        monthly = bio.groupby("_month")["ABOVEGROUND_BIOMASS"].mean()
        self.obs_monthly_biomass = np.array([
            monthly.get(m, np.nan) for m in range(1, 13)
        ])
        # Normalisation denominator: mean of the non-NaN monthly means
        valid = np.isfinite(self.obs_monthly_biomass)
        self.obs_biomass_mean = float(self.obs_monthly_biomass[valid].mean())

        os.makedirs(OPT_RUN_DIR, exist_ok=True)
        os.makedirs(OUT_DIR, exist_ok=True)

    def _make_config(self, run_id: str, output_dir: str = OPT_RUN_DIR) -> PlotConfig:
        return PlotConfig(
            site_name="north_inlet",
            plot_id=run_id,
            surface_elevation_m=self.site_info["elevation_m"],
            distance_from_creek_m=self.site_info["distance_m"],
            tides=north_inlet_default_tides(),
            met=north_inlet_default_met(),
            n_years=RUN_YEARS,
            sea_level_rise_m_yr=0.006,
            mean_aboveground_biomass_kg_m2=self.obs_biomass_mean / 1000.0,
            output_dir=output_dir,
        )

    def objective(self, log_params: np.ndarray) -> float:
        """Combined ANPP + biomass loss.  Inputs are log-transformed parameters."""
        params = np.exp(log_params)
        extra = {name: float(v) for name, v in zip(PARAM_NAMES, params)}

        self._eval_count += 1
        run_id = f"ni_{self.site_key}_e{self._eval_count:04d}"
        yaml_path = os.path.join(OPT_RUN_DIR, f"{run_id}.yaml")
        nc_path = yaml_path.replace(".yaml", ".nc")

        try:
            config = self._make_config(run_id)
            write_config(config, yaml_path, extra_parameters=extra)
            run_model(yaml_path, cli_binary=self.cli_binary, silent=True)

            # --- ANPP term (log-space) ---
            # log(mod/obs)² is symmetric around the right answer and diverges to
            # +∞ as mod→0, so dead-plant states are never a local minimum.
            # The floor of 1 g/m²/yr avoids log(0) while still penalising
            # near-zero ANPP heavily (loss ≈ 47 when plant is dead vs ~0.5 when
            # mod is 2× target).
            mod_anpp = mean_annual_anpp_g_m2_yr(nc_path, skip_spinup_years=SKIP_SPINUP)
            anpp_loss = (np.log(max(mod_anpp, 1.0) / self.target_anpp)) ** 2

            # --- Biomass seasonal-cycle term ---
            if self.biomass_weight > 0.0:
                mod_monthly = _model_monthly_biomass(nc_path, SKIP_SPINUP)
                valid = np.isfinite(mod_monthly) & np.isfinite(self.obs_monthly_biomass)
                bio_loss = float(np.mean(
                    ((mod_monthly[valid] - self.obs_monthly_biomass[valid])
                     / self.obs_biomass_mean) ** 2
                )) if valid.any() else 0.0
            else:
                bio_loss = 0.0

            loss = anpp_loss + self.biomass_weight * bio_loss

        except Exception as exc:
            anpp_loss = bio_loss = loss = 1e6
            if self.verbose:
                print(f"  [eval {self._eval_count:4d}] ERROR: {exc}")
            return float(loss)
        finally:
            for path in (yaml_path, nc_path):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass

        if loss < self._best_loss:
            self._best_loss = loss

        if self.verbose:
            elapsed = time.time() - self._t0
            print(
                f"  [eval {self._eval_count:4d}] "
                f"cap={params[0]:.3f}  mort_base={params[1]:.5f}  cold_slope={params[2]:.5f}"
                f"  lue={params[3]:.2e}"
                f"  → ANPP={mod_anpp:.1f}  "
                f"L_anpp={anpp_loss:.4f}  L_bio={bio_loss:.4f}  "
                f"loss={loss:.4f}  best={self._best_loss:.4f}  ({elapsed:.0f}s)"
            )

        return float(loss)

    def run(
        self,
        method: str = "nelder-mead",
        n_starts: int = 3,
    ) -> dict:
        """Run calibration.  Returns dict with best parameters and diagnostics.

        Checkpoints are written to CHECKPOINT_DIR after every completed start so
        that the run can be resumed after an interruption.  On startup the
        checkpoint is validated against the current method and n_starts; if they
        differ (e.g. you increased n_starts) the checkpoint is ignored and a
        fresh run begins.
        """
        log_bounds = [(np.log(lo), np.log(hi)) for lo, hi in PARAM_BOUNDS]
        log_defaults = np.log(PARAM_DEFAULTS)

        # ---- load checkpoint ------------------------------------------------
        ckpt = _load_checkpoint(self.site_key, method, n_starts)
        starts_done = 0
        best_result = None

        if ckpt is not None:
            starts_done = ckpt["starts_completed"]
            self._eval_count = ckpt["n_evals_total"]
            self._best_loss = ckpt["best_loss"]
            # Reconstruct a minimal result-like object from the saved best_x.
            best_result = type(
                "_Ckpt", (),
                {"x": np.array(ckpt["best_x"]), "fun": ckpt["best_loss"]},
            )()
            if self.verbose and starts_done > 0:
                print(f"  Resuming from checkpoint: {starts_done}/{n_starts} "
                      f"starts done, {self._eval_count} evals so far, "
                      f"best loss {self._best_loss:.4f}")

        if self.verbose:
            print(f"\n{'=' * 65}")
            print(f"Site : {self.site_key}  "
                  f"({self.site_info['site']} {self.site_info['location']} "
                  f"{self.site_info['treatment']})")
            print(f"Target ANPP        : {self.target_anpp:.1f} g m-2 yr-1 "
                  f"(n={len(self.obs)} obs years)")
            print(f"Obs biomass mean   : {self.obs_biomass_mean:.1f} g m-2 "
                  f"(normalisation denominator)")
            print(f"Biomass weight     : {self.biomass_weight}")
            print(f"Elevation          : {self.site_info['elevation_m']} m, "
                  f"Distance: {self.site_info['distance_m']} m")
            print(f"Method             : {method}, starts: {n_starts} "
                  f"(resuming from start {starts_done + 1})" if starts_done else
                  f"Method             : {method}, starts: {n_starts}")
            print(f"{'=' * 65}")

        if method == "de":
            if starts_done == 0:
                result = differential_evolution(
                    self.objective,
                    bounds=log_bounds,
                    maxiter=60,
                    popsize=8,
                    tol=1e-3,
                    seed=42,
                    polish=True,
                )
                best_result = result
                _save_checkpoint(
                    self.site_key, method, n_starts,
                    starts_completed=1,
                    best_x=best_result.x,
                    best_loss=float(best_result.fun),
                    n_evals=self._eval_count,
                )
            # DE is a single call — nothing to resume mid-run.
        else:
            # Nelder-Mead with multiple restarts, checkpointed per start.
            rng = np.random.default_rng(42)
            # Burn the RNG forward to stay consistent with a fresh run when
            # resuming partway through (start 0 uses log_defaults, not rng).
            for _ in range(max(0, starts_done - 1)):
                [rng.uniform(lo, hi) for lo, hi in log_bounds]

            for i in range(starts_done, n_starts):
                x0 = log_defaults.copy() if i == 0 else np.array(
                    [rng.uniform(lo, hi) for lo, hi in log_bounds]
                )
                if self.verbose:
                    print(f"\n  Start {i + 1}/{n_starts}  "
                          f"x0 = {np.exp(x0).round(5)}")
                result = minimize(
                    self.objective,
                    x0,
                    method="Nelder-Mead",
                    options={"maxiter": 600, "xatol": 1e-3, "fatol": 1e-5},
                )
                if best_result is None or result.fun < best_result.fun:
                    best_result = result

                _save_checkpoint(
                    self.site_key, method, n_starts,
                    starts_completed=i + 1,
                    best_x=best_result.x,
                    best_loss=float(best_result.fun),
                    n_evals=self._eval_count,
                )

        best_params = {
            name: float(np.exp(v))
            for name, v in zip(PARAM_NAMES, best_result.x)
        }

        # Final run with best parameters — saved permanently for later plotting
        final_run_id = f"ni_{self.site_key}_best"
        yaml_path = os.path.join(OUT_DIR, f"{final_run_id}.yaml")
        nc_path = yaml_path.replace(".yaml", ".nc")
        config_final = self._make_config(final_run_id, output_dir=OUT_DIR)
        write_config(config_final, yaml_path, extra_parameters=best_params)
        run_model(yaml_path, cli_binary=self.cli_binary, silent=True)
        mod_anpp = mean_annual_anpp_g_m2_yr(nc_path, skip_spinup_years=SKIP_SPINUP)
        mod_monthly = _model_monthly_biomass(nc_path, SKIP_SPINUP)
        valid = np.isfinite(mod_monthly) & np.isfinite(self.obs_monthly_biomass)
        biomass_rmse = float(np.sqrt(np.mean(
            (mod_monthly[valid] - self.obs_monthly_biomass[valid]) ** 2
        ))) if valid.any() else np.nan

        record = {
            **best_params,
            "_site_key": self.site_key,
            "_site": self.site_info["site"],
            "_location": self.site_info["location"],
            "_treatment": self.site_info["treatment"],
            "_elevation_m": self.site_info["elevation_m"],
            "_biomass_weight": self.biomass_weight,
            "_target_anpp": self.target_anpp,
            "_modelled_anpp": float(mod_anpp),
            "_anpp_ratio": float(mod_anpp / self.target_anpp),
            "_obs_biomass_mean": self.obs_biomass_mean,
            "_modelled_biomass_mean": float(mod_monthly[valid].mean()) if valid.any() else np.nan,
            "_biomass_rmse_g_m2": biomass_rmse,
            "_loss": float(best_result.fun),
            "_n_evals": self._eval_count,
            "_nc_path": nc_path,
        }

        # Save JSON summary
        json_path = os.path.join(OUT_DIR, f"ni_{self.site_key}_best_params.json")
        with open(json_path, "w") as fh:
            json.dump(record, fh, indent=2)

        if self.verbose:
            print(f"\n  Best parameters:")
            for name in PARAM_NAMES:
                print(f"    {name}: {best_params[name]:.6f}")
            print(f"  Target ANPP      : {self.target_anpp:.1f} g m-2 yr-1")
            print(f"  Modelled ANPP    : {mod_anpp:.1f} g m-2 yr-1")
            print(f"  ANPP ratio       : {mod_anpp / self.target_anpp:.3f}")
            print(f"  Obs biomass mean : {self.obs_biomass_mean:.1f} g m-2")
            print(f"  Mod biomass mean : {record['_modelled_biomass_mean']:.1f} g m-2")
            print(f"  Biomass RMSE     : {biomass_rmse:.1f} g m-2")
            print(f"  Best NC: {nc_path}")
            print(f"  Params : {json_path}")

        # Remove the checkpoint now that the final outputs are safely written.
        ckpt_file = _checkpoint_path(self.site_key)
        try:
            os.remove(ckpt_file)
        except FileNotFoundError:
            pass

        return record


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(results: List[dict]) -> None:
    header = (
        f"{'Site':<12} {'ANPP_obs':>8} {'ANPP_mod':>9} {'ratio':>6}  "
        f"{'bio_obs':>7} {'bio_mod':>7} {'bio_rmse':>9}  "
        f"{'capacity':>10} {'mort_base':>10} {'cold_slope':>11} {'lue':>9}"
    )
    print(f"\n{'=' * len(header)}")
    print(header)
    print(f"{'-' * len(header)}")
    for r in results:
        print(
            f"{r['_site_key']:<12} "
            f"{r['_target_anpp']:>8.1f} "
            f"{r['_modelled_anpp']:>9.1f} "
            f"{r['_anpp_ratio']:>6.3f}  "
            f"{r['_obs_biomass_mean']:>7.1f} "
            f"{r['_modelled_biomass_mean']:>7.1f} "
            f"{r['_biomass_rmse_g_m2']:>9.1f}  "
            f"{r[PARAM_NAMES[0]]:>10.4f} "
            f"{r[PARAM_NAMES[1]]:>10.5f} "
            f"{r[PARAM_NAMES[2]]:>11.5f} "
            f"{r[PARAM_NAMES[3]]:>9.2e}"
        )
    print(f"{'=' * len(header)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate marsh_muddpile against North Inlet LTER ANPP."
    )
    parser.add_argument(
        "--site",
        choices=list(NI_SITES),
        default=None,
        help="Site to calibrate (default: all sites).",
    )
    parser.add_argument(
        "--method",
        choices=["nelder-mead", "de"],
        default="nelder-mead",
        help="Optimisation method: nelder-mead (default) or de (differential evolution).",
    )
    parser.add_argument(
        "--n-starts",
        type=int,
        default=3,
        help="Number of random restarts for Nelder-Mead (default: 3).",
    )
    parser.add_argument(
        "--cli",
        default=CLI_BINARY,
        help=f"Path to marsh_cli binary (default: '{CLI_BINARY}').",
    )
    parser.add_argument(
        "--biomass-weight",
        type=float,
        default=1.0,
        help="Weight on the seasonal biomass term relative to the ANPP term (default: 1.0). "
             "Set to 0 for ANPP-only calibration.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-evaluation progress output.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sites = [args.site] if args.site else list(NI_SITES)

    results = []
    for site_key in sites:
        cal = SiteCalibrator(
            site_key=site_key,
            cli_binary=args.cli,
            biomass_weight=args.biomass_weight,
            verbose=not args.quiet,
        )
        record = cal.run(method=args.method, n_starts=args.n_starts)
        results.append(record)

    if len(results) > 1:
        print_summary_table(results)


if __name__ == "__main__":
    main()
