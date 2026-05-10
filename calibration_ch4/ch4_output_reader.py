# ch4_output_reader.py
#
# Extends output_reader.py with CH4-specific diagnostics from marsh_muddpile
# NetCDF output.
#
# Layer-resolved variables (layer_porewater_so4, layer_porewater_ch4,
# layer_porewater_nh4, layer_top_elevation, layer_thickness) are stored as
# 2-D arrays (time × layer) with _FillValue = -9999 for unused layer slots.
# Layers are ordered deepest-first: index 0 = bottom of column,
# index n_layers-1 = surface layer.

from __future__ import annotations
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import netCDF4 as nc
except ImportError as exc:
    raise ImportError("netCDF4 is required: pip install netCDF4") from exc

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "calibration"))
from output_reader import read_time_series


# ---------------------------------------------------------------------------
# Layer variable reader
# ---------------------------------------------------------------------------

def read_layer_variables(nc_path: str) -> Dict[str, np.ndarray]:
    """Read 2-D (time × layer) variables from a marsh_muddpile NetCDF file.

    Fill values (-9999) are replaced with np.nan.  Returns a dict keyed by
    variable name, values shape (n_time, n_layers).
    """
    layer_vars: Dict[str, np.ndarray] = {}
    with nc.Dataset(nc_path, "r") as ds:
        for name, var in ds.variables.items():
            if var.ndim == 2 and "layer" in var.dimensions:
                data = np.array(var[:], dtype=np.float64)
                fv = getattr(var, "_FillValue", -9999.0)
                data[np.isclose(data, fv)] = np.nan
                layer_vars[name] = data
    return layer_vars


def layer_depths_m(nc_path: str) -> np.ndarray:
    """Return midpoint depths below surface (m) for all time steps.

    Shape: (n_time, n_layers).  NaN where layer slot is empty.
    Surface = depth 0; depth increases downward.
    """
    lv = read_layer_variables(nc_path)
    ts = read_time_series(nc_path)

    top_elev = lv.get("layer_top_elevation")
    thickness = lv.get("layer_thickness")
    surface_elev = ts.get("surface_elevation")

    if top_elev is None or thickness is None or surface_elev is None:
        raise KeyError("layer_top_elevation, layer_thickness, or surface_elevation "
                       "not found in NetCDF. Check that the model was run with "
                       "porewater output enabled.")

    # surface_elev shape: (n_time,) → broadcast to (n_time, n_layers)
    surf = surface_elev[:, np.newaxis]
    depth = surf - top_elev + 0.5 * thickness
    return depth


# ---------------------------------------------------------------------------
# Monthly averaging helpers
# ---------------------------------------------------------------------------

def _month_index(time_days: np.ndarray) -> np.ndarray:
    """Return 1-based calendar month for each timestep."""
    return ((time_days % 365.25) / (365.25 / 12.0)).astype(int) % 12 + 1


def _post_spinup_mask(time_days: np.ndarray, skip_spinup_years: int) -> np.ndarray:
    return time_days >= skip_spinup_years * 365.25


def mean_ch4_flux_monthly(
    nc_path: str,
    skip_spinup_years: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monthly mean, std, and count of surface CH4 flux (μmol m-2 s-1).

    Returns
    -------
    mean_by_month : shape (12,), nan where no data
    std_by_month  : shape (12,)
    n_by_month    : shape (12,)
    """
    ts = read_time_series(nc_path)
    time_days = ts["model_time_days"]
    flux = ts.get("surface_ch4_flux_umol_m2_s")

    if flux is None:
        raise KeyError("surface_ch4_flux_umol_m2_s not in NetCDF. "
                       "Enable methane_model_name: sulfate_methane.")

    mask = _post_spinup_mask(time_days, skip_spinup_years)
    months = _month_index(time_days)

    means = np.full(12, np.nan)
    stds = np.full(12, np.nan)
    ns = np.zeros(12, dtype=int)
    for m in range(1, 13):
        sel = mask & (months == m)
        vals = flux[sel]
        if len(vals) > 0:
            means[m - 1] = vals.mean()
            stds[m - 1]  = vals.std()
            ns[m - 1]    = len(vals)

    return means, stds, ns


def mean_porewater_profile(
    nc_path: str,
    variable: str,
    skip_spinup_years: int = 3,
    depth_bins_cm: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Time-averaged porewater profile for a layer variable.

    Parameters
    ----------
    variable : one of layer_porewater_so4, layer_porewater_ch4, layer_porewater_nh4
    depth_bins_cm : if supplied, interpolate onto these depths (cm below surface)

    Returns
    -------
    depth_m   : shape (n_layers,)  — median depth over time (m below surface)
    conc_mean : shape (n_layers,)  — time-mean concentration
    """
    lv = read_layer_variables(nc_path)
    ts = read_time_series(nc_path)
    time_days = ts["model_time_days"]

    if variable not in lv:
        raise KeyError(f"'{variable}' not in layer variables. "
                       f"Available: {list(lv)}")

    conc = lv[variable]
    depths = layer_depths_m(nc_path)

    mask = _post_spinup_mask(time_days, skip_spinup_years)

    conc_sub = conc[mask]
    depth_sub = depths[mask]

    # Median depth per layer slot (depths shift slightly as column compacts/grows)
    depth_m = np.nanmedian(depth_sub, axis=0)
    conc_mean = np.nanmean(conc_sub, axis=0)

    # Drop empty layer slots
    valid = ~(np.isnan(depth_m) | np.isnan(conc_mean))
    depth_m = depth_m[valid]
    conc_mean = conc_mean[valid]

    # Sort by depth (shallow → deep)
    order = np.argsort(depth_m)
    depth_m = depth_m[order]
    conc_mean = conc_mean[order]

    if depth_bins_cm is not None:
        depth_bins_m = depth_bins_cm / 100.0
        conc_mean = np.interp(depth_bins_m, depth_m, conc_mean,
                              left=np.nan, right=np.nan)
        depth_m = depth_bins_m

    return depth_m, conc_mean


def smtz_depth_m(
    nc_path: str,
    skip_spinup_years: int = 3,
    so4_threshold_fraction: float = 0.10,
) -> float:
    """Estimate SMTZ depth (m below surface) from the time-mean SO4 profile.

    SMTZ = deepest depth at which SO4 >= threshold_fraction × creek SO4.
    Below this depth SO4 is considered depleted and methanogenesis dominates.

    Returns NaN if SO4 never drops below the threshold (column is fully
    flushed) or if SO4 data is unavailable.
    """
    try:
        depth_m, so4 = mean_porewater_profile(nc_path, "layer_porewater_so4",
                                               skip_spinup_years)
    except KeyError:
        return np.nan

    if len(so4) == 0:
        return np.nan

    surface_so4 = so4[0]   # shallowest layer
    threshold = surface_so4 * so4_threshold_fraction

    # Walk from shallow to deep; find the deepest layer still above threshold
    above = depth_m[so4 >= threshold]
    if len(above) == 0:
        return 0.0   # SO4 depleted even at surface
    return float(above[-1])


def monthly_porewater_profiles(
    nc_path: str,
    variable: str,
    skip_spinup_years: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Porewater profile averaged by calendar month.

    Returns
    -------
    months    : shape (12,)          — 1–12
    depth_m   : shape (n_layers,)    — median depth over all post-spinup steps
    conc      : shape (12, n_layers) — monthly mean concentration
    """
    lv = read_layer_variables(nc_path)
    ts = read_time_series(nc_path)
    time_days = ts["model_time_days"]
    depths = layer_depths_m(nc_path)

    if variable not in lv:
        raise KeyError(f"'{variable}' not available")

    conc = lv[variable]
    mask = _post_spinup_mask(time_days, skip_spinup_years)
    months_arr = _month_index(time_days)

    depth_m = np.nanmedian(depths[mask], axis=0)

    conc_monthly = np.full((12, conc.shape[1]), np.nan)
    for m in range(1, 13):
        sel = mask & (months_arr == m)
        if sel.any():
            conc_monthly[m - 1] = np.nanmean(conc[sel], axis=0)

    # Drop unused layer slots
    valid_layers = ~np.isnan(depth_m)
    depth_m = depth_m[valid_layers]
    conc_monthly = conc_monthly[:, valid_layers]

    order = np.argsort(depth_m)
    return np.arange(1, 13), depth_m[order], conc_monthly[:, order]


# ---------------------------------------------------------------------------
# Summary diagnostic
# ---------------------------------------------------------------------------

def summarise_ch4_run(
    nc_path: str,
    skip_spinup_years: int = 3,
) -> Dict:
    """Return a dict of key CH4 diagnostics averaged over the post-spinup period."""
    ts = read_time_series(nc_path)
    time_days = ts["model_time_days"]
    mask = _post_spinup_mask(time_days, skip_spinup_years)

    def _mean(key):
        v = ts.get(key)
        return float(np.mean(v[mask])) if v is not None else None

    ch4_monthly, ch4_std, _ = mean_ch4_flux_monthly(nc_path, skip_spinup_years)

    diag = {
        "mean_ch4_flux_umol_m2_s":     float(np.nanmean(ch4_monthly)),
        "peak_ch4_flux_umol_m2_s":     float(np.nanmax(ch4_monthly)),
        "seasonal_ch4_cv":             (float(np.nanstd(ch4_monthly) /
                                         np.nanmean(ch4_monthly))
                                        if np.nanmean(ch4_monthly) > 0 else np.nan),
        "mean_surface_so4_umol_L":     _mean("surface_so4_umol_L"),
        "smtz_depth_m":                smtz_depth_m(nc_path, skip_spinup_years),
        "mean_surface_elevation_m":    _mean("surface_elevation"),
        "mean_inundation_fraction":    _mean("inundation_fraction"),
    }
    return diag
