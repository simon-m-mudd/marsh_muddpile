# lter_reader.py
#
# Reads LTER observational data from the lter_data/ directory and returns
# pandas DataFrames ready for comparison with model output.
#
# Supported datasets:
#   - North Inlet LTREB (edi.135.12): annual productivity (Smalley method)
#     and monthly aboveground biomass
#   - PIE (knb-lter-pie.32.15): monthly aboveground biomass, S. alterniflora
#     control plots (treatment='C') only
#   - GCE (knb-lter-gce.759.27): annual biomass statistics
#
# All ANPP values are returned in g m-2 yr-1 to match model output.
# All biomass values are returned in g m-2 to match the existing LTER units.

from __future__ import annotations
from pathlib import Path
from typing import Optional

import pandas as pd

_LTER_ROOT = Path(__file__).parent.parent / "lter_data"


# ---------------------------------------------------------------------------
# North Inlet LTREB
# ---------------------------------------------------------------------------

def read_ni_annual_productivity(
    treatment: str = "C",
) -> pd.DataFrame:
    """Read NI annual productivity (g m-2 yr-1, Smalley method).

    Parameters
    ----------
    treatment:
        Filter to this treatment code.  'C' = control, 'NP' = nutrient-
        amended plots (NP is the code used in the file for nutrient plots
        at some sites; check the metadata for the exact coding used at NI).

    Returns
    -------
    DataFrame with columns: SITE, LOCATION, YEAR, TREATMENT, PRODUCTIVITY,
    STDERR.  PRODUCTIVITY is in g m-2 yr-1.
    """
    path = _LTER_ROOT / "edi.135.12" / "NILTREB_plants_annual_productivity.csv"
    df = pd.read_csv(path)
    if treatment:
        df = df[df["TREATMENT"] == treatment].copy()
    return df.reset_index(drop=True)


def read_ni_monthly_biomass(
    treatment: str = "C",
    location: Optional[str] = None,
) -> pd.DataFrame:
    """Read NI monthly aboveground biomass density (g m-2).

    Parameters
    ----------
    treatment:
        Filter to this treatment code ('C' = control).
    location:
        Optional location code to filter on (e.g. 'HM' = high marsh).

    Returns
    -------
    DataFrame with columns: SITE, LOCATION, PLOT, SUBPLOT, TREATMENT, DATE,
    PLANT_DENSITY, ABOVEGROUND_BIOMASS.  DATE is parsed as datetime.
    """
    path = (
        _LTER_ROOT
        / "edi.135.12"
        / "NILTREB_plants_aboveground_biomass_density.csv"
    )
    df = pd.read_csv(path, parse_dates=["DATE"])
    if treatment:
        df = df[df["TREATMENT"] == treatment].copy()
    if location:
        df = df[df["LOCATION"] == location].copy()
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# PIE - Parker River / Plum Island Estuary
# Note: LTE-MP-LPA plots are S. alterniflora low marsh.
#       LTE-MP-LPP plots are S. patens - exclude from S. alterniflora calibration.
# ---------------------------------------------------------------------------

def read_pie_monthly_biomass(treatment: str = "C") -> pd.DataFrame:
    """Read PIE monthly aboveground biomass (g m-2), S. alterniflora only.

    Treatment column is labelled 'TRT' in this dataset.
    Returns DataFrame with columns: SITE, YEAR, MONTH, DAY, TRT,
    MEAN BIOMASS, BIOMASS SE, MEAN PLANT DENSITY, DENSITY SE.
    """
    path = _LTER_ROOT / "knb-lter-pie.32.15" / "LTE-MP-LPA-biomass.csv"
    df = pd.read_csv(path)
    if treatment:
        df = df[df["TRT"] == treatment].copy()
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# GCE - Georgia Coastal Ecosystems
# Note: GCE data integrate over a spatial footprint (multiple zones).
# Best used for validation rather than plot-level calibration until plot
# elevations and positions are known.
# ---------------------------------------------------------------------------

def read_gce_biomass_stats() -> pd.DataFrame:
    """Read GCE annual biomass statistics.

    Returns the raw DataFrame from PLT-GCES-1609c_Biomass_Stats_5_0.CSV.
    The first two rows are a header block; real column names start on row 3.
    """
    path = (
        _LTER_ROOT
        / "knb-lter-gce.759.27"
        / "PLT-GCES-1609c_Biomass_Stats_5_0.CSV"
    )
    # Skip the two-line title/blank block at the top of the file.
    df = pd.read_csv(path, skiprows=2)
    return df


def read_gce_observations() -> pd.DataFrame:
    """Read GCE individual biomass observations."""
    path = (
        _LTER_ROOT
        / "knb-lter-gce.759.27"
        / "PLT-GCES-1609c_Observations_5_0.CSV"
    )
    df = pd.read_csv(path, skiprows=2)
    return df
