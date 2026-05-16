# site_config.py
#
# Dataclasses that describe a single 1-D model column for calibration.
# All tidal and meteorological values marked PLACEHOLDER must be replaced
# with site-specific data before any calibration results are meaningful.
#
# Typical workflow:
#   1. Measure (or derive from lidar) the surface elevation of each LTER plot.
#   2. Download harmonic tidal constituents from the nearest NOAA gauge.
#   3. Fill in seasonal temperature and PAR from a met station or reanalysis.
#   4. Pass a PlotConfig to forcing_builder.build_monthly_forcing() and
#      yaml_writer.write_config() to produce a run-ready YAML file.

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class TideRecord:
    """Tidal parameters for one site.

    All values are PLACEHOLDERS derived from order-of-magnitude estimates.
    Replace with data from the nearest NOAA CO-OPS gauge before use.
    See https://tidesandcurrents.noaa.gov/stations.html for gauge data.
    """

    mean_sea_level_m: float = 0.0          # PLACEHOLDER: MSL relative to model datum (m)
    mean_high_tide_m: float = 0.8          # PLACEHOLDER: mean high water (m)
    tidal_amplitude_m: float = 0.8         # PLACEHOLDER: half tidal range (m)
    tidal_period_hours: float = 12.42      # semidiurnal period

    # Harmonic constituents (amplitude in metres, phase in degrees).
    # PLACEHOLDER values are rough North American Atlantic coast typical values.
    M2_amplitude_m: float = 0.60           # PLACEHOLDER
    M2_phase_deg: float = 0.0              # PLACEHOLDER
    S2_amplitude_m: float = 0.12           # PLACEHOLDER
    S2_phase_deg: float = 30.0             # PLACEHOLDER
    N2_amplitude_m: float = 0.12           # PLACEHOLDER
    N2_phase_deg: float = 340.0            # PLACEHOLDER
    K1_amplitude_m: float = 0.05           # PLACEHOLDER
    K1_phase_deg: float = 200.0            # PLACEHOLDER
    O1_amplitude_m: float = 0.04           # PLACEHOLDER
    O1_phase_deg: float = 180.0            # PLACEHOLDER

    # Water column properties
    creek_salinity_ppt: float = 28.0       # PLACEHOLDER
    suspended_sediment_concentration_kg_m3: float = 0.020   # PLACEHOLDER
    fine_sediment_concentration_kg_m3: float = 0.005        # PLACEHOLDER


@dataclass
class MetForcingParams:
    """Seasonal meteorological parameters used to build sinusoidal forcing.

    All values are PLACEHOLDERS.  Replace with data from a met station or
    reanalysis product (e.g. PRISM, ERA5) co-located with the LTER site.
    """

    # Air temperature: T(doy) = mean + amp * cos(2*pi*(doy - peak_doy) / 365)
    temperature_mean_c: float = 15.0       # PLACEHOLDER: annual mean (degC)
    temperature_amplitude_c: float = 12.0  # PLACEHOLDER: half-range (degC)
    temperature_peak_day: float = 198.0    # PLACEHOLDER: day of year of warmest day

    # PAR: PAR(doy) = mean + amp * cos(2*pi*(doy - peak_doy) / 365)
    par_mean_umol_m2_d: float = 25_000_000.0    # PLACEHOLDER: annual mean (umol m-2 d-1)
    par_amplitude_umol_m2_d: float = 13_000_000.0  # PLACEHOLDER: half-range
    par_peak_day: float = 172.0                 # PLACEHOLDER: summer solstice ~ day 172

    # Long-term trends (applied as linear drift on top of the seasonal cycle).
    # Useful for climate-change sensitivity experiments.
    temperature_trend_c_per_yr: float = 0.0
    par_trend_umol_m2_d_per_yr: float = 0.0

    # Precipitation
    precipitation_mean_mm_d: float = 3.0   # PLACEHOLDER
    freshwater_input_mm_d: float = 0.0     # PLACEHOLDER


@dataclass
class PlotConfig:
    """All information needed to drive one model column for one LTER plot.

    surface_elevation_m is the only value that is truly plot-specific.
    Everything else may be shared across plots at the same site.
    """

    site_name: str                           # e.g. "north_inlet", "pie", "gce"
    plot_id: str                             # e.g. "NI_HM_plot5"

    # Elevation of the sediment surface relative to MSL (m).
    # Derive from site lidar data; this is the key plot-specific variable.
    surface_elevation_m: float

    # Distance from the nearest tidal creek (m).
    # Controls salinity flushing and sediment supply in the model.
    distance_from_creek_m: float = 50.0    # PLACEHOLDER

    tides: TideRecord = field(default_factory=TideRecord)
    met: MetForcingParams = field(default_factory=MetForcingParams)

    # Run duration
    n_years: int = 10

    # Sea level rise rate (m/yr).  Applied as a linear trend to mean_sea_level
    # and mean_high_tide in the forcing steps.  Set to site-specific value;
    # e.g. 0.006 for North Inlet, SC (NOAA gauge 8661070).
    sea_level_rise_m_yr: float = 0.0

    # Initial soil column — built as a multi-layer organic profile
    # (see yaml_writer.build_initial_column_layers).
    # n_layers × layer_thickness_m = total column depth (default 1 m).
    initial_column_n_layers: int = 20
    initial_column_layer_thickness_m: float = 0.05
    initial_column_porosity: float = 0.60

    # Mean annual aboveground biomass (kg m-2) for this plot, derived from
    # LTER observations.  Used to:
    #   (a) set initial_aboveground_biomass_kg_m2 in the ecohydrology state, and
    #   (b) distribute root mass (root_to_shoot_ratio × this value) through
    #       the initial sediment column.
    mean_aboveground_biomass_kg_m2: float = 0.30

    # Root : shoot ratio for initial column root loading.
    initial_root_to_shoot_ratio: float = 3.0

    # Plant species — controls which parameter preset is applied.
    # One of: "spartina_alterniflora", "spartina_patens", "juncus_roemerianus".
    # See calibration/species_presets.py for the parameter values associated
    # with each species.
    species: str = "spartina_alterniflora"

    # Initial ecophysiology state (derived from mean_aboveground_biomass_kg_m2
    # if not explicitly overridden; set in yaml_writer.write_config).
    initial_salinity_ppt: float = 20.0

    # Output
    output_dir: str = "calibration_runs"


# ---------------------------------------------------------------------------
# Placeholder site records - update these as real data are gathered
# ---------------------------------------------------------------------------

def north_inlet_default_tides() -> TideRecord:
    """North Inlet, SC
    Harmonic constituents from NOAA 8661070 (Springmaid Pier / North Inlet, SC),
    then damped by factor 0.85 to account for the amplitude attenuation observed
    between the NOAA open-water gauge and the local creek gauge inside the marsh
    (site-specific analysis of in-situ vs NOAA water level records).
    Phases are unchanged — damping is amplitude-only.
    """
    _damp = 0.85
    t = TideRecord()
    t.mean_sea_level_m            = 0.0
    t.mean_high_tide_m            = round(0.762 * _damp, 4)   # 0.648 m
    t.tidal_amplitude_m           = round(0.766 * _damp, 4)   # 0.651 m
    t.tidal_period_hours          = 12.42
    t.M2_amplitude_m              = round(0.730 * _damp, 4)   # 0.621 m
    t.M2_phase_deg                = 357.9
    t.S2_amplitude_m              = round(0.123 * _damp, 4)   # 0.105 m
    t.S2_phase_deg                = 20.3
    t.N2_amplitude_m              = round(0.171 * _damp, 4)   # 0.145 m
    t.N2_phase_deg                = 340.6
    t.K1_amplitude_m              = round(0.102 * _damp, 4)   # 0.087 m
    t.K1_phase_deg                = 187.6
    t.O1_amplitude_m              = round(0.076 * _damp, 4)   # 0.065 m
    t.O1_phase_deg                = 191.4
    t.creek_salinity_ppt          = 28.0
    t.suspended_sediment_concentration_kg_m3 = 0.020  # PLACEHOLDER
    t.fine_sediment_concentration_kg_m3      = 0.005  # PLACEHOLDER
    return t


def north_inlet_default_met() -> MetForcingParams:
    """North Inlet, SC met 
    Generated from ERA5
    """
    m = MetForcingParams()
    m.temperature_mean_c          = 18.92
    m.temperature_amplitude_c     = 8.46
    m.temperature_peak_day        = 202.8
    m.par_mean_umol_m2_d          = 34172288.0
    m.par_amplitude_umol_m2_d     = 14160082.0
    m.par_peak_day                = 164.8
    m.precipitation_mean_mm_d     = 3.0   # PLACEHOLDER — add ERA5 tp variable
    m.freshwater_input_mm_d       = 0.0
    return m


def pie_default_tides() -> TideRecord:
    """Parker River / PIE, MA
    Nearest gauge (with data): 8419870 (Seavey Island, NH, 41 km away)
    """
    t = TideRecord()
    t.mean_sea_level_m            = 0.0
    t.mean_high_tide_m            = 1.224
    t.tidal_amplitude_m           = 1.243
    t.tidal_period_hours          = 12.42
    t.M2_amplitude_m              = 1.214
    t.M2_phase_deg                = 110.9
    t.S2_amplitude_m              = 0.177
    t.S2_phase_deg                = 147.2
    t.N2_amplitude_m              = 0.278
    t.N2_phase_deg                = 81.3
    t.K1_amplitude_m              = 0.132
    t.K1_phase_deg                = 207.0
    t.O1_amplitude_m              = 0.106
    t.O1_phase_deg                = 189.1
    t.creek_salinity_ppt          = 22.0
    t.suspended_sediment_concentration_kg_m3 = 0.020  # PLACEHOLDER
    t.fine_sediment_concentration_kg_m3      = 0.005  # PLACEHOLDER
    return t


def pie_default_met() -> MetForcingParams:
    """PIE, MA met 
        Generated from ERA5
    """
    m = MetForcingParams()
    m.temperature_mean_c          = 9.91
    m.temperature_amplitude_c     = 12.0
    m.temperature_peak_day        = 208.9
    m.par_mean_umol_m2_d          = 29073008.0
    m.par_amplitude_umol_m2_d     = 16267152.0
    m.par_peak_day                = 169.5
    m.precipitation_mean_mm_d     = 3.0   # PLACEHOLDER — add ERA5 tp variable
    m.freshwater_input_mm_d       = 0.0
    return m


def gce_default_tides() -> TideRecord:
    """GCE, Sapelo Island GA
    Nearest gauge: NOAA 8670870 (Fort Pulaski, GA) or Sapelo Sound.
    """
    t = TideRecord()
    t.mean_sea_level_m            = 0.0
    t.mean_high_tide_m            = 1.009
    t.tidal_amplitude_m           = 1.054
    t.tidal_period_hours          = 12.42
    t.M2_amplitude_m              = 1.01
    t.M2_phase_deg                = 17.5
    t.S2_amplitude_m              = 0.16
    t.S2_phase_deg                = 45.3
    t.N2_amplitude_m              = 0.227
    t.N2_phase_deg                = 2.1
    t.K1_amplitude_m              = 0.109
    t.K1_phase_deg                = 200.0
    t.O1_amplitude_m              = 0.08
    t.O1_phase_deg                = 205.2
    t.creek_salinity_ppt          = 25.0
    t.suspended_sediment_concentration_kg_m3 = 0.020  # PLACEHOLDER
    t.fine_sediment_concentration_kg_m3      = 0.005  # PLACEHOLDER
    return t


def gce_default_met() -> MetForcingParams:
    """GCE, GA met placeholder.
        Generated from ERA5
    """
    m = MetForcingParams()
    m.temperature_mean_c          = 20.28
    m.temperature_amplitude_c     = 7.96
    m.temperature_peak_day        = 204.3
    m.par_mean_umol_m2_d          = 35685717.0
    m.par_amplitude_umol_m2_d     = 14122451.0
    m.par_peak_day                = 164.9
    m.precipitation_mean_mm_d     = 3.0   # PLACEHOLDER — add ERA5 tp variable
    m.freshwater_input_mm_d       = 0.0
    return m
