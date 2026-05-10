# calibration/

Python scaffolding for calibrating the marsh_muddpile NPP model against LTER
aboveground productivity observations.

## Files

| File | Purpose |
|------|---------|
| `site_config.py` | `PlotConfig`, `TideRecord`, `MetForcingParams` dataclasses; placeholder site records for NI, PIE, GCE |
| `site_config_filled.py` | Auto-generated replacement — real tidal and met values from NOAA and ERA5 |
| `forcing_builder.py` | Builds monthly forcing-step list from a `PlotConfig` using sinusoidal seasonal cycles |
| `yaml_writer.py` | Writes a complete run YAML from a `PlotConfig` and optional parameter overrides |
| `model_runner.py` | Thin subprocess wrapper around `marsh_cli` |
| `output_reader.py` | Reads NetCDF output; computes annual ANPP and mean diagnostics |
| `lter_reader.py` | Reads NI, PIE, and GCE LTER CSV files into DataFrames |
| `write_yaml.py` | CLI to generate run YAMLs from site_config.py (`--site`, `--elevation`) |
| `example_ni.py` | End-to-end example: one North Inlet plot, placeholder elevation |
| `calibrate_ni.py` | Optimisation wrapper — fits three vegetation parameters to NI LTER ANPP |
| `plot_timeseries.py` | Plotting functions — model time series and model-vs-obs comparisons |
| `site_data_<key>.json` | Raw tidal datums and harmonic constituents as downloaded from NOAA |

---

## Data gathering workflow

All data-gathering scripts live in `../scripts/`.  Run them in the order
described below before attempting any calibration run.

### Step 1 — Download ERA5 meteorological data

ERA5-Land temperature and solar radiation are fetched from the Open-Meteo
archive API (no account or API key required).  Run one site at a time to
avoid rate limits:

```bash
cd scripts/

python download_era5_met.py --site ni  --start 1985 --end 2024 --outdir ../era5_data
python download_era5_met.py --site pie --start 1985 --end 2024 --outdir ../era5_data
python download_era5_met.py --site gce --start 1985 --end 2024 --outdir ../era5_data
```

Each call produces a daily CSV in `era5_data/`:

```
era5_land_ni_1985_2024_daily.csv
era5_land_pie_1985_2024_daily.csv
era5_land_gce_1985_2024_daily.csv
```

Columns: `date`, `temperature_mean_c`, `temperature_max_c`,
`temperature_min_c`, `par_umol_m2_d`, `precipitation_mm_d`.

PAR is computed from ERA5 shortwave radiation using the McCree (1972)
conversion (0.45 × shortwave fraction, 4.57 µmol J⁻¹) and expressed in
µmol m⁻² d⁻¹.  Typical annual means are 20–35 million µmol m⁻² d⁻¹ for
the US East Coast sites.

Add `--process` to print fitted sinusoidal parameters immediately after
download:

```bash
python download_era5_met.py --site ni --process
```

### Step 2 — Gather tidal and met parameters for all sites

`gather_site_data.py` queries the NOAA CO-OPS API for tidal datums and
harmonic constituents, then fits sinusoidal seasonal parameters to the ERA5
CSVs produced in Step 1.  It writes:

- `calibration/site_data_<key>.json` — raw downloaded values (datums,
  harmonics, fitted met) for archiving and inspection.
- `calibration/site_config_filled.py` — a drop-in replacement for the
  factory functions in `site_config.py` with all `PLACEHOLDER` values
  replaced by real data.

```bash
# From the project root:
python scripts/gather_site_data.py --era5-dir era5_data

# Or one site at a time:
python scripts/gather_site_data.py --site ni --era5-dir era5_data
```

NOAA gauges used:

| Site | Gauge | Distance |
|------|-------|----------|
| North Inlet, SC (ni) | 8661070 — North Inlet | local |
| Plum Island Estuary, MA (pie) | 8419870 — Seavey Island, NH | ~41 km |
| Georgia Coastal Ecosystems, GA (gce) | 8670870 — Fort Pulaski, GA | ~80 km |

If ERA5 CSVs are absent, tidal parameters are still filled in and met
parameters are left as improved placeholders.

### Step 3 (optional) — Explore nearby water-level gauges

`compare_coastal_water_levels.py` lets you search for USGS water-level
stations near any location and compare them against the nearest NOAA CO-OPS
reference gauge:

```bash
# Find stations within 50 km of North Inlet and save as FlatGeobuf:
python scripts/compare_coastal_water_levels.py search \
    --lat 33.35 --lon -79.18 --radius 50 \
    --outfile scripts/ni_stations.fgb

# Compare water levels and produce tidal plots + phase/amplitude table:
python scripts/compare_coastal_water_levels.py plot \
    --lat 33.35 --lon -79.18 \
    --stations scripts/ni_stations.fgb \
    --year 2023 \
    --outdir viz/
```

This is useful for verifying that the CO-OPS gauge chosen in Step 2 is
representative of water levels at the marsh site, and for quantifying any
phase or amplitude offsets between the gauge and the marsh interior.

---

## Using `site_config_filled.py`

`site_config_filled.py` contains updated versions of the three factory
functions (`north_inlet_default_tides`, `north_inlet_default_met`, etc.)
with real data substituted for every `PLACEHOLDER`.

**Option A — use it directly** (quickest for a one-off run):

```python
import sys
sys.path.insert(0, "calibration")
from site_config_filled import (
    north_inlet_default_tides,
    north_inlet_default_met,
)
from site_config import PlotConfig
from yaml_writer import write_config

config = PlotConfig(
    site_name="north_inlet",
    plot_id="ni_plot5",
    surface_elevation_m=0.30,        # from lidar
    distance_from_creek_m=75.0,      # from creek network
    tides=north_inlet_default_tides(),
    met=north_inlet_default_met(),
    n_years=20,
)
write_config(config, "calibration_runs/ni_plot5.yaml")
```

**Option B — merge into `site_config.py`** (cleaner for ongoing work):

Copy the filled factory functions from `site_config_filled.py` back into
`site_config.py`, replacing the corresponding placeholder functions.  The
two files share the same `TideRecord` and `MetForcingParams` dataclasses.

### What still needs to be set manually

`site_config_filled.py` fills in everything that can be derived from public
data APIs.  The following values remain plot-specific and must be set before
running:

| Field | Source |
|-------|--------|
| `surface_elevation_m` | Lidar survey of the plot surface (relative to MSL) |
| `distance_from_creek_m` | Distance to nearest tidal creek from lidar-derived creek network |
| `suspended_sediment_concentration_kg_m3` | Water-column SSC measurements or literature |
| `fine_sediment_concentration_kg_m3` | As above |
| `precipitation_mean_mm_d` | ERA5 daily CSV (column `precipitation_mm_d`) — mean over record |

`surface_elevation_m` is the most important plot-specific variable: it
controls inundation frequency and is the primary control on modelled ANPP.

---

## Generating run YAMLs

`write_yaml.py` translates the filled `site_config.py` into run-ready YAML
files.  Surface elevation (relative to MSL) is the only required argument
because it is the one value that varies plot to plot.

```bash
cd calibration/

# One plot at 0.30 m elevation:
python write_yaml.py --site ni --elevation 0.30

# Sweep across elevations for a sensitivity run:
python write_yaml.py --site ni --elevation 0.10 0.20 0.30 0.40 0.50

# Override distance from creek and run length:
python write_yaml.py --site pie --elevation 0.80 --distance 120 --years 20

# Choose a custom output directory:
python write_yaml.py --site gce --elevation 0.50 --outdir ../runs/gce/
```

Each call writes one YAML per elevation value to `calibration_runs/` (or
`--outdir`).  The filename encodes the site and elevation, e.g.
`ni_elev0p300.yaml`.  A matching `.nc` output path is embedded in the YAML
so the model knows where to write results.

## Running the model

Once a YAML is written, pass it to `marsh_cli`:

```bash
marsh_cli calibration_runs/ni_elev0p300.yaml
```

Or from Python via `model_runner.py`:

```python
from model_runner import run_model
nc_path = run_model("calibration_runs/ni_elev0p300.yaml")
```

Output is written to the NetCDF path specified in the YAML
(`calibration_runs/ni_plot5.nc`).

### Reading output

```python
from output_reader import annual_anpp_g_m2_yr, summarise_run

anpp = annual_anpp_g_m2_yr("calibration_runs/ni_plot5.nc")
print(anpp)          # g m-2 yr-1, one value per model year

diag = summarise_run("calibration_runs/ni_plot5.nc", skip_spinup_years=2)
print(diag)          # post-spinup mean diagnostics
```

### Comparing against LTER observations

```python
from lter_reader import read_ni_annual_productivity

obs = read_ni_annual_productivity(treatment="C")
print(obs[["SITE", "LOCATION", "YEAR", "PRODUCTIVITY"]].to_string(index=False))
```

See `example_ni.py` for a complete end-to-end script.

---

## Calibration target

Annual ANPP is computed from model output as:

```
ANPP (g m-2 yr-1) = sum_over_year( aboveground_growth_kg_m2_d × dt_days ) × 1000
```

This matches the Smalley (1968) clip-and-weigh method used at North Inlet
and PIE.  Use post-spinup years only (discard the first 2–3 years).

---

## Notes on LTER datasets

- **North Inlet (NI)**: annual productivity (`NILTREB_plants_annual_productivity.csv`)
  and monthly biomass. Use control plots (`TREATMENT='C'`). Best primary
  calibration target once plot elevations are available from lidar.
- **PIE**: monthly biomass, *S. alterniflora* low marsh (`LTE-MP-LPA`).
  Control plots only (`TRT='C'`). Exclude `LTE-MP-LPP` (*S. patens*).
- **GCE**: biomass statistics and observations. Spatial footprint integrates
  across multiple zones — best used for validation rather than plot-level
  calibration.


# Calibration targets

GPP  = lue_gC_per_umol × APAR × temp_modifier × hydroperiod_stress × salinity_stress
NPP  = carbon_use_efficiency × GPP
aboveground_growth = NPP × shoot_fraction × (1 − bio/capacity) / (1000 × C_fraction)
ANPP = ∫ aboveground_growth dt

With PAR ≈ 34 million µmol m⁻² d⁻¹ and the default LUE of 2.5×10⁻⁴ gC/µmol, even with heavy stress scalars, GPP is
enormous and the biomass races to vegetation_aboveground_capacity_kg_m2 = 3.0 kg/m² almost immediately. Once there,
the capacity modifier (1 − bio/capacity) clamps to zero, and the system is capacity-limited, not light-limited. At
that point ANPP is set by steady-state:

ANPP ≈ mortality_rate × capacity_biomass
     ≈ (baseline_0.001 + seasonal_mean_0.001) /d × 365 d × 3000 g/m²
     ≈ 2190 g DW/m²/yr

vegetation_lue_gC_per_umol is therefore not the right lever — you can reduce it by 10× and ANPP barely moves, because
you still saturate to capacity, just slightly more slowly.

The parameter that most directly and cleanly controls ANPP in this regime is vegetation_aboveground_capacity_kg_m2,
because at steady state ANPP scales linearly with it. The default of 3.0 kg/m² is too high for S. alterniflora — peak
standing crop at North Inlet is typically 0.5–1.2 kg/m². Reducing capacity to 0.5 kg/m² would alone reduce ANPP by
~6×.

The second lever is the mortality parameters, specifically vegetation_aboveground_mortality_seasonal_amp_per_day. The
seasonal term currently contributes as much to annual mortality as the baseline, and both together drive ANPP through
the mortality × capacity product. These two parameters interact multiplicatively, so a combined reduction in both is
needed for a 15× correction.

To get ANPP ≈ 150 g/m²/yr from the current ~2190 g/m²/yr you need the mortality × capacity product to drop by ~15×,
e.g.:

  ┌───────────────────────────────────────────────────────┬─────────┬────────┬─────────────────┐
  │                       Parameter                       │ Current │ Target │     Effect      │
  ├───────────────────────────────────────────────────────┼─────────┼────────┼─────────────────┤
  │ vegetation_aboveground_capacity_kg_m2                 │ 3.0     │ 0.8    │ 3.75× reduction │
  ├───────────────────────────────────────────────────────┼─────────┼────────┼─────────────────┤
  │ vegetation_aboveground_mortality_base_per_day         │ 0.001   │ 0.0003 │ ~3× reduction   │
  ├───────────────────────────────────────────────────────┼─────────┼────────┼─────────────────┤
  │ vegetation_aboveground_mortality_seasonal_amp_per_day │ 0.002   │ 0.0003 │ ~6× reduction   │
  └───────────────────────────────────────────────────────┴─────────┴────────┴─────────────────┘

The right calibration sequence is: fix capacity first (it has a direct field measurement analogue — peak standing
crop), then adjust the mortality amplitude to tune the annual cycle shape, and only touch LUE once the system is no
longer capacity-saturated throughout the whole growing season.

---

## Automated calibration — `calibrate_ni.py`

`calibrate_ni.py` wraps the model in a numerical optimiser and finds the
combination of the three key parameters that best reproduces the observed
mean annual ANPP for each North Inlet site.

### Required data files

`calibrate_ni.py` reads two LTER CSV files via `lter_reader.py`.  Both must be
present before the script is run:

| File | Location | Contents |
|------|----------|----------|
| `NILTREB_plants_annual_productivity.csv` | `lter_data/edi.135.12/` | Annual ANPP (g m⁻² yr⁻¹) by site/location/treatment, 1984–2025 — primary calibration target |
| `NILTREB_plants_aboveground_biomass_density.csv` | `lter_data/edi.135.12/` | Monthly aboveground biomass (g m⁻²) by plot — used for seasonal biomass term |

Both files are part of the **edi.135.12** dataset (North Inlet LTREB aboveground
plant data).  Download from the EDI data portal and unpack into
`lter_data/edi.135.12/` before running calibration.

The script also requires the compiled `marsh_cli` binary to be on `$PATH`, or
its full path supplied via `--cli`.

### NI sites

North Inlet has five calibration targets, distinguished by site, marsh zone,
and treatment.  Elevations are NAVD88 values measured at the plot locations;
distances are to the nearest tidal creek.

| Site key   | SITE | LOCATION | TREATMENT | Elevation (m NAVD88) | Distance from creek (m) |
|------------|------|----------|-----------|:--------------------:|:-----------------------:|
| `gi_hm_c`  | GI   | HM       | C         | 0.50                 | 62                      |
| `gi_hm_np` | GI   | HM       | NP        | 0.50                 | 62                      |
| `gi_lm_c`  | GI   | LM       | C         | 0.20                 | 42                      |
| `ol_hm_c`  | OL   | HM       | C         | 0.45                 | 33                      |
| `ol_lm_c`  | OL   | LM       | C         | 0.25                 | 33                      |

GI = Goat Island, OL = Oyster Landing.  HM = high marsh, LM = low marsh.
C = control, NP = nitrogen + phosphorus fertilisation.

**Note on the NP site**: the model has no nutrient-limitation module, so
calibrated parameters for `gi_hm_np` reflect a high-productivity vegetation
regime rather than a mechanistic nutrient response.  Treat those parameter
values separately from the control sites.

### Sea level rise

A linear sea level rise trend of **6 mm yr⁻¹** is applied to both
`mean_sea_level` and `mean_high_tide` at every forcing step, derived from
the long-term record at NOAA gauge 8661070 (North Inlet, SC).  Over the
default 30-year run this raises the tidal frame by 0.18 m.

Sea level rise is configured via `PlotConfig.sea_level_rise_m_yr` (set to
`0.006` in `_make_config`) and applied in `forcing_builder.build_monthly_forcing`.
To run without SLR (e.g. for a sensitivity test) pass `sea_level_rise_m_yr=0.0`
when constructing the `PlotConfig`.

### Initial sediment column

Rather than a single sand layer, the model is initialised with a **1 m organic
sediment column** (20 layers × 5 cm) built by `yaml_writer.build_initial_column_layers`.
The profile is:

| Material | Profile |
|----------|---------|
| Refractory organic | 10 % of solid volume throughout the full column |
| Labile organic | Exponential decay: 10 % at surface, e-folding depth 10 cm (≈ 0.5 % at 30 cm) |
| Silt | Remainder of solid volume |
| Roots | Total = `mean_aboveground_biomass × root_to_shoot_ratio` (default 3×), distributed with the same 10 cm e-folding depth |

The mean aboveground biomass used to set the initial root loading is derived
from the observed monthly biomass record for that site (`obs_biomass_mean`
in `SiteCalibrator.__init__`), converted from g m⁻² to kg m⁻².  This ensures
the initial root mass is consistent with the long-term observed vegetation state
at each plot rather than a generic placeholder.

### Optimised parameters and search bounds

| Parameter | Lower | Upper | Default |
|-----------|:-----:|:-----:|:-------:|
| `vegetation_aboveground_capacity_kg_m2` | 0.10 | 3.0 | 3.0 |
| `vegetation_aboveground_mortality_base_per_day` | 1×10⁻⁴ | 5×10⁻³ | 1×10⁻³ |
| `vegetation_aboveground_mortality_seasonal_amp_per_day` | 1×10⁻⁴ | 1×10⁻² | 2×10⁻³ |

All other parameters are held at their `yaml_writer.py` defaults.  The
optimiser works in log-transformed parameter space because the parameters span
roughly two orders of magnitude; this keeps the search well-conditioned.

### Objective function

The loss combines two dimensionless terms:

```
loss = L_ANPP + w × L_biomass

L_ANPP    = ((mod_mean_ANPP − obs_mean_ANPP) / obs_mean_ANPP)²

L_biomass = mean over 12 months of
            ((mod_monthly_biomass_m − obs_monthly_biomass_m) / obs_biomass_mean)²
```

- `mod_mean_ANPP` — post-spinup mean annual ANPP from the model (years 3–30).
- `obs_mean_ANPP` — mean of all available annual productivity values for that
  site/location/treatment combination.
- `mod_monthly_biomass_m` — mean aboveground biomass for calendar month *m*,
  averaged over post-spinup model years.
- `obs_monthly_biomass_m` — mean aboveground biomass for month *m*, averaged
  over all plots, subplots, and observation years in the LTER record.  Pooling
  across subplots is consistent with the site-mean ANPP target.
- `obs_biomass_mean` — the grand mean of the 12 monthly observed means; used
  as the normalisation denominator so both terms are scale-free.
- `w` — biomass weight (default 1.0, set via `--biomass-weight`).

Both `L_ANPP` and `L_biomass` are of order 1 when the model is a factor of ~2
off, so equal weighting is a reasonable starting point.  Setting
`--biomass-weight 0` reverts to ANPP-only calibration.

The biomass term constrains the *shape* of the seasonal cycle and the mean
standing crop independently of the annual total, helping to pin
`vegetation_aboveground_capacity_kg_m2` (which sets the peak) and
`vegetation_aboveground_mortality_seasonal_amp_per_day` (which sets the timing
and depth of the winter trough).

### Optimisation methods

Two methods are available via `--method`:

- **`nelder-mead`** (default) — Nelder-Mead simplex with multiple random
  restarts.  Each restart runs ≤600 model evaluations; the default of 3
  restarts gives a good balance between runtime and robustness.  Expect
  ~150–250 model runs per site (~5–15 minutes depending on hardware).

- **`de`** — Scipy differential evolution (global).  More likely to find the
  true optimum at the cost of ~5× more model evaluations.  Useful if
  Nelder-Mead solutions look inconsistent across sites.

### Outputs

For each site the calibrator writes two files to `calibration_runs/`:

| File | Contents |
|------|----------|
| `ni_{site_key}_best.yaml` | Run YAML with best-fit parameters |
| `ni_{site_key}_best.nc` | NetCDF output for the best-fit run |
| `ni_{site_key}_best_params.json` | Best parameters + diagnostics (target ANPP, modelled ANPP, ratio, loss) |

### Usage

```bash
cd calibration/

# Calibrate all five sites with default settings:
python3 calibrate_ni.py

# One site only:
python3 calibrate_ni.py --site gi_hm_c

# Differential evolution (more thorough):
python3 calibrate_ni.py --site gi_hm_c --method de

# More Nelder-Mead restarts, suppress per-evaluation output:
python3 calibrate_ni.py --n-starts 5 --quiet

# ANPP-only calibration (no biomass term):
python3 calibrate_ni.py --biomass-weight 0

# Upweight the biomass seasonal cycle (e.g. 2× vs ANPP):
python3 calibrate_ni.py --biomass-weight 2.0

# Custom model binary:
python3 calibrate_ni.py --cli /path/to/marsh_cli
```

After all sites are done a summary table is printed:

```
Site         ANPP_obs  ANPP_mod  ratio  bio_obs  bio_mod  bio_rmse    capacity  mort_base   mort_amp
gi_hm_c         975.0     978.3  1.003    310.0    312.4      48.3      0.3821    0.00041    0.00073
gi_lm_c        1343.0    1351.2  1.006    479.8    481.1      61.7      0.5104    0.00063    0.00091
...
```

`bio_rmse` is the root-mean-squared error of the monthly biomass seasonal cycle
in g m⁻² and is a useful indicator of how well the model captures the shape and
amplitude of the annual growth cycle, independently of the ANPP total.

---

## Visualising results — `plot_timeseries.py`

`plot_timeseries.py` provides four plotting functions and a CLI that compares
model output from a NetCDF file against NI LTER observations.

### Functions

| Function | Description |
|----------|-------------|
| `plot_variable_timeseries` | Any model variable(s) vs model time, with spin-up shading |
| `plot_anpp_comparison` | Modelled annual ANPP time series overlaid on the observed mean ± 1σ band |
| `plot_seasonal_biomass` | Mean seasonal cycle of aboveground biomass: model line vs observed monthly mean ± SE |
| `plot_site_dashboard` | 2 × 2 panel combining all three comparisons plus biomass compartments |

All functions accept an optional `ax` argument so they can be embedded in
larger figures.

### Model-vs-observation alignment

The model does not run at specific calendar years, so annual ANPP is compared
statistically: the modelled time series is plotted against the observed
multi-decadal mean ± 1 standard deviation as a horizontal band.

For the seasonal cycle the model is averaged over post-spinup years by
calendar month (the forcing repeats the same seasonal cycle every year).
Observations are averaged over all available years by calendar month with
standard-error bars.

Model biomass (`aboveground_biomass_kg_m2`) is converted from kg m⁻² to
g m⁻² (× 1000) to match the LTER units before plotting.

### Usage

```bash
cd calibration/

# Full dashboard for a calibrated run:
python3 plot_timeseries.py \
    --nc calibration_runs/ni_gi_hm_c_best.nc \
    --site GI --location HM --treatment C

# Save to a file instead of showing interactively:
python3 plot_timeseries.py \
    --nc calibration_runs/ni_gi_hm_c_best.nc \
    --site GI --location HM --treatment C \
    --out viz/figures/ni_gi_hm_c_dashboard.png

# Plot one or more specific variables:
python3 plot_timeseries.py \
    --nc calibration_runs/ni_gi_hm_c_best.nc \
    --variable aboveground_biomass_kg_m2 surface_elevation

# List available variable names in a NetCDF file:
python3 plot_timeseries.py \
    --nc calibration_runs/ni_gi_hm_c_best.nc --list-vars
```

From Python:

```python
from plot_timeseries import plot_site_dashboard, plot_seasonal_biomass

# Dashboard saved to file
plot_site_dashboard(
    "calibration_runs/ni_gi_hm_c_best.nc",
    site="GI", location="HM", treatment="C",
    outpath="viz/figures/ni_gi_hm_c.png",
)

# Seasonal cycle only — embedded in an existing figure
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_seasonal_biomass(
    "calibration_runs/ni_gi_hm_c_best.nc",
    site="GI", location="HM", treatment="C",
    ax=axes[0],
)
plot_seasonal_biomass(
    "calibration_runs/ni_ol_lm_c_best.nc",
    site="OL", location="LM", treatment="C",
    ax=axes[1],
)
plt.tight_layout()
plt.savefig("viz/figures/ni_seasonal_comparison.png", dpi=150)
```