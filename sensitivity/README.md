# sensitivity

Scripts for numerical and parameter sensitivity analysis of the marsh_muddpile
model.  Each script is self-contained and can be run from the project root.

All scripts use **North Inlet ERA5-derived meteorological forcing** as the
reference configuration unless otherwise stated.  The deposition model is
`edge_distance_deposition` and the compaction model is `mixing_compaction`
(depth-dependent mixing law calibrated to CCN synthesis data) throughout.

---

## Scripts

### `timestep_sensitivity.py`

Tests how the **outer forcing timestep** affects a 50-year simulation.

Five timesteps are tested (1 d / 7 d / monthly / quarterly / annual).
`water_level_substeps_per_step` is scaled proportionally so every run
maintains 20 substeps per tidal cycle.

**Outputs** (`figures/`):
`timestep_sensitivity_timeseries.png`, `timestep_sensitivity_last_year.png`

```bash
python sensitivity/timestep_sensitivity.py --binary ./build/marsh_cli
```

---

### `inner_cycle_sensitivity.py`

Tests how the **inner tidal substep count** affects results at fixed outer
timestep (1 day).  Seven substep densities are tested: 2 / 4 / 8 / 16 / 32 /
64 / 128 samples per tidal cycle.

**Outputs** (`figures/`):
`inner_cycle_timeseries.png`, `inner_cycle_last_year.png`,
`inner_cycle_inundation.png`

```bash
python sensitivity/inner_cycle_sensitivity.py --binary ./build/marsh_cli
```

---

### `gpp_parameter_sensitivity.py`

Tests the four vegetation parameters optimised during North Inlet GPP
calibration (aboveground capacity, base mortality, cold-mortality slope, LUE),
each swept across three values in log space around the calibrated default.
Total: 12 runs × 10 years.

**Outputs** (`figures/`):
`gpp_sensitivity_capacity.png`, `gpp_sensitivity_mort_base.png`,
`gpp_sensitivity_cold_slope.png`, `gpp_sensitivity_lue.png`

```bash
python sensitivity/gpp_parameter_sensitivity.py --binary ./build/marsh_cli
```

---

### `forcing_sensitivity.py`

Tests mean annual air temperature (8 / 18 / 28 °C) and creek salinity (5 / 20
/ 35 ppt) effects over 10-year simulations.  Total: 6 runs.

**Outputs** (`figures/`):
`forcing_sensitivity_temperature.png`, `forcing_sensitivity_salinity.png`

```bash
python sensitivity/forcing_sensitivity.py --binary ./build/marsh_cli
```

---

### `temperature_forcing_sensitivity.py`

Compares three temperature-forcing approaches (sinusoidal, ERA5 daily,
ERA5 monthly) on spring biomass onset.  All runs use the 2015–2024 ERA5
period (10 years) with identical NI parameters.

**Outputs** (`figures/`):
`temperature_forcing_aboveground.png`, `temperature_forcing_timeseries.png`

```bash
python sensitivity/temperature_forcing_sensitivity.py --binary ./build/marsh_cli
```

---

### `longterm_sensitivity.py`

Full-factorial **200-year** sensitivity runs across SSC (3 values), distance
from creek (2), starting elevation (2), SLR rate (2), and temperature mean
(2).  Total: 48 runs.

**Outputs** (`figures/`):
`longterm_total_carbon.png`, `longterm_labile_carbon.png`,
`longterm_labile_rate.png`, `longterm_elevation.png`

```bash
python sensitivity/longterm_sensitivity.py --binary ./build/marsh_cli
```

---

### `salinity_sensitivity.py`

10-year factorial simulations exploring how creek salinity (5 / 28 ppt),
marsh elevation (0.2 / 0.5 m), distance from creek (2 / 20 m), and mean
temperature (18 / 22 °C) interact to drive root-zone salinity and biomass.
Total: 16 runs.

**Outputs** (`figures/`):
`salinity_sensitivity_rootzone.png`, `salinity_sensitivity_biomass.png`

```bash
python sensitivity/salinity_sensitivity.py --binary ./build/marsh_cli
```

---

### `salinity_elevation_sweep.py`

1-year sweep of root-zone salinity vs surface elevation at North Inlet,
comparing near-creek (2 m) and far-from-creek (30 m) positions across the
full intertidal range (0.05–0.85 m MSL).

**Outputs** (`figures/`):
`salinity_elevation_sweep.png`

```bash
python sensitivity/salinity_elevation_sweep.py --binary ./build/marsh_cli
```

---

### `ni_biomass_rmse_scan.py`

Grid scan of the three biomass-curve parameters **(K, σ_H, σ_R)** to
minimise RMSE against the 14 Morris et al. (2013) North Inlet *S. alterniflora*
biomass–elevation data points.  Identifies the calibrated NI defaults used
across all other North Inlet runs.

These calibrated values (K = 0.95, σ_H = 0.766, σ_R = 0.36) are the North
Inlet defaults in `yaml_writer.py` and must not be changed without re-running
this scan.

**Outputs** (`figures/`):
`ni_biomass_rmse_heatmap.png`, `ni_biomass_best_fit.png`

```bash
python sensitivity/ni_biomass_rmse_scan.py --binary ./build/marsh_cli
```

---

### `test_biomass_curve_ni.py`

Plots peak aboveground biomass vs surface elevation for *S. alterniflora* at
North Inlet using 1-year simulations across the intertidal elevation range.
Validates that the calibrated biomass curve reproduces the observed
biomass–elevation relationship before longer runs are attempted.

**Outputs** (`figures/`):
`ni_biomass_curve.png`

```bash
python sensitivity/test_biomass_curve_ni.py --binary ./build/marsh_cli
```

---

### `ni_parameter_test.py`

Spot-checks the calibrated NI parameters (biomass, accretion, root:shoot
ratio, porewater salinity) in a single 50-year reference run at Goat Island
High Marsh conditions (elevation 0.75 m, SSC 28 mg/L, distance 20 m).

**Outputs** (`figures/`):
`ni_parameter_test_timeseries.png`, `ni_parameter_test_profiles.png`

```bash
python sensitivity/ni_parameter_test.py --binary ./build/marsh_cli
```

---

### `ni_carbon_profile_sensitivity.py`

Sensitivity scan to tune the model to reproduce OM depth profiles at North
Inlet, South Carolina (Stevens 2024 CB_H core).

**Scenario A — OL High Marsh (Goat Island)**
Target: ~5 % OM near surface, ~2 % OM deep.
Grid: 2 elevations × 6 refractory fractions × 5 mortality rates × 3 root
e-folding depths = **180 runs**, 50 years each.
Calibrated result (mixing compaction): elev = 0.75 m, refrac = 0.50,
mort = 8×10⁻⁴/d, efolding = 0.05 m.

**Scenario B — Morris tube**
Target: ~20 % OM surface, ~8 % at 20 cm.  Fresh mineral substrate, SSC ≈ 0,
5-year run.  Grid: 6 refrac × 5 mort = 30 runs.
(Target unachievable in 5 years — result documents why.)

Output directories: `runs/ni_carbon_ol_hm_v5/`, `runs/ni_carbon_morris_tube_v4/`

**Outputs** (`figures/`):
`ni_olhm_elevation_summary.png`, `ni_olhm_heatmap_e*.png`,
`ni_olhm_diagnostic_overall_best.png`, `ni_olhm_profiles_best_elev.png`,
`ni_morris_tube_heatmap_v3.png`, `ni_morris_tube_profiles_v3.png`,
`ni_morris_tube_diagnostic_best_v3.png`

```bash
python sensitivity/ni_carbon_profile_sensitivity.py --binary ./build/marsh_cli
python sensitivity/ni_carbon_profile_sensitivity.py --plot-only
```

---

### `carbon_pool_test.py`

Single 20-year diagnostic run to verify carbon-pool mass balance (refractory,
labile, live roots) under North Inlet forcing with an elevated root:shoot ratio
(~3–4).  Used to confirm the root allocation and decay models are self-consistent.

**Outputs** (`figures/`):
`carbon_pool_test.png` (column totals vs time),
`carbon_pool_test_depth.png` (depth profiles at end)

```bash
python sensitivity/carbon_pool_test.py --binary ./build/marsh_cli
```

---

### `brain_compaction_test.py`

Compares the **Brain (2012) two-stage compaction model** against the new
**depth-dependent mixing compaction model** across 108 prescribed LOI columns
(100 layers × 1 cm, no vegetation/deposition/decay) and benchmarks both
against CCN synthesis DBD data for USA and UK cores.

Grid: 36 mean LOI values (0.00–0.70, step 0.02) × 3 profile types
(uniform / top-heavy / bottom-heavy) = 108 runs per model, 216 total.

**Key finding**: Brain model systematically over-compacts (DBD ~2–3× higher
than CCN observations); mixing model sits within the CCN data cloud.

**Outputs** (`figures/`):
`brain_compaction_dbd_loi.png` — mixing model vs CCN (USA / UK panels)
`brain_vs_mixing_compaction_dbd_loi.png` — three-panel comparison:
Brain vs CCN | Mixing vs CCN | direct Brain vs Mixing overlay

```bash
python sensitivity/brain_compaction_test.py --binary ./build/marsh_cli
python sensitivity/brain_compaction_test.py --plot-only   # uses cached runs
python sensitivity/brain_compaction_test.py --force       # re-run everything
```

---

### `plot_ni_porewater.py`

Plots monthly depth profiles and time series of porewater chemistry (NH₄,
SO₄, pH, H₂S, CH₄) from the North Inlet LTREB dataset (edi.136.11) for
control plots.  Does not run the model.

**Outputs** (`figures/`):
`depth_profiles_<site>_<loc>_<year>.png`,
`timeseries_<site>_<loc>.png`

```bash
python sensitivity/plot_ni_porewater.py
```

---

### `porewater_chemistry_north_inlet.py`

Investigates drivers of porewater chemistry at North Inlet LTREB sites by
combining porewater data (edi.136.11), aboveground biomass (edi.135.12), and
NOAA tidal harmonics.  Explores correlations between NH₄, SO₄ and biomass,
elevation, and inundation.  Does not run the model.

**Outputs** (`figures/`): correlation matrices, scatter plots per species.

```bash
python sensitivity/porewater_chemistry_north_inlet.py
```

---

### `ni_long_profile.py`

**300-year** marsh column simulations at North Inlet starting from MHW on a
sand substrate.  Four sea-level rise scenarios are compared to examine how SLR
rate controls long-term accretion, organic carbon accumulation, and depth
profiles of LOI, root carbon, labile organic C, and refractory organic C.

**Site and forcing**

| Parameter | Value |
|-----------|-------|
| Site | North Inlet, SC (Goat Island / high marsh) |
| Start elevation | 0.625 m NAVD88 (MHW) |
| SSC | 10 mg/L |
| Distance from creek | 30 m |
| SLR scenarios | 2, 3, 4, 5 mm/yr |
| Duration | 300 yr |
| Snapshot interval | every 36 steps (~3 yr) |

**Key model settings**

| Parameter | Value | Notes |
|-----------|-------|-------|
| `fine_silt` settling velocity | 1×10⁻⁵ m/s | Back-calculated from Morris (2013) Goat Island mineral deposition rate (0.0635 g cm⁻² yr⁻¹, SSC = 28 mg/L, 30 m from creek) |
| `vegetation_hydroperiod_sigma_fraction` | 0.25 | |
| `vegetation_inundation_sigma_fraction` | 0.22 | |
| `vegetation_aboveground_capacity_kg_m2` | 0.95 | Calibrated against Morris (2013) biomass–elevation data |
| `vegetation_lue_gC_per_umol` | 1.6×10⁻⁶ | |
| `layer_merging_enable` | 0 | **Disabled.** Layer merging caused periodic ~1–1.4 mm amplitude sawtooth artifacts in accretion rate: when ~1500 accumulated monthly layers simultaneously crossed the first depth threshold (0.5 m), a batch of ~750 merge pairs fired over a few timesteps, producing compaction-driven surface elevation drops. The artifact timing was SLR-dependent (each scenario's surface reached the 0.5 m threshold at a different year), confirming layer merging as the cause. |
| Initial column | 100 layers × 1 cm, pure sand | 100 layers chosen to match the 1 cm depth-profile bin resolution; coarser layers (e.g. 20 × 5 cm) produce step artifacts in depth profiles at the 5 cm layer boundaries. |

**Plots**

*`ni_long_profile.png`* — four-panel depth profiles (0–200 cm) at end of run:
LOI (%), live root wt%, labile organic wt%, refractory organic wt%.  Dashed
horizontal lines mark the depth of the original sand surface for each SLR
scenario.

*`ni_long_profile_rates.png`* — 2×4 time series from year 5 to year 300:
surface accretion rate (mm yr⁻¹), mineral accumulation rate (kg m⁻² yr⁻¹),
organic accumulation rate (kg m⁻² yr⁻¹), mean inundation depth (m),
inundation fraction, aboveground biomass (kg m⁻²), belowground biomass
(kg m⁻²), surface elevation (m NAVD88).  First 5 years omitted to hide
biomass establishment transient.  No temporal smoothing applied.

**Outputs** (`runs/ni_long_profile/`, `figures/`):
`ni_lp_slr{2,3,4,5}.{yaml,nc}`,
`ni_long_profile.png`, `ni_long_profile_rates.png`

```bash
python sensitivity/ni_long_profile.py --binary ./build/marsh_cli
python sensitivity/ni_long_profile.py --plot-only
python sensitivity/ni_long_profile.py --force --binary ./build/marsh_cli
```

---

### `restoration_experiment.py`

50-year colonisation runs starting from **bare sand** — simulating marsh
restoration or initial platform establishment.  Explores how starting elevation
and SSC control the trajectory of biomass establishment and carbon accumulation
on bare intertidal substrate.

**Outputs** (`figures/`):
`restoration_elevation_timeseries.png`, `restoration_carbon_timeseries.png`

```bash
python sensitivity/restoration_experiment.py --binary ./build/marsh_cli
```

---

## Common flags

| Flag | Effect |
|------|--------|
| `--binary PATH` | Path to `marsh_cli` executable (default: `marsh_cli` on PATH) |
| `--plot-only` | Plot from existing outputs; skip all model runs |
| `--force` | Re-run all simulations even if outputs already exist |
| `--skip-runs` | Skip runs where output `.nc` exists; plot only new runs |

---

## Output layout

```
sensitivity/
  runs/
    timestep/                   timestep_sensitivity runs
    inner_cycle/                inner_cycle_sensitivity runs
    gpp_params/                 gpp_parameter_sensitivity runs
    forcing/                    forcing_sensitivity runs
    longterm/                   longterm_sensitivity (48 runs)
    ni_biomass_curve/           test_biomass_curve_ni runs
    ni_biomass_curve_k120/      ni_biomass_rmse_scan runs
    ni_carbon_ol_hm_v5/         ni_carbon_profile_sensitivity Scenario A (mixing compaction)
    ni_carbon_morris_tube_v4/   ni_carbon_profile_sensitivity Scenario B (mixing compaction)
    brain_compaction_test/
      brain/                    Brain two-stage model runs
      mixing/                   Mixing compaction model runs
    ni_long_profile/            ni_long_profile runs (4 SLR scenarios × 300 yr)
    carbon_pool_test/           carbon_pool_test run
  figures/                      All saved PNG figures
```

Intermediate YAML files are kept alongside NetCDF outputs so any run can be
reproduced exactly with `marsh_cli <yaml_path>`.

---

## Compaction model note

All scripts now use `mixing_compaction` (depth-dependent mixing law) as the
default compaction model.  This model is calibrated against 91,780 CCN
synthesis core observations and produces DBD values consistent with observed
data across the full LOI range (0–0.7).  The previous default
(`two_stage_compaction`, Brain et al. 2012) over-compacted by a factor of
2–3× relative to CCN observations and is retained only in
`brain_compaction_test.py` for comparison.

The polynomial coefficients (depth in metres, DBD in g cm⁻³) fitted to all
CCN data:

| Parameter | a₀ | a₁ | a₂ | R² |
|-----------|-----|-----|-----|-----|
| k₁ (organic) | 0.1059 | −0.0129 | 0.0276 | 0.85 |
| k₂ (mineral) | 1.3997 | 0.1701 | −0.3136 | 0.95 |

---

## Relationship to calibration

`ni_biomass_rmse_scan.py` produces the K, σ_H, σ_R values that are the NI
defaults in `calibration/yaml_writer.py`.  If those defaults change, this
scan must be re-run.  `ni_carbon_profile_sensitivity.py` imports those
defaults and the NI tidal constituents from `calibration/site_config.py`.
