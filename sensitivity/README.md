# sensitivity

Scripts for numerical and parameter sensitivity analysis of the marsh_muddpile
model.  Each script is self-contained and can be run from the project root.

All scripts use a **pure M2 sinusoidal tide** (no spring-neap variability) and
**North Inlet ERA5-derived meteorological forcing** as the reference
configuration, unless otherwise stated.  The deposition model is
`edge_distance_deposition` throughout.

---

## Scripts

### `timestep_sensitivity.py`

Tests how the **outer forcing timestep** affects a 50-year simulation.

The outer timestep (`dt_days` in `forcing.steps`) controls how frequently
climate and tidal statistics (temperature, PAR, suspended sediment
concentration, salinity) are updated.  Five timesteps are tested:

| Label | dt (days) |
|-------|-----------|
| 1 d | 1 |
| 7 d | 7 |
| monthly | 30.4 |
| quarterly | 91.3 |
| annual | 365.25 |

The `water_level_substeps_per_step` parameter is scaled proportionally so
every run maintains **20 substeps per tidal cycle**, keeping the inundation
fraction estimate consistent across timesteps.

**Outputs** (`figures/`):

- `timestep_sensitivity_timeseries.png` — surface elevation and total column OC
  over 50 years, one line per timestep
- `timestep_sensitivity_last_year.png` — aboveground and live belowground
  biomass over the last year of the run

```bash
python sensitivity/timestep_sensitivity.py --binary ./build/marsh_cli
python sensitivity/timestep_sensitivity.py --skip-runs   # plot only, skip re-runs
```

---

### `inner_cycle_sensitivity.py`

Tests how the **inner tidal substep count** affects results, with the outer
forcing timestep fixed at **1 day**.

Within each forcing step, the `composite_water_level_model` samples water level
at `water_level_substeps_per_step` evenly-spaced time points and counts the
fraction above the marsh surface.  This `inundation_fraction` is used by the
GPP, salinity, and ET models.  For a high-marsh site briefly inundated each
tide, too few substeps per tidal cycle causes the inundation fraction to be
poorly estimated (coarse quantisation: 0 %, 25 %, 50 %, ...).

Note: the `edge_distance_deposition` model is **not** affected by this
parameter — it loops over every tidal cycle independently and uses an
analytical arcsine formula for inundation fraction (see the main `readme.md`
Timestepping section for details).

Seven substep densities are tested (number of water-level samples per 12.42 h
tidal cycle):

2 / 4 / 8 / 16 / 32 / 64 / 128

**Outputs** (`figures/`):

- `inner_cycle_timeseries.png` — surface elevation and total column OC over 50
  years
- `inner_cycle_last_year.png` — aboveground and live belowground biomass over
  the last year
- `inner_cycle_inundation.png` — annual mean inundation fraction, showing
  convergence with substep count (the most sensitive diagnostic to this
  parameter)

```bash
python sensitivity/inner_cycle_sensitivity.py --binary ./build/marsh_cli
python sensitivity/inner_cycle_sensitivity.py --skip-runs
```

---

### `gpp_parameter_sensitivity.py`

Tests the effect of the four vegetation parameters that are optimised during
North Inlet GPP calibration (see `calibration/calibrate_ni.py`).

Each parameter is varied across three values while all others are held at the
`yaml_writer` defaults.  The three test values are placed symmetrically around
the calibration default in log space:

```
low  = sqrt(lower_bound × default)
mid  = calibration default
high = sqrt(default × upper_bound)
```

| Parameter | Low | Mid (default) | High |
|-----------|-----|---------------|------|
| Aboveground capacity (kg m⁻²) | 0.224 | 0.5 | 1.22 |
| Base mortality (d⁻¹) | 3.16×10⁻⁴ | 1×10⁻³ | 2.24×10⁻³ |
| Cold mortality slope (d⁻¹ °C⁻¹) | 4.47×10⁻⁴ | 2×10⁻³ | 4.47×10⁻³ |
| LUE (gC μmol⁻¹) | 5×10⁻⁵ | 2.5×10⁻⁴ | 3.54×10⁻⁴ |

Runs: 10 years, monthly forcing, sine M2 tide, elevation 0.30 m.
Total: 12 runs (4 parameters × 3 values).

**Outputs** (`figures/`) — one figure per parameter, each with two subplots:

- Top: aboveground biomass over the **last year** (seasonal cycle)
- Bottom: belowground root mortality rate (`belowground_mortality_kg_m2_d`,
  kg m⁻² d⁻¹) over the **full 10-year time series**

```
figures/gpp_sensitivity_capacity.png
figures/gpp_sensitivity_mort_base.png
figures/gpp_sensitivity_cold_slope.png
figures/gpp_sensitivity_lue.png
```

```bash
python sensitivity/gpp_parameter_sensitivity.py --binary ./build/marsh_cli
python sensitivity/gpp_parameter_sensitivity.py --skip-runs
```

---

### `forcing_sensitivity.py`

Tests the effect of two environmental forcing variables on vegetation and
sediment dynamics over a **10-year** simulation.

**Mean annual air temperature (°C):** 8 / 18 / 28
(cool temperate → warm temperate → subtropical)

**Creek salinity (ppt):** 5 / 20 / 35
(mesohaline → polyhaline → euhaline)

Each parameter is swept while all others are held at the North Inlet /
`yaml_writer` defaults.  For salinity runs the initial root-zone salinity is
set to match the creek salinity, eliminating any spin-up transient from
mismatched initial and boundary conditions.

Total runs: 2 variables × 3 values = 6

**Outputs** (`figures/`):

- `forcing_sensitivity_temperature.png` — aboveground biomass (last year, top)
  and root mortality timeseries (bottom) across the three temperature regimes
- `forcing_sensitivity_salinity.png` — same layout across the three salinity
  regimes

Each figure has two stacked subplots:
- **Top:** aboveground biomass over the last year (seasonal cycle, day-of-year
  x-axis)
- **Bottom:** belowground root mortality rate over the full 10-year run

```bash
python sensitivity/forcing_sensitivity.py --binary ./build/marsh_cli
python sensitivity/forcing_sensitivity.py --skip-runs   # plot only
```

---

## Common options

All four scripts accept the same set of flags:

| Flag | Effect |
|------|--------|
| `--binary PATH` | Path to `marsh_cli` executable (default: `marsh_cli` on PATH) |
| `--skip-runs` | Skip runs if output `.nc` files already exist; go straight to plotting |
| `--plot-only` | Plot from existing outputs; do not run the model at all |
| `--force` | Re-run all simulations even if outputs exist |
| `--no-save` | Display figures interactively instead of saving to `figures/` |

---

## Output layout

```
sensitivity/
  runs/
    timestep/          YAML configs and NetCDF outputs for timestep_sensitivity
    inner_cycle/       YAML configs and NetCDF outputs for inner_cycle_sensitivity
    gpp_params/        YAML configs and NetCDF outputs for gpp_parameter_sensitivity
    forcing/           YAML configs and NetCDF outputs for forcing_sensitivity
  figures/             All saved PNG figures
```

Intermediate YAML files are kept alongside the NetCDF outputs so any run can be
reproduced exactly with `marsh_cli <yaml_path>`.

---

## Relationship to the calibration scripts

The GPP parameter sensitivity script imports `PARAM_NAMES`, `PARAM_BOUNDS`, and
`PARAM_DEFAULTS` directly from `calibration/calibrate_ni.py`.  If the
calibration parameter set or bounds are updated, the sensitivity test values
update automatically on the next run.

The `yaml_writer.py` defaults (shared by both calibration and sensitivity runs)
use `edge_distance_deposition` and set `deposition_distance_from_edge_m` from
the plot's `distance_from_creek_m`.  See the main `readme.md` for a description
of the deposition and timestepping models.
