# marsh_muddpile

`marsh_muddpile` is a modular C++ marsh stratigraphy model for simulating the evolution of subsurface sediment layers in tidal marshes.

This code is an update and redesign of the **Mudd et al. (2009, Estuarine, Coastal and Shelf Science)** marsh model framework. The aim of this new version is to make the model:

- more modular
- easier to extend
- faster
- better suited to parameter exploration and future inversion workflows

The long-term goal is to support both:

- **forward modelling**, where marsh evolution is simulated under prescribed forcing and parameter scenarios
- **inversion / calibration**, where measured marsh profiles are used to constrain uncertain model parameters

## Current model components

The code currently includes modular implementations of:

- **surface deposition**
  - TKE-based tidal settling and trapping
- **biomass**
  - seasonal aboveground and belowground biomass
- **root allocation**
  - exponential emplacement of roots through the sediment column
- **decay**
  - first-order decay of materials with depth and temperature effects
- **compaction**
  - a two-stage compaction model motivated by **Brain et al. (2012)**
- **material catalog**
  - support for multiple particle / material types
- **YAML-based configuration**
  - materials, parameters, forcing, and initial state can be loaded from a config file

## Why this code exists

The original marsh model design was scientifically useful, but much of the logic was held in monolithic driver code and tightly coupled classes. This new version is intended to modernise that structure by:

- moving process logic into explicit modules
- storing state in array-based form using **Eigen**
- allowing process selection from configuration files
- making the forward model kernel reusable for future inversion workflows
- preparing the codebase for performance improvements, including OpenMP

## Scientific scope

The model tracks a 1D vertical sediment column in a marsh and can represent:

- mineral sediment
- labile organic matter
- refractory organic matter
- live roots
- isotopes and tracers, where configured

Each timestep can include:

- mineral deposition to the surface
- plant biomass production
- belowground root emplacement
- organic matter decay
- compaction and porosity evolution

## Current status

This is an active redevelopment codebase. The architecture is now in place for modular forward modelling, but the code should still be treated as a research model under development.

In particular:

- some parameter values in example configurations are placeholders
- model calibration is still an ongoing task
- inversion support is planned for a later design phase

---

# Requirements

## Core dependencies

The code currently depends on:

- **C++17**
- **CMake** >= 3.16
- **Eigen3**
- **yaml-cpp**

Optional:

- **OpenMP** for parallel speedup in selected components

## Ubuntu / Debian installation

Install the required packages with:

```bash
sudo apt-get update
sudo apt-get install build-essential cmake libeigen3-dev libyaml-cpp-dev
```

If you want OpenMP support, a standard GCC toolchain is usually sufficient, but installing `g++` explicitly is also fine:

```bash
sudo apt-get install g++
```

---

# Building the code

From the project root:

```bash
mkdir -p build
cd build
cmake ..
cmake --build . -j
```

This should build:

- the library target
- the command-line executable `marsh_cli`

## OpenMP

OpenMP detection is controlled in `CMakeLists.txt`.

By default, OpenMP support is enabled if available.

To configure normally:

```bash
cmake ..
```

To disable OpenMP explicitly:

```bash
cmake -Dmarsh_muddpile_use_openmp=OFF ..
```

During configuration, CMake should report whether it found:

- a C++17 compiler
- Eigen3
- yaml-cpp
- OpenMP

---

# Running the model

## Basic usage

The model is run from the command line with a YAML configuration file.

From the build directory:

```bash
./marsh_cli ../example_run.yaml
```

Or with another config file:

```bash
./marsh_cli ../path/to/your_run.yaml
```

If no filename is supplied, the CLI defaults to `example_run.yaml`.

## CLI options

| Option | Description |
|--------|-------------|
| `<config.yaml>` | Path to the YAML run configuration (default: `example_run.yaml`) |
| `--silent` | Suppress all stdout output; errors still go to stderr |

The `--silent` flag can appear before or after the config file:

```bash
./marsh_cli ../example_run.yaml --silent
./marsh_cli --silent ../example_run.yaml
```

## What the CLI prints

On a normal run the CLI prints:

1. the config file being loaded
2. the total model duration (years and days)
3. a progress line every ~3 months of model time, showing current and total time in years
4. the path of the written NetCDF output file

Example output:

```text
loading config: ../example_runs/run_edge_distance_50yr.yaml
total run duration: 50.0 yr (18262.5 days)
  0.25 yr / 50.0 yr
  0.50 yr / 50.0 yr
  ...
wrote netcdf output: results/run_edge_distance_50yr.nc
```

Use `--silent` to suppress all of the above (useful when running ensembles or calling from scripts).

---

# YAML configuration

The model is configured through a YAML file.

A typical run file contains:

- `simulation`
- `parameters`
- `materials`
- `forcing`
- `initial_state`

## Example structure

```yaml
simulation:
  start_year: 0
  end_year: 1
  dt_days: 30.0

  deposition_model_name: tke_deposition
  biomass_model_name: seasonal_biomass
  root_allocation_model_name: exponential_root_allocation
  decay_model_name: first_order_decay
  compaction_model_name: two_stage_compaction

parameters:
  water_density_kg_m3: 1000.0
  gravity_m_s2: 9.81

materials:
  - name: silt
    category: mineral
    density: 2600.0
    allow_surface_deposition: true
    settling:
      diameter: 2.0e-5
      settling_velocity: 3.7e-5

forcing:
  steps:
    - model_time_days: 30.0
      dt_days: 30.0
      mean_sea_level: 0.0
      mean_high_tide: 0.5
      tidal_amplitude: 0.5
      tidal_period_hours: 12.4
      temperature: 20.0
      suspended_sediment_concentration: 0.02

initial_state:
  layers:
    - thickness_m: 0.10
      porosity: 0.70
      top_elevation_m: 0.10
      age_days: 0.0
      mass_kg_m2:
        silt: 20.0
```

---

# Timestepping

The model operates on two distinct temporal scales that must be configured
independently.

## Outer forcing timestep

The `dt_days` field in each entry of `forcing.steps` sets the duration of one
forcing step — how often temperature, PAR, suspended sediment concentration,
salinity, and tidal statistics are updated.  Typical choices range from one day
(capturing synoptic weather variation) to one month (sufficient for long
equilibrium runs).

A forcing step does not need to equal the tidal period.  Processes that require
sub-step tidal integration handle that internally (see below).

## Inner hydrological substeps

Within each forcing step, the `composite_water_level_model` evaluates water
level at `water_level_substeps_per_step` evenly-spaced time points and counts
what fraction are above the marsh surface.  This `inundation_fraction` (and the
derived mean water depth and inundation duration) is then used by the
**GPP, salinity, and ET models**.

The key constraint is the number of substeps per tidal cycle:

```
substeps_per_tidal_cycle ≈ water_level_substeps_per_step
                            × tidal_period_hours / (24 × dt_days)
```

For a high-marsh site that is inundated for only 15–25 % of each tidal cycle,
at least **~20 substeps per tidal cycle** are recommended to estimate the
inundation fraction accurately.  With only 4 substeps per cycle the estimated
fraction is coarsely quantised (0 %, 25 %, 50 %, …) and can introduce
significant bias in GPP and salinity.

For a monthly forcing step (dt_days ≈ 30.4) this requires roughly
`water_level_substeps_per_step ≈ 20 × 30.4 × 24 / 12.42 ≈ 1175`.
The historic default of 240 corresponds to ~4 substeps per cycle and
is too coarse for high-marsh simulations.

## Deposition model: independent per-cycle integration

The `edge_distance_deposition` model does **not** use
`water_level_substeps_per_step`.  Instead, it:

1. Loops over every individual tidal cycle within the forcing step
   (`n_cycles = round(24 × dt_days / tidal_period_hours)`).
2. Samples 60 points within each cycle to find the cycle's min and max
   water level (the *cycle envelope*).
3. Computes the inundation fraction for that cycle analytically using the
   exact arcsine formula for a sinusoidal tide:

   ```
   f = 0.5 − arcsin((z − mean) / amplitude) / π
   ```

This is exact regardless of how coarse the outer forcing step is, and correctly
handles high-marsh sites that are barely inundated.  The `water_level_substeps_per_step`
parameter has no effect on deposition computed by this model.

## Parameter summary

| Parameter | Where set | Effect |
|-----------|-----------|--------|
| `dt_days` in `forcing.steps` | forcing YAML | Duration of each forcing step; controls how often climate/tidal statistics update |
| `water_level_substeps_per_step` | `parameters` block | Number of time samples within each forcing step used by the water level model to compute inundation fraction for GPP, salinity, ET |

## Rule of thumb

For a mid-marsh site (~20 % inundation fraction per tide), target at least
20 substeps per tidal cycle.  Scale `water_level_substeps_per_step`
proportionally with `dt_days`:

```
water_level_substeps_per_step ≈ 20 × dt_days × 24 / tidal_period_hours
```

For a semidiurnal (12.42 h) site:

| dt_days | Recommended substeps |
|---------|----------------------|
| 1       | 39                   |
| 7       | 271                  |
| 30.4    | 1175                 |
| 91.3    | 3526                 |

The sensitivity scripts in `sensitivity/` quantify the effect of both
parameters on a 50-year simulation.

---

# Available process models

The code currently supports the following model names.

## Deposition

- `zero_deposition`
- `tke_deposition`

## Biomass

- `null_biomass`
- `seasonal_biomass`

## Root allocation

- `null_root_allocation`
- `exponential_root_allocation`

## Decay

- `identity_decay`
- `first_order_decay`

## Compaction

- `identity_compaction`
- `two_stage_compaction`

These are selected using the `*_model_name` fields in the `simulation` section of the YAML file.

---

# Model design

## State representation

The marsh column is represented using array-based state storage:

- layer-by-material mass matrix
- layer thickness
- layer porosity
- layer top elevation
- layer age

This structure is designed for:

- simpler process coupling
- better numerical transparency
- better memory locality than object-heavy per-particle designs

## Materials

Material definitions are stored in a `material_catalog`.

A material can include properties such as:

- density
- settling properties
- decay properties
- category
- flags for root input or surface deposition

This allows the same forward model to work with different sediment and tracer mixtures.

---

# Compaction model

The current compaction module is a two-stage formulation motivated by:

- **Brain et al. (2012)**

It uses:

- a reference void ratio
- a recompression index
- a virgin compression index
- a yield stress

The current implementation also includes hooks to let compression properties depend on:

- organic content, estimated from loss on ignition
- grain size

This makes it more flexible than the older single-stage compaction treatment.

---

# Relationship to the older marsh model

This code is intended as a modern update to the **Mudd et al. (2009, ECSS)** marsh model family.

The scientific ideas retained from that modelling tradition include:

- marsh elevation control on biomass
- seasonal biomass and mortality
- root emplacement into the sediment column
- marsh surface deposition linked to hydrodynamics and vegetation
- organic matter decay
- vertical sediment-column evolution

The main changes in this new code are architectural:

- modular process interfaces
- YAML-based setup
- array-based state storage
- support for multiple interchangeable process implementations
- preparation for faster forward ensembles and future inversion

---

# Development goals

Near-term goals include:

- validating the forward model against expected behaviour
- improving runtime diagnostics and output writing
- adding result export
- strengthening runtime checks
- improving performance in hotspots

Longer-term goals include:

- inversion and calibration against measured marsh profiles
- MPI-based or ensemble-scale workflow support
- richer observation operators
- more alternative process modules

---

# Project layout

A simplified project structure is:

```text
apps/
  marsh_cli.cpp

include/marsh_model/
  core/
  engine/
  io/
  processes/

src/
  core/
  engine/
  io/
  processes/
```

## Key parts

- `core/`
  - state, materials, forcing, parameters, results
- `processes/`
  - deposition, biomass, roots, decay, compaction
- `engine/`
  - simulator and process factory
- `io/`
  - YAML configuration loading
- `apps/`
  - command-line interface

---

# Example workflow

## 1. Build

```bash
mkdir -p build
cd build
cmake ..
cmake --build . -j
```

## 2. Run example config

```bash
./marsh_cli ../example_runs/run_edge_distance_50yr.yaml
```

## 3. Check screen output

The CLI prints the total run duration, then a progress line every ~3 months of
model time, and finally the path of the written NetCDF output file.
All results (time series, final state, material totals) are in the NetCDF file.

---

# Troubleshooting

## yaml-cpp not found

Install:

```bash
sudo apt-get install libyaml-cpp-dev
```

## Eigen3 not found

Install:

```bash
sudo apt-get install libeigen3-dev
```

## Compiler does not support C++17

Install a newer compiler, for example:

```bash
sudo apt-get install g++
```

and rerun CMake.

## OpenMP not found

This is not fatal unless you specifically require OpenMP. The code will still build without it if CMake is configured to allow that.

---

# Licence

This project is released under the **GNU General Public Licence v3 (GPL-3.0)**.

See the `LICENSE` file for details.

---

# Citation and acknowledgements

This code is an update and redesign of the marsh modelling framework associated with:

- Mudd, S.M., Howell, S.M., Morris, J.T., 2009. Impact of dynamic feedbacks between sedimentation, sea-level rise, and biomass production on near-surface marsh stratigraphy and carbon accumulation. Estuarine, Coastal and Shelf Science 82, 377–389. https://doi.org/10.1016/j.ecss.2009.01.028

This model evolved with a more complex particle settling component build on the basis of

- Mudd, S.M., D’Alpaos, A., Morris, J.T., 2010. How does vegetation affect sedimentation on tidal marshes? Investigating particle capture and hydrodynamic controls on biologically mediated sedimentation. Journal of Geophysical Research: Earth Surface 115. https://doi.org/10.1029/2009JF001566

Which was then used in:

- Kirwan, M.L., Mudd, S.M., 2012. Response of salt-marsh carbon accumulation to climate change. Nature 489, 550–553. https://doi.org/10.1038/nature11440

However, recent papers have suggested the mixing of the sediment column and low velocities on marsh surface mean that deposition is dominated by length from channel, initial suspended sediment concentration, and nothing else, and we have implemented such a model in this software:

- Duran Vinent, O., Herbert, E.R., Coleman, D.J., Himmelstein, J.D., Kirwan, M.L., 2021. Onset of runaway fragmentation of salt marshes. One Earth 4, 506–516. https://doi.org/10.1016/j.oneear.2021.02.013


The compaction model has been updated to reflect:

- Brain, M.J., Long, A.J., Woodroffe, S.A., Petley, D.N., Milledge, D.G., Parnell, A.C., 2012. Modelling the effects of sediment compaction on salt marsh reconstructions of recent sea-level rise. Earth and Planetary Science Letters 345–348, 180–193. https://doi.org/10.1016/j.epsl.2012.06.045


