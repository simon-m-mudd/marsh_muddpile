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

The model is currently run from the command line using a YAML configuration file.

From the build directory:

```bash
./marsh_cli ../example_run.yaml
```

Or with another config file:

```bash
./marsh_cli ../path/to/your_run.yaml
```

If no filename is supplied, the CLI currently defaults to:

```text
example_run.yaml
```

## What the CLI does

The CLI currently:

1. loads the YAML configuration
2. builds the selected process modules
3. runs the forward model
4. prints diagnostics to screen

These diagnostics include:

- selected process models
- number of layers
- final surface elevation
- top-layer thickness and porosity
- total mass by material
- a timestep-by-timestep summary of biomass and elevation

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
./marsh_cli ../example_run.yaml
```

## 3. Inspect screen output

Check:

- number of layers
- final surface elevation
- top-layer thickness and porosity
- total mass by material
- time series of biomass and surface elevation

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

- **Mudd et al. (2009, ECSS)**

The current two-stage compaction formulation is informed by:

- **Brain et al. (2012)**

