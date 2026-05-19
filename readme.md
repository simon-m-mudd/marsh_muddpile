# marsh_muddpile

`marsh_muddpile` is a modular 1-D tidal-marsh sediment-column model written in C++17.
It simulates the vertical evolution of a sediment column in a tidal marsh: mineral
deposition, plant biomass, root emplacement, organic-matter decay, porewater chemistry
(NH₄⁺, SO₄²⁻, CH₄), and compaction — all driven by tidal water levels and climate
forcing read from a YAML configuration file.

The code is an update and redesign of the **Mudd et al. (2009, ECSS)** marsh model
framework, recast as a fully modular architecture where every process is an
interchangeable plug-in.

---

## Current model components

Each step in the per-timestep loop selects one named implementation from the
list below; implementations marked *(default)* are the recommended starting point.

### Water level
| Name | Description |
|------|-------------|
| `composite_water_level` *(default)* | Harmonic reconstruction from up to 11 tidal constituents (M2, S2, N2, K1, O1, P1, K2, Q1, 2N2, M4, MS4) with nodal factors; falls back to a single-sine wave if constituents are absent |

### Evapotranspiration
| Name | Description |
|------|-------------|
| `simple_canopy_et` *(default)* | Penman-Monteith-style ET scaled by LAI and limited by available water |

### Salinity
| Name | Description |
|------|-------------|
| `distance_flushing_salinity` *(default)* | Relaxes root-zone salinity toward creek salinity; rate decays exponentially with distance from creek; ODE solved exactly per step |

### Vegetation / GPP / biomass
| Name | Description |
|------|-------------|
| `marsh_gpp_biomass` *(default)* | Light-use efficiency GPP model (Monteith 1972) with five stress scalars (temperature, hydroperiod, salinity, inundation range, low-inundation desiccation); exact ODE solution for both biomass pools; dynamic belowground capacity tied to root:shoot ratio |
| `seasonal_biomass` | Simple seasonal biomass cycle without GPP |
| `null_biomass` | No vegetation |

### Root allocation
| Name | Description |
|------|-------------|
| `exponential_root_allocation` *(default)* | Distributes live roots and mortality fluxes through the column via an exponential depth profile |
| `null_root_allocation` | No root input to column |

### Deposition
| Name | Description |
|------|-------------|
| `edge_distance_deposition` *(default)* | SSC decays exponentially inland from the creek edge; inundation computed analytically per tidal cycle to capture spring-neap variation |
| `tke_deposition` | TKE-based settling and vegetation trapping |
| `zero_deposition` | No mineral deposition |

### Organic-matter decay
| Name | Description |
|------|-------------|
| `marsh_decay` *(default)* | First-order decay with depth attenuation, temperature sensitivity, and a hydrology–salinity modifier that correctly scales the rate (not the survival fraction) |
| `first_order_decay` | Simpler first-order decay without modifier |
| `identity_decay` | No decay |

### Porewater chemistry — NH₄⁺ (optional)
| Name | Description |
|------|-------------|
| `nh4_porewater` | Per-layer NH₄⁺: production from decomposition, tidal flushing, Fickian diffusion, Michaelis-Menten root uptake |
| `none` *(default)* | Disabled |

Enable with `porewater_chemistry_model_name: nh4_porewater` in the `simulation` block.

### Methane / sulfate (optional)
| Name | Description |
|------|-------------|
| `sulfate_methane` | SO₄²⁻–CH₄ Michaelis-Menten competition, tidal SO₄ replenishment, Fickian diffusion, CH₄ oxidation, ebullition, and plant-mediated (aerenchyma) transport; outputs surface CH₄ flux (µmol m⁻² s⁻¹) |
| `none` *(default)* | Disabled |

Enable with `methane_model_name: sulfate_methane` in the `simulation` block.

### Compaction
| Name | Description |
|------|-------------|
| `mixing_compaction` *(default)* | Depth-dependent Morris et al. (2016) mixing model; DBD = 1 / (LOI/k₁(d) + (1−LOI)/k₂(d)); end-member polynomials calibrated to 85,122 CCN synthesis layers |
| `two_stage_compaction` | Brain et al. (2012) stress-based model (retained for comparison; over-compacts by 2–3× relative to CCN data) |
| `identity_compaction` | No compaction |

---

## Current status

The forward model is calibrated and actively used for research.  Calibration
targets include the North Inlet LTER site (South Carolina, US-HB1 AmeriFlux
tower) with SET accretion records from the GI control plots (1996–2025) and
biomass–elevation data from Morris et al. (2013, *Oceanography*).

Extension sites (PIE/GCE/VCR LTER) are documented in `lter_data/README.md`.
Sensitivity scripts are in `sensitivity/`; calibration helper code is in
`calibration/`.  Output is NetCDF; Python scripts for post-processing and
visualisation are in `scripts/` and `viz/`.

Porewater CH₄ outputs are calibrated against edi.1828.3 (North Inlet CH₄ flux
chamber data).  Porewater NH₄⁺ outputs are compared against edi.136.11
(North Inlet porewater profiles).

---

## Requirements

### Core dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| C++17 compiler | — | Required language standard |
| CMake | ≥ 3.16 | Build system |
| Eigen3 | any 3.x | Dense linear algebra (mass matrix) |
| yaml-cpp | any | YAML configuration loading |
| NetCDF-C | any | NetCDF output (via pkg-config) |
| NetCDF-C++4 | any | C++ NetCDF API |

Optional: **OpenMP** for parallel speedup in the decay, compaction, and
deposition inner loops (controlled by `-Dmarsh_muddpile_use_openmp=ON/OFF`).

---

## Installing dependencies

### Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install \
    build-essential \
    cmake \
    libeigen3-dev \
    libyaml-cpp-dev \
    libnetcdf-dev \
    libnetcdf-c++4-dev \
    pkg-config
```

OpenMP is included in standard GCC; no extra package is needed.

To verify NetCDF is visible to pkg-config after installation:

```bash
pkg-config --modversion netcdf
pkg-config --modversion netcdf-cxx4
```

### conda / mamba (no root access, or for isolated environments)

All required libraries are available from [conda-forge](https://conda-forge.org/):

```bash
mamba install -c conda-forge cmake eigen yaml-cpp libnetcdf netcdf-cxx4 compilers
```

Or with plain conda:

```bash
conda install -c conda-forge cmake eigen yaml-cpp libnetcdf netcdf-cxx4 compilers
```

The `compilers` metapackage provides a GCC C++ compiler pinned to the
conda-forge toolchain.  If your system compiler is already C++17-capable you
can omit it.

After activating the environment, expose the conda pkg-config files:

```bash
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
```

Add this line to `~/.bashrc` to make it permanent, then build with:

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
cmake --build . -j
```

---

## Building

```bash
mkdir -p build
cd build
cmake ..
cmake --build . -j
```

CMake reports which dependencies were found:

```
-- C++17 compiler support: found
-- Eigen3 found: 3.4.0
-- yaml-cpp found: 0.7.0
-- NetCDF C found: 4.9.2
-- NetCDF C++4 found: 4.3.1
-- OpenMP found
```

### Build options

| Option | Default | Description |
|--------|---------|-------------|
| `marsh_muddpile_use_openmp` | `ON` | Enable OpenMP (inner-loop parallelism in decay, compaction, deposition) |
| `CMAKE_BUILD_TYPE` | `Release` | Set to `Debug` for assertions and debug symbols |

Example — disable OpenMP:

```bash
cmake -Dmarsh_muddpile_use_openmp=OFF ..
```

---

## Running

```bash
# from the build directory
./marsh_cli ../example_runs/run_edge_distance_50yr.yaml
```

### CLI options

| Argument | Description |
|----------|-------------|
| `<config.yaml>` | Path to YAML run file (default: `example_run.yaml`) |
| `--silent` | Suppress stdout progress output; errors still go to stderr |

Example output:

```
loading config: ../example_runs/run_edge_distance_50yr.yaml
total run duration: 50.0 yr (18262.5 days)
  0.25 yr / 50.0 yr
  0.50 yr / 50.0 yr
  ...
wrote netcdf output: results/run_edge_distance_50yr.nc
```

---

## YAML configuration

A minimal YAML run file contains these top-level keys:

```yaml
simulation:
  water_level_model_name:         "composite_water_level"
  evapotranspiration_model_name:  "simple_canopy_et"
  salinity_model_name:            "distance_flushing_salinity"
  vegetation_model_name:          "marsh_gpp_biomass"
  root_allocation_model_name:     "exponential_root_allocation"
  deposition_model_name:          "edge_distance_deposition"
  decay_model_name:               "marsh_decay"
  compaction_model_name:          "mixing_compaction"
  porewater_chemistry_model_name: "none"   # or "nh4_porewater"
  methane_model_name:             "none"   # or "sulfate_methane"
  dt_days: 30.4375                         # outer forcing timestep

site:
  distance_from_creek_m:    10.0
  creek_bank_elevation_m:    0.0
  local_tidal_offset_m:      0.0

parameters:
  # tidal constituents
  M2_amplitude_m:    0.766
  M2_phase_deg:      0.0
  tidal_period_hours: 12.42
  # vegetation (S. alterniflora defaults)
  vegetation_lue_gC_per_umol:              1.6e-6
  vegetation_aboveground_capacity_kg_m2:   0.95
  vegetation_root_shoot_ratio:             2.0
  # deposition
  deposition_distance_from_edge_m:  10.0
  deposition_basin_length_m:        50.0

materials:
  - name: silt
    category: mineral
    density: 2650.0
    allow_surface_deposition: true
    settling: { diameter: 2.0e-5, settling_velocity: 3.7e-5 }
  - name: labile_organic
    category: organic_labile
    density: 1200.0
    decay: { k_0: 0.01, gamma: 0.10 }
  - name: refractory_organic
    category: organic_refractory
    density: 1400.0
    decay: { k_0: 0.0, gamma: 2.0 }
  - name: roots
    category: live_root
    density: 1100.0

forcing:
  generated:
    n_steps: 600                        # 50 years × 12 months
    dt_days: 30.4375
    start_time_days: 0.0
    initial_mean_sea_level: 0.0
    sea_level_rise_rate_m_per_yr: 0.003
    temperature_mean_c:       18.92
    temperature_amplitude_c:   8.46
    temperature_peak_day:    202.8
    par_mean_umol_m2_d:      34172288.0
    par_amplitude_umol_m2_d: 14160082.0
    par_peak_day:            164.8
    tidal_amplitude:         0.766
    tidal_period_hours:      12.42
    creek_salinity_ppt:      28.0
    suspended_sediment_concentration: 0.020
    fine_sediment_concentration:      0.005

initial_state:
  layers:
    - top_elevation_m: 0.30
      thickness_m:     0.50
      porosity:        0.60
      age_days:        0.0
      fill_material:   silt

initial_ecohydrology_state:
  aboveground_biomass_kg_m2: 0.5
  belowground_biomass_kg_m2: 1.5
  root_zone_salinity_ppt:   28.0
  lai: 1.5
  litter_kg_m2: 0.0

output:
  file: output.nc
  write_time_series: true
  write_column_snapshots: true
  snapshot_times_days: [18262.5]    # end of 50-year run
```

### Forcing options

Three mutually exclusive ways to specify forcing:

| Key | Use case |
|-----|---------|
| `forcing.steps:` | Explicit list of steps — ERA5-driven or short test runs |
| `forcing.constant:` | Fixed climate, no SLR or seasonal cycle |
| `forcing.generated:` | Synthetic seasonal cycle with optional linear SLR and warming trends (preferred for long runs) |

See `example_runs/` for working examples of each format.

---

## Python calibration and sensitivity layer

The Python code in `calibration/` and `sensitivity/` does **not** call C++
directly.  It writes YAML input files and runs `marsh_cli` as a subprocess,
then reads the NetCDF output.

| File | Purpose |
|------|---------|
| `calibration/site_config.py` | `PlotConfig` dataclass (site, met, tidal parameters); `north_inlet_default_tides()` |
| `calibration/yaml_writer.py` | Canonical parameter defaults (`_DEFAULT_PARAMETERS`, `_DEFAULT_MATERIALS`); writes complete YAML from a `PlotConfig` |
| `calibration/forcing_builder.py` | Builds monthly forcing steps with sinusoidal temperature and PAR |
| `calibration/model_runner.py` | `run_model()` subprocess wrapper |
| `sensitivity/*.py` | Factorial parameter sweeps; outputs go to `sensitivity/runs/`, figures to `sensitivity/figures/` |

---

## Model state

The sediment column (`column_state`) stores:

- layer-by-material mass matrix (kg m⁻²)
- layer thickness (m)
- layer porosity
- layer top elevation (m, relative to MSL)
- layer age (days)
- per-layer porewater NH₄⁺ (µmol L⁻¹)
- per-layer porewater SO₄²⁻ (µmol L⁻¹)
- per-layer porewater CH₄ (µmol L⁻¹)

Layers are ordered **deepest first** (index 0 = oldest).

The ecohydrology state (`ecohydrology_state`) stores aboveground and belowground
biomass (kg m⁻²), LAI, root-zone salinity (ppt), and litter (kg m⁻²).

---

## NetCDF output

### Time-series variables (dimension: `time`)
`model_time_days`, `aboveground_biomass_kg_m2`, `belowground_biomass_kg_m2`,
`gpp_gC_m2_d`, `npp_gC_m2_d`, `aboveground_mortality_kg_m2_d`,
`belowground_mortality_kg_m2_d`, `root_zone_salinity_ppt`, `lai`,
`inundation_fraction`, `mean_water_level_m`, `surface_nh4_umol_L`,
`surface_ch4_flux_umol_m2_s`, `surface_so4_umol_L`,
`total_mass_by_material(time, material)`.

### Column snapshot variables (dimensions: `snapshot`, `layer`)
`layer_mass(snapshot, layer, material)`, `layer_top_elevation`,
`layer_thickness`, `layer_age`, `layer_porewater_nh4`,
`layer_porewater_so4`, `layer_porewater_ch4`.

Layer 0 in a snapshot is the **deepest** (oldest) layer; the surface layer is
at index `snapshot_n_layers − 1`.

---

## Project layout

```
marsh_muddpile/
├── CMakeLists.txt
├── apps/
│   └── marsh_cli.cpp          # command-line entry point
├── include/marsh_model/
│   ├── core/                  # data structures (headers only)
│   ├── engine/                # simulator, process factory, layer merger
│   ├── io/                    # config and result I/O headers
│   └── processes/             # abstract base classes + concrete headers
├── src/
│   ├── core/
│   ├── engine/
│   ├── io/
│   └── processes/             # one .cpp per process implementation
├── calibration/               # Python helpers for YAML generation and running
├── sensitivity/               # Python sensitivity-analysis scripts
├── scripts/                   # data-processing utilities
├── example_runs/              # working YAML configs
├── docs/
│   ├── marsh_muddpile_architecture.md   # developer reference
│   └── marsh_muddpile_model.tex         # governing equations (LaTeX)
└── lter_data/                 # calibration data (LTER, AmeriFlux)
```

---

## Troubleshooting

### NetCDF not found

Ubuntu/Debian:
```bash
sudo apt-get install libnetcdf-dev libnetcdf-c++4-dev pkg-config
```

Conda:
```bash
mamba install -c conda-forge libnetcdf netcdf-cxx4
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
```

### yaml-cpp not found

```bash
sudo apt-get install libyaml-cpp-dev
```

### Eigen3 not found

```bash
sudo apt-get install libeigen3-dev
```

### Compiler does not support C++17

```bash
sudo apt-get install g++   # GCC ≥ 7 is sufficient
```

### CMake cannot find NetCDF after conda install

Pass `CMAKE_PREFIX_PATH` explicitly:
```bash
cmake .. -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
```

---

## Licence

Released under the **GNU General Public Licence v3 (GPL-3.0)**.
See the `LICENSE` file for details.

---

## Citation and acknowledgements

If you use this model please cite the original framework paper and the
relevant process-model papers for the components you use:

**Framework**

- Mudd, S.M., Howell, S.M., Morris, J.T., 2009. Impact of dynamic feedbacks
  between sedimentation, sea-level rise, and biomass production on near-surface
  marsh stratigraphy and carbon accumulation. *Estuarine, Coastal and Shelf
  Science* 82, 377–389. https://doi.org/10.1016/j.ecss.2009.01.028

**Deposition (edge-distance model)**

- Duran Vinent, O., Herbert, E.R., Coleman, D.J., Himmelstein, J.D., Kirwan, M.L.,
  2021. Onset of runaway fragmentation of salt marshes. *One Earth* 4, 506–516.
  https://doi.org/10.1016/j.oneear.2021.02.013

**GPP / biomass**

- Morris, J.T., Sundberg, K., Hopkinson, C.S., 2013. Salt marsh primary production
  and its responses to relative sea level and nutrients in estuaries at Plum Island,
  Massachusetts, and North Inlet, South Carolina, USA. *Oceanography* 26, 78–84.
  https://doi.org/10.5670/oceanog.2013.48

**Compaction (mixing model)**

- Morris, J.T., Barber, D.C., Callaway, J.C., Chambers, R., Hagen, S.C.,
  Hopkinson, C.S., Johnson, B.J., Megonigal, P., Neubauer, S.C., Troxler, T.,
  Wigand, C., 2016. Contributions of organic and inorganic matter to sediment
  volume and accretion in tidal wetlands at steady state. *Earth's Future* 4,
  110–121. https://doi.org/10.1002/2015EF000334

**Compaction (two-stage model, alternative)**

- Brain, M.J., Long, A.J., Woodroffe, S.A., Petley, D.N., Milledge, D.G.,
  Parnell, A.C., 2012. Modelling the effects of sediment compaction on salt marsh
  reconstructions of recent sea-level rise. *Earth and Planetary Science Letters*
  345–348, 180–193. https://doi.org/10.1016/j.epsl.2012.06.045
