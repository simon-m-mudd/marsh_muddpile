# CH4 Model Calibration — North Inlet, SC and Extension to Other LTER Sites

## 1. Is the AmeriFlux dataset correct?

**Yes, US-HB1 is the right site.** The BIF metadata confirms:

- **Site name:** North Inlet Crab Haul Creek
- **Coordinates:** 33.3455°N, 79.1957°W
- **Ecosystem (IGBP):** WET — permanent, natural tidal wetland
- **Vegetation:** ~40% tall *Spartina alterniflora*, 40% short *S. alterniflora*, 20% creek
- **Climate:** Köppen Cfa (humid subtropical)
- **Period:** 2018–2024 (measured), 1981–2025 (ERA5 downscaled record)
- **PIs:** Tom O'Halloran (Clemson), Erik M. Smith
- **Affiliation:** North Inlet–Winyah Bay NERR, Belle W. Baruch Foundation

**Important caveat:** The standard FLUXNET/ONEFlux product (FLUXMET files) does **not**
contain FCH4. ONEFlux only processes CO2 and energy fluxes. The FLUXMET monthly file
gives `GPP_NT_VUT_REF`, `RECO_NT_VUT_REF`, `NEE_VUT_REF`, `TA_F`, `SW_IN_F`, and
soil temperatures (`TS_F_MDS_1`, `TS_F_MDS_2`) — all useful for the vegetation/GPP
module but not for the CH4 model. The CH4 flux data is in a **separate dataset**
described below.

---

## 2. Data inventory for CH4 calibration at North Inlet

### 2.1 Primary CH4 target — edi.1828.3 (BICEFS)

`lter_data/edi.1828.3/241206_BICEFS_flux_data.csv`

Chamber-based CH4 and CO2 flux measurements from 2022–2023. Key columns:

| Column | Use |
|---|---|
| `CH4 flux (μmol CH4 m-2 s-1)` | Primary calibration target: compare to model `surface_ch4_flux_umol_m2_s` |
| `R2 value for CH4 flux` | Quality filter (retain R2 > 0.9) |
| `chamber type` | Separates `root and shoot` (total flux) from root-only (diffusive + ebullition); difference = plant-mediated transport, constrains `ch4_plant_transport_factor` |
| `ORP at 10/20/30/40 cm depth (mV)` | Negative ORP indicates anoxia and active methanogenesis; constrains the depth of the sulfate–methane transition zone |
| `mean salinity (PSU)` | Site-level salinity; cross-check against `ch4_creek_so4_umol_L` (19600 μmol/L at 35 PSU ≈ 19,000–20,000 μmol/L) |
| `mean water depth (cm)` | Proxy for inundation state at measurement time |
| `site` | Two sites: **Salt marsh** (active tidal) and **Impounded wetland** (isolated, permanently flooded). Calibrate CH4 model against salt marsh only; impounded wetland has different hydrology. |

`lter_data/edi.1828.3/241206_BICEFS_biogeochem_data.csv`

Soil biogeochemistry at 10 cm depth, paired by month with the flux campaign:

| Column | Use |
|---|---|
| `Average C concentration (%)`, `C:N ratio` | Constrain `ch4_carbon_fraction`, `nh4_c_to_n_molar_ratio` |
| `Soil moisture (g water g soil-1)` | Cross-check porosity estimate |
| `organic matter content (g OM g soil-1)` | Constrain initial organic-matter loading in the column |
| `alive root biomass (g)`, `dead root biomass (g)` | Constrain initial belowground biomass and root allocation |
| `Average sq (mcrA copies ng DNA-1)` | mcrA gene copies — proxy for active methanogen density; higher = more methanogenesis capacity |

### 2.2 Porewater depth profiles — edi.136.11 (NILTREB)

`lter_data/edi.136.11/NILTREB_porewater.csv`

Monthly porewater chemistry from 1993–2025, depths 10–100 cm, at three sites
(OL = Old Landing, GI = Georgetown Island, SB) × two positions
(HM = high marsh, LM = low marsh). Columns:

| Column | Use |
|---|---|
| `NH4` (μmol/L) | Calibration target for the NH4 porewater model; compare to `porewater_nh4` output per layer depth |
| `S2` (sulfide, μmol/L) | Proxy for SO4 depletion — high S2- at depth = active SRB, SO4 depleted. Constrains `ch4_km_so4_umol_L` and `ch4_flushing_depth_scale_m` |
| `SALINITY` (ppt) | Constrains `ch4_creek_so4_umol_L` via salinity–SO4 relationship (~560 μmol SO4 per ppt at North Inlet seawater) |
| `DEPTH` | Layer midpoint depths 10, 25, 50, 75, 100 cm; map to model layer indices |

The depth profiles give the shape of the sulfate–methane transition zone (SMTZ).
A good CH4 calibration should reproduce:
- High S2- and near-zero SO4 below ~20–30 cm in high-marsh cores
- Lower S2- at all depths in low marsh (more tidal flushing)

### 2.3 Climate and vegetation forcing — AMF US-HB1

`lter_data/AMF_US-HB1_FLUXNET_2018-2024_v1.3_r1/`

Use the ERA5 downscaled monthly file (`FLUXMET_ERA5_MM_1981-2025`) for long-term
simulations. The measured monthly file (`FLUXMET_MM_2018-2024`) gives observed
`GPP_NT_VUT_REF` and `RECO_NT_VUT_REF` that constrain the vegetation module.
Key extraction steps:

```python
import pandas as pd

# Climate forcing (1981-2025, monthly)
era5 = pd.read_csv("AMF_US-HB1_FLUXNET_ERA5_MM_1981-2025_v1.3_r1.csv")
# TA_ERA = air temperature (deg C)
# SW_IN_ERA = incoming shortwave (W m-2); convert to PAR: PAR_umol = SW_IN * 2.1

# Observed fluxes for validation (2018-2024, monthly)
obs = pd.read_csv("AMF_US-HB1_FLUXNET_FLUXMET_MM_2018-2024_v1.3_r1.csv")
# GPP_NT_VUT_REF in gC m-2 d-1 (already daily mean)
# NEE_VUT_REF in umol m-2 s-1
```

---

## 3. Calibration workflow for North Inlet

### Step 1 — Constrain vegetation parameters from US-HB1 GPP

Run the model with CH4 disabled (`methane_model_name: none`) and optimise
`vegetation_lue_gC_per_umol`, `vegetation_temperature_optimum_c`, and
`vegetation_aboveground_capacity_kg_m2` to match monthly `GPP_NT_VUT_REF`
from the AmeriFlux record.

Objective: minimise RMSE between `gpp_gC_m2_d` (model time-series) and
`GPP_NT_VUT_REF` (AmeriFlux monthly, 2018–2024).

### Step 2 — Constrain organic matter from biogeochemistry

From `241206_BICEFS_biogeochem_data.csv` (salt marsh only, 10 cm depth):

- Set `ch4_carbon_fraction` from mean C concentration (~6–7%)
- Set `nh4_c_to_n_molar_ratio` from C:N ratio (~17–19 at this site)
- Set initial labile/refractory fractions from OM content and C:N

### Step 3 — Constrain porewater NH4 from edi.136.11

Enable the NH4 module and optimise `nh4_tidal_flushing_rate_per_d` and
`nh4_flushing_depth_scale_m` to match observed NH4 depth profiles at OL/HM
(the high-marsh site closest to the BICEFS chamber plots).

Target: observed NH4 gradient (low at surface ~5–30 μmol/L, higher at depth
up to several hundred μmol/L).

### Step 4 — Constrain SO4 and SMTZ from porewater S2-

Enable the sulfate-methane module. The S2- profile constrains:

- `ch4_tidal_flushing_rate_per_d` and `ch4_flushing_depth_scale_m` — set SO4
  near-zero at depth where S2- exceeds ~10 μmol/L
- `ch4_km_so4_umol_L` — the half-saturation constant for sulfate inhibition of
  methanogenesis; higher Km = methanogenesis begins at shallower depth

Target metric: depth at which modelled SO4 falls to 10% of creek concentration
should match depth of S2- peak in porewater profiles.

### Step 5 — Calibrate total CH4 flux against edi.1828.3

Enable CH4 output. Filter BICEFS data to `site == "Salt marsh"` and average by
month across replicates.

Key parameters to optimise in order of sensitivity:

| Parameter | Effect | Typical range |
|---|---|---|
| `ch4_km_so4_umol_L` | Depth of SMTZ; lower = more methanogenesis | 500–2000 μmol/L |
| `ch4_oxidation_rate_per_d` | Surface CH4 attenuation | 0.02–0.15 d-1 |
| `ch4_oxidation_depth_scale_m` | How deep oxidation penetrates | 0.03–0.10 m |
| `ch4_ebullition_threshold_umol_L` | Burst flux events | 200–1000 μmol/L |
| `ch4_plant_transport_factor` | Total/diffusive flux ratio | 1.5–4.0 |

Constrain `ch4_plant_transport_factor` directly from the root+shoot vs.
root-only chamber pair ratio in the BICEFS dataset (empirical at this site ≈ 3).

Objective: match monthly mean CH4 flux (μmol m-2 s-1) within measurement
uncertainty (replicate SD typically 0.5–2 μmol m-2 s-1).

### Step 6 — Validate against temporal pattern

The BICEFS campaign spans summer 2022 and summer 2023. Check that the model
reproduces the summer peak and any between-year difference.

---

## 4. Extension to other LTER sites

The calibration approach above uses three data types: (a) tower-based GPP for
vegetation, (b) porewater chemistry for SMTZ depth, and (c) chamber CH4 flux
as the final target. The following sites have the best combination.

### 4.1 Data available in lter_data/ by site

| Network | Datasets present | GPP | Porewater | CH4 flux |
|---|---|---|---|---|
| **NIN (North Inlet, SC)** | edi.136.11, edi.1828.3, AMF US-HB1 | Yes (AmeriFlux) | NH4, S2-, sal | Yes (BICEFS chambers) |
| **PIE (Plum Island, MA)** | knb-lter-pie.120.12, .127.12, .131.15, .32.15, .519.5, .624.2, .625.1 | No tower | Soil C/N, nutrients | No |
| **GCE (Georgia Coast)** | knb-lter-gce.274.18, .275.18, .277.20, .586.30, .645.4, .759.27, .762.17, .838.3, .839.3, .842.3, .843.5, .858.12 | knb-lter-gce.858.12 = eddy covariance | Nutrients, salinity | No |
| **VCR (Virginia Coast)** | knb-lter-vcr.148.29, .17.16, .170.23, .171.22, .189.16, .247.20, .286.6, .41.18, .6.15 | No tower | Soil C/N, porewater cond. | No |

### 4.2 What is needed to expand to each site

**PIE-LTER (Plum Island Ecosystems, MA)**

- Climate and GPP: AmeriFlux site **US-PLM** (Plum Island Low Marsh) or
  **US-PLT** (Plum Island Transition), if accessible; otherwise use ERA5
- Porewater: knb-lter-pie.625.1 has soil data; knb-lter-pie.519.5 has nutrient
  exchange — extract porewater NH4 from the lateral exchange dataset
- CH4 flux: not in current lter_data. PIE has published chamber CH4 data
  (Moseman-Valtierra et al. 2011; Tang et al. 2020) — obtain EDI dataset
  `knb-lter-pie.174` or similar
- Additional forcing: tidal harmonics for Plum Island Sound (NOAA station 8440900,
  Rockport or Ipswich Bay); salinity ~28–32 ppt

**GCE-LTER (Georgia Coastal Ecosystems, Sapelo Island)**

- Climate: knb-lter-gce.858.12 contains a 30-minute eddy covariance record
  (`MSH-GCEM-2508_30min`, `_daily`, `_yearly`) — has NEE, GPP; check columns
- Porewater: knb-lter-gce.277.20 (organic matter), .586.30 (nutrients, salinity);
  porewater sulfate not directly measured but salinity allows conversion
- CH4 flux: no chamber data in current lter_data; GCE has published bubble trap
  and chamber data (Craft et al.; Segarra et al. 2013 GRL) — find EDI package
- Additional forcing: tidal harmonics for Doboy Sound (NOAA station 8677344);
  salinity ~25–35 ppt; warm climate (mean annual T ~20°C) will push the CH4
  flux higher relative to North Inlet

**VCR-LTER (Virginia Coast Reserve, Hog Island Bay)**

- Climate: no AmeriFlux tower at VCR; use ERA5 or NOAA climate records
- Porewater: knb-lter-vcr.41.18 (organic C), .286.6 (soil C/N, porewater
  conductivity) — conductivity can constrain salinity but not directly SO4
- CH4 flux: VCR marshes are lower in organic content and have higher salinity —
  expect lower CH4 flux. No chamber data in current lter_data; literature values
  from Kathilankal et al. or Megonigal group. Consider whether CH4 module is
  worth running for VCR (SO4 rarely depleted in high-salinity systems).
- Additional forcing: tidal harmonics for Wachapreague Inlet (NOAA station
  8631044); salinity ~30–35 ppt

### 4.3 Parameters that change between sites

When moving from North Inlet to another site, the following parameters are
site-specific and must be re-calibrated:

| Parameter | Site-specific driver | How to set |
|---|---|---|
| `ch4_creek_so4_umol_L` | Salinity (~560 μmol/L per PSU) | From mean salinity × 560 |
| `ch4_initial_so4_umol_L` | Same as creek initially | = creek value |
| `nh4_creek_umol_L` | Creek water chemistry | From nutrient monitoring data |
| `vegetation_temperature_optimum_c` | Latitude / species | ~25°C (subtropical), ~20°C (temperate) |
| `vegetation_aboveground_capacity_kg_m2` | Species productivity | From biomass datasets |
| `ch4_tidal_flushing_rate_per_d` | Tidal prism / distance to creek | From ORP or S2- profiles |
| `ch4_plant_transport_factor` | Species aerenchyma | S. alterniflora ≈ 3; P. australis ≈ 5 |

Parameters that are expected to transfer across sites without re-calibration:
`ch4_km_so4_umol_L`, `ch4_oxidation_rate_per_d`, `ch4_c_to_ch4`,
`nh4_diffusion_coeff_m2_per_d`, `ch4_diffusion_coeff_m2_per_d`.

### 4.4 Suggested workflow for a new site

1. Locate tidal harmonics (NOAA CO-OPS) and build the forcing file
2. Set `ch4_creek_so4_umol_L` from mean creek salinity
3. Copy North Inlet yaml and update site-specific parameters above
4. Calibrate vegetation against any available biomass time-series or
   AmeriFlux GPP record
5. Calibrate SMTZ from porewater S2- or SO4 profiles if available
6. If no chamber CH4 data: run the model and compare to published flux
   ranges for the site type; treat the output as a forward prediction

---

## 5. Files expected in this directory

```
calibration_ch4/
  README.md           — this file
  extract_ni_ch4.py   — read BICEFS, aggregate to monthly means, write to CSV
  extract_ni_porewater.py — read NILTREB, pivot to depth-profile CSV
  calibrate_ch4_ni.py — run model sweeps, compute RMSE vs BICEFS CH4 flux
  ni_ch4_params.yaml  — best-fit parameter set for North Inlet CH4
  plots/              — output comparison figures
```

Scripts to be written; this README defines the calibration design.
