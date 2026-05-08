# LTER Data Summary

Twelve datasets from three LTER sites covering salt-marsh plant biomass, productivity, porewater chemistry, surface elevation, and carbon fluxes.

---

## EDI / North Inlet LTREB — Georgetown, SC
*PI: James Morris, Belle W. Baruch Institute / University of South Carolina*
*Site: Spartina alterniflora salt marsh in North Inlet, a pristine, tidally-dominated estuary; tidal range ~1.4 m. Control and N+P fertilised plots at 4 locations across 2 sites (low marsh + high marsh).*

### edi.135.12 — Aboveground plant data (1984–2025)
`NILTREB_plants_aboveground_biomass_density.csv` — 13,844 rows  
Columns: `SITE, LOCATION, PLOT, SUBPLOT, TREATMENT, DATE, PLANT_DENSITY, ABOVEGROUND_BIOMASS`  
Non-destructive monthly biomass estimates (g m⁻²) computed from allometric equations applied to plant height measurements. Also includes:
- `NILTREB_plants_annual_productivity.csv` — annual above-ground productivity (g m⁻² yr⁻¹) with SE, by site/year/treatment
- `NILTREB_plants_plantheights.csv` — individual plant height measurements (large file)
- `NILTREB_plants_snail_observations.csv` — *Littoraria* and *Melampus* snail counts per plot

### edi.136.11 — Porewater chemistry (1993–2025)
`NILTREB_porewater.csv` — 27,107 rows  
Columns: `SITE, LOCATION, DATE, TREATMENT, DEPTH, REP, NH4, PO4, S2, Fe2, SALINITY` (each with a quality flag)  
Porewater sampled at multiple depths from diffusion samplers; ~monthly at 5 locations. Analytes: ammonium (NH₄), phosphate (PO₄), sulfide (S²⁻), dissolved iron (Fe²⁺), salinity.

---

## PIE — Plum Island Ecosystems LTER, Massachusetts

### knb-lter-pie.120.12 — NIN site biomass (1984–2016)
`LTE-MP-NIN-biomass.csv` — 11,020 rows  
Columns: `year, month, day, site, location, treatment, plot, subplot, biomass`  
Long-term monthly biomass (g m⁻²) at the North Inlet NIN plots.

### knb-lter-pie.32.15 — LPA site biomass (1999–2018)
`LTE-MP-LPA-biomass.csv` — 224 rows  
Columns: `SITE, YEAR, MONTH, DAY, TRT, MEAN BIOMASS, BIOMASS SE, MEAN PLANT DENSITY, DENSITY SE`  
Seasonal mean biomass and plant density with standard errors.

### knb-lter-pie.127.12 — LAC site annual productivity (1999–2019)
`LTE-MP-LAC-productivitymeans.csv` — 21 rows  
Columns: `SITE, YEAR, MEAN PRODUCTIVITY, PRODUCTIVITY SE`  
Annual above-ground productivity means.

### knb-lter-pie.131.15 — LPP site annual productivity (2000–2025)
`PIE_MP_LPP_productivity.csv` — 50 rows  
Columns: `SITE, TRT, YEAR, MEAN PRODUCTIVITY, STANDARD ERROR`  
Annual productivity by treatment.

---

## GCE — Georgia Coastal Ecosystems LTER, Sapelo Island, GA

### knb-lter-gce.759.27 — Long-term plant biomass, Altamaha River transition sites (2012–present)
*PI: Steven C. Pennings*  
Multi-species monitoring along a plant-zone gradient (creek bank to high marsh) at sites near the Altamaha River. Three tables:
- `PLT-GCES-1609c_Biomass_Stats_5_0.CSV` — 288 rows; plot-level statistics: `Year, Site, Zone, Plot, Num_Plant_Biomass_m2, Min/Max/Mean/Median/Std_Plant_Biomass_m2` (g m⁻²)
- `PLT-GCES-1609c_Observations_5_0.CSV` — large; individual shoot measurements: `Date, Site, Zone, Species, Shoot_Height, Flowering, Dead, Shoot_Biomass`
- `PLT-GCES-1609c_Shoots_Flowering_6_0.CSV` — 424 rows; plot-level shoot density and flowering counts: `Year, Site, Zone, Species_Code, Tallest_Shoot, Num_Shoots_m2, Num_Flowers_m2`

### knb-lter-gce.762.17 — Monthly *Spartina alterniflora* vegetation, multiple Georgia coast sites (2016–2021)
*PI: Jessica L. O'Connell*  
Four tables from a Belowground Ecosystem Resiliency Model campaign:
- `PLT-GCED-2106_coords_1_0.CSV` — 49 rows; site coordinates with UTM and lat/lon
- `PLT-GCED-2106_vegplot_1_0.CSV` — vegetation plot data: LAI, chlorophyll, % cover by species (*Juncus*, *Spartina*, dead, bare), canopy heights, stem densities
- `PLT-GCED-2106_rootcore_1_0.CSV` — 182 rows; root core data: live/dead stem counts and masses, root and rhizome biomass (live and dead, g)
- `PLT-GCED-2106_leafN_1_0.CSV` — 174 rows; leaf chemistry: `%N, %C, C:N ratio`

### knb-lter-gce.858.12 — Eddy covariance carbon fluxes, Georgia tidal salt marsh (2014–2024)
*PI: Peter Hawman*  
Vertical carbon flux time series at a single Georgia marsh tower. Three temporal aggregations:
- `MSH-GCEM-2508_30min_2_0.CSV` — 192,866 rows; 30-minute NEE, ER, GPP with 95% CI (μmol m⁻² s⁻¹), plus wind speed, air temperature, humidity, and gap-fill fraction
- `MSH-GCEM-2508_daily_3_0.CSV` — ~3,650 rows; daily totals/means in g C m⁻² day⁻¹
- `MSH-GCEM-2508_yearly_3_0.CSV` — 11 rows (2014–2024); annual NEE, ER, GPP in g C m⁻² yr⁻¹. NEE ranges from −95 to −249 g C m⁻² yr⁻¹, consistently a net carbon sink.

---

## VCR — Virginia Coast Reserve LTER, Eastern Shore, VA
*Site: Upper Phillips Creek (UPM) marsh (~37.46°N, −75.83°W)*

### knb-lter-vcr.148.29 — Surface elevation, Upper Phillips Creek (1997–2025)
*PI: L.K. Blum, R.R. Christian et al.*  
`VCR06136.csv` — 999 rows  
Columns: `DATE, SITE, DATATYPE, INCREMENTAL, CUMULATIVE, COMMENTS`  
Incremental and cumulative surface elevation change measurements at multiple sites within the marsh.

### knb-lter-vcr.170.23 — End-of-year biomass at SET plots (2001–2017)
*PI: R.R. Christian*  
`UPC_SET_data.csv` — 941 rows  
Columns: `EOYBYear, marshName, siteName, locationID, speciesName, liveMass, deadMass, unknownMass, totalMass, latitude, longitude`  
Annual end-of-year biomass (live, dead, total) at Surface Elevation Table (SET) plots; primarily *Spartina alterniflora*.

### knb-lter-vcr.171.22 — End-of-year biomass in marsh transition plots (2001–2016)
*PI: R.R. Christian*  
`UPT_data.csv` — 273 rows  
Same column structure as above, but in marsh-to-upland transition zone. Species include *Distichlis spicata* in addition to *Spartina alterniflora*.

---

## Quick-reference table

| Dataset ID | Site / Location | Variable(s) | Period | Rows |
|---|---|---|---|---|
| edi.135.12 | North Inlet, SC | Aboveground biomass, plant density, productivity, heights, snails | 1984–2025 | ~13,844+ |
| edi.136.11 | North Inlet, SC | Porewater NH₄, PO₄, S²⁻, Fe²⁺, salinity (multi-depth) | 1993–2025 | 27,107 |
| knb-lter-pie.120.12 | PIE NIN, MA | Biomass | 1984–2016 | 11,020 |
| knb-lter-pie.32.15 | PIE LPA, MA | Mean biomass & plant density | 1999–2018 | 224 |
| knb-lter-pie.127.12 | PIE LAC, MA | Annual productivity | 1999–2019 | 21 |
| knb-lter-pie.131.15 | PIE LPP, MA | Annual productivity | 2000–2025 | 50 |
| knb-lter-gce.759.27 | GCE Altamaha, GA | Biomass, shoot density, flowering | 2012–present | ~1,000 |
| knb-lter-gce.762.17 | GCE multi-site, GA | Veg cover, LAI, root cores, leaf %N/%C | 2016–2021 | ~500 |
| knb-lter-gce.858.12 | GCE Sapelo, GA | NEE, ER, GPP (eddy covariance) | 2014–2024 | ~197,000 |
| knb-lter-vcr.148.29 | VCR Phillips Creek, VA | Surface elevation (incremental + cumulative) | 1997–2025 | 999 |
| knb-lter-vcr.170.23 | VCR Phillips Creek, VA | End-of-year biomass at SET plots | 2001–2017 | 941 |
| knb-lter-vcr.171.22 | VCR Phillips Creek, VA | End-of-year biomass, marsh transition plots | 2001–2016 | 273 |
