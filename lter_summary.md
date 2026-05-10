# LTER Data Summary

Thirty-three datasets from four LTER sites covering salt-marsh plant biomass, productivity, porewater chemistry, soil properties, surface elevation, decomposition, carbon fluxes, water quality, creek nutrient and sediment dynamics, and marsh-to-upland vegetation dynamics.

---

## Baruch Institute / North Inlet — Methane Fluxes

### edi.1828.3 — Chamber CH4 and CO2 fluxes with soil biogeochemistry, Baruch Institute tidal marshes (~2022–2023)
*PI: Seyfried et al. (Clemson University / Baruch Institute of Coastal Ecology and Forest Science)*  
*Site: Belle W. Baruch Institute, Georgetown, SC — physically at North Inlet. Two marsh types: (1) euhaline tidal salt marsh (Spartina alterniflora) and (2) mesohaline impounded (non-tidal) brackish wetland.*

Two tables:
- `241206_BICEFS_flux_data.csv` — 1,075 rows; monthly diurnal chamber campaigns  
  Columns: `site, month, start time, end time, date, chamber ID, chamber type, replicate, chamber height (cm), leaf area (cm2), dead stems, tide height (ft), oxidation reduction potential at 10/20/30/40 cm depth (mV), Pressure (kPa), incoming solar radiation (W m⁻²), relative humidity (%), Chamber temperature (C), wind speed, CO2 flux (μmol m⁻² s⁻¹), R² CO2, CH4 flux (μmol m⁻² s⁻¹), R² CH4, mean salinity (PSU), mean water temperature (C), mean water depth (cm)`  
  Three chamber types: root+shoot (whole plant), root-only (soil + roots, no shoots), plant-free (bare sediment). Salt marsh: 444 rows; impounded wetland: 631 rows.  
  **Salt marsh CH4 flux: n = 435, mean ≈ 0.0046 μmol m⁻² s⁻¹ (~2.3 g CH4 m⁻² yr⁻¹).** Root-and-shoot chambers yield ~3× higher mean flux than root-only, showing strong plant-mediated transport. Redox potential at 10 cm: mean −212 mV (range −238 to −160 mV) — consistently reducing.

- `241206_BICEFS_biogeochem_data.csv` — 174 rows; soil cores at 10 cm and 50 cm depth  
  Columns: `Sample ID, site, soil depth, month, Average sq (mcrA copies ng DNA⁻¹), Average N (%), Average C (%), C:N ratio, Soil moisture, pH, conductivity (mS cm⁻¹), organic matter content, dead root biomass (g), alive root biomass (g)`  
  mcrA gene copies (direct proxy for active methanogen abundance): salt marsh mean 2,837 copies ng DNA⁻¹, range 10–17,680. Salt marsh soil %C: mean 1.97 %, range 0.81–4.53 %.

---

## NIN — North Inlet LTER (historical), Georgetown, SC
*Site: North Inlet estuary, ~2,630 ha of tidal marshes, ocean-dominated; three creek sampling stations — Town Creek (TC), Clambank Creek (CB), Oyster Landing / Crab Haul Creek (OL). Data collected under Vernberg/Blood PIs, now archived via LTER Network.*

### knb-lter-nin.1.1 — Daily creek water nutrient chemistry (1978–1992)
*PI: F. Vernberg, Elizabeth Blood*  
`LTER.NIN.DWS.csv` — 13,176 rows (4,982 TC / 4,097 CB / 4,097 OL)  
Columns: `DATE, transect, water_temp, SAL, TNW, TNF, TPW, TPF, POP, NHN, NNN, CHEM, TOC, DOC, POC`  
Daily water samples from three creek stations measuring temperature, salinity, total/filtered N and P (μmol L⁻¹), orthophosphate, ammonia (NHN), nitrate/nitrite, total organic carbon, dissolved organic carbon, and particulate organic carbon. Valid NH₄⁺ measurements: n = 10,387, range 0–31.7 μmol L⁻¹, mean ~1.5 μmol L⁻¹. Salinity complete (mean ~24.5 ppt across all stations). No direct sulfate measurements; salinity can proxy creek SO₄²⁻ (≈ 0.8 × salinity in mM, giving mean ~19.6 mM).

### knb-lter-nin.8.1 — Daily creek suspended sediment and physical parameters (1981–1992)
*PI: J. Vernberg, Elizabeth Blood, Robert Gardner*  
`LTER.NIN.sedi.csv` — 12,364 rows (same three stations: TC, CB, OL)  
Columns: `DATE, Transect, TIME, Water_temp, SAL, Tide_ele, Wave_ele, Secchi_disk, Sedi_filter, Volume_water, Total_sedi, Inorganic_sedi, Organic_sedi, Tide_dir, Tide_stage, Water_sur, Sky_cond, Sky_cover`  
Daily water-column samples (0.5 m depth) with TSS split into inorganic and organic fractions (mg L⁻¹), tide elevation, wave height, Secchi depth, and a tide direction code (1 = ebb, 2 = flood, 3 = slack). Tide direction is coded for only ~594 rows (~5 % of records), representing single-point-in-tidal-cycle observations rather than paired ebb/flood samples on the same tidal cycle. Temporal overlap with knb-lter-nin.1.1 is 1981–1992.

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

### knb-lter-pie.519.5 — Tidal creek lateral exchange nutrients (2016–2019)
*PI: Nathaniel Weston, Villanova University*  
`EST-RO-TC-LateralExchange-NUT_06.csv` — 2,212 rows  
Columns: `Station, Sample, Date, Time, DIC, TSS, PO4, NH4, NO3, DOC, TDN, PN, PC, Comments`  
Water samples at ~15-minute intervals from the start of flood tide through to the following low tide at five tidal creek stations in Rowley, MA: two low-marsh creeks (LM1, LM2 — Shad Creek branches) and three high-marsh creeks (HM1, HM2/Nelson, HM3/West). Analytes: dissolved inorganic carbon (μmol L⁻¹), total suspended solids (mg L⁻¹), PO₄, NH₄⁺, NO₃ (all μmol L⁻¹), DOC, TDN (μmol L⁻¹), particulate N and C (μg L⁻¹). The within-tidal-cycle time resolution makes this suitable for computing net lateral nutrient fluxes by integrating concentration × tidal flow over flood–ebb sequences. Companion to the Space-for-Time biomass and soil datasets (same PI, same site).

### knb-lter-pie.624.2 — Space for Time aboveground biomass (2018–2024)
*PI: Nathaniel Weston, Villanova University*  
`MAR-RO-ST-Biomass_V.03.csv` — 342 rows  
Columns: `Date, Site, Transect, Quadrat, Biomass, Latitude, Longitude, Elevation`  
Annual peak aboveground biomass clipped from 0.053 m² quadrats along transects at seven "Space for Time" gradient plots (2 low-marsh, 5 high-marsh sites) in PIE LTER, 2018–2024. Dried at 60°C and weighed (g m⁻²). Designed to capture productivity variation across a marsh elevation gradient as a space-for-time substitute.

### knb-lter-pie.625.1 — Space for Time soil cores (2019–2022)
*PI: Nathaniel Weston*  
`MAR_RO_ST_Soil.csv` — 593 rows  
Columns: `Date, Site, Transect, Quadrat, Depth_start, Depth_end, LOI, BD, Salinity, NH4, Latitude, Longitude, Elevation`  
Soil cores (30 cm deep, 8 cm diameter) from Space for Time plots, sectioned at three depth intervals (0–5, 10–15, 25–30 cm). Analyses: loss-on-ignition (organic matter %), bulk density, porewater salinity, and porewater ammonium (NH₄⁺). Companion dataset to knb-lter-pie.624.2.

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

### knb-lter-gce.274.18 — Marsh soil characteristics, nine GCE sites (May 2001)
*PI: Christopher Craft, Indiana University*  
`GEL-GCEM-0508a_1_2.CSV` — 285 rows  
Columns: `Site, Marsh_Location, Depth_Core_Top, Depth_Core_Bottom, Replicate, Bulk_Density, Carbon_Percent, Nitrogen_Percent, Phosphorous_Conc, Cesium137, Lead210`  
Single-event soil-core campaign (two locations per site: levee and interior) across nine GCE sampling sites spanning three estuaries with contrasting freshwater and sediment inputs. Core sections analysed for bulk density, organic carbon (%C), total nitrogen (%N), phosphorus, ¹³⁷Cs and ²¹⁰Pb.

### knb-lter-gce.275.18 — Soil respiration and temperature, five GCE sites (2003–2005)
*PI: Christopher Craft*  
`ORG-GCEM-0508a_1_2.CSV` — 406 rows  
Columns: `Site, Year, Month, Marsh_Location, Replicate, Temp_Soil, CO2_Conc_Start, CO2_Conc_End, Respiration_Rate`  
Soil CO₂ efflux and soil temperature measured at five GCE sites (levee, interior, and dieback zones) across seven dates (June 2003 – March 2005) to evaluate freshwater input effects on soil respiration rates.

### knb-lter-gce.277.20 — Root decomposition and in-growth, sites 6–9 (2003–2004)
*PI: Christopher Craft*  
`ORG-GCEM-0508b_2_0.CSV` — 158 rows  
Columns: `Site, Date, Days_Buried, Location_Code, Replicate, Wetland_Type, Plant_Species, Root_Dry_Weight, Carbon_Percent, Nitrogen_Percent, Phosphorus_Total, Root_Growth` (plus QA flags)  
Nylon mesh bags filled with dominant-species roots buried at 20 cm depth across four tidal marshes; retrieved quarterly (June 2003 – June 2004). Records dry weight, %C, %N, total P, and in-growth root mass to quantify root decomposition rates.

### knb-lter-gce.586.30 — SALTEx porewater chemistry (2014–2020)
*PI: Christopher Craft*  
Seawater Addition Long-Term Experiment (SALTEx): 31 plots (2.5 × 2.5 m) in a freshwater tidal marsh subject to Press (continuous seawater), Pulse (seasonal), and Freshwater control treatments. Two tables:
- `NUT-GCES-1608_SALTEx_5_0.CSV` — 511 rows; quarterly porewater: `Year, Month, Day, Treatment, Replicate, Elevation, Dissolved_Reactive_Phosphorus, Total_Phosphorus, Ammonium, Nitrate_Nitrite, Nitrite, Total_Nitrogen, Organic_Nitrogen, Dissolved_Organic_Carbon, C:N, Sulfides, Chloride, Sulfate, pH, Salinity` (plus flags)
- `NUT-GCES-1608_SourceWater_5_0.CSV` — 20 rows; source water chemistry used to make up treatments

### knb-lter-gce.645.4 — SALTEx sediment elevation (SET), 2013–2022
*PI: Christopher Craft*  
`GEL-GCED-1802_2_0.CSV` — 7,695 rows  
Columns: `Sample_Date, Plot, Treatment, Replicate, Arm, Pin_Number, Elevation, Flag_Elevation`  
Rod Surface Elevation Tables (SETs) installed in SALTEx plots; semi-annual measurements from July 2013 to August 2022 tracking soil accretion/subsidence response to chronic saltwater intrusion.

### knb-lter-gce.838.3 — SALTEx plant percent cover (2013–2022)
*PI: Christopher Craft*  
`PLT-GCED-2404_1_0.CSV` — 1,232 rows  
Columns: `Date, Treatment, Replicate, Percent_Cover, Species`  
Annual July visual percent-cover estimates (peak biomass) for four dominant freshwater marsh species (*Zizaniopsis miliacea*, *Pontederia cordata*, *Persicaria hydropiperoides*, *Ludwigia repens*) in SALTEx plots, 2013–2022.

### knb-lter-gce.839.3 — SALTEx photosynthetically active radiation (PAR), 2014–2022
*PI: Christopher Craft*  
`PLT-GCED-2404a_1_0.CSV` — 369 rows  
Columns: `Year, Month, Treatment, Replicate, PAR_above, PAR_below, Light_penetration, Light_lost`  
Above- and below-canopy PAR measured with SunScan Canopy Analysis System (4–5 times per growing season during active treatment; 1–2 times during recovery). Quantifies canopy light interception as a response variable to saltwater stress.

### knb-lter-gce.842.3 — SALTEx belowground biomass (2017)
*PI: Christopher Craft*  
`PLT-GCED-2411_1_0.CSV` — 124 rows  
Columns: `Year, Month, Treatment, Replicate, Depth, Biomass`  
Single-event root-core campaign (7.62 cm diameter, 20 cm deep) in SALTEx plots after 3 years of treatment, sectioned at 0–5, 5–10, and 10–20 cm depth intervals. Quantifies belowground biomass response to chronic saltwater intrusion.

### knb-lter-gce.843.5 — Vertical accretion and C/N burial: natural vs. restored salt marsh, Sapelo Island (2020–2022)
*PI: Christopher Craft*  
`HYD-GCED-2310_2_0.CSV` — 166 rows  
Columns: `Site_ID, Core_ID, Latitude, Longitude, Depth_range, Bulk_Density, Organic_C, Total_N, Depth_Increment, Carbon, Nitrogen, Carbon.Density, Nitrogen.Density`  
Paired soil-core study comparing a natural, never-diked tidal salt marsh with a hydrologically restored marsh (diked 1948, breach restored 1956) on Sapelo Island. Cores (8.5 cm diameter, 60 cm deep) sectioned at 2 cm increments for bulk density, organic carbon, and total nitrogen; assesses whether restoration recovers carbon sequestration and nitrogen burial functions.

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

### knb-lter-vcr.189.16 — Soil characterisation of end-of-year biomass marshes (1999)
*PI: Cassondra Thomas, Linda Blum, Robert Christian*  
`VCR11181.csv` — 634 rows  
Columns: `CORECODE, SITE, TRANSECT, ZONE, SEGMENT, DEPTH1, DEPTH2, LENGTH, WETWT, DRYWT, PANWT, CORERAD, BULKDENS, SALINITY, PH, ORGANICSPER, WATERPER, DATE, MONTH, DAY, YEAR`  
Soil core analyses supporting the end-of-year biomass studies at Virginia Coast Reserve megasite marshes (single year, 1999). Cores from 10 sites across five marsh zones measure bulk density, pH, salinity, organic matter %, and water content.

### knb-lter-vcr.247.20 — Integrated water quality, Virginia Coast (1992–2025)
*PI: Karen McGlathery, Sophia Hoffman, Robert Christian*  
`WQ_Integrated.csv` — 2,896 rows  
Columns include: `STATION, SESSIONDATE, LATITUDE, LONGITUDE, TIDESTAGE, SCTTEMP, SCTSAL, DO, SECCHI, CHLA, NH4, PO4, NO3_NO2, TDN, ORGANIC_CONTENT, POROSITY, TSS, POM, PIM, SEDPERC_C, SEDPERC_N` (each analyte has SD, N, and quality-flag columns)  
Long-term water quality at 16 VCR stations: monthly 1998–2008, seasonal 2008–present. Physical parameters (temperature, salinity, dissolved oxygen, Secchi depth), nutrients (NH₄⁺, PO₄³⁻, NO₃/NO₂, TDN), chlorophyll *a*, suspended solids (TSS, POM, PIM), sediment %C and %N, occasional macroalgae (*Gracilaria*, *Ulva*). Covers 33 years of coastal water chemistry.

### knb-lter-vcr.286.6 — Marsh-to-upland vegetation monitoring, coastal Virginia (2018–2022)
*PI: Keryn Gedan, George Washington University*  
`PlotLocations.csv` — 71 rows (site coordinates, zone, habitat)  
`NonWoody.csv` — 3,570 rows; `Shrubs.csv` — 653 rows; `Trees.csv` — 915 rows; `Soil_CN.csv` — 1,023 rows; `Soil_Water_Conductivity.csv` — 88 rows  
Annual vegetation monitoring along marsh-to-upland transects at three VCR sites (Boxtree, Cushmans, GATR). Four transects per site pass through five zones (low marsh → high marsh → ecotone → shrub → forest). Non-woody cover estimated by Domin scale per species; shrubs measured by canopy height and width; trees by DBH and canopy dieback %. Soil %C, %N, and electrical conductivity (as salinity proxy) sampled per zone. Tracks marsh migration under sea-level rise.

### knb-lter-vcr.17.16 — Tidal creek water chemistry, Northampton County (1990–1992)
*PI: Luis Lagera*  
`compodat.asc` — ~1,168 data rows (20-line header)  
Columns: `DATE, TIME, SITEID, REPC, NH4, NO2, NOx, TIN, PO4, CHLA, PHAE, TSSW, TSSM, TSSO, TSSOPER, BOD5`  
Biweekly (summer) to monthly (winter) water samples from multiple tidal creek stations across Northampton County, VA: Old Plantation Creek, Cherrystone Creek, Sand Shoal Channel, Phillips Creek, and Machipongo River. Analytes: NH₄⁺, NO₂⁻, NOₓ, total inorganic N, PO₄³⁻ (all μmol L⁻¹), chlorophyll *a*, phaeophytin (μg L⁻¹), TSS split into mineral and organic fractions (mg L⁻¹), and biochemical oxygen demand (BOD₅). Upstream and downstream stations per creek provide spatial context; tidal stage noted at sampling. No direct sulfate measurements.

### knb-lter-vcr.41.18 — Crab burrows, soil nutrients, and *Spartina* organic carbon, Brownsville VA (1992)
*PI: Winli Lin / Linda Blum*  
`orgc.csv` — 35 data rows (23-line header)  
Columns: `FOIL_WT, FOIL_ASH_DRY_WT, DRY_WT, ASH_DRY_WT, ORG_CARBON`  
Short-term (9-week) split-block experiment testing effects of added fiddler crab (*Uca pugnax*) burrows on porewater chemistry and *Spartina alterniflora* production at No Name Creek, Brownsville, VA. Ten blocks with control vs. treatment subplots; porewater sippers at 10 cm and 20 cm depth sampled weekly for sulfide, NH₄⁺, PO₄³⁻, and salinity (those porewater data are not included in this file). **This file contains only root ingrowth litter-bag organic carbon values** (foil weight, ash dry weight, dry weight, and derived organic carbon). Notable context: sulfide was significantly lower in burrow-addition plots (p < 0.05), and aboveground biomass trended 4% higher, linking sediment aeration to productivity.

### knb-lter-vcr.6.15 — *Spartina alterniflora* decomposition, Phillips Creek Marsh (1988–1990)
*PI: Linda Blum*  
`linda.dat` — 38 rows  
Columns: `DAYS, YEAR, MONTH, DAY, CRKSURF, CRKSSTD, CRKBELO, CRKBSTD, INTSURF, INTSSTD, INTBELO, INTBSTD`  
One-year decomposition experiment using 50 cm litter bags inserted vertically (0–40 cm depth) in Phillips Creek marsh sediment. Compares weight loss at creek bank vs. marsh interior, and surface vs. buried, to evaluate the effect of sediment aeration on *Spartina* decomposition rates.

---

## Quick-reference table

| Dataset ID | Site / Location | Variable(s) | Period | Rows |
|---|---|---|---|---|
| edi.1828.3 | Baruch Inst. (North Inlet), SC | CH4 + CO2 chamber fluxes, redox potential, mcrA gene copies, soil C/N | ~2022–2023 | 1,249 |
| AmeriFlux US-HB1 | North Inlet Crab Haul Creek, SC | Eddy covariance CO2 + CH4 (LI-7700 active 2021–2022); **download manually** from ameriflux.lbl.gov/sites/siteinfo/US-HB1 (free account required) | 2018–2024 | — |
| knb-lter-nin.1.1 | NIN creeks (TC/CB/OL), SC | Creek water NH₄, NO₃, TN, TP, TOC/DOC/POC, salinity, temp | 1978–1992 | 13,176 |
| knb-lter-nin.8.1 | NIN creeks (TC/CB/OL), SC | Creek TSS (organic + inorganic), tide elevation, tide direction | 1981–1992 | 12,364 |
| edi.135.12 | North Inlet, SC | Aboveground biomass, plant density, productivity, heights, snails | 1984–2025 | ~13,844+ |
| edi.136.11 | North Inlet, SC | Porewater NH₄, PO₄, S²⁻, Fe²⁺, salinity (multi-depth) | 1993–2025 | 27,107 |
| knb-lter-pie.519.5 | PIE Rowley creeks, MA | Creek NH₄, NO₃, DIC, DOC, TSS, PN/PC (~15-min tidal cycle) | 2016–2019 | 2,212 |
| knb-lter-pie.120.12 | PIE NIN, MA | Biomass | 1984–2016 | 11,020 |
| knb-lter-pie.32.15 | PIE LPA, MA | Mean biomass & plant density | 1999–2018 | 224 |
| knb-lter-pie.127.12 | PIE LAC, MA | Annual productivity | 1999–2019 | 21 |
| knb-lter-pie.131.15 | PIE LPP, MA | Annual productivity | 2000–2025 | 50 |
| knb-lter-pie.624.2 | PIE Space-for-Time, MA | Peak aboveground biomass across elevation gradient | 2018–2024 | 342 |
| knb-lter-pie.625.1 | PIE Space-for-Time, MA | Soil LOI, bulk density, porewater NH₄, salinity | 2019–2022 | 593 |
| knb-lter-gce.274.18 | GCE 9 sites, GA | Soil bulk density, %C, %N, P, ¹³⁷Cs, ²¹⁰Pb | 2001 | 285 |
| knb-lter-gce.275.18 | GCE 5 sites, GA | Soil CO₂ respiration, soil temperature | 2003–2005 | 406 |
| knb-lter-gce.277.20 | GCE sites 6–9, GA | Root decomposition & in-growth (%C, %N, P) | 2003–2004 | 158 |
| knb-lter-gce.586.30 | GCE SALTEx, GA | Porewater nutrients, C, sulfide, salinity (saltwater intrusion expt) | 2014–2020 | 511+20 |
| knb-lter-gce.645.4 | GCE SALTEx, GA | SET sediment elevation | 2013–2022 | 7,695 |
| knb-lter-gce.759.27 | GCE Altamaha, GA | Biomass, shoot density, flowering | 2012–present | ~1,000 |
| knb-lter-gce.762.17 | GCE multi-site, GA | Veg cover, LAI, root cores, leaf %N/%C | 2016–2021 | ~500 |
| knb-lter-gce.838.3 | GCE SALTEx, GA | Plant % cover (4 dominant species) | 2013–2022 | 1,232 |
| knb-lter-gce.839.3 | GCE SALTEx, GA | PAR above/below canopy | 2014–2022 | 369 |
| knb-lter-gce.842.3 | GCE SALTEx, GA | Belowground root biomass by depth | 2017 | 124 |
| knb-lter-gce.843.5 | GCE Sapelo, GA | Soil %C, %N, bulk density — natural vs. restored marsh | 2020–2022 | 166 |
| knb-lter-gce.858.12 | GCE Sapelo, GA | NEE, ER, GPP (eddy covariance) | 2014–2024 | ~197,000 |
| knb-lter-vcr.17.16 | VCR Northampton creeks, VA | Creek NH₄, NOₓ, TIN, PO₄, TSS, Chl-a, BOD (multiple creeks) | 1990–1992 | ~1,168 |
| knb-lter-vcr.41.18 | VCR Brownsville, VA | Root ingrowth organic carbon (crab burrow experiment) | 1992 | 35 |
| knb-lter-vcr.6.15 | VCR Phillips Creek, VA | *Spartina* litter decomposition (surface vs. buried, creek vs. interior) | 1988–1990 | 38 |
| knb-lter-vcr.148.29 | VCR Phillips Creek, VA | Surface elevation (incremental + cumulative) | 1997–2025 | 999 |
| knb-lter-vcr.170.23 | VCR Phillips Creek, VA | End-of-year biomass at SET plots | 2001–2017 | 941 |
| knb-lter-vcr.171.22 | VCR Phillips Creek, VA | End-of-year biomass, marsh transition plots | 2001–2016 | 273 |
| knb-lter-vcr.189.16 | VCR megasite marshes, VA | Soil bulk density, pH, salinity, organic %, water % | 1999 | 634 |
| knb-lter-vcr.247.20 | VCR 16 stations, VA | Water quality: nutrients, Chl-a, TSS, sediment C/N | 1992–2025 | 2,896 |
| knb-lter-vcr.286.6 | VCR marsh-to-upland, VA | Veg cover, shrub/tree structure, soil C/N, soil salinity | 2018–2022 | ~6,250 |
