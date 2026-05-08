"""
build_lter_sites_fgb.py

Compile LTER site locations from metadata into:
  - lter_data/lter_sites.fgb   (FlatGeobuf, all sites/plots with attributes)
  - lter_data/lter_observatories.csv  (4-row CSV, one point per observatory,
                                       for use with query_cores_by_location.py)

Run from the repo root:
    python scripts/build_lter_sites_fgb.py
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# ---------------------------------------------------------------------------
# Site records
# Each dict: site_code, site_name, lter_site, dataset_ids,
#            latitude, longitude, location_type,
#            elevation_m (optional), elevation_datum (optional), notes
# ---------------------------------------------------------------------------
SITES = [

    # ------------------------------------------------------------------
    # North Inlet LTREB / PIE, Georgetown SC
    # Coordinates from knb-lter-pie.120.12 metadata (EML bounding coords)
    # Elevations from the same metadata file (NAVD88)
    # ------------------------------------------------------------------
    dict(
        site_code="NI-GIHM",
        site_name="Goat Island High Marsh",
        lter_site="North_Inlet_SC",
        dataset_ids="edi.135.12; edi.136.11; knb-lter-pie.120.12",
        latitude=33.331659,
        longitude=-79.198208,
        location_type="plot",
        elevation_m=0.50,
        elevation_datum="NAVD88",
        distance_to_creek_m=62.0,
        notes="High marsh Spartina alterniflora; N+P fertilisation experiment (GIHM)",
    ),
    dict(
        site_code="NI-GILM",
        site_name="Goat Island Low Marsh",
        lter_site="North_Inlet_SC",
        dataset_ids="edi.135.12; edi.136.11; knb-lter-pie.120.12",
        latitude=33.331603,
        longitude=-79.197942,
        location_type="plot",
        elevation_m=0.20,
        elevation_datum="NAVD88",
        distance_to_creek_m=42.0,
        notes="Low marsh Spartina alterniflora (GILM)",
    ),
    dict(
        site_code="NI-OLHM",
        site_name="Oyster Landing High Marsh",
        lter_site="North_Inlet_SC",
        dataset_ids="edi.135.12; edi.136.11; knb-lter-pie.120.12",
        latitude=33.351389,
        longitude=-79.191944,
        location_type="plot",
        elevation_m=0.45,
        elevation_datum="NAVD88",
        distance_to_creek_m=33.0,
        notes="High marsh Spartina alterniflora, north side of causeway (OLHM)",
    ),
    dict(
        site_code="NI-OLLM",
        site_name="Oyster Landing Low Marsh",
        lter_site="North_Inlet_SC",
        dataset_ids="edi.135.12; edi.136.11; knb-lter-pie.120.12",
        latitude=33.348102,
        longitude=-79.189171,
        location_type="plot",
        elevation_m=0.25,
        elevation_datum="NAVD88",
        distance_to_creek_m=33.0,
        notes="Low marsh Spartina alterniflora, south of pier (OLLM)",
    ),

    # ------------------------------------------------------------------
    # Plum Island Ecosystems LTER, Law's Point, Rowley River MA
    # Coordinates from EML bounding coords in each dataset's metadata
    # Elevations from PIE 131 metadata narrative
    # ------------------------------------------------------------------
    dict(
        site_code="PIE-LPA",
        site_name="Law's Point Low Marsh (LPA)",
        lter_site="PIE_MA",
        dataset_ids="knb-lter-pie.32.15; knb-lter-pie.127.12",
        latitude=42.731743,
        longitude=-70.842468,
        location_type="plot",
        elevation_m=1.17,
        elevation_datum="NAVD88 (2019)",
        notes="Low marsh Spartina alterniflora; N+P fertilisation experiment",
    ),
    dict(
        site_code="PIE-LPP",
        site_name="Law's Point High Marsh (LPP)",
        lter_site="PIE_MA",
        dataset_ids="knb-lter-pie.131.15",
        latitude=42.730954,
        longitude=-70.842915,
        location_type="plot",
        elevation_m=1.46,
        elevation_datum="NAVD88 (2019)",
        notes="High marsh Spartina patens; N+P fertilisation experiment",
    ),

    # ------------------------------------------------------------------
    # Georgia Coastal Ecosystems LTER
    # GCE-758 / 759: Altamaha River plant-transition sites (KML polygon centroids)
    # GCE-762: plot-level coords from PLT-GCED-2106_coords_1_0.CSV
    # GCE-858: flux tower point from MSH-GCEM-2508.kml
    # ------------------------------------------------------------------

    # GCE 759 – Altamaha River transition sites
    dict(
        site_code="GCE-SCSA",
        site_name="SCSA Altamaha Plant Transition Site",
        lter_site="GCE_GA",
        dataset_ids="knb-lter-gce.759.27",
        latitude=31.322985,
        longitude=-81.374147,
        location_type="site_centroid",
        elevation_m=None,
        elevation_datum=None,
        notes="Broughton Island; Spartina alterniflora / S. cynosuroides transition; ~9 km upriver",
    ),
    dict(
        site_code="GCE-ZSC1",
        site_name="ZSC1 Altamaha Plant Transition Site",
        lter_site="GCE_GA",
        dataset_ids="knb-lter-gce.759.27",
        latitude=31.328070,
        longitude=-81.450924,
        location_type="site_centroid",
        elevation_m=None,
        elevation_datum=None,
        notes="Champney Island; S. cynosuroides / Zizaniopsis miliacea transition; ~20 km upriver",
    ),
    dict(
        site_code="GCE-ZSC2",
        site_name="ZSC2 Altamaha Plant Transition Site",
        lter_site="GCE_GA",
        dataset_ids="knb-lter-gce.759.27",
        latitude=31.340693,
        longitude=-81.453783,
        location_type="site_centroid",
        elevation_m=None,
        elevation_datum=None,
        notes="Butler Island / Champney River; S. cynosuroides / Zizaniopsis transition; ~20 km upriver",
    ),

    # GCE 762 – four sub-sites with plot-level coordinates
    dict(
        site_code="GCE-FLUXA",
        site_name="Flux Tower A (GCE 762)",
        lter_site="GCE_GA",
        dataset_ids="knb-lter-gce.762.17",
        latitude=31.450238,
        longitude=-81.284726,
        location_type="plot_representative",
        elevation_m=None,
        elevation_datum=None,
        notes="Representative plot t1; Spartina alterniflora near GCE flux tower",
    ),
    dict(
        site_code="GCE-FLUXB",
        site_name="Flux Tower B (GCE 762)",
        lter_site="GCE_GA",
        dataset_ids="knb-lter-gce.762.17",
        latitude=31.441526,
        longitude=-81.277428,
        location_type="plot_representative",
        elevation_m=None,
        elevation_datum=None,
        notes="Representative plot fb1; Spartina alterniflora",
    ),
    dict(
        site_code="GCE-SKIDA",
        site_name="Skidaway (GCE 762)",
        lter_site="GCE_GA",
        dataset_ids="knb-lter-gce.762.17",
        latitude=31.986534,
        longitude=-81.030370,
        location_type="plot_representative",
        elevation_m=None,
        elevation_datum=None,
        notes="Representative plot sk1; Spartina alterniflora, Skidaway Island",
    ),
    dict(
        site_code="GCE-UGAMI",
        site_name="UGAMI Marsh (GCE 762)",
        lter_site="GCE_GA",
        dataset_ids="knb-lter-gce.762.17",
        latitude=31.399591,
        longitude=-81.281787,
        location_type="plot_representative",
        elevation_m=None,
        elevation_datum=None,
        notes="Representative plot ug1; Spartina alterniflora, UGAMI",
    ),

    # GCE 858 – eddy covariance flux tower
    dict(
        site_code="GCE-FLUX-TOWER",
        site_name="GCE Eddy Covariance Flux Tower",
        lter_site="GCE_GA",
        dataset_ids="knb-lter-gce.858.12",
        latitude=31.444094,
        longitude=-81.283461,
        location_type="observatory",
        elevation_m=None,
        elevation_datum=None,
        notes="Sapelo Island; Spartina alterniflora marsh; 187 ha footprint; NEE/ER/GPP 2014-2024",
    ),

    # ------------------------------------------------------------------
    # Virginia Coast Reserve LTER, Upper Phillips Creek
    # More precise coords from knb-lter-vcr.148.29 metadata (37.485744, -75.8325)
    # VCR data tables only record 37.46, -75.833 (rounded)
    # ------------------------------------------------------------------
    dict(
        site_code="VCR-UPM",
        site_name="Upper Phillips Creek Marsh",
        lter_site="VCR_VA",
        dataset_ids="knb-lter-vcr.148.29; knb-lter-vcr.170.23; knb-lter-vcr.171.22",
        latitude=37.485744,
        longitude=-75.8325,
        location_type="observatory",
        elevation_m=None,
        elevation_datum=None,
        notes="Eastern Shore VA; surface elevation (SET), end-of-year biomass at SET and transition plots",
    ),
]

# Representative point per observatory (for core searching)
OBSERVATORIES = [
    dict(
        lter_site="North_Inlet_SC",
        latitude=33.340,   # centroid of the 4 NI plots
        longitude=-79.194,
    ),
    dict(
        lter_site="PIE_MA",
        latitude=42.731,
        longitude=-70.843,
    ),
    dict(
        lter_site="GCE_GA",
        latitude=31.444,   # flux tower
        longitude=-81.283,
    ),
    dict(
        lter_site="VCR_VA",
        latitude=37.486,
        longitude=-75.833,
    ),
]


def build_fgb(out_path: Path):
    rows = []
    for s in SITES:
        row = dict(s)
        row["geometry"] = Point(s["longitude"], s["latitude"])
        rows.append(row)

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    # drop bare lat/lon columns — geometry carries the location
    gdf = gdf.drop(columns=["latitude", "longitude"])
    gdf.to_file(out_path, driver="FlatGeobuf")
    print(f"Wrote {len(gdf)} features to {out_path}")
    return gdf


def build_observatory_csv(out_path: Path):
    df = pd.DataFrame(OBSERVATORIES)[["latitude", "longitude", "lter_site"]]
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    return df


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    fgb_path = root / "lter_data" / "lter_sites.fgb"
    csv_path = root / "lter_data" / "lter_observatories.csv"

    gdf = build_fgb(fgb_path)
    print(gdf[["site_code", "lter_site", "location_type", "elevation_m"]].to_string())
    print()
    df = build_observatory_csv(csv_path)
    print(df.to_string(index=False))
