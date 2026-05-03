# Run with
# python3 download_3dhp_bbox.py --bbox -79.29387360027498 33.252629271294964 -79.14606055065641 33.3740440497309 --out-prefix oregon_test --format fgb


#!/usr/bin/env python3



import argparse
import sys
import requests
import geopandas as gpd

BASE_URL = "https://3dhp.nationalmap.gov/arcgis/rest/services/usgs_3dhp_all/FeatureServer"

LAYERS = {
    "flowline": 50,
    "waterbody": 60,
}

DEFAULT_PAGE_SIZE = 2000


def query_feature_layer(layer_id, bbox, page_size=DEFAULT_PAGE_SIZE, timeout=120):
    """
    Query an ArcGIS FeatureServer layer by bounding box, with pagination.

    bbox = (xmin, ymin, xmax, ymax) in EPSG:4326
    """
    xmin, ymin, xmax, ymax = bbox
    url = f"{BASE_URL}/{layer_id}/query"

    all_features = []
    offset = 0

    while True:
        params = {
            "f": "geojson",
            "where": "1=1",
            "geometry": f"{xmin},{ymin},{xmax},{ymax}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": 4326,
            "resultOffset": offset,
            "resultRecordCount": page_size,
        }

        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()

        if "error" in data:
            raise RuntimeError(f"ArcGIS server error on layer {layer_id}: {data['error']}")

        features = data.get("features", [])
        print(f"Layer {layer_id}: fetched {len(features)} features at offset {offset}")

        if not features:
            break

        all_features.extend(features)

        if len(features) < page_size:
            break

        offset += page_size

    if not all_features:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    return gpd.GeoDataFrame.from_features(all_features, crs="EPSG:4326")


def main():
    parser = argparse.ArgumentParser(
        description="Download USGS 3DHP flowline and waterbody data for a bounding box."
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        required=True,
        help="Bounding box in EPSG:4326"
    )
    parser.add_argument(
        "--out-gpkg",
        default=None,
        help="Output GeoPackage path. If given, writes both layers into one GPKG."
    )
    parser.add_argument(
        "--out-prefix",
        default="3dhp",
        help="Output prefix for separate files if --out-gpkg is not used"
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help="Records per request (default: 2000)"
    )
    parser.add_argument(
        "--format",
        choices=["gpkg", "fgb"],
        default="fgb",
        help="Output format for separate files when --out-gpkg is not used"
    )

    args = parser.parse_args()
    bbox = tuple(args.bbox)

    print("Downloading flowline...")
    flowline_gdf = query_feature_layer(LAYERS["flowline"], bbox, page_size=args.page_size)

    print("Downloading waterbody...")
    waterbody_gdf = query_feature_layer(LAYERS["waterbody"], bbox, page_size=args.page_size)

    print(f"Flowline features: {len(flowline_gdf)}")
    print(f"Waterbody features: {len(waterbody_gdf)}")

    if args.out_gpkg:
        flowline_gdf.to_file(args.out_gpkg, layer="flowline", driver="GPKG")
        waterbody_gdf.to_file(args.out_gpkg, layer="waterbody", driver="GPKG")
        print(f"Wrote GeoPackage: {args.out_gpkg}")
    else:
        if args.format == "fgb":
            flowline_out = f"{args.out_prefix}_flowline.fgb"
            waterbody_out = f"{args.out_prefix}_waterbody.fgb"
            flowline_gdf.to_file(flowline_out, driver="FlatGeobuf")
            waterbody_gdf.to_file(waterbody_out, driver="FlatGeobuf")
        else:
            flowline_out = f"{args.out_prefix}_flowline.gpkg"
            waterbody_out = f"{args.out_prefix}_waterbody.gpkg"
            flowline_gdf.to_file(flowline_out, driver="GPKG")
            waterbody_gdf.to_file(waterbody_out, driver="GPKG")

        print(f"Wrote: {flowline_out}")
        print(f"Wrote: {waterbody_out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
