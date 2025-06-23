#Week13-1.clipped EU-27 coutries map
import arcpy
from arcpy.sa import ExtractByMask
arcpy.CheckOutExtension("Spatial")

# path
input_raster = r"D:\work\database\arcgis_pro\data_manure_app\Glb_Cattle_CC2006_AD.tif"
eu_shapefile = r"D:\work\database\arcgis_pro\EU-27_4326\CNTR_RG_10M_2024_4326.shp"
output_raster = r"D:\work\database\arcgis_pro\data_manure_app\EU-27_clipped.tif"

# clipping
out_extract = ExtractByMask(in_raster=input_raster, in_mask_data=eu_shapefile)

# save
out_extract.save(output_raster)

print("finished, save as:", output_raster)

#2.Week13-tif->csv
import os
import rasterio
import csv

# data input
input_tif = r"D:\manure_app_thesis\clipped\EU-27_clipped.tif"
output_folder = r"D:\manure_app_thesis\Raster_point"
output_name = "livestock_data.csv"
output_path = os.path.join(output_folder, output_name)

os.makedirs(output_folder, exist_ok=True)

# read and output data
with rasterio.open(input_tif) as src, open(output_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['X', 'Y', 'Value'])  # column title

    band = src.read(1)
    transform = src.transform
    nodata = src.nodata
    rows, cols = band.shape

    count = 0
    for row in range(rows):
        for col in range(cols):
            value = band[row, col]
            if value != nodata:
                x, y = transform * (col + 0.5, row + 0.5)
                writer.writerow([x, y, value])
                count += 1

print("finished，extracted {} grids，save as：\n{}".format(count, output_path))

#Week13-3.match with NUTS-2
import geopandas as gpd
from rasterstats import zonal_stats
import pandas as pd

# Load NUTS-0 (country-level) boundary shapefile
# Make sure this is a polygon layer (e.g., NUTS_RG_01M_2016_4326_LEVL_0.shp)
nuts0 = gpd.read_file(r'D:\manure_app_thesis\01M_NUTS0\NUTS_RG_01M_2021_4326_LEVL_0.shp')

# Filter for NUTS level 0 (national level)
nuts0 = nuts0[nuts0['LEVL_CODE'] == 0]

# Perform zonal statistics: calculate the total number of cattle within each country
stats = zonal_stats(
    vectors=nuts0,  # vector polygons (countries)
    raster=r'D:\manure_app_thesis\poultry\duck\GLb_Ducks_CC2006_AD.tif',  # raster with cattle counts
    stats=['sum'],  # compute the sum of values within each polygon
    geojson_out=True  # return output as GeoJSON-like features
)

# Convert the results to a GeoDataFrame
nuts0_stats = gpd.GeoDataFrame.from_features(stats)

# Extract useful columns: NUTS_ID (country code) and cattle count sum
result = nuts0_stats[['NUTS_ID', 'sum']].rename(columns={'sum': 'cattle_sum'})

# Export the result to CSV
result.to_csv(r'D:\manure_app_thesis\Raster_point\nuts0_duck_sum.csv', index=False)

#Week14-1.LandCoverRaster1km
import rasterio
from rasterio.enums import Resampling
import numpy as np
import os

# ====== Set input and output paths ======
input_path = r"D:\ManureAppThesis\CORINERaster100m\DATA\CORINERaster100m2020.tif"  # Please change to your .tif file path
output_path = r"D:\ManureAppThesis\CORINERaster100m2020_1km.tif"

# ====== Set aggregation factor (10x downsampling) ======
scale_factor = 10

# ====== Open source .tif file ======
with rasterio.open(input_path) as src:
    # Calculate the new shape for output
    out_height = src.height // scale_factor
    out_width = src.width // scale_factor
    out_shape = (src.count, out_height, out_width)

    # Resample data (mode)
    data = src.read(
        out_shape=out_shape,
        resampling=Resampling.mode  # Aggregate each 10x10 pixels by mode
    )

    # Calculate new transform matrix (georeferencing)
    transform = src.transform * src.transform.scale(
        (src.width / out_width),
        (src.height / out_height)
    )

    # ====== Write to output .tif file ======
    profile = src.profile
    profile.update({
        "height": out_height,
        "width": out_width,
        "transform": transform,
        "compress": "lzw"  # Optional compression to save space
    })

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data)

print("Aggregation finished, output file path:", output_path)

#Week14-2








