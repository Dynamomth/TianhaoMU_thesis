# 1.clipped EU-27 coutries map
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

# 2.tif->csv
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

# 3.match with NUTS-0
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

# 4.LandCoverRaster1km
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
import rasterio
import numpy as np
from simpledbf import Dbf5
import os
from osgeo import gdal  # GDAL is required for embedding the legend


# --- Helper function to convert Hex color to RGB tuple ---
def hex_to_rgb(hex_color):
    """Converts a hex color string to an (R, G, B) tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def embed_legend_in_raster(raster_path, dbf_path):
    """
    Creates and embeds a Value Attribute Table (VAT) and a Color Table
    into an existing raster file using a DBF file as a source for labels.

    Args:
        raster_path (str): Path to the target GeoTIFF file (with correct 111, 211... values).
        dbf_path (str): Path to the .vat.dbf file to get the 'Value' -> 'LABEL3' mapping.
    """
    # --- This section maps the official CORINE colors to the class codes ---
    # We use this predefined dictionary for color accuracy
    color_map_hex = {
        111: '#E6004D', 112: '#FF0000', 121: '#CC4DF2', 122: '#CC0000', 123: '#E6CCCC',
        124: '#E6CCE6', 131: '#A600CC', 132: '#A64DCC', 133: '#FF4DFF', 141: '#FFA6FF',
        142: '#FFE6FF', 211: '#FFFFA8', 212: '#FFFF00', 213: '#E6E600', 221: '#E68000',
        222: '#F2A64D', 223: '#E6A600', 231: '#E6E64D', 241: '#FFE6A6', 242: '#FFEE00',
        243: '#FFFFA6', 244: '#E6E6A6', 311: '#80FF00', 312: '#00A600', 313: '#4DFF00',
        321: '#CCFF4D', 322: '#A6FF80', 323: '#A6E64D', 324: '#A6F200', 331: '#E6E6E6',
        332: '#CCCCCC', 333: '#CCF2CC', 334: '#000000', 335: '#A6FFFF', 411: '#4DE6F2',
        412: '#00A6A6', 421: '#A6E6CC', 422: '#A6F2E6', 423: '#E6F2F2', 511: '#00CCFF',
        512: '#0080FF', 521: '#A6CCFF', 522: '#80A6F2', 523: '#004DFF'
    }

    print("\nStarting legend embedding process...")
    # --- Read labels from the DBF file ---
    dbf = Dbf5(dbf_path)
    df = dbf.to_dataframe()
    # Create a dictionary for Value -> LABEL3 mapping
    label_lookup = dict(zip(df['CODE_18'].astype(int), df['LABEL3']))

    # --- Open the raster in update mode with GDAL ---
    dataset = gdal.Open(raster_path, gdal.GA_Update)
    if dataset is None:
        print(f"Error: GDAL could not open {raster_path} for updating.")
        return
    band = dataset.GetRasterBand(1)

    # --- Create and set a Color Table ---
    print("Creating and applying color table...")
    colors = gdal.ColorTable()
    for code, hex_val in color_map_hex.items():
        r, g, b = hex_to_rgb(hex_val)
        colors.SetColorEntry(code, (r, g, b, 255))  # RGBA
    band.SetColorTable(colors)

    # --- Create and set a Raster Attribute Table (for labels) ---
    print("Creating and applying raster attribute table (VAT)...")
    rat = gdal.RasterAttributeTable()
    rat.CreateColumn("Value", gdal.GFT_Integer, gdal.GFU_Generic)
    rat.CreateColumn("LABEL3", gdal.GFT_String, gdal.GFU_Generic)
    # Populate the RAT
    row_idx = 0
    for code, label in label_lookup.items():
        rat.SetValueAsInt(row_idx, 0, code)
        rat.SetValueAsString(row_idx, 1, label)
        row_idx += 1
    band.SetDefaultRAT(rat)

    # --- Flush changes to disk and close ---
    dataset = None
    print("Legend embedding complete.")


# Week14-2 processes on_irrigation data

import os
import glob
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
import numpy as np

# --- CONFIGURATION ---
# Please change these paths to match your local file system.

# 1. Directory containing your source 100m .tif files.
INPUT_DIR = r"C:\path\to\your\GMIE_tiles"

# 2. Path to the EU27 boundary vector file (e.g., .shp or .gpkg).
SHAPEFILE_PATH = r"C:\path\to\your\EU27_boundary.shp"

# 3. Directory where output files will be saved.
OUTPUT_DIR = r"C:\path\to\your\output_data"

# 4. Aggregation factor. To go from 100m to 1km (1000m), the factor is 10.
AGG_FACTOR = 10


# --- END OF CONFIGURATION ---

def merge_rasters(input_dir, output_path):
    """
    Merges all GeoTIFF files in a directory into a single file.
    """
    print("--- Step 1: Merging rasters ---")

    # Find all .tif files in the input directory
    search_criteria = "*.tif"
    q = os.path.join(input_dir, search_criteria)
    src_files_to_mosaic = glob.glob(q)

    if not src_files_to_mosaic:
        print(f"No .tif files found in {input_dir}. Exiting.")
        return False

    print(f"Found {len(src_files_to_mosaic)} files to merge.")

    # Open all source files
    src_datasets = [rasterio.open(fp) for fp in src_files_to_mosaic]

    # Merge function returns a tuple (merged_array, output_transform)
    mosaic, out_trans = merge(src_datasets)

    # Copy metadata from the first file and update it
    out_meta = src_datasets[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
    })

    # Write the merged raster to disk
    print(f"Saving merged file to: {output_path}")
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    # Close all source datasets
    for src in src_datasets:
        src.close()

    print("Merge complete.")
    return True


def aggregate_raster(input_path, output_path, factor):
    """
    Aggregates a raster by a given factor using the mean.
    This function resamples the data to a lower resolution.
    """
    print("\n--- Step 2: Aggregating resolution ---")
    print(f"Input file: {input_path}")
    print(f"Aggregation factor: {factor}")

    with rasterio.open(input_path) as src:
        # Get metadata and nodata value
        meta = src.meta.copy()
        nodata = src.nodata

        # Resample data to a new resolution
        # The data is read into an array with a new shape
        data = src.read(
            out_shape=(
                src.count,
                int(src.height / factor),
                int(src.width / factor)
            ),
            resampling=Resampling.average  # Use average for aggregation (similar to mean)
        )

        # Update the transform (affine) matrix for the new resolution
        out_transform = src.transform * src.transform.scale(
            (src.width / data.shape[-1]),
            (src.height / data.shape[-2])
        )

        # Update metadata for the output file
        meta.update({
            "height": data.shape[-2],
            "width": data.shape[-1],
            "transform": out_transform
        })

        # Write the aggregated raster to disk
        print(f"Saving aggregated file to: {output_path}")
        with rasterio.open(output_path, "w", **meta) as dest:
            dest.write(data)
            if nodata is not None:
                dest.nodata = nodata

    print("Aggregation complete.")


def clip_raster_by_mask(input_path, mask_path, output_path):
    """
    Clips a raster file using a vector mask.
    """
    print("\n--- Step 3: Clipping raster by mask ---")
    print(f"Input raster: {input_path}")
    print(f"Mask vector: {mask_path}")

    # Read the vector mask using geopandas
    try:
        mask_gdf = gpd.read_file(mask_path)
    except Exception as e:
        print(f"Could not read shapefile: {e}")
        return

    with rasterio.open(input_path) as src:
        # Ensure the mask's CRS matches the raster's CRS
        if mask_gdf.crs != src.crs:
            print("CRS mismatch. Reprojecting mask to match raster CRS...")
            mask_gdf = mask_gdf.to_crs(src.crs)

        # Get geometries from the geodataframe
        geoms = mask_gdf.geometry.values

        # Clip the raster with the mask
        # The 'mask' function returns a tuple (clipped_array, output_transform)
        out_image, out_transform = mask(src, geoms, crop=True)

        out_meta = src.meta.copy()

        # Update metadata for the clipped raster
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        # Write the final clipped raster to disk
        print(f"Saving final clipped file to: {output_path}")
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)

    print("Clipping complete.")


if __name__ == '__main__':
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Define full file paths for intermediate and final products
    merged_fp = os.path.join(OUTPUT_DIR, "GMIE_Europe_100m_merged.tif")
    aggregated_fp = os.path.join(OUTPUT_DIR, "GMIE_Europe_1km_aggregated.tif")
    final_fp = os.path.join(OUTPUT_DIR, "GMIE_EU27_1km_final.tif")

    # --- Execute Workflow ---

    # Step 1: Merge all input tiles
    if merge_rasters(INPUT_DIR, merged_fp):
        # Step 2: Aggregate the merged raster
        aggregate_raster(merged_fp, aggregated_fp, factor=AGG_FACTOR)

        # Step 3: Clip the aggregated raster to the EU27 boundary
        clip_raster_by_mask(aggregated_fp, SHAPEFILE_PATH, final_fp)

        print("\nWorkflow successfully completed!")
        print(f"Final output saved at: {final_fp}")

# week 15-1 (manure application-cattle->standardization by Eurostat)
import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np
from rasterio.features import rasterize
import os

# === Input paths ===
cattle_raster_path = r'D:\ManureAppThesis\AnimalDistribution\cattle\Glb_Cattle_CC2006_AD.tif'
nuts0_shp_path = r'D:\ManureAppThesis\01M_NUTS0_WSG84_4326\NUTS_RG_01M_2021_4326_LEVL_0.shp'
standardization_csv = r'D:\ManureAppThesis\AnimalDistribution\Comparison_GLIMS\DairyCow&Bovine_csv.csv'
output_path = r'D:\ManureAppThesis\standardized_cattle_1km.tif'

# === Read original cattle distribution raster ===
with rasterio.open(cattle_raster_path) as src:
    cow_data = src.read(1)
    cow_meta = src.meta.copy()
    transform = src.transform
    raster_shape = src.shape
    raster_crs = src.crs

# === Read NUTS0 boundaries and reproject to match raster CRS ===
nuts = gpd.read_file(nuts0_shp_path)
nuts = nuts.to_crs(raster_crs)

# === Read CSV and merge attributes ===
factors = pd.read_csv(standardization_csv)
nuts = nuts.merge(factors, on='NUTS_ID', how='left')

# === Rasterize scale factor (assigned by country) ===
scale_factor_raster = rasterize(
    [(row.geometry, row['scalefactor_Cattle']) for idx, row in nuts.iterrows()],
    out_shape=raster_shape,
    transform=transform,
    fill=1.0,
    dtype='float32'
)

# === Rasterize NUTS0 mask ===
nuts_mask_raster = rasterize(
    [(geom, 1) for geom in nuts.geometry],
    out_shape=raster_shape,
    transform=transform,
    fill=0,
    dtype='uint8'
)

# === Calculate adjusted cattle raster ===
adjusted_cattle = cow_data.astype('float32') * scale_factor_raster

# === Mask out non-NUTS areas (set areas outside region to 0) ===
adjusted_cattle[nuts_mask_raster == 0] = 0

# === Replace abnormal values ===
adjusted_cattle = np.nan_to_num(adjusted_cattle, nan=0.0, posinf=0.0, neginf=0.0)

# === Update metadata and write output ===
cow_meta.update({
    'dtype': 'float32',
    'compress': 'lzw'
})

with rasterio.open(output_path, 'w', **cow_meta) as dst:
    dst.write(adjusted_cattle, 1)

print("✅ Successfully generated the calibrated cattle distribution map for Europe only:", output_path)

# week 15-2 calculation of manure production
import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np
from rasterio.features import rasterize

# === Input paths ===
calibrated_cattle_raster = r'D:\ManureAppThesis\standardized_cattle_1km.tif'  # Calibrated cattle distribution raster
nuts0_shp_path = r'D:\ManureAppThesis\01M_NUTS0_WSG84_4326\NUTS_RG_01M_2021_4326_LEVL_0.shp'
standardization_csv = r'D:\ManureAppThesis\AnimalDistribution\Comparison_GLIMS\DairyCow&Bovine_csv.csv'  # Contains excretion_rate
output_manure_path = r'D:\ManureAppThesis\manure_production_1km.tif'

# === Read the calibrated cattle distribution raster ===
with rasterio.open(calibrated_cattle_raster) as src:
    cattle_data = src.read(1)
    meta = src.meta.copy()
    transform = src.transform
    shape = src.shape
    crs = src.crs

# === Read NUTS0 boundaries and align projection ===
nuts = gpd.read_file(nuts0_shp_path)
nuts = nuts.to_crs(crs)

# === Read CSV and merge attributes ===
factors = pd.read_csv(standardization_csv)
nuts = nuts.merge(factors, on='NUTS_ID', how='left')

# === Rasterize excretion rates ===
excretion_rate_raster = rasterize(
    [(row.geometry, row['excretion_cattle(total)']) for idx, row in nuts.iterrows()],
    out_shape=shape,
    transform=transform,
    fill=0.0,
    dtype='float32'
)

# === Calculate manure production ===
manure_production = cattle_data.astype('float32') * excretion_rate_raster
manure_production = np.nan_to_num(manure_production, nan=0.0, posinf=0.0, neginf=0.0)

# === Write the result ===
meta.update(dtype='float32', compress='lzw')

with rasterio.open(output_manure_path, 'w', **meta) as dst:
    dst.write(manure_production, 1)

print("✅ Manure production raster calculation completed and saved to:", output_manure_path)
