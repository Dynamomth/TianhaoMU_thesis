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

# week 15-1 cattle distribution from CLIMS->standardization by Eurostat (example:cattle)
import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np
from rasterio.features import rasterize
import os

# === Input paths ===
cattle_raster_path = r'D:\ManureAppThesis\AnimalDistribution\goat\Glb_bovine_GTAD_2006.tif'
nuts0_shp_path = r'D:\ManureAppThesis\01M_NUTS0_WSG84_4326\NUTS_RG_01M_2021_4326_LEVL_0.shp'
standardization_csv = r'D:\ManureAppThesis\ManureProduction\bovine_csv.csv'
output_path = r'D:\ManureAppThesis\standardized_bovine_1km.tif'

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
    [(row.geometry, row['ScaleFactor_bovine']) for idx, row in nuts.iterrows()],
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

print("Successfully generated the calibrated cattle distribution map for Europe only:", output_path)

# week 15-2 calculation of manure production (example: cattle)
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

print("Manure production raster calculation completed and saved to:", output_manure_path)


# for sheep
import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np
from rasterio.features import rasterize
import os

# === Input paths ===
cattle_raster_path = r'D:\ManureAppThesis\AnimalDistribution\sheep\Glb_sheep_SHAD_2006.tif'
nuts0_shp_path = r'D:\ManureAppThesis\01M_NUTS0_WSG84_4326\NUTS_RG_01M_2021_4326_LEVL_0.shp'
standardization_csv = r'D:\ManureAppThesis\ManureProduction\DairyCow&Bovine_csv.csv'
output_path = r'D:\ManureAppThesis\standardized_sheep_1km.tif'

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
    [(row.geometry, row['ScaleFactor_sheep']) for idx, row in nuts.iterrows()],
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

print("Successfully generated the calibrated cattle distribution map for Europe only:", output_path)


import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np
from rasterio.features import rasterize

# === Input paths ===
calibrated_cattle_raster = r'D:\ManureAppThesis\standardized_sheep_1km.tif'  # Calibrated livestock distribution map
nuts0_shp_path = r'D:\ManureAppThesis\01M_NUTS0_WSG84_4326\NUTS_RG_01M_2021_4326_LEVL_0.shp'
standardization_csv = r'D:\ManureAppThesis\ManureProduction\DairyCow&Bovine_csv.csv'  # Contains excretion_rate
output_manure_path = r'D:\ManureAppThesis\sheep_manure_production_1km.tif'

# === Read calibrated livestock distribution raster ===
with rasterio.open(calibrated_cattle_raster) as src:
    cattle_data = src.read(1)
    meta = src.meta.copy()
    transform = src.transform
    shape = src.shape
    crs = src.crs

# === Read NUTS0 boundaries and reproject to match raster ===
nuts = gpd.read_file(nuts0_shp_path)
nuts = nuts.to_crs(crs)

# === Read CSV and merge attributes ===
factors = pd.read_csv(standardization_csv)
nuts = nuts.merge(factors, on='NUTS_ID', how='left')

# === Rasterize the excretion rate ===
excretion_rate_raster = rasterize(
    [(row.geometry, row['excretion_sheep']) for idx, row in nuts.iterrows()],
    out_shape=shape,
    transform=transform,
    fill=0.0,
    dtype='float32'
)

# === Calculate manure production ===
manure_production = cattle_data.astype('float32') * excretion_rate_raster
manure_production = np.nan_to_num(manure_production, nan=0.0, posinf=0.0, neginf=0.0)

# === Write the result to output raster ===
meta.update(dtype='float32', compress='lzw')

with rasterio.open(output_manure_path, 'w', **meta) as dst:
    dst.write(manure_production, 1)

print("Manure production raster calculation completed and saved to:", output_manure_path)



# for swine
import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np
from rasterio.features import rasterize
import os

# === Input paths ===
cattle_raster_path = r'D:\ManureAppThesis\AnimalDistribution\swine\Glb_swine_CC2006_AD.tif'
nuts0_shp_path = r'D:\ManureAppThesis\01M_NUTS0_WSG84_4326\NUTS_RG_01M_2021_4326_LEVL_0.shp'
standardization_csv = r'D:\ManureAppThesis\ManureProduction\DairyCow&Bovine_csv.csv'
output_path = r'D:\ManureAppThesis\standardized_swine_1km.tif'

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
    [(row.geometry, row['ScaleFactor_swine']) for idx, row in nuts.iterrows()],
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

print("Successfully generated the calibrated cattle distribution map for Europe only:", output_path)


import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np
from rasterio.features import rasterize

# === Input paths ===
calibrated_cattle_raster = r'D:\ManureAppThesis\standardized_cattle_1km.tif'  # Calibrated livestock distribution map
nuts0_shp_path = r'D:\ManureAppThesis\01M_NUTS0_WSG84_4326\NUTS_RG_01M_2021_4326_LEVL_0.shp'
standardization_csv = r'D:\ManureAppThesis\ManureProduction\DairyCow&Bovine_csv.csv'  # Contains excretion_rate
output_manure_path = r'D:\ManureAppThesis\cattle_manure_production_1km.tif'

# === Read calibrated livestock distribution raster ===
with rasterio.open(calibrated_cattle_raster) as src:
    cattle_data = src.read(1)
    meta = src.meta.copy()
    transform = src.transform
    shape = src.shape
    crs = src.crs

# === Read NUTS0 boundaries and reproject to match raster ===
nuts = gpd.read_file(nuts0_shp_path)
nuts = nuts.to_crs(crs)

# === Read CSV and merge attributes ===
factors = pd.read_csv(standardization_csv)
nuts = nuts.merge(factors, on='NUTS_ID', how='left')

# === Rasterize the excretion rate ===
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

# === Write the result to output raster ===
meta.update(dtype='float32', compress='lzw')

with rasterio.open(output_manure_path, 'w', **meta) as dst:
    dst.write(manure_production, 1)

print("Manure production raster calculation completed and saved to:", output_manure_path)







# For chicken
import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np
from rasterio.features import rasterize
import os

# === Input paths ===
cattle_raster_path = r'D:\ManureAppThesis\AnimalDistribution\poultry\chicken\Glb_chicken_AD_2006_paper.tif'
nuts0_shp_path = r'D:\ManureAppThesis\01M_NUTS0_WSG84_4326\NUTS_RG_01M_2021_4326_LEVL_0.shp'
standardization_csv = r'D:\ManureAppThesis\ManureProduction\chicken_csv.csv'
output_path = r'D:\ManureAppThesis\standardized_chicken_1km.tif'

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
    [(row.geometry, row['ScaleFactor_chicken']) for idx, row in nuts.iterrows()],
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

print("✅ Successfully generated the calibrated chicken distribution map for Europe only:", output_path)


import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np
from rasterio.features import rasterize

# === Input paths ===
calibrated_cattle_raster = r'D:\ManureAppThesis\standardized_chicken_1km.tif'  # Calibrated livestock distribution map
nuts0_shp_path = r'D:\ManureAppThesis\01M_NUTS0_WSG84_4326\NUTS_RG_01M_2021_4326_LEVL_0.shp'
standardization_csv = r'D:\ManureAppThesis\ManureProduction\chicken_csv.csv'  # Contains excretion_rate
output_manure_path = r'D:\ManureAppThesis\chicken_manure_production_1km.tif'

# === Read calibrated livestock distribution raster ===
with rasterio.open(calibrated_cattle_raster) as src:
    cattle_data = src.read(1)
    meta = src.meta.copy()
    transform = src.transform
    shape = src.shape
    crs = src.crs

# === Read NUTS0 boundaries and reproject to match raster ===
nuts = gpd.read_file(nuts0_shp_path)
nuts = nuts.to_crs(crs)

# === Read CSV and merge attributes ===
factors = pd.read_csv(standardization_csv)
nuts = nuts.merge(factors, on='NUTS_ID', how='left')

# === Rasterize the excretion rate ===
excretion_rate_raster = rasterize(
    [(row.geometry, row['Excretion_chicken']) for idx, row in nuts.iterrows()],
    out_shape=shape,
    transform=transform,
    fill=0.0,
    dtype='float32'
)

# === Calculate manure production ===
manure_production = cattle_data.astype('float32') * excretion_rate_raster
manure_production = np.nan_to_num(manure_production, nan=0.0, posinf=0.0, neginf=0.0)

# === Write the result to output raster ===
meta.update(dtype='float32', compress='lzw')

with rasterio.open(output_manure_path, 'w', **meta) as dst:
    dst.write(manure_production, 1)

print("✅ Manure production raster calculation completed and saved to:", output_manure_path)






# week 17-1 (classification of livestock production systems ->classification of smallholder and industry of swine and chicken)
import rasterio
import numpy as np
import os

# --- Input raster file paths ---
input_pig_raster = r"D:\ManureAppThesis\ManureApplication\StandardizedLivestockDistribution\standardized_swine_1km.tif"
input_chicken_raster = r"D:\ManureAppThesis\ManureApplication\StandardizedLivestockDistribution\standardized_chicken_1km.tif"

# --- Output directory and file ---
output_dir = r"D:\ManureAppThesis\ManureApplication"
os.makedirs(output_dir, exist_ok=True)
output_combined_tif = os.path.join(output_dir, "Classification_of_Smallhold_and_industry.tif")

# --- Read input rasters ---
try:
    with rasterio.open(input_pig_raster) as src_pig:
        pig_data = src_pig.read(1)
        pig_profile = src_pig.profile
        pig_nodata = src_pig.nodata

    with rasterio.open(input_chicken_raster) as src_chicken:
        chicken_data = src_chicken.read(1)
        chicken_nodata = src_chicken.nodata

except FileNotFoundError as e:
    print(f"Error: Input file not found. Check the path: {e}")
    print(f"Current working directory: {os.getcwd()}")
    exit()

# --- Check if raster shapes match ---
if pig_data.shape != chicken_data.shape:
    print("Error: Pig and chicken raster dimensions do not match.")
    exit()

# --- Initialize output raster with zeros ---
combined_data = np.zeros(pig_data.shape, dtype=np.uint8)

print("Applying classification rules...")

# --- Apply classification rules (in priority order) ---

# 1. Pigs between 125 and 700 → class 1
mask_pig_1 = (pig_data >= 125) & (pig_data < 700) & (pig_data != pig_nodata)
combined_data[mask_pig_1] = 1

# 2. Pigs > 700 → class 2
mask_pig_2 = (pig_data > 700) & (pig_data != pig_nodata)
combined_data[mask_pig_2] = 2

# 3. Chickens between 5000 and 21400 → class 3
mask_chicken_3 = (chicken_data >= 5000) & (chicken_data < 21400) & (chicken_data != chicken_nodata)
combined_data[mask_chicken_3] = 3

# 4. Chickens > 21400 → class 4 (highest priority)
mask_chicken_4 = (chicken_data > 21400) & (chicken_data != chicken_nodata)
combined_data[mask_chicken_4] = 4

# --- Prepare output raster profile ---
output_profile = pig_profile.copy()
output_profile.update({
    "dtype": combined_data.dtype,
    "count": 1,
    "nodata": 0
})

# --- Write output GeoTIFF ---
try:
    with rasterio.open(output_combined_tif, 'w', **output_profile) as dst:
        dst.write(combined_data, 1)
    print(f"\nClassification completed. Output saved to: {output_combined_tif}")
except Exception as e:
    print(f"Error writing output file: {e}")


# week 17-2 classification of livestock-only system
import rasterio
import numpy as np

# Define input and output file paths
input_file = r"D:\ManureAppThesis\ManureApplication\CORINERaster1km\COEINERaster1km_withLegend.tif"
output_file = r"D:\ManureAppThesis\ManureApplication\livestockOnlySytem.tif"

try:
    # Open the input raster file
    with rasterio.open(input_file) as src:
        # Read the first band
        data = src.read(1)

        # Create a new array with the same shape and type as the original
        isolated_pastures = np.zeros_like(data, dtype=data.dtype)

        # Set pixels with value 231 (pasture) to 5
        isolated_pastures[data == 231] = 5

        # Copy metadata from the source file
        profile = src.profile

        # Optionally, set NoData value if needed
        # profile.update(nodata=0)

        # Write the output raster with modified data
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(isolated_pastures, 1)

    print("Pixels with value 231 (pasture) have been set to 5.")
    print("All other areas have been set to 0.")
    print(f"Output saved to: {output_file}")

except FileNotFoundError:
    print(f"Error: File not found '{input_file}'. Please check the path.")
except Exception as e:
    print(f"An error occurred: {e}")


# week 17-3 mixed irrigated farming system
import rasterio
import numpy as np

# Define input and output file paths
input_file = r"D:\ManureAppThesis\ManureApplication\GMIE_Irrigation\GMIE_Irrigation1km\GMIE_EU27_1km_final.tif"
output_file = r"D:\ManureAppThesis\ManureApplication\MixedIrrigatedFarmingSystem.tif"

try:
    # Open the input raster file
    with rasterio.open(input_file) as src:
        # Read the first band of the raster
        data = src.read(1)

        # ✅ Correct way to get dtype: use data.dtype or src.dtypes[0]
        modified_data = np.zeros_like(data, dtype=data.dtype)

        # Set pixels with values greater than 0 in the original raster to 6 in the new array
        modified_data[data > 0] = 6

        # Get metadata from the source file
        profile = src.profile

        # If you want to treat 0 as NoData, uncomment the following line:
        # profile.update(nodata=0)

        # Write the modified data to a new file
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(modified_data, 1)

    print(f"Pixels with values greater than 0 in the irrigation map '{input_file}' were successfully filtered.")
    print(f"These pixel values have been reset to 6.")
    print(f"The new TIFF file has been saved to: {output_file}")

except FileNotFoundError:
    print(f"Error: File not found '{input_file}'. Please make sure the path is correct.")
except Exception as e:
    print(f"An error occurred while processing the file: {e}")


# week 17-4 mixed rainfed farming system
import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling

# Define input file paths
corine_land_use_file = r"D:\ManureAppThesis\ManureApplication\CORINERaster1km\COEINERaster1km_withLegend.tif"
mixed_irrigated_farming_system_file = r"D:\ManureAppThesis\ManureApplication\MixedIrrigatedFarmingSystem.tif"

# Define output file path
output_file = r"D:\ManureAppThesis\ManureApplication\StandardizedMixedIrrigatedFarmingSystem.tif"

try:
    # 1. Open the CORINE land use raster file (used as reference)
    with rasterio.open(corine_land_use_file) as src_corine:
        corine_data = src_corine.read(1)
        corine_profile = src_corine.profile

    # 2. Open the mixed irrigated farming system raster file
    with rasterio.open(mixed_irrigated_farming_system_file) as src_irrigated:
        irrigated_data_original = src_irrigated.read(1)
        irrigated_profile_original = src_irrigated.profile

        # Create an empty array to store the resampled irrigated data
        # The shape will match the CORINE data
        irrigated_data_reprojected = np.empty_like(corine_data, dtype=irrigated_data_original.dtype)

        # Perform resampling: resample the irrigated raster to match the geometry of the CORINE raster
        reproject(
            source=irrigated_data_original,
            destination=irrigated_data_reprojected,
            src_transform=irrigated_profile_original['transform'],  # Affine transform of the original irrigated raster
            src_crs=irrigated_profile_original['crs'],  # CRS of the original irrigated raster
            dst_transform=corine_profile['transform'],  # Target (CORINE) affine transform
            dst_crs=corine_profile['crs'],  # Target (CORINE) CRS
            resampling=Resampling.nearest,  # Resampling method: nearest neighbor (suitable for categorical data)
            src_nodata=irrigated_profile_original.get('nodata'),  # NoData value of the original data
            dst_nodata=0  # NoData value for the resampled target data, set to 0
        )

    # Now, irrigated_data_reprojected has the same shape as corine_data
    # All non-irrigated areas (<= 0 in the original irrigated raster) are now set to 0

    # Create a new array with the same shape and data type as CORINE data, initialized with 0
    result_data = np.zeros_like(corine_data, dtype=corine_data.dtype)

    # 3. Create a mask for areas in the CORINE raster where pixel values are between 200 and 250
    agricultural_mask = (corine_data >= 200) & (corine_data <= 250)

    # 4. Exclude areas identified as mixed irrigated farming system
    # In the resampled irrigated data, pixel value 6 indicates mixed irrigated area, 0 indicates non-irrigated
    # So we create a mask for areas not equal to 6 (i.e., non-mixed irrigated areas)
    non_irrigated_mask = (irrigated_data_reprojected != 6)  # '!= 6' means exclude irrigated areas

    # 5. Combine logic: within agricultural_mask, exclude irrigated_mask
    # Final condition is met when both agricultural_mask and non_irrigated_mask are True
    final_selection_mask = agricultural_mask & non_irrigated_mask

    # 6. Set the pixel value to 7 for the selected areas
    result_data[final_selection_mask] = 7

    # Update output file profile
    # Use the CORINE profile and ensure data type matches result_data
    # Uncomment the next line if you want 0 to explicitly indicate NoData
    output_profile = corine_profile
    output_profile.update(dtype=result_data.dtype, count=1)
    # output_profile.update(nodata=0)  # Uncomment if 0 should be treated as NoData

    # 7. Write the result to a new TIFF file
    with rasterio.open(output_file, 'w', **output_profile) as dst:
        dst.write(result_data, 1)

    print(f"Raster alignment and processing completed successfully!")
    print(f"Areas in CORINE with pixel values between 200 and 250, excluding mixed irrigated areas, have been selected,")
    print(f"and their pixel values have been set to 7.")
    print(f"New TIFF file has been saved as: {output_file}")

except FileNotFoundError as e:
    print(f"Error: File not found. Please make sure the following files exist or that the paths are correct:")
    print(f"- '{corine_land_use_file}'")
    print(f"- '{mixed_irrigated_farming_system_file}'")
    print(f"Detailed error: {e}")
except ValueError as e:
    print(f"Data processing error: {e}")
except Exception as e:
    print(f"An unexpected error occurred while processing the files: {e}")



# week 17-5 clip AI global map to EU-27
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np

# Define input and output file paths
# Make sure these files are in the same directory as your Python script, or provide full paths
aridity_index_raster = r"D:\ManureAppThesis\ManureApplication\AridityIndex\AridityIndex_yr.tif"
nuts_shapefile = r"D:\ManureAppThesis\01M_NUTS0_WSG84_4326\NUTS_RG_01M_2021_4326_LEVL_0.shp"
output_clipped_raster = r"D:\ManureAppThesis\ManureApplication\AridityIndex_EUclipped.tif"

try:
    print(f"Reading Shapefile: {nuts_shapefile}")
    # 1. Read Eurostat NUTS shapefile
    # geopandas automatically reads the file and returns a GeoDataFrame
    gdf = gpd.read_file(nuts_shapefile)

    # Extract all geometries from the shapefile
    # rasterio.mask.mask requires a list of GeoJSON-like geometries
    # If the shapefile contains multiple features (e.g., multiple NUTS regions),
    # all these will be used for masking
    geometries = [geom for geom in gdf.geometry]

    print(f"Opening Aridity Index raster: {aridity_index_raster}")
    # 2. Open the global aridity index raster file
    with rasterio.open(aridity_index_raster) as src:
        # Get raster metadata and CRS
        raster_profile = src.profile
        raster_crs = src.crs

        # Check if the raster and shapefile CRS match
        # geopandas gdf.crs is already a pyproj.CRS object
        if raster_crs != gdf.crs:
            print(f"Warning: Raster CRS ({raster_crs}) and Shapefile CRS ({gdf.crs}) do not match.")
            print("rasterio.mask.mask will try to reproject automatically, but this may impact performance or accuracy.")
            print("It is recommended to manually reproject one dataset to match the other before clipping.")
            # To manually reproject shapefile, uncomment below:
            # gdf = gdf.to_crs(raster_crs)
            # geometries = [geom for geom in gdf.geometry]

        # 3. Perform the mask (clip) operation
        # 'crop=True' crops to the bounding box of the geometries
        # 'filled=True' fills outside the mask with nodata value
        # 'invert=False' keeps only the inside of the geometries
        out_image, out_transform = mask(src, geometries, crop=True, filled=True, nodata=raster_profile['nodata'])

        # Update metadata for the output raster
        # The clip changes the image dimensions and affine transform
        out_meta = raster_profile.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],  # out_image shape is (bands, height, width)
            "width": out_image.shape[2],
            "transform": out_transform,
            "crs": raster_crs  # Keep CRS same as original raster
        })

        # 4. Save the clipped image as a new TIFF file
        print(f"Saving clipped raster to: {output_clipped_raster}")
        with rasterio.open(output_clipped_raster, "w", **out_meta) as dest:
            dest.write(out_image)

    print(f"Successfully clipped '{aridity_index_raster}' to the extent of '{nuts_shapefile}'.")
    print(f"The clipped raster has been saved as: {output_clipped_raster}")

except FileNotFoundError:
    print(f"Error: File not found. Please ensure these files exist in the current directory or check their paths:")
    print(f"- '{aridity_index_raster}'")
    print(f"- '{nuts_shapefile}'")
except Exception as e:
    print(f"An error occurred while processing the files: {e}")



# week 17-6 merge four classification 1-4 in order swine-chicken->livestock only->mixed irrigated farming->mixed rainfed farming
import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling

# Define input file paths and their priority order (from high to low)
# Ensure these files are in the same directory as your Python script, or provide full paths.
input_files = [
    r"D:\ManureAppThesis\ManureApplication\Classification_of_Smallhold_and_industry.tif",  # Priority 1 (Highest)
    r"D:\ManureAppThesis\ManureApplication\livestockOnlySytem.tif",  # Priority 2
    r"D:\ManureAppThesis\ManureApplication\MixedIrrigatedFarmingSystem.tif",  # Priority 3
    r"D:\ManureAppThesis\ManureApplication\MixedRainfedFarmingSystem.tif"  # Priority 4 (Lowest)
]

# Define the output file path
output_merged_raster = "D:\ManureAppThesis\ManureApplication\LivestockProductionSystems.tif"

try:
    # 1. Open the first (highest priority) raster file as the base.
    with rasterio.open(input_files[0]) as src_base:
        base_data = src_base.read(1)
        base_profile = src_base.profile
        base_nodata = base_profile.get('nodata')

    # Initialize the final merged array.
    # It will have the same dimensions and data type as the base raster,
    # and will be filled with the base raster's nodata value (or 0 if nodata is None).
    final_merged_data = np.full(base_data.shape, base_nodata if base_nodata is not None else 0, dtype=base_data.dtype)

    # If the base raster's nodata is None, and its data type is integer, we default 0 as background/nodata.
    # Ensure final_merged_data is initially entirely "empty" or background values.
    if base_nodata is None:
        final_merged_data.fill(0)  # Initialize all pixels to 0

    # Iterate through all input files, from lowest priority to highest priority, overwriting progressively.
    # This ensures that non-zero values from higher priority layers overwrite lower priority ones.
    for i in range(len(input_files) - 1, -1, -1):  # Iterate in reverse, from 3 down to 0
        file_path = input_files[i]
        print(f"Processing file with priority {i + 1}: {file_path}")

        with rasterio.open(file_path) as src:
            current_data = src.read(1)
            current_profile = src.profile
            current_nodata = current_profile.get('nodata')

            # If the current file is not the base file, it needs to be reprojected.
            if file_path != input_files[0]:
                print(f"  - Reprojecting '{file_path}' to match the base raster...")
                reprojected_data = np.empty_like(base_data, dtype=current_data.dtype)

                reproject(
                    source=current_data,
                    destination=reprojected_data,
                    src_transform=current_profile['transform'],
                    src_crs=current_profile['crs'],
                    dst_transform=base_profile['transform'],
                    dst_crs=base_profile['crs'],
                    resampling=Resampling.nearest,  # Nearest neighbor is typically used for categorical data
                    src_nodata=current_nodata,
                    # If the current file's nodata is None, and we assume 0 as background,
                    # treat 0 as nodata during reprojection as well.
                    dst_nodata=0 if current_nodata is None else current_nodata
                )
                data_to_merge = reprojected_data
            else:
                data_to_merge = current_data

            # Merging logic: if the current layer has non-zero and non-nodata values, update the final result.
            # Note: This assumes 0 is background/no data. If your layers have other background values, adjust this.
            # The logic here is: "If the current layer has valid data, and the corresponding position in the final result
            # is background (0 or initial nodata), OR the current layer has higher priority, then overwrite."

            # More general high-priority overlay logic:
            # Find pixels in the current layer that have "data" (i.e., not 0 and not nodata).
            # Alternatively, simply consider anything not 0 as "having data."

            # Use a boolean mask to identify where the current layer has non-background values.
            if current_nodata is not None:
                # Exclude current_nodata locations
                has_data_mask = (data_to_merge != 0) & (data_to_merge != current_nodata)
            else:
                # Assume only 0 is the background value
                has_data_mask = (data_to_merge != 0)

            # Overwrite positions in final_merged_data that are 0 (or base_nodata) with data from the current layer.
            # Or, if the current layer has non-zero data, simply overwrite directly (higher priority overwrites lower).
            # Since we iterate in reverse (from low priority to high priority), direct assignment achieves
            # higher priority overwriting lower priority.
            final_merged_data[has_data_mask] = data_to_merge[has_data_mask]

    # Update the output raster's metadata
    output_profile = base_profile.copy()
    output_profile.update({
        "driver": "GTiff",
        "height": final_merged_data.shape[0],
        "width": final_merged_data.shape[1],
        "transform": base_profile['transform'],
        "crs": base_profile['crs'],
        "dtype": final_merged_data.dtype,
        "count": 1,
        "nodata": base_nodata  # Use the base raster's NoData value as the output NoData
    })

    # If the base raster has no explicit NoData, and we've used 0 as background,
    # we can explicitly set nodata=0.
    if output_profile.get('nodata') is None:
        output_profile['nodata'] = 0

    # Save the final merged raster
    print(f"\nAll rasters merged and saved to: {output_merged_raster}")
    with rasterio.open(output_merged_raster, "w", **output_profile) as dest:
        dest.write(final_merged_data, 1)

    print("Merging complete!")

except FileNotFoundError as e:
    print(f"Error: File not found. Please ensure all input files exist in the current directory or the paths are correct.")
    print(f"Detailed error: {e}")
except Exception as e:
    print(f"An unexpected error occurred while processing files: {e}")


# week 17-7 merge sheep and goats distribution
import rasterio
import numpy as np
import os

# --- Input Paths ---
input_sheep_raster = r"D:\ManureAppThesis\ManureApplication\StandardizedLivestockDistribution\standardized_sheep_1km.tif"
input_goats_raster = r"D:\ManureAppThesis\ManureApplication\StandardizedLivestockDistribution\standardized_goat_1km.tif"

# --- Output Paths ---
output_dir = r"D:\ManureAppThesis\ManureApplication\StandardizedLivestockDistribution"
os.makedirs(output_dir, exist_ok=True)
output_combined_raster = os.path.join(output_dir, "standardized_sheep&goats_1km.tif")

# --- Read Data ---
try:
    with rasterio.open(input_sheep_raster) as src_sheep:
        sheep_data = src_sheep.read(1)
        sheep_profile = src_sheep.profile
        sheep_nodata = src_sheep.nodata

    with rasterio.open(input_goats_raster) as src_goats:
        goats_data = src_goats.read(1)
        goats_profile = src_goats.profile
        goats_nodata = src_goats.nodata

except FileNotFoundError as e:
    print(f"❌ Input file not found: {e}")
    exit()

# --- Consistency Check ---
if sheep_data.shape != goats_data.shape:
    print("❌ Raster dimensions do not match, cannot merge.")
    exit()

# --- Clean Data: Treat all non-positive values as invalid (including negative, NaN, NoData, extreme values) ---
def clean(data, nodata_val):
    EXTREME_NEG = -3.4028235e+38  # Common representation for extreme negative float values
    cleaned = np.where(
        (data == nodata_val) | (np.isnan(data)) | (np.isinf(data)) | (data == EXTREME_NEG),
        0,
        data
    )
    # If there are still negative values after cleaning, set them to 0
    cleaned = np.where(cleaned < 0, 0, cleaned)
    return cleaned

sheep_data_cleaned = clean(sheep_data, sheep_nodata)
goats_data_cleaned = clean(goats_data, goats_nodata)

# --- Combine Data: Only non-zero values participate in the merge ---
combined_data = sheep_data_cleaned + goats_data_cleaned

# Ensure all non-positive values are set to 0 (for a clean output)
combined_data = np.where(combined_data > 0, combined_data, 0).astype(np.float32)

# --- Output Configuration ---
output_profile = sheep_profile.copy()
output_profile.update({
    "dtype": np.float32,
    "count": 1,
    "nodata": 0  # Explicitly set NoData to 0 for easier QGIS recognition
})

# --- Write Result ---
try:
    with rasterio.open(output_combined_raster, 'w', **output_profile) as dst:
        dst.write(combined_data, 1)
    print(f"✅ Merging complete, only non-zero pixels retained. Output saved to: {output_combined_raster}")
except Exception as e:
    print(f"❌ Write failed: {e}")


# week 17-8  classification with aridity index and animal categories (experiment on definition of subsystems)
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import os
import gc


def reclassify_livestock_systems_bulletproof():
    """
    Final solution: combines block processing with dynamic alignment.
    1. Processes data in blocks to ensure low memory consumption.
    2. Dynamically reprojects (aligns) conditional rasters for each block to resolve shape mismatches.
    3. Includes all previous fixes and logical optimizations.
    """
    # --- 1. Define file paths ---
    base_path = 'D:/ManureAppThesis/ManureApplication/'
    lps_file = os.path.join(base_path, 'LivestockProductionSystems.tif')
    aridity_file = os.path.join(base_path, 'AridityIndex_EUclipped.tif')
    cattle_file = os.path.join(base_path, 'StandardizedLivestockDistribution/standardized_cattle_1km.tif')
    sg_file = os.path.join(base_path, 'StandardizedLivestockDistribution/standardized_sheep&goats_1km.tif')
    output_file = os.path.join(base_path, 'Reclassified_LivestockSystems_Bulletproof.tif')

    print("Starting raster processing (Block Processing + Dynamic Alignment version)...")

    # --- 2. Open all source rasters in read mode ---
    with rasterio.open(lps_file) as lps_src, \
            rasterio.open(aridity_file) as aridity_src, \
            rasterio.open(cattle_file) as cattle_src, \
            rasterio.open(sg_file) as sg_src:

        # Get the profile from the source raster for the output file
        profile = lps_src.profile
        nodata_value = 0
        # Update the profile for the output GeoTIFF
        profile.update(dtype=rasterio.uint8, nodata=nodata_value, compress='lzw')

        with rasterio.open(output_file, 'w', **profile) as dst:
            # Get an iterator for the processing windows (blocks) from the source raster
            windows = lps_src.block_windows(1)
            # Calculate the total number of blocks to show progress
            total_windows = len(list(lps_src.block_windows(1)))

            print(f"The raster will be processed in {total_windows} blocks...")

            # --- 3. Loop through and process each data block ---
            for index, (ji, window) in enumerate(windows):
                print(f"Processing block {index + 1}/{total_windows}...", end='\r')

                # Read the current block from the main raster
                lps_block = lps_src.read(1, window=window)

                # If the current block contains no pixels to reclassify, write it directly and skip to the next block
                reclass_mask = (lps_block == 5) | (lps_block == 6) | (lps_block == 7)
                if not np.any(reclass_mask):
                    dst.write(lps_block.astype(profile['dtype']), 1, window=window)
                    continue

                # --- Dynamic Alignment Logic ---
                # Get the precise geographic transformation for the current window
                window_transform = lps_src.window_transform(window)

                # Create empty arrays to receive the aligned data
                aridity_block = np.zeros(lps_block.shape, dtype=np.float32)
                cattle_block = np.zeros(lps_block.shape, dtype=np.float32)
                sg_block = np.zeros(lps_block.shape, dtype=np.float32)

                # Perform on-the-fly reprojection for each conditional raster to align its data to the current window
                reproject(source=rasterio.band(aridity_src, 1), destination=aridity_block,
                          src_transform=aridity_src.transform, src_crs=aridity_src.crs,
                          dst_transform=window_transform, dst_crs=lps_src.crs,
                          resampling=Resampling.bilinear)

                reproject(source=rasterio.band(cattle_src, 1), destination=cattle_block,
                          src_transform=cattle_src.transform, src_crs=cattle_src.crs,
                          dst_transform=window_transform, dst_crs=lps_src.crs,
                          resampling=Resampling.bilinear)

                reproject(source=rasterio.band(sg_src, 1), destination=sg_block,
                          src_transform=sg_src.transform, src_crs=sg_src.crs,
                          dst_transform=window_transform, dst_crs=lps_src.crs,
                          resampling=Resampling.bilinear)
                # --- Dynamic alignment finished. All *_block arrays are now guaranteed to have the same shape. ---

                output_block = lps_block.copy()
                # Set all candidate pixels to the NoData value by default.
                # They will only be assigned a new value if they meet one of the conditions below.
                output_block[reclass_mask] = nodata_value

                # --- Apply Classification Logic ---
                # Cast to float64 for safe multiplication to prevent overflow
                cattle_64 = cattle_block.astype(np.float64)
                sg_64 = sg_block.astype(np.float64)

                # Create boolean masks for the conditions
                aridity_hyper_arid = aridity_block < 500
                aridity_arid = (aridity_block >= 500) & (aridity_block <= 5000)
                aridity_humid = aridity_block > 5000

                # The 'equal to' case is classified as 'cattle dominant'
                cattle_dominant = (5.80 * cattle_64) >= sg_64
                sg_dominant = (5.80 * cattle_64) < sg_64

                for original_value in [5, 6, 7]:
                    # NOTE: system_mask must be based on the original lps_block
                    system_mask = (lps_block == original_value)
                    base_new_value = original_value * 10

                    # Apply the new classification values based on the combined conditions
                    output_block[(system_mask) & (aridity_hyper_arid) & (cattle_dominant)] = base_new_value + 1
                    output_block[(system_mask) & (aridity_hyper_arid) & (sg_dominant)] = base_new_value + 2
                    output_block[(system_mask) & (aridity_arid) & (cattle_dominant)] = base_new_value + 3
                    output_block[(system_mask) & (aridity_arid) & (sg_dominant)] = base_new_value + 4
                    output_block[(system_mask) & (aridity_humid) & (cattle_dominant)] = base_new_value + 5
                    output_block[(system_mask) & (aridity_humid) & (sg_dominant)] = base_new_value + 6

                # Write the processed block to the output file
                dst.write(output_block.astype(profile['dtype']), 1, window=window)

                # Clean up memory (optional, but good practice in a loop)
                del lps_block, aridity_block, cattle_block, sg_block, output_block, cattle_64, sg_64, reclass_mask
                gc.collect()

    print(f"\nProcessing complete! The final file has been generated: {output_file}")


if __name__ == '__main__':
    reclassify_livestock_systems_bulletproof()


# week 17-9 classification only based on aridity index
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import os
import gc


def reclassify_by_aridity_simplified():
    """
    Simplified Classification: Reclassifies based only on the Aridity Index.

    This final version uses a robust block-processing and dynamic-alignment approach
    to ensure low memory usage and handle potentially misaligned raster files.
    It also correctly handles the 10,000x scaling factor of the aridity index file.
    """
    # --- 1. 定义文件路径 ---
    # 使用您指定的原始路径
    base_path = 'D:/ManureAppThesis/ManureApplication/'
    lps_file = os.path.join(base_path, 'LivestockProductionSystems.tif')
    aridity_file = os.path.join(base_path, 'AridityIndex_EUclipped.tif')

    # 为本次简化逻辑的输出文件定义一个新名称
    output_file = os.path.join(base_path, 'Reclassified_By_Aridity_Only.tif')

    print("开始处理栅格文件（简化版逻辑）...")

    # --- 2. Open source rasters in read mode ---
    with rasterio.open(lps_file) as lps_src, \
            rasterio.open(aridity_file) as aridity_src:

        # Get the profile from the source raster for the output file
        profile = lps_src.profile
        nodata_value = 0
        # Update the profile for the output GeoTIFF
        profile.update(dtype=rasterio.uint8, nodata=nodata_value, compress='lzw')

        with rasterio.open(output_file, 'w', **profile) as dst:
            # Get an iterator for the processing windows (blocks)
            windows = lps_src.block_windows(1)
            total_windows = len(list(lps_src.block_windows(1)))

            print(f"栅格将被分为 {total_windows} 块进行处理...")

            # --- 3. Loop through and process each data block ---
            for index, (ji, window) in enumerate(windows):
                print(f"正在处理块 {index + 1}/{total_windows}...", end='\r')

                # Read the current block from the main raster
                lps_block = lps_src.read(1, window=window)

                # If the block contains no pixels to reclassify (values 5, 6, 7),
                # write it directly and skip to the next block for efficiency.
                reclass_mask = (lps_block == 5) | (lps_block == 6) | (lps_block == 7)
                if not np.any(reclass_mask):
                    dst.write(lps_block.astype(profile['dtype']), 1, window=window)
                    continue

                # --- Dynamic Alignment Logic ---
                # Get the precise geographic transformation for the current window
                window_transform = lps_src.window_transform(window)

                # Create an empty array to receive the aligned aridity data
                aridity_block = np.zeros(lps_block.shape, dtype=np.float32)

                # Perform on-the-fly reprojection to align the aridity data to the current window
                reproject(source=rasterio.band(aridity_src, 1), destination=aridity_block,
                          src_transform=aridity_src.transform, src_crs=aridity_src.crs,
                          dst_transform=window_transform, dst_crs=lps_src.crs,
                          resampling=Resampling.bilinear)
                # --- Dynamic alignment finished ---

                output_block = lps_block.copy()
                # By default, set candidate pixels to NoData. They will be assigned a new value
                # only if they meet a specific aridity condition.
                output_block[reclass_mask] = nodata_value

                # --- Apply Simplified Classification Logic ---
                # Create boolean masks for aridity conditions, accounting for the 10,000x scaling factor
                aridity_hyper_arid = aridity_block < 500  # Equivalent to Aridity Index < 0.05
                aridity_arid = (aridity_block >= 500) & (aridity_block <= 5000)  # Equivalent to 0.05 <= AI <= 0.5
                aridity_humid = aridity_block > 5000  # Equivalent to Aridity Index > 0.5

                # Reclassify pixels with original value 5
                mask_5 = (lps_block == 5)
                output_block[(mask_5) & (aridity_hyper_arid)] = 51
                output_block[(mask_5) & (aridity_arid)] = 52
                output_block[(mask_5) & (aridity_humid)] = 53

                # Reclassify pixels with original value 6
                mask_6 = (lps_block == 6)
                output_block[(mask_6) & (aridity_hyper_arid)] = 61
                output_block[(mask_6) & (aridity_arid)] = 62
                output_block[(mask_6) & (aridity_humid)] = 63

                # Reclassify pixels with original value 7
                mask_7 = (lps_block == 7)
                output_block[(mask_7) & (aridity_hyper_arid)] = 71
                output_block[(mask_7) & (aridity_arid)] = 72
                output_block[(mask_7) & (aridity_humid)] = 73

                # Write the processed block to the output file
                dst.write(output_block.astype(profile['dtype']), 1, window=window)

                # Clean up memory
                del lps_block, aridity_block, output_block, reclass_mask
                gc.collect()

    print(f"\n处理完成！最终文件已生成: {output_file}")


if __name__ == '__main__':
    reclassify_by_aridity_simplified()





# week 17-10 upscale on length of growing and process/clip on average temperature July
import os
import rasterio
from rasterio.mask import mask
from rasterio.enums import Resampling
import geopandas as gpd
import numpy as np

# --- 1. Define input and output files ---
# Input shapefile for clipping
shp_filename = r"D:\ManureAppThesis\01M_NUTS0_WSG84_4326\NUTS_RG_01M_2021_4326_LEVL_0.shp"
# Input growing season raster data
input_raster_filename = r"D:\ManureAppThesis\ManureApplication\CORINE_Length_of_Growing\Length_of_Growing.tiff"

# Intermediate output: temporary file after clipping
clipped_raster_filename = r"D:\ManureAppThesis\ManureApplication\CORINE_Length_of_Growing\Length_of_Growing_Europe_300m.tif"
# Final output: raster aggregated to 1km resolution
output_raster_filename = r"D:\ManureAppThesis\ManureApplication\CORINE_Length_of_Growing\Length_of_Growing_Europe_1km_mode.tif"

print("Processing started...")
print(f"Input Shapefile: {shp_filename}")
print(f"Input Raster: {input_raster_filename}")

# --- 2. Clip raster (Step 1) ---
# Read shapefile using geopandas
print("\nStep 1: Reading Europe boundary Shapefile...")
try:
    europe_gdf = gpd.read_file(shp_filename)
except Exception as e:
    print(f"Error: Failed to read Shapefile '{shp_filename}'. Please make sure the file exists and is complete.")
    print(e)
    exit()

# Ensure the coordinate reference system is EPSG:4326
if europe_gdf.crs.to_epsg() != 4326:
    print("Warning: The CRS of the Shapefile is not EPSG:4326. Converting now.")
    europe_gdf = europe_gdf.to_crs(epsg=4326)

print("Step 2: Clipping global growing season raster to European boundaries...")
try:
    with rasterio.open(input_raster_filename) as src:
        # Extract geometries from the shapefile for clipping
        geometries = europe_gdf.geometry

        # Perform the clipping
        out_image, out_transform = mask(src, geometries, crop=True)
        out_meta = src.meta.copy()

    # Update metadata to reflect new size and transform
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

    # Save clipped result to a temporary intermediate file
    with rasterio.open(clipped_raster_filename, "w", **out_meta) as dest:
        dest.write(out_image)

    print(f"Clipping completed. Temporary file saved to: {clipped_raster_filename}")

except Exception as e:
    print(f"Error: An issue occurred during raster clipping. Please check if the input raster '{input_raster_filename}' exists and is valid.")
    print(e)
    exit()

# --- 3. Aggregate (resample) to 1km resolution (Step 2) ---
print("\nStep 3: Aggregating 300m resolution to 1km...")

# Calculate downscale factor. 1km / 300m ≈ 3.333
downscale_factor = 1000 / 300

try:
    with rasterio.open(clipped_raster_filename) as src:
        # Compute new height and width for the resampled raster
        new_height = int(src.height / downscale_factor)
        new_width = int(src.width / downscale_factor)

        # Create a numpy array to hold the resampled data
        # Only reading the first band here
        data = src.read(
            1,  # Read only the first band
            out_shape=(new_height, new_width),
            resampling=Resampling.mode  # Use mode (most frequent value) for resampling
        )

        # Copy and update metadata
        out_meta = src.meta.copy()

        # Calculate new transform by scaling the original
        new_transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height)
        )

        out_meta.update({
            "height": new_height,
            "width": new_width,
            "transform": new_transform
        })

        # Write the final result to output file
        with rasterio.open(output_raster_filename, 'w', **out_meta) as dest:
            dest.write(data, 1)

    print(f"Aggregation complete! Final result saved to: {output_raster_filename}")

except Exception as e:
    print(f"Error: An issue occurred during aggregation.")
    print(e)
    exit()

finally:
    # --- 4. Clean up temporary file ---
    if os.path.exists(clipped_raster_filename):
        print(f"\nStep 4: Cleaning up temporary file {clipped_raster_filename}...")
        os.remove(clipped_raster_filename)
        print("Cleanup completed.")

print("\nAll processing steps completed successfully!")



# week 17-11 reclassification of livestock production subsystems
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
import numpy as np
import os


def reclassify_livestock_systems():
    """
    Reclassify Livestock Production Systems (LPS) raster based on Length of Growing Period (LGP).
    This version automatically resamples the LGP raster to match the LPS raster alignment.
    """
    try:
        lps_filename = r'D:\ManureAppThesis\ManureApplication\ClassificationofLPS\LivestockProductionSystems.tif'
        lgp_filename = r'D:\ManureAppThesis\ManureApplication\CORINE_Length_of_Growing\Length_of_Growing_Europe_1km_mode.tif'
        output_filename = r'D:\ManureAppThesis\ManureApplication\LivestockProduction_SubSystems.tif'

        # Check file existence
        if not os.path.exists(lps_filename):
            raise FileNotFoundError(f"Error: LPS file not found: {lps_filename}")
        if not os.path.exists(lgp_filename):
            raise FileNotFoundError(f"Error: LGP file not found: {lgp_filename}")

    except Exception as e:
        print(e)
        return

    print("Starting processing...")
    print(f"LPS raster: {lps_filename}")
    print(f"LGP raster: {lgp_filename}")

    with rasterio.open(lps_filename) as lps_src, rasterio.open(lgp_filename) as lgp_src:
        # Print transform to check alignment
        print("LPS transform:", lps_src.transform)
        print("LGP transform:", lgp_src.transform)

        # Resample LGP to match LPS alignment
        print("Resampling LGP raster to match LPS raster...")
        lgp_resampled = np.empty((lps_src.height, lps_src.width), dtype=np.float32)

        reproject(
            source=rasterio.band(lgp_src, 1),
            destination=lgp_resampled,
            src_transform=lgp_src.transform,
            src_crs=lgp_src.crs,
            dst_transform=lps_src.transform,
            dst_crs=lps_src.crs,
            resampling=Resampling.nearest
        )

        print("Resampling complete. Starting block-wise processing and reclassification...")

        # Prepare output file metadata
        output_profile = lps_src.profile
        output_profile.update({
            'dtype': 'uint8',
            'compress': 'lzw'
        })

        with rasterio.open(output_filename, 'w', **output_profile) as dst:
            block_size = 512
            total_blocks = int(np.ceil(lps_src.height / block_size)) * int(np.ceil(lps_src.width / block_size))
            current_block = 0

            for j in range(0, lps_src.height, block_size):
                for i in range(0, lps_src.width, block_size):
                    current_block += 1
                    print(f"Processing block {current_block}/{total_blocks}...")

                    h = min(block_size, lps_src.height - j)
                    w = min(block_size, lps_src.width - i)
                    window = Window(i, j, w, h)

                    try:
                        lps_data = lps_src.read(1, window=window)
                        lgp_data = lgp_resampled[j:j + h, i:i + w]
                    except Exception as e:
                        print(f"Failed to read window. Skipping block. Reason: {e}")
                        continue

                    # Initialize result block
                    result_data = lps_data.copy()

                    # Reclassification rules
                    # Rule 1: Pure livestock (original LPS = 5)
                    mask_lga = (lps_data == 5) & (lgp_data < 180)
                    mask_lgh = (lps_data == 5) & (lgp_data >= 180)
                    result_data[mask_lga] = 51
                    result_data[mask_lgh] = 52

                    # Rule 2: Mixed irrigated (LPS =





# week 17-12 calculation of manure application
import rasterio
from rasterio.windows import Window, bounds as window_bounds
from rasterio.transform import from_bounds
from rasterio.enums import Resampling
import numpy as np
import os
from contextlib import ExitStack

def calculate_manure_application():
    print("--- Manure Application Calculation Script ---")

    # --- 1. Input Raster Files and Output Settings ---

    lps_filename = r'D:\ManureAppThesis\ManureApplication\LivestockProduction_SubSystems.tif'

    ANIMAL_PROCESSING_TASKS = {
        'cattle': {
            'input_files': [r'D:\ManureAppThesis\ManureProduction\ManureProduction_tif\cattle_manure_production_1km.tif'],
            'output_file': r'D:\ManureAppThesis\ManureApplication\cattle_manure_application.tif'
        },
        'sheep_goat': {
            'input_files': [
                r'D:\ManureAppThesis\ManureProduction\ManureProduction_tif\sheep_manure_production_1km.tif',
                r'D:\ManureAppThesis\ManureProduction\ManureProduction_tif\goat_manure_production_1km.tif'],
            'output_file': r'D:\ManureAppThesis\ManureApplication\sheep_goat_manure_application.tif'
        },
        'chicken': {
            'input_files': [r'D:\ManureAppThesis\ManureProduction\ManureProduction_tif\chicken_manure_production_1km.tif'],
            'output_file': r'D:\ManureAppThesis\ManureApplication\chicken_manure_application.tif'
        },
        'swine': {
            'input_files': [r'D:\ManureAppThesis\ManureProduction\ManureProduction_tif\swine_manure_production_1km.tif'],
            'output_file': r'D:\ManureAppThesis\ManureApplication\swine_manure_application.tif'
        }
    }

    FRACTION_MAP = {
        51: {'cattle': 0.138, 'sheep_goat': 0.04},
        52: {'cattle': 0.336, 'sheep_goat': 0.11},
        71: {'cattle': 0.298, 'sheep_goat': 0.04},
        72: {'cattle': 0.304, 'sheep_goat': 0.04},
        61: {'cattle': 0.298, 'sheep_goat': 0.04},
        62: {'cattle': 0.289, 'sheep_goat': 0.04},
        3: {'chicken': 0.27},
        4: {'chicken': 0.49},
        1: {'swine': 0.23},
        2: {'swine': 0.50}
    }

    # --- 2. Loop Through Each Animal Type ---
    if not os.path.exists(lps_filename):
        print(f"ERROR: LPS file not found: {lps_filename}")
        return

    for animal_key, task in ANIMAL_PROCESSING_TASKS.items():
        print(f"\n--- Processing: {animal_key.upper()} ---")

        input_manure_files = task['input_files']
        output_file = task['output_file']

        # --- 3. Check File Existence ---
        all_files = [lps_filename] + input_manure_files
        for f in all_files:
            if not os.path.exists(f):
                print(f"Skipping {animal_key}: Missing file {f}")
                break
        else:
            with ExitStack() as stack:
                lps_src = stack.enter_context(rasterio.open(lps_filename))
                manure_src_list = [stack.enter_context(rasterio.open(f)) for f in input_manure_files]

                # --- 4. Calculate Overlapping Area ---
                bounds = lps_src.bounds
                for src in manure_src_list:
                    bounds = (
                        max(bounds[0], src.bounds.left),
                        max(bounds[1], src.bounds.bottom),
                        min(bounds[2], src.bounds.right),
                        min(bounds[3], src.bounds.top)
                    )

                if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
                    print(f"Skipping {animal_key}: No overlap")
                    continue

                print(f"Found overlapping area: {bounds}")

                # --- 5. Output Raster Settings ---
                output_transform = from_bounds(*bounds, width=int((bounds[2] - bounds[0]) / lps_src.res[0]),
                                               height=int((bounds[3] - bounds[1]) / abs(lps_src.res[1])))

                output_width = int((bounds[2] - bounds[0]) / lps_src.res[0])
                output_height = int((bounds[3] - bounds[1]) / abs(lps_src.res[1]))

                output_profile = lps_src.profile.copy()
                output_profile.update({
                    'driver': 'GTiff',
                    'height': output_height,
                    'width': output_width,
                    'transform': output_transform,
                    'dtype': 'float32',
                    'compress': 'lzw'
                })

                # --- 6. Chunked Processing ---
                block_size = 512
                with rasterio.open(output_file, 'w', **output_profile) as dst:
                    for j in range(0, output_height, block_size):
                        for i in range(0, output_width, block_size):
                            w = min(block_size, output_width - i)
                            h = min(block_size, output_height - j)
                            window = Window(i, j, w, h)

                            win_bounds = window_bounds(window, output_transform)
                            lps_window = lps_src.window(*win_bounds)
                            lps_data = lps_src.read(1, window=lps_window,
                                                    out_shape=(h, w), resampling=Resampling.nearest)

                            total_manure_data = np.zeros((h, w), dtype=np.float32)

                            for src in manure_src_list:
                                manure_window = src.window(*win_bounds)
                                manure_data = src.read(1, window=manure_window,
                                                       out_shape=(h, w), resampling=Resampling.bilinear)
                                np.nan_to_num(manure_data, copy=False)
                                total_manure_data += manure_data

                            # --- 7. Apply Application Fractions ---
                            fraction_data = np.zeros_like(lps_data, dtype=np.float32)
                            for lps_val, fractions in FRACTION_MAP.items():
                                if animal_key in fractions:
                                    fraction_data[lps_data == lps_val] = fractions[animal_key]

                            application_data = total_manure_data * fraction_data
                            dst.write(application_data.astype(np.float32), 1, window=window)

                    print(f"SUCCESS: Finished writing {output_file}")

    print("\n--- All manure application rasters created successfully ---")


if __name__ == '__main__':
    calculate_manure_application()








# week 18-1 calculation of manure application (combine fractions from INTEGRATOR)
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.warp import reproject, Resampling
from rasterio.io import MemoryFile
import numpy as np
import os


def load_manure_fractions_from_excel(excel_path):
    print(f"--- Step 1/4: Loading parameters from Excel file: {excel_path}... ---")
    df = pd.read_excel(excel_path)
    df = df.rename(columns={
        'AvaliableforApplicationLiquid_Fraction_inTotalManure': 'Frac_Liquid_Available',
        'AvaliableforApplicationSolid_Fraction_inTotalManure': 'Frac_Solid_Available',
        'ApplictionLiquid_Arableland_swine_Fraction': 'Frac_Liquid_Arable',
        'ApplictionSolid_Arableland_swine_Fraction': 'Frac_Solid_Arable',
        'ApplictionLiquid_Grasslandland_swine_Fraction': 'Frac_Liquid_Grassland',
        'ApplictionSolid_Grassland_swine_Fraction': 'Frac_Solid_Grassland'
    })
    df = df.set_index('NUTS_ID')

    # Standardize index: remove spaces + uppercase
    df.index = df.index.str.strip().str.upper()

    # Fix common country code differences
    code_map = {
        'GR': 'EL',  # Greece
        'GB': 'UK'   # United Kingdom
    }
    df.rename(index=code_map, inplace=True)

    print("Parameters loaded successfully.\n")
    return df


def reclassify_land_use(corine_raster_path, classification_dict, output_path):
    print(f"--- Step 2/4: Reclassifying land use data: {corine_raster_path}... ---")
    with rasterio.open(corine_raster_path) as src:
        corine_data = src.read(1)
        profile = src.profile
        reclassified_data = np.zeros(corine_data.shape, dtype=np.uint8)
        for pixel_value, land_class in classification_dict.items():
            reclassified_data[corine_data == pixel_value] = land_class
        profile.update(dtype=rasterio.uint8, count=1, nodata=0)
        print(f"Writing reclassified data to: {output_path}...")
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(reclassified_data, 1)
    print("Land use reclassification completed.\n")
    return output_path


def calculate_manure_application(
        reclassified_land_use_path,
        swine_manure_production_path,
        nuts0_shapefile_path,
        manure_fractions_df,
        output_path
):
    print("--- Step 3/4: Loading and aligning raster data... ---")

    with rasterio.open(reclassified_land_use_path) as land_use_src:
        land_use_data = land_use_src.read(1)
        dst_profile = land_use_src.profile

        with rasterio.open(swine_manure_production_path) as manure_prod_src:
            if (manure_prod_src.shape == land_use_src.shape and
                    manure_prod_src.transform == land_use_src.transform):
                print("Rasters are aligned. No resampling needed.")
                manure_prod_data = manure_prod_src.read(1)
            else:
                print("Rasters are not aligned. Performing automatic resampling...")
                with MemoryFile() as memfile:
                    with memfile.open(**dst_profile) as dst:
                        reproject(
                            source=rasterio.band(manure_prod_src, 1),
                            destination=rasterio.band(dst, 1),
                            src_transform=manure_prod_src.transform,
                            src_crs=manure_prod_src.crs,
                            dst_transform=dst_profile['transform'],
                            dst_crs=dst_profile['crs'],
                            resampling=Resampling.bilinear
                        )
                        manure_prod_data = dst.read(1)
                print("Automatic resampling completed.")

    print("Raster alignment step completed.\n")
    print("--- Step 4/4: Calculating swine manure application... ---")

    print("Loading NUTS0 boundaries and rasterizing...")
    nuts0_gdf = gpd.read_file(nuts0_shapefile_path)
    nuts0_gdf = nuts0_gdf.to_crs(dst_profile['crs'])
    nuts0_gdf['CNTR_CODE'] = nuts0_gdf['CNTR_CODE'].str.strip().str.upper()
    nuts0_gdf['country_id'] = range(1, len(nuts0_gdf) + 1)
    country_id_map = dict(zip(nuts0_gdf['CNTR_CODE'], nuts0_gdf['country_id']))

    # Print matching info
    print("\n✔ Excel country codes:")
    print(sorted(manure_fractions_df.index.tolist()))
    print("\n✔ Shapefile country codes:")
    print(sorted(nuts0_gdf['CNTR_CODE'].unique().tolist()))

    matched = [c for c in country_id_map if c in manure_fractions_df.index]
    unmatched = [c for c in country_id_map if c not in manure_fractions_df.index]
    print(f"\n✅ Successfully matched countries: {matched}")
    print(f"⚠️ Unmatched countries (will be skipped): {unmatched}")

    country_raster = features.rasterize(
        shapes=[(geom, value) for geom, value in zip(nuts0_gdf.geometry, nuts0_gdf['country_id'])],
        out_shape=land_use_data.shape,
        transform=dst_profile['transform'],
        fill=0,
        all_touched=True,
        dtype=np.uint16
    )

    print("Starting calculation for each country...")
    manure_application_data = np.zeros(land_use_data.shape, dtype=np.float32)
    for nuts_id, country_id in country_id_map.items():
        if nuts_id not in manure_fractions_df.index:
            continue
        country_mask = (country_raster == country_id)
        if not np.any(country_mask):
            continue
        fractions = manure_fractions_df.loc[nuts_id]
        arable_mask = country_mask & (land_use_data == 1)
        if np.any(arable_mask):
            liquid_to_arable = manure_prod_data[arable_mask] * fractions['Frac_Liquid_Available'] * fractions['Frac_Liquid_Arable']
            solid_to_arable = manure_prod_data[arable_mask] * fractions['Frac_Solid_Available'] * fractions['Frac_Solid_Arable']
            manure_application_data[arable_mask] = liquid_to_arable + solid_to_arable
        grassland_mask = country_mask & (land_use_data == 2)
        if np.any(grassland_mask):
            liquid_to_grassland = manure_prod_data[grassland_mask] * fractions['Frac_Liquid_Available'] * fractions['Frac_Liquid_Grassland']
            solid_to_grassland = manure_prod_data[grassland_mask] * fractions['Frac_Solid_Available'] * fractions['Frac_Solid_Grassland']
            manure_application_data[grassland_mask] = liquid_to_grassland + solid_to_grassland

    profile = dst_profile
    profile.update(dtype=rasterio.float32, count=1, nodata=0.0)
    print(f"Calculation complete. Saving final map to {output_path}...\n")
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(manure_application_data, 1)


if __name__ == '__main__':
    # --- 1. Define file paths ---
    FRACTION_EXCEL_PATH = r'D:\ManureAppThesis\MnanureApplication_INTEGRATOR\Fractions\Fraction_swine.xlsx'
    INPUT_CORINE_TIF = r'D:\ManureAppThesis\ManureApplication\CORINE_LandUseRaster1km\COEINERaster1km_withLegend.tif'
    INPUT_NUTS0_SHP = r'D:\ManureAppThesis\01M_NUTS0_WSG84_4326\NUTS_RG_01M_2021_4326_LEVL_0.shp'
    INPUT_SWINE_MANURE_TIF = r'D:\ManureAppThesis\ManureProduction\ManureProduction_tif\swine_manure_production_1km.tif'
    OUTPUT_DIR = r'D:\ManureAppThesis\MnanureApplication_INTEGRATOR'

    # --- 2. Define output file names ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_RECLASSIFIED_TIF = os.path.join(OUTPUT_DIR, 'reclassified_land_use.tif')
    OUTPUT_MANURE_APPLICATION_TIF = os.path.join(OUTPUT_DIR, 'swine_manure_application_map.tif')

    # --- 3. Define land use classification logic ---
    classification_logic = {212: 1, 213: 1, 221: 1, 222: 1, 223: 1, 241: 1, 242: 1, 243: 1, 244: 1, 231: 2}

    # --- 4. Execute workflow ---
    manure_fractions = load_manure_fractions_from_excel(FRACTION_EXCEL_PATH)
    reclassified_path = reclassify_land_use(INPUT_CORINE_TIF, classification_logic, OUTPUT_RECLASSIFIED_TIF)
    calculate_manure_application(
        reclassified_land_use_path=reclassified_path,
        swine_manure_production_path=INPUT_SWINE_MANURE_TIF,
        nuts0_shapefile_path=INPUT_NUTS0_SHP,
        manure_fractions_df=manure_fractions,
        output_path=OUTPUT_MANURE_APPLICATION_TIF
    )

    print("\n--- ✅ All processing complete! ---")
    print(f"Final swine manure application map saved to: {OUTPUT_MANURE_APPLICATION_TIF}")










