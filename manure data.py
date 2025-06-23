import rasterio
from rasterio.enums import Resampling
import numpy as np
import os

# ====== 设置输入输出路径 ======
input_path = r"D:\ManureAppThesis\CORINERaster100m\DATA\CORINERaster100m2020.tif"  # 请改成你的.tif文件路径
output_path = r"D:\ManureAppThesis\CORINERaster100m2020_1km.tif"

# ====== 设置聚合因子（10x 降采样）======
scale_factor = 10

# ====== 打开源.tif文件 ======
with rasterio.open(input_path) as src:
    # 计算输出的新形状
    out_height = src.height // scale_factor
    out_width = src.width // scale_factor
    out_shape = (src.count, out_height, out_width)

    # 重采样数据（众数）
    data = src.read(
        out_shape=out_shape,
        resampling=Resampling.mode  # 用众数聚合每10x10像素
    )

    # 计算新的变换矩阵（地理参考）
    transform = src.transform * src.transform.scale(
        (src.width / out_width),
        (src.height / out_height)
    )

    # ====== 写入输出.tif文件 ======
    profile = src.profile
    profile.update({
        "height": out_height,
        "width": out_width,
        "transform": transform,
        "compress": "lzw"  # 可选压缩，节省空间
    })

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data)

print("✅ 聚合完成，输出文件路径：", output_path)









