# %%
from pathlib import Path
import json
import os
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from rasterio import MemoryFile, windows
from PIL import Image
import rasterio
import geopandas as gpd
import numpy as np
import shapely

from gdstools import multithreaded_execution, image_collection, infer_utm

collection = '3dep_dtm'

images = image_collection(f'data/dev/interim/{collection}')
outpath = Path(f'data/dev/processed/{collection}')
plots = gpd.read_file('data/dev/features/plot_features.geojson')

buffer_size = 60
naip_path = 'data/dev/stac/naip_reprojected'

if collection == 'landsat8':
    match_naip = True
    res=30
    max_pixval = 0.3
    dtype = rasterio.float32
    viz_bands = [1,2,3]
elif collection == 'naip':
    match_naip = False
    res=.5
    max_pixval = 255
    dtype = rasterio.uint8
    viz_bands = [0,1,2]
elif collection == '3dep':
    match_naip = True
    res=1
    max_pixval = 3000
    dtype = rasterio.float32
    viz_bands = [0]
elif collection == '3dep_dtm':
    match_naip = True
    res=10
    max_pixval = 100
    dtype = rasterio.float32
    viz_bands = [0]

def reproject_image(filepath, outpath, match_naip=False, naip_path=None):
    with rasterio.open(filepath) as src:
        
        data = src.read()
        dst_crs = infer_utm(list(src.bounds))
        year = Path(filepath).name.split('_')[1]
        uuid = Path(filepath).name.split('_')[0]

        # %%
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )

        kwargs = src.meta.copy()
        kwargs.update(
            {
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
                "dtype": dtype,
                "count": src.count,
            }
        )

        cog_profile = cog_profiles.get("deflate")
        with MemoryFile() as memfile:
            with memfile.open(**kwargs) as dst:
                output = np.zeros((src.count, dst.shape[0], dst.shape[1]), dtype)
                reproject(
                    source=data,
                    destination=output,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                )

                if year.isdigit():
                    out_path = Path(outpath) / year / Path(filepath).name
                else:
                    out_path = Path(outpath) / Path(filepath).name
                out_path.parent.mkdir(parents=True, exist_ok=True)
                dst.write(output)

                # Extract 120x120m window using plot bbox
                plot = plots[plots.uuid == uuid]
                new_bbox = [plot.utm_xmin.iloc[0], plot.utm_ymin.iloc[0], plot.utm_xmax.iloc[0], plot.utm_ymax.iloc[0]]
                # if match_naip:
                #     naip_collection = image_collection(naip_path)
                #     uuid = Path(filepath).name.split('_')[0]
                #     naip = [i for i in naip_collection if Path(i).name.split('_')[0] == uuid][0]

                #     if naip:
                #         with rasterio.open(naip) as match:
                #             new_bbox = list(match.bounds)
                # else:
                #     geom = shapely.geometry.box(*list(dst.bounds))
                #     centroid = gpd.GeoSeries(geom, crs=dst_crs).centroid
                #     new_bbox = centroid.buffer(buffer_size, join_style=2).bounds.values[0]

                new_geom = shapely.geometry.box(*list(new_bbox))
                window = windows.from_bounds(*new_bbox, dst.transform)

                # Update metadata
                metadatapath = filepath.replace('-cog.tif', '-metadata.json')
                if os.path.exists(metadatapath):
                    f = open(filepath.replace('-cog.tif', '-metadata.json'))
                    metadata = json.load(f)

                    for band in metadata['bands']:
                        band['crs'] = f'EPSG:{dst_crs.to_epsg()}'
                        band['origin'] = [new_bbox[0], new_bbox[-1]]

                    if 'system:footprint' in metadata['properties']:
                        metadata['properties']['system:footprint']['coordinates'] = [list(l) for l in list(new_geom.exterior.coords)]
                    else:
                        metadata['properties']['system:footprint'] = {
                            'type': 'Polygon',
                            'coordinates': [list(l) for l in list(new_geom.exterior.coords)]
                        }

                    with open(out_path.parent / out_path.name.replace('-cog.tif', '-metadata.json'), 'w') as f:
                        json.dump(metadata, f, indent=4)

                transform, width, height = calculate_default_transform(
                    dst_crs, dst_crs, buffer_size*2/res, buffer_size*2/res, *new_bbox
                )

                new_data = dst.read(window=window)

                # Save preview
                if len(viz_bands) > 1:
                    preview = Image.fromarray(
                        np.moveaxis((new_data[viz_bands] / max_pixval) * 255, 0, -1).astype(np.uint8)).convert('RGB')
                else:
                    preview = Image.fromarray(
                        (np.squeeze(new_data[viz_bands]) / max_pixval) * 255).convert('RGB')

                preview.save(out_path.parent / out_path.name.replace('-cog.tif', '-preview.png'), optimize=True)

                kwargs.update(
                    {
                        "transform": transform,
                        "width": width,
                        "height": height,
                    }
                )

                with MemoryFile() as memfile2:
                    with memfile2.open(**kwargs) as dst2:
                        output = np.zeros((src.count, new_data.shape[0], new_data.shape[1]), dtype)
                        dst2.write(new_data)

                        cog_translate(dst2, out_path, cog_profile, in_memory=True, quiet=False)

params = [
    {
        'filepath': i,
        'outpath': outpath,
        'match_naip': match_naip,
        'naip_path': naip_path
    }
    for i in images
]

multithreaded_execution(reproject_image, params)
