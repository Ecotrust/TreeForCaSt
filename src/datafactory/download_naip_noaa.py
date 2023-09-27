"""
Download NAIP imagery from NOAA

Find the tif URL from NOAA tile index shapefiles. For each plot, we will find the tile 
that intersects the plot bbox in the tile index shp. Since there are two shapefiles, 
one for OR and one for WA, we will import and concaternate them into a single geodataframe. 

The steps are:
1. Import the tile index shapefiles as geodataframes
2. Concaternate both geodataframes
3. Import the plot data
4. Iterate over the plots and 
  a. find the tile that intersects with plot bbox. Different plots can inteserct the same tile.
  b. retrieve the url of the tif file
  c. retrieve the name of the tile
  d. open tif url with rasterio

NOAA seems to have NAIP data only for 2021 (WA), 2020 (OR), 2016 (OR), 2015 (WA) 
"""

# %%
from datetime import datetime
from pathlib import Path
import json
from PIL import Image

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import transform, MemoryFile, windows
from rasterio.warp import reproject, Resampling
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from shapely.geometry import mapping, box
from pyproj import CRS

from gdstools import degrees_to_meters, infer_utm

# %%
# DATADIR = Path('/mnt/data/FESDataRepo/stac_plots/data/')
DATADIR = Path('/mnt/data/users/ygalvan/fbstac_plots/data/dev/')
replace = True

def center_crop_array(new_size, array):
    xpad, ypad = (np.subtract(array.shape, new_size)/2).astype(int)
    dx, dy = np.subtract(new_size, array[xpad:-xpad, ypad:-ypad].shape)
    return array[xpad:-xpad+dx, ypad:-ypad+dy]

def bbox_padding(geom:object, padding:int=1e3):
    """
    Add padding to a bounding box.

    :param geom: shapely.geometry.Polygon
        The geometry to add padding to.
    :type geom: shapely.geometry.Polygon
    :param padding: float, optional
        The amount of padding to add to the geometry, in meters. Default is 1000.
    :type padding: float

    :return: tuple
        A tuple of four floats representing the padded bounding box coordinates (minx, miny, maxx, maxy).
    :rtype: tuple
    """
    p_crs = infer_utm(geom.bounds)
    p_geom = gpd.GeoSeries(geom, crs=4326).to_crs(p_crs)
    if padding > 0:
        p_geom = p_geom.buffer(padding, join_style=2)

    return p_geom.to_crs(4326).bounds.values[0]

crs4326 = CRS.from_epsg(4326)

# Load plots and naip file info
plots = gpd.read_file(DATADIR / 'features/plot_features.geojson')
plots = plots.iloc[:10].sort_values(by='uuid')
orwa_tileidx = gpd.read_file(DATADIR / 'features/orwa_naip_tileidx.geojson')

# Four vrt files, two for OR and two for WA
# URL0 = 'https://coast.noaa.gov/htdata/raster5/imagery/WA_NAIP_2021_9586/WA_NAIP_2021.0.vrt'
# URL1 = 'https://coast.noaa.gov/htdata/raster5/imagery/WA_NAIP_2021_9586/WA_NAIP_2021.0.vrt'
# URL2 = 'https://coastalimagery.blob.core.windows.net/digitalcoast/OR_NAIP_2020_9504/or_naip_2020_10.vrt'
# URL3 = 'https://coastalimagery.blob.core.windows.net/digitalcoast/OR_NAIP_2020_9504/or_naip_2020_11.vrt'
# -- We don't need the vrt files as we are using the tif urls directly. Keeping links for reference --

cog_profile = cog_profiles.get("deflate")

# %% 
errors = []
for i, row in plots.iterrows():

    geom = plots[plots.index == i].geometry.values[0]
    geom = box(*bbox_padding(geom.centroid))
    bbox = geom.bounds

    # Find the tile that intersects with plot bbox
    tile = orwa_tileidx[orwa_tileidx.intersects(box(*bbox))]
    tile_url = tile.url.iloc[0]
    tile_name = tile.location.iloc[0]
    date = tile.date.iloc[0]
    state = tile.state.iloc[0]

    outfile = DATADIR / f"naip/{date.year}" / f"{row.uuid}_{date.year}_{row.source}_NAIP_NOAA-cog.tif"
    if outfile.exists() and replace is False:
        print('File exists, skipping', outfile)
        continue

    outfile.parent.mkdir(parents=True, exist_ok=True)

    print('Fetching', row.uuid)
    with rasterio.open(tile_url) as src:
        geom = gpd.GeoSeries(geom, crs=4326).to_crs(src.crs)

        window = windows.from_bounds(*geom[0].bounds, src.transform)
        src_transform = rasterio.windows.transform(window, src.transform)
        data = src.read(window=window)           
    
        width = int(np.ceil(degrees_to_meters(bbox[2]-bbox[0])/src.res[0]))
        height = int(np.ceil(degrees_to_meters(bbox[-1]-bbox[1])/src.res[1]))

        dst_transform = transform.from_bounds(*bbox, width, height)

        PROFILE = {
            'driver': 'GTiff',
            'interleave': 'band',
            'tiled': True,
            'crs': crs4326,
            'transform': dst_transform,
            'width': width,
            'height': height,
            'blockxsize': 256,
            'blockysize': 256,
            'compress': 'lzw',
            'count': src.count,
            'dtype': rasterio.uint8,
        }

        print('Writing', outfile)
        with MemoryFile() as memfile:
            with memfile.open(**PROFILE) as dst:
                output = np.zeros((src.count, height, width), rasterio.uint8)
                try:
                    reproject(
                        source=data,#np.array(bands),
                        destination=output,#rasterio.band(dst, i+1),
                        src_transform=src_transform,#.window_transform(window),
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=rasterio.crs.CRS.from_epsg(4326),
                        resampling=Resampling.bilinear
                    )
                    dst.write(output)

                    cog_translate(
                        dst,
                        outfile,
                        cog_profile,
                        in_memory=True,
                        quiet=True
                    )
                except Exception as e:
                    print('Failed to write', outfile, e)
                    errors.append(row.uuid)
                    continue

                # Generate and save preview
                print('Writing preview ...')
                preview = Image.fromarray(
                    np.moveaxis(output[:3], 0, -1)).convert('RGB')
                h, w = preview.size
                # Change preview res to 30m
                new_w = int(w * src.res[0] / 3)
                new_h = int(h * src.res[1] / 3)
                preview = preview.resize((new_w, new_h))
                preview.save(outfile.parent / outfile.name.replace('-cog.tif',
                            '-preview.png'), optimize=True)


                # Generate and save metadata
                print('Writing metadata ...')
                xoff, yoff = (dst_transform.xoff, dst_transform.yoff)
                coordinates = mapping(row.geometry)['coordinates'][0][0]
                # date = datetime.strptime(
                #     tile_url.split('_')[-1].replace('.tif',''), "%Y%m%d")
                unixdate = int(datetime.timestamp(date)*1000)

                metadata = {
                    "type": "ImageCollection",
                    "bands": [
                        {
                        "id": "R",
                        "data_type": {
                            "type": "PixelType",
                            "precision": "double",
                            "min": 0,
                            "max": 255
                        },
                        "dimensions": [1,1],
                        "origin": [xoff, yoff],
                        "crs": "EPSG:4326",
                        "crs_transform": [1,0,0,0,1,0]
                        },
                        {
                        "id": "G",
                        "data_type": {
                            "type": "PixelType",
                            "precision": "double",
                            "min": 0,
                            "max": 255
                        },
                        "dimensions": [1,1],
                        "origin": [xoff, yoff],
                        "crs": "EPSG:4326",
                        "crs_transform": [1,0,0,0,1,0]
                        },
                        {
                        "id": "B",
                        "data_type": {
                            "type": "PixelType",
                            "precision": "double",
                            "min": 0,
                            "max": 255
                        },
                        "dimensions": [1,1],
                        "origin": [xoff, yoff],
                        "crs": "EPSG:4326",
                        "crs_transform": [1,0,0,0,1,0]
                        },
                        {
                        "id": "N",
                        "data_type": {
                            "type": "PixelType",
                            "precision": "double",
                            "min": 0,
                            "max": 255
                        },
                        "dimensions": [1,1],
                        "origin": [xoff, yoff],
                        "crs": "EPSG:4326",
                        "crs_transform": [1,0,0,0,1,0]
                        }
                    ],
                    "properties": {
                        "system:footprint": {
                        "geodesic": 'false',
                        "type": "Polygon",
                        "coordinates": [coordinates]
                        },
                        "system:time_start": unixdate,
                        "system:time_end": unixdate,
                        "description": "<p>The National Agriculture Imagery Program (NAIP) acquires aerial imagery\nduring the agricultural growing seasons in the continental U.S.</p><p>NAIP projects are contracted each year based upon available funding and the\nimagery acquisition cycle. Beginning in 2003, NAIP was acquired on\na 5-year cycle. 2008 was a transition year, and a three-year cycle began\nin 2009.</p><p>NAIP imagery is acquired at a one-meter ground sample distance (GSD) with a\nhorizontal accuracy that matches within six meters of photo-identifiable\nground control points, which are used during image inspection.</p><p>Older images were collected using 3 bands (Red, Green, and Blue: RGB), but\nnewer imagery is usually collected with an additional near-infrared band\n(RGBN). RGB asset ids begin with &#39;n<em>&#39;, NRG asset ids begin with &#39;c</em>&#39;, RGBN\nasset ids begin with &#39;m_&#39;.</p><p><b>Provider: <a href=\"https://www.fsa.usda.gov/programs-and-services/aerial-photography/imagery-programs/naip-imagery/\">USDA Farm Production and Conservation - Business Center, Geospatial Enterprise Operations</a></b><br><p><b>Resolution</b><br>1 meter\n</p><p><b>Bands</b><table class=\"eecat\"><tr><th scope=\"col\">Name</th><th scope=\"col\">Description</th></tr><tr><td>R</td><td><p>Red</p></td></tr><tr><td>G</td><td><p>Green</p></td></tr><tr><td>B</td><td><p>Blue</p></td></tr><tr><td>N</td><td><p>Near infrared</p></td></tr></table><p><b>Terms of Use</b><br><p>Most information presented on the FSA Web site is considered public domain\ninformation. Public domain information may be freely distributed or copied,\nbut use of appropriate byline/photo/image credits is requested. For more\ninformation visit the <a href=\"https://www.fsa.usda.gov/help/policies-and-links\">FSA Policies and Links</a>\nwebsite.</p><p>Users should acknowledge USDA Farm Production and Conservation -\nBusiness Center, Geospatial Enterprise Operations when using or\ndistributing this data set.</p><p><b>Suggested citation(s)</b><ul><li><p>USDA Farm Production and Conservation - Business Center, Geospatial Enterprise Operations</p></li></ul><style>\n  table.eecat {\n  border: 1px solid black;\n  border-collapse: collapse;\n  font-size: 13px;\n  }\n  table.eecat td, tr, th {\n  text-align: left; vertical-align: top;\n  border: 1px solid gray; padding: 3px;\n  }\n  td.nobreak { white-space: nowrap; }\n</style>"
                    },
                    "id": "image"
                }

                with open(outfile.parent / outfile.name.replace('-cog.tif', '-metadata.json'), 'w') as f:
                    json.dump(metadata, f, indent=2)

if len(errors) > 0:
    print('The following plots encountered errors while fetching:\n', errors)

print('Done!')
