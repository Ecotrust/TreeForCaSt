"""
Oregon Statewide Imagery Program (OSIP) - Oregon Imagery Framework Implementation Team
"""

import json
from typing import Tuple
import requests
from pathlib import Path
from PIL import Image

import numpy as np
import geopandas as gpd
from shapely import geometry
import rasterio
from rasterio import MemoryFile
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

from gdstools import (
    degrees_to_meters,
    print_message,
    multithreaded_execution,
    ConfigLoader, 
    infer_utm,
)

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


def naip_from_rest(bbox, res=1, crs=4326, **kwargs):
    """
    Retrieve a Digital Elevation Model (DEM) image from The National Map (TNM)
    web service.

    :param bbox: list-like
        List of bounding box coordinates (minx, miny, maxx, maxy).
    :type bbox: list
    :param res: numeric
        Spatial resolution to use for returned DEM (grid cell size).
    :type res: float
    :param inSR: int
        Spatial reference for bounding box, such as an EPSG code (e.g., 4326).
    :type inSR: int

    :returns: numpy array
        DEM image as array.
    """
    xmin, ymin, xmax, ymax = bbox

    if crs == 4326:
        dx = degrees_to_meters(xmax - xmin)
        dy = degrees_to_meters(ymax - ymin, angle='lat')
    else:
        dx = xmax - xmin
        dy = ymax - ymin

    width = int(abs(dx) // res)  # type: ignore
    height = int(abs(dy) // res)  # type: ignore

    BASE_URL = ''.join([
        'https://imagery.oregonexplorer.info/arcgis/rest/',
        'services/OSIP_2018/OSIP_2018_SL/ImageServer/exportImage'
    ])

    params = dict(
        bbox=','.join([str(x) for x in bbox]),
        bboxSR=crs,
        size=f'{width},{height}',
        imageSR=crs,
        time=None,
        format='tiff',
        pixelType='U8',
        noData=None,
        noDataInterpretation='esriNoDataMatchAny',
        interpolation='RSP_NearestNeighbor',
        compression=None,
        compressionQuality=None,
        bandIds=None,
        mosaicRule=None,
        renderingRule=None,
        f='image'
    )

    for key, value in kwargs.items():
        params.update({key: value})

    r = requests.get(BASE_URL, params=params)
    with MemoryFile(r.content) as memfile:
        try:
            src = memfile.open()
            image = src.read()
            profile = src.profile
        except rasterio.errors.RasterioIOError:
            print('No data returned from server ...')
            image = None
            profile = None

    return image, profile


def fetch_naip(
        bbox: Tuple[float, float, float, float],
        prefix: str,
        outdir: Path,
        epsg:int=4326,
        overwrite:bool=False, 
    ) -> None:

    band_info = {
        'R': 'Red',
        'G': 'Green',
        'B': 'Blue',
        'N': 'Near Infrared',
    }

    outfile = outdir / f'{prefix}_NAIP_OSIP-cog.tif'
    if outfile.exists() and not overwrite:
        print_message(f'File exists: {outfile.name}')
        return

    # Get the NAIP image
    image, profile = naip_from_rest(bbox, res=1, crs=epsg)
    if image is None:
        print_message(f'Failed to fetch {outfile.name}')
        return
    
    cog_profile = cog_profiles.get("deflate")
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(image)
            for i, band in enumerate(band_info.keys()):
                dst.set_band_description(i + 1, band_info[band])
        
            cog_translate(
                dst,
                outfile,
                cog_profile,
                in_memory=True,
                quiet=True
            )

            # Generate preview
            preview = Image.fromarray(np.moveaxis(image[:3], 0, -1)).convert('RGB')
            preview.save(outfile.parent / outfile.name.replace('-cog.tif',
                        '-preview.png'), optimize=True)

            # Create metadata file
            xoff, yoff = dst.bounds.left, dst.bounds.top
            geom = geometry.box(*dst.bounds)
            coordinates = [list(geom.exterior.coords)]
            unixdate = 1514764800 #2018-01-01

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
                    "description": "A State Lambert mosaic derived from one-foot resolution color Digital Orthophoto Quadrangles (DOQ) of the western half of the state of Oregon.This digital, geographically referenced data set was developed for the Oregon GIS department to provide updated state wide imagery. Digital 4 band ortho imagery covering the state of Oregon was flown in 2018. The 4 Band imagery was rectified and cut into a DOQs."
                },
                "id": "image"
            }

            with open(outfile.parent / outfile.name.replace('-cog.tif', '-metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
    print_message(f'Image saved as {outfile}')
    return


if __name__ == '__main__':

    run_as = 'dev'
    res = 1 

    # Load config file
    conf = ConfigLoader(Path(__file__).parent.parent).load()
    OUTLIERS = Path(conf.OUTLIERS)
    PLOTS = Path(conf.PLOTS)
    plots = gpd.read_file(PLOTS)

    if run_as == 'dev':
        DATADIR = Path(conf.DEV_DATADIR)
        plots = gpd.read_file(PLOTS)
        plots = plots.sort_values('uuid').iloc[:20]
    else:
        DATADIR = Path(conf.DATADIR)

    # bbox = plots[plots.source == 'BLM-COOS'].iloc[0].geometry.bounds
    
    out_path = DATADIR / 'interim' / 'naip' / '2018'
    out_path.mkdir(exist_ok=True, parents=True)

    params = [
        {
            'bbox': bbox_padding(row.geometry.centroid, 90),
            'prefix': f'{row.uuid}_2018_{row.source}',
            'outdir': out_path,
            'overwrite': True
        } 
        for row in plots.itertuples()
    ]

    # fetch_naip(**params[0])
    multithreaded_execution(fetch_naip, params)
