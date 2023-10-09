"""
Fetch NAIP images from Google Earth Engine (GEE)
"""
# %%
from pathlib import Path
import os
import time
from multiprocessing.pool import ThreadPool
from functools import partial
from PIL import Image
import json

import numpy as np
from affine import Affine
from shapely.geometry import box
import ee
import geopandas as gpd

from gdstools import (
    GEEImageLoader,
    ConfigLoader,
    multithreaded_execution,
    infer_utm,
    split_bbox,
    save_cog
)

def timeit(method):
    """Decorator to time the execution of a function."""
    def timed(*args, **kwargs):
        start_time = time.time()
        result = method(*args, **kwargs)
        end_time = time.time()
        print(f"{method.__name__} took {end_time - start_time:.2f} seconds to run.")
        return result
    return timed

# %%
def naip_from_gee(
    bbox: list,
    year: int,
    outfilepref: str,
    epsg:int=4326,
    scale:int=1,
):
    """
    Fetch NAIP image url from Google Earth Engine (GEE) using a bounding box.

    :param bbox: Bounding box in the form [xmin, ymin, xmax, ymax].
    :type bbox: list
    :param year: Year (e.g. 2019)
    :type year: int
    :param epsg: EPSG code for coordinate reference system. Default is 4326.
    :type epsg: int, optional
    :param scale: Resolution in meters of the image to fetch. Default is 1.
    :type scale: int, optional
    :return: Returns a tuple containing the image as a numpy array and its metadata as a dictionary.
    :rtype: Tuple[np.ndarray, Dict]
    """
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    # get the naip image collection for our aoi and timeframe
    eebbox = ee.Geometry.BBox(*bbox)
    collection = (
        ee.ImageCollection("USDA/NAIP/DOQQ")
        .filterDate(start_date, end_date)
        .filterBounds(eebbox)
    )

    if collection.size().getInfo() == 0:
        print(f"No images found for {year}")
        return 1

    date_range = collection.reduceColumns(ee.Reducer.minMax(), ['system:time_start'])
    ts_end, ts_start = date_range.getInfo().values()

    # outpath.mkdir(exist_ok=True, parents=True)
    try:
        image = GEEImageLoader(collection.median().clip(eebbox))
        image.metadata_from_collection(collection)
        image.set_property("system:time_start", ts_start)# * 1000)
        image.set_property("system:time_end", ts_end)# * 1000)
        image.set_params("crs", f"EPSG:{epsg}")
        image.set_params("scale", scale)
        image.set_params("region", eebbox)
        image.set_viz_params("min", 0)
        image.set_viz_params("max", 255)
        image.set_viz_params("bands", ["R", "G", "B"])
        image.id = outfilepref

        return image.to_array(), image.metadata 

    except Exception as e:
        print(f"Failed to load image for {outfilepref}: {e}")
        return 1


def quad_fetch(
        bbox: tuple, 
        dim:int=1, 
        num_threads:int=None, 
        **kwargs
    ):
    """
    Breaks user-provided bounding box into quadrants and retrieves data
    using `fetcher` for each quadrant in parallel using a ThreadPool.

    :param collection: Earth Engine image collection.
    :type collection: ee.Collection
    :param bbox: Coordinates of x_min, y_min, x_max, and y_max for bounding box of tile.
    :type bbox: tuple
    :param dim: Dimension of the quadrants to split the bounding box into. Default is 1.
    :type dim: int, optional
    :param num_threads: Number of threads to use for parallel executing of data requests. Default is None.
    :type num_threads: int, optional

    :return: Returns a tuple containing the image as a numpy array and its metadata as a dictionary.
    :rtype: tuple
    """
    # def clip_image(bbox, scale, epsg):
    #     ee_bbox = ee.Geometry.BBox(*bbox)
    #     image = GEEImageLoader(collection.median().clip(ee_bbox))
    #     image.set_params("scale", scale)
    #     image.set_params("crs", f"EPSG:{epsg}")
    #     return image.to_array()

    if dim > 1:
        if num_threads is None:
            num_threads = dim**2

        bboxes = split_bbox(dim, bbox)

        get_quads = partial(naip_from_gee, **kwargs)
        with ThreadPool(num_threads) as p:
            quads = p.map(get_quads, bboxes)

        # Split quads list in tuples of size dim
        images = [x[0][0] for x in quads]
        quad_list = [images[x:x + dim] for x in range(0, len(images), dim)]

        # Reverse order of rows to match rasterio's convention
        [x.reverse() for x in quad_list]
        image = np.concatenate(
            [
                np.hstack(quad_list[x]) for x in range(0, len(quad_list))
            ], 2
        )

        profile = quads[0][0][1]
        first = quads[0][0][1]['transform']
        last = quads[-1][0][1]['transform']
        profile['transform'] = Affine(
            first.a,
            first.b,
            first.c,
            first.d,
            first.e,
            last.f
        )
        h, w = image.shape[1:]
        profile.update(width=w, height=h)
        profile['dtype'] = 'uint8'
        metadata = quads[0][1]
        metadata['type'] = 'Image'
        metadata['properties']['system:footprint']['coordinates'] = list(box(*bbox).exterior.coords)

        return image, profile, metadata

    else:
        return naip_from_gee(bbox, **kwargs)
    

@timeit
def get_naip(
    bbox:tuple, 
    outpath: str or Path, 
    outfilepref: str,
    year:int, 
    dim:int=3, 
    overwrite:bool=False, 
    # num_threads:int=None
):
    """
    Downloads a NAIP image from Google Earth Engine and save it as a Cloud-Optimized GeoTIFF (COG) file.

    :param bbox: list-like
        list of bounding box coordinates (minx, miny, maxx, maxy)
    :type bbox: list
    :param year: int
        year of the NAIP image to download
    :type year: int
    :param outpath: str or Path
        path to save the downloaded image
    :type outpath: str or Path
    :param dim: int, optional
        dimension of the image to download (default is 3)
    :type dim: int
    :param overwrite: bool, optional
        whether to overwrite an existing file with the same name (default is False)
    :type overwrite: bool
    :param num_threads: int, optional
        number of threads to use for downloading (default is None)
    :type num_threads: int

    :return: None
    :rtype: None
    """
    print(f"Fetching {outfilepref}...")
    if os.path.exists(outpath / f'{outfilepref}-cog.tif') and not overwrite:
        print(f"{outpath} already exists. Skipping...")
        return

    try:
        image, profile, metadata = quad_fetch(bbox, dim, None, outfilepref=outfilepref, year=year)
    except Exception as e:
        print(f"Failed to fetch {outfilepref}: {e}")
        return

    preview = Image.fromarray(
        np.moveaxis(image[:3], 0, -1).astype(np.uint8)
    ).convert('RGB')
    h, w = preview.size
    preview = preview.resize((w//10, h//10))
    preview.save(outpath / f'{outfilepref}-preview.png', optimize=True)
    # profile = ['properties']['profile']
    # metadata['properties'].pop('profile')

    with open(outpath / f'{outfilepref}-metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
        
    save_cog(image, profile, outpath / f'{outfilepref}-cog.tif', overwrite=overwrite)

    return


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


if "__main__" == __name__:

    run_as = 'dev'
    res = 1
    # Load config file
    conf = ConfigLoader(Path(__file__).parent.parent).load()
    api_url = conf['items']['naip']['providers']['Google']['api']
    GRID = Path(conf.GRID)
    grid = gpd.read_file(GRID)

    if run_as == 'dev':
        PLOTS = Path(conf.DEV_PLOTS)
        DATADIR = Path(conf.DEV_DATADIR)
        gdf = gpd.read_file(PLOTS)
        gdf = gdf.sort_values('uuid').iloc[:20]
        ovly = grid.overlay(gdf)
        gdf = grid[grid.CELL_ID.isin(ovly['CELL_ID'].unique())]
        
    else:
        PLOTS = conf.DEV_PLOTS
        DATADIR = Path(conf.DATADIR)
        gdf = gpd.read_file(PLOTS)

    ee.Initialize(opt_url=api_url)

    # Overwrite years if needed
    years = [2009, 2011, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

    for year in years:
        outpath = Path(DATADIR) / 'fulltiles'/ 'naip' / str(year)
        outpath.mkdir(exist_ok=True, parents=True)

        params = [
            {
                "bbox": row.geometry.bounds, #bbox_padding(row.geometry.centroid, padding=300), 
                "year": year,
                "dim": 6,
                "outpath": outpath,
                "outfilepref": f"{row.CELL_ID}_{row.PRIMARY_STATE}_{year}_NAIP_DOQQ", #f"{row.uuid}_{year}_{row.source}_NAIP_DOQQ",
                "overwrite": True
            } for row in gdf.itertuples()
        ]

        # Product dim^2 x threads must be < 30 or else GEE will throw an error
        multithreaded_execution(get_naip, params, 1)
        # get_naip(**params[0])
