"""
Fetch NAIP images from Google Earth Engine (GEE)
"""
# %%
from pathlib import Path
import time

import ee
import geopandas as gpd

from gdstools import (
    GEEImageLoader,
    ConfigLoader,
    multithreaded_execution,
    infer_utm
)

def timeit(method):
    """Decorator that times the execution of a method and prints the time taken."""
    def timed(*args, **kwargs):
        start_time = time.time()
        result = method(*args, **kwargs)
        end_time = time.time()
        print(f"{method.__name__} took {end_time - start_time:.2f} seconds to run.")
        return result
    return timed

# %%
@timeit
def naip_from_gee(
    bbox: list,
    year: int,
    outpath: str or Path,
    outfilepref: str,
    epsg:int=4326,
    scale:int=1,
    overwrite:bool=False
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

    colsize = collection.size().getInfo()

    if colsize == 0:
        print(f"No images found for {year}")

        return 1

    date_range = collection.reduceColumns(ee.Reducer.minMax(), ['system:time_start'])
    ts_end, ts_start = date_range.getInfo().values()

    outpath.mkdir(exist_ok=True, parents=True)

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

        image.to_geotif(outpath, overwrite=overwrite)
        image.save_metadata(outpath)
        image.save_preview(outpath, overwrite=overwrite)

    except Exception as e:
        print(f"Failed to load image for {outfilepref}: {e}")
        return 1

    return 0


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
    if run_as == 'dev':
        PLOTS = Path(conf.DEV_PLOTS)
        DATADIR = Path(conf.DEV_DATADIR)
        gdf = gpd.read_file(PLOTS)
        gdf = gdf.iloc[:10].sort_values('uuid')
        
    else:
        PLOTS = conf.DEV_PLOTS
        DATADIR = Path(conf.DATADIR)
        gdf = gpd.read_file(PLOTS)

    ee.Initialize(opt_url=api_url)

    # Overwrite years if needed
    years = [2013, 2014] #[2009, 2011, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

    for year in years:
        outpath = Path(DATADIR) / 'naip' / str(year)

        params = [
            {
                "bbox": bbox_padding(row.geometry.centroid), #bbox_padding(row.geometry),
                "year": year,
                "outpath": outpath,
                "outfilepref": f"{row.uuid}_{year}_{row.source}_NAIP_DOQQ",
                "overwrite": True
            } for row in gdf.itertuples()
        ]

        multithreaded_execution(naip_from_gee, params, 10)
        #naip_from_gee(**params[0])
