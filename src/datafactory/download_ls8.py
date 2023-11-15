# %%
import os
from pathlib import Path

import pandas as pd
import geopandas as gpd
import ee

from gdstools import (
    create_directory_tree,
    multithreaded_execution, 
    GEEImageLoader,
    ConfigLoader,
)

def prepSrL8(image):
  """A function to prepare cloud free Landsat 8 (C2) surface reflectance images.
  
  Adapted from https://gis.stackexchange.com/a/425160/72937
  """
  # Develop masks for unwanted pixels (fill, cloud, cloud shadow).
  qaMask = image.select('QA_PIXEL').bitwiseAnd(int('11111', 2)).eq(0);
  saturationMask = image.select('QA_RADSAT').eq(0);

  # Apply the scaling factors to the appropriate bands.
  def getFactorImg(factorNames): 
    factorList = image.toDictionary().select(factorNames).values();
    return ee.Image.constant(factorList);

  scaleImg = getFactorImg([
    'REFLECTANCE_MULT_BAND_.|TEMPERATURE_MULT_BAND_ST_B10']);
  offsetImg = getFactorImg([
    'REFLECTANCE_ADD_BAND_.|TEMPERATURE_ADD_BAND_ST_B10']);
  scaled = image.select('SR_B.|ST_B10').multiply(scaleImg).add(offsetImg);

  # Replace original bands with scaled bands and apply masks.
  return image.addBands(scaled, None, True).updateMask(qaMask).updateMask(saturationMask)


# %%
def get_landsat8(
    bbox,
    year,
    out_path,
    prefix,
    season="leafon",
    epsg=4326,
    scale=30,
    overwrite=False,
):
    """
    Fetch LandSat 8 image from GEE and save to disk.

    Parameters
    ----------
    month : int
        Month of year (1-12)
    year : int
        Year (e.g. 2019)
    bbox : list
        Bounding box in the form [xmin, ymin, xmax, ymax].

    Returns
    -------
    url : str
        GEE generated URL from which the raster will be downloaded.
    metadata : dict
        Image metadata.
    """
    # TODO: define method to get collection, then pass an instance of that 
    # collection to the GEEImageLoader to avoid instantiating the collection
    # every time this function is called.
    filename = f"{prefix}_Landsat8_{season}"
    if os.path.exists(os.path.join(out_path, filename + '-cog.tif')) and not overwrite:
        print(f"File exists: {filename}-cog.tif")
        return 0

    def apply_scale(image): 
        optical = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        thermal = image.select('ST_B.*').multiply(0.00341802).add(149.0)
    
        return image.addBands(optical, None, True).addBands(thermal, None, True);

    if season == "leafoff":
        start_date = f"{year - 1}-10-01"
        end_date = f"{year}-03-31"
    elif season == "leafon":
        start_date = f"{year}-04-01"
        end_date = f"{year}-09-30"
    else:
        raise ValueError(f"Invalid season: {season}")

    collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")\
        .filterDate(start_date, end_date)\
        .map(prepSrL8)\
        # .select('SR.*')\
        # .median()
    # ts_start = datetime.timestamp(datetime.strptime(start_date, "%Y-%m-%d"))
    # ts_end = datetime.timestamp(datetime.strptime(end_date, "%Y-%m-%d"))

    bbox = ee.Geometry.BBox(*bbox)
    # image = apply_scale(collection.median().clip(bbox))
    image = collection.median().clip(bbox)
    
    # Mask clouds
    # scored = ee.Algorithms.Landsat.simpleCloudScore(image)
    # mask = scored.select(["cloud"]).lte(20)
    # image = image.updateMask(mask)
    
    image = GEEImageLoader(image)
    
    # Set image metadata and params
    image.metadata_from_collection(collection)
    # image.set_property("system:time_start", ts_start * 1000)scale
    # image.set_property("system:time_end", ts_end * 1000)
    image.set_params("scale", scale)
    image.set_params("crs", f"EPSG:{epsg}")
    image.set_params("region", bbox)
    image.set_viz_params("min", 0)
    image.set_viz_params("max", 0.4)
    image.set_viz_params("bands", ['SR_B6', 'SR_B5', 'SR_B3'])
    image.id = filename

    # Download cog
    # out_path = path / image.id
    # out_path.mkdir(parents=True, exist_ok=True)

    image.to_geotif(out_path, overwrite=overwrite)
    image.save_preview(out_path, overwrite=overwrite)
    image.save_metadata(out_path)
    return 0


def bbox_padding(geom, padding=1e3):
    from gdstools import infer_utm
    p_crs = infer_utm(geom.bounds)
    p_geom = gpd.GeoSeries(geom, crs=4326).to_crs(p_crs)
    if padding > 0:
        p_geom = p_geom.buffer(padding, join_style=2)

    return p_geom.to_crs(4326).bounds.values[0]


if __name__ == "__main__":

    run_as = "prod"
    conf = ConfigLoader(Path(__file__).parent.parent).load()
    api_url = conf['items']['landsat8']['providers']['Google']['api']
    OUTLIERS = Path(conf.OUTLIERS)
    PLOTS = Path(conf.PLOTS)
    # GRID = Path(conf.GRID)
    # grid = gpd.read_file(GRID)
    outliers = pd.read_csv(OUTLIERS)
    outliers['uuid'] = outliers.outlier_uuid.apply(lambda x: x.split('-')[0])
    outliers = outliers.set_index('uuid')
    plots = gpd.read_file(PLOTS)
    plots = plots.set_index('uuid')
    plots = plots[~plots.index.isin(outliers.index)]

    if run_as == 'dev':
        DATADIR = Path(conf.DEV_DATADIR)
        plots = plots.sort_index().iloc[:20]
        overwrite = True
        
    else:
        PLOTS = conf.PLOTS
        DATADIR = Path(conf.DATADIR)
        overwrite = False
        # plots = gpd.read_file(PLOTS)

    # ovly = grid.overlay(plots)
    # plots = grid[grid.CELL_ID.isin(ovly['CELL_ID'].unique())]

    # Initialize the Earth Engine module.
    # Setup your Google Earth Engine API key before running this script.
    # %%
    ee.Initialize(opt_url=api_url)

    # Overwrite years if needed
    years = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

    for year in years:

        out_path = create_directory_tree(DATADIR, 'interim', 'landsat8', str(year))

        params = [
            {
                "bbox": bbox_padding(row.geometry.centroid, padding=90), #row.geometry.bounds, 
                "year": year,
                "out_path": out_path,
                "prefix": f"{row[0]}_{year}_{row.source}",
                "season": "leafon",
                "overwrite": overwrite,
            } for row in plots.itertuples()
        ]

        multithreaded_execution(get_landsat8, params, 10)
