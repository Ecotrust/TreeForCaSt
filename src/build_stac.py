"""
Build the STAC catalog for Oregon BLM and Washington St DNR forest stands. 
"""

# %%
import re
import json
from datetime import datetime
from pathlib import Path
import pandas as pd

from pystac import (
    Extent, 
    SpatialExtent, 
    TemporalExtent,
    Item,
    Asset,
    MediaType,
    Catalog,
    Collection,
    CatalogType,
    Provider,
    Link
)
from pystac.extensions.eo import Band, EOExtension
from pystac.extensions.label import LabelExtension, LabelType, LabelClasses
from pystac.extensions.projection import ProjectionExtension

import numpy as np
import geopandas as gpd
import rasterio
from shapely import geometry

from gdstools import ConfigLoader, image_collection, multithreaded_execution

# %%
def bbox_to_json(bbox):
    """
    Generate GeoJSON geometry from bounding box.

    :param bbox: A list of four coordinates representing the bounding box in the order 
        [minx, miny, maxx, maxy].
    :type bbox: list
    :return: A GeoJSON geometry representing the bounding box.
    :rtype: dict
    """
    geom = geometry.box(*bbox, ccw=True)
    return json.loads(gpd.GeoSeries(geom).to_json())

# %%
def create_label_item(
    label_path: str,
    attr_dict: dict,
    crs='EPSG:4326'
):
    """
    Create a label item.

    :param label_path: The path to label geojson file.
    :type label_path: str
    :param attr_dict: Dictionary with attributes for the label item. Expected attributes: 
        label_properties, label_date, label_description, label_type, label_tasks, 
        and label_name.
    :type attr_dict: dict
    :param crs: Coordinate Reference System (CRS) of the label data, defaults to 'EPSG:4326'.
    :type crs: str, optional
    :return: A PySTAC Item representing the label data.
    :rtype: pystac.Item
    """
    # Read label data
    label_path = Path(label_path)
    label_id = label_path.stem.replace('-cog', '')
    label_data = gpd.read_file(label_path)
    bbox = label_data.total_bounds.tolist()

    if attr_dict['label_task'] in ['classification', 'segmentation']:
        label_classes = [
            LabelClasses.create(classes=attr_dict['label_classes'],
                                name=attr_dict['label_name'])
        ]
    else:
        label_classes = None

    label_data = label_data.to_crs(crs)
    label_data = json.loads(label_data.to_json())

    # Create label item
    label_item = Item(
        id=f'{label_id}',
        geometry=label_data,
        bbox=bbox,
        datetime=attr_dict['label_date'],
        properties={}
    )

    if attr_dict['label_type'] == 'vector':
        label_type = LabelType.VECTOR
    elif attr_dict['label_type'] == 'raster':
        label_type = LabelType.RASTER

    label_ext = LabelExtension.ext(label_item, add_if_missing=True)
    label_ext.apply(
        label_description=attr_dict['label_description'], 
        label_type=label_type,
        label_properties=attr_dict['label_properties'],
        label_tasks=attr_dict['label_task'],
        label_classes=label_classes
    )

    # Add link to label data
    idx = label_path.as_posix().split('/').index('labels')
    subdir = '/'.join(label_path.as_posix().split('/')[idx:-1])
    url = 'https://fbstac-stands.s3.amazonaws.com/data/' + subdir + '/' + label_path.name
    label_ext.add_geojson_labels(href=url)

    return label_item, label_ext


def create_item(
    image_path: str,
    thumb_path: str,
    metadata_path: str,
    asset_path_url: str,
):
    """
    Create a STAC item.

    :param image_path: Path to local COG image.
    :type image_path: str
    :param thumb_path: Path to local thumbnail image.
    :type thumb_path: str
    :param metadata_path: Path to local metadata file.
    :type metadata_path: str
    :param asset_path_url: URL to the asset path. This is the access point where users can download the catalog assets.
    :type asset_path_url: str
    :return: A STAC item.
    :rtype: pystac.Item
    """
    # Load image data
    image_path = Path(image_path)
    thumb_path = Path(thumb_path)
    with rasterio.open(image_path) as src:
        crs = src.crs
        bbox = list(src.bounds)

    # Load metadata
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Collect image properties
    image_date = datetime.utcfromtimestamp(
        metadata['properties']['system:time_start']/1000)
    image_id = image_path.stem.replace('-cog', '')

    image_geom = bbox_to_json(bbox)
    image_bands = metadata['bands']

    # Create item
    item = Item(
        id=image_id,
        geometry=image_geom,
        bbox=bbox,
        datetime=image_date,
        properties={},
    )

    # Add bands and projection
    bands = [Band.create(name=b['id'], common_name=b.get('name'))
             for b in image_bands]
    eo = EOExtension.ext(item, add_if_missing=True)
    eo.apply(bands=bands)

    proj = ProjectionExtension.ext(item, add_if_missing=True)
    proj.apply(epsg=crs.to_epsg())

    # Add links to assets
    item.add_asset('image', Asset(href=asset_path_url +
                   image_path.name, media_type=MediaType.COG))
    # item.add_asset('metadata', pystac.Asset(href=github_url +
    #                metadata_path[3:], media_type=pystac.MediaType.JSON))
    item.add_asset('thumbnail', Asset(href=asset_path_url +
                   thumb_path.name, media_type=MediaType.PNG))

    return item


def paths_to_dict(paths_list, idx):
    """
    Create a dictionary from a list of paths.

    :param paths_list: A list of file paths.
    :type paths_list: list
    :param idx: The index of the path to use for the collection name.
    :type idx: int
    :return: A dictionary with the collection name as the key and a list of file paths as the value.
    :rtype: dict
    """
    _dict = {}
    for p in paths_list:
        plist = p.split('/')
        # expects name in format cellid_year_state_agency/dataset
        nameparts = Path(p).stem.split('_')
        collection = plist[idx + 1]
        year = nameparts[1]
        if not re.match(r'^\d+$', year):
            _dict.setdefault(collection, []).append(p)
        else:
            _dict.setdefault(collection, {}).setdefault(year, []).append(p)
    return _dict


def build_stac(rootpath: Path, run_as: str = 'dev'):
    """
    Builds the Plot STAC (SpatioTemporal Asset Catalog).

    :param conf: A dictionary containing configuration information for the catalog.
    :type conf: Dict[str, Any]
    :param qq_shp: A GeoDataFrame containing the quad boundaries.
    :type qq_shp: gpd.GeoDataFrame
    :return: A STAC Catalog object.
    :rtype: Catalog
    """
    # Load config file
    conf = ConfigLoader(rootpath).load()
    if run_as == 'dev':
        PLOTS = Path(conf.DEV_PLOTS)
        DATADIR = Path(conf.DEV_DATADIR)
        PLOTATTRS = Path(conf.DEV_PLOTATTRS)
    else:
        PLOTS = conf.DEV_PLOTS
        DATADIR = Path(conf.DATADIR)
        PLOTATTRS = Path(conf.PLOTATTRS)

    plots_shp = gpd.read_file(PLOTS)
    
    # Build catalog
    """
    Label collections. One per agency-survey comb.

         1. BLM - Coos Bay
         2. BLM - Rogue Valley
         3. BLM - Lane County
         4. USFS - Blue Mountains
         5. USFS - Umpqua
         6. USFS - Willamette
         7. USFS - Gifford Pinchot
         8. USFS - Fremont-Winema
         9. USFS - Rogue  
        10. USFS - Umatilla
        11. USFS - Wenatchee
        12. WADNR  
    
    Data collections
        1. NAIP
        2. 3DEP
        3. Landsat 8
        4. LIDAR
        5. Attribute data

    Name convention
        labels: <uuid>_<year>_<agency-survey>-label
        datasets: 
            - <uuid>_<year>_<agency-survey>_<dataset>-cog
            - <uuid>_<year>_<agency-survey>_<dataset>-preview

    Steps:
        1. Load plot geojson file 
        2. Load image paths and convert list to dict
        3. Load plot attributes (inventory_features.csv) and merge with plot geojson features
        4. Organize items by agency-survey comb: {'agency-survey': 'labels': {}, 'datasets': {}}
        
    """
    fbench = Catalog(
        id='fbstac-plots',
        description='A STAC implementation for forest benchmarking and modeling',
        title='Forest Benchmarking and Modeling STAC',
    )

    datasets = image_collection(DATADIR)
    images_dict = paths_to_dict(datasets, 5)
    attrs = pd.read_csv(PLOTATTRS)

    # Drop tree species columns
    attrs.drop(attrs.columns[18:], axis=1, inplace=True)
    merged = attrs.merge(plots_shp[['uuid', 'source', 'geometry']], on='uuid')
    merged = gpd.GeoDataFrame(merged, crs=plots_shp.crs, geometry=merged.geometry)

    items_dict = {}
    for survey, year in merged[['year', 'source']].groupby(
        ['source','year']).count().index.tolist():

        for year, year_dict in year.items():
            items_dict.setdefault(survey, {}).setdefault(
                year, {'labels': year_dict})
            cellids = [Path(p).stem.split('_')[0] for p in year_dict]

            # add images
            for coll, coll_dict in images_dict.items():

                if isinstance(coll_dict, dict):
                    _year = min(coll_dict.keys(),
                                key=lambda x: abs(int(x)-int(year)))

                    if int(year) - int(_year) <= np.abs(2):
                        _year_dict = coll_dict[_year]
                        items_dict[survey][year]\
                            .setdefault('items', {})\
                            .setdefault(
                                coll, [p for p in _year_dict if Path(p).stem.split('_')[0] in cellids])

                else:
                    items_dict[survey][year]\
                        .setdefault('items', {})\
                        .setdefault(
                            coll, [p for p in coll_dict if Path(p).stem.split('_')[0] in cellids])


if __name__ == '__main__':
    # build_stac(Path(__file__).parent.parent)
    rootpath = Path(__file__).parent
    conf = ConfigLoader(rootpath).load()

    build_stac(rootpath, run_as='dev')
