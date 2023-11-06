import os
from pathlib import Path
import pdal
import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon 

from gdstools import image_collection, multithreaded_execution

def classify(infile, outfile, to_epsg):
    # out_base = os.path.basename(infile)
    # outfile = os.path.join(out_dir, out_name)
    
    reader = pdal.Reader.las(infile)
    reproject = pdal.Filter.reprojection(out_srs=f"EPSG:{to_epsg}")
    blank = pdal.Filter.assign(assignment="Classification[:]=0")
    outlier = pdal.Filter.outlier()
    elm = pdal.Filter.elm()
    smrf = pdal.Filter.smrf()
    hag = pdal.Filter.hag_nn()

    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    writer = pdal.Writer.las(
        outfile,
        extra_dims="HeightAboveGround=float32",
        a_srs=f"EPSG:{to_epsg}",
    )

    pipeline = reader | blank | outlier | elm | smrf | hag | writer
    pipeline.execute()

    return outfile

def make_rasters(infile, prefix, bbox, resolution=0.5):
    USEFUL = "Classification[0:6],Classification[8:17]"
    # out_base = os.path.basename(infile).split(".")[0]
    reader = pdal.Reader.las(
        infile,
        extra_dims='HeightAboveGround=float32'
    )

    parent_dir = Path(infile).parent

    xmin, ymin, xmax, ymax = bbox
    bounds = f"([{xmin}, {xmax}], [{ymin}, {ymax}])"

    rng = pdal.Filter.range(limits=USEFUL)
    
    dtm = pdal.Writer.gdal(
        os.path.join(parent_dir, f'{prefix}-DTM.tif'),
        resolution=resolution,
        bounds=bounds,
        dimension="Z",
        radius=resolution,
        data_type="float32",
        output_type="mean",
        where="Classification==2"
    )

    chm = pdal.Writer.gdal(
        os.path.join(parent_dir, f'{prefix}-CHM.tif'),
        resolution=resolution,
        bounds=bounds,
        dimension="HeightAboveGround",
        data_type="float32",
        output_type="max",
    )
    
    pipeline = reader | rng | dtm | chm
    pipeline.execute()

    assign = pdal.Filter.assign(value = [
          "Intensity = Intensity / 256"
      ])
    intensity = pdal.Writer.gdal(
        os.path.join(parent_dir, f'{prefix}-Intensity.tif'),
        resolution=resolution,
        bounds=bounds,
        dimension="Intensity",
        data_type="float32",
        output_type="mean"
    )
    
    if np.max(pipeline.arrays[0]['Intensity']) > 255:
        is16Bit = True
    else:
        is16Bit = False
        
    ipipeline = reader | rng
    if is16Bit:
      ipipeline |= assign
    ipipeline |= intensity
    ipipeline.execute()
    
    return True

def calc_crown_widths(hull):
    """
    Finds the smallest bounding rectangle from a convex hull and
    returns lengths of major and minor axes.

    Adapted from: https://stackoverflow.com/a/33619018
    """
    from scipy.ndimage import rotate
    
    pi2 = np.pi/2.

    # get points of the convex hull
    hull_points = hull.points[hull.vertices]

    # calculate edge angles
    edges = hull_points[1:] - hull_points[:-1]
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T

    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]

    ll = np.array((x1, y1))
    ul = np.array((x1, y2))
    lr = np.array((x2, y1))

    width = np.linalg.norm(lr - ll)
    height = np.linalg.norm(ul - ll)

    return max(width, height), min(width, height)

def get_crowns(arr, srs):
    clusters = np.unique(arr['ClusterID'])
    clusters = clusters[clusters>0]
    geoms = []
    heights = []
    topx = []
    topy = []
    max_widths = []
    orth_widths = []
    hull_surfs = []
    hull_vols = []
    for c in clusters:
        msk = arr['ClusterID'] == c
        x = arr['X'][msk]
        y = arr['Y'][msk]
        z = arr['HeightAboveGround'][msk]
        xy = np.stack((x, y), axis=-1)
        xyz = np.stack((x, y, z), axis=-1)
        hullxy = ConvexHull(xy)
        hullxyz = ConvexHull(xyz)
        
        geoms.append(Polygon(hullxy.points[hullxy.vertices]))
        heights.append(z.max())
        topx.append(x[z.argmax()])
        topy.append(y[z.argmax()])
        max_width, orth_width = calc_crown_widths(hullxy)
        max_widths.append(max_width)
        orth_widths.append(orth_width)
        hull_surfs.append(hullxyz.area)
        hull_vols.append(hullxyz.volume)
    gdf = gpd.GeoDataFrame(
        data=np.array([clusters, topx, topy, heights, max_widths, orth_widths, hull_surfs, hull_vols]).T, 
        columns=['TAO_ID', 'TOP_X', 'TOP_Y', 'TOP_HAG', 'MCW', 'MCW90', 'HULL_AREA', 'HULL_VOLUME'], 
        geometry=geoms,
        crs=srs
    )
    gdf['TAO_ID'] = gdf['TAO_ID'].astype(int)

    pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(gdf.TOP_X, gdf.TOP_Y),
        crs=srs
    )
    pts = pts.to_crs(epsg=4326)
    gdf['TOP_X'] = pts.geometry.x
    gdf['TOP_Y'] = pts.geometry.y
    
    return gdf.to_crs(epsg=4326)

def treeseg(infile, prefix, to_epsg, overwrite=False):
    # out_base = os.path.basename(infile).split('.')[0]
    parent_dir = Path(infile).parent
    outfile = os.path.join(parent_dir, f"{prefix}-Crowns.geojson")
    if os.path.exists(outfile) and not overwrite:
        print(f"{outfile} already exist, skipping")
        return 

    print(f"Processing {infile} ...")
    USEFUL = "Classification[0:6],Classification[8:17]"
    
    reader = pdal.Reader.las(
        infile,
        extra_dims='HeightAboveGround=float32'
    )
    rng = pdal.Filter.range(limits=USEFUL)
    sort_hag = pdal.Filter.sort(dimension="HeightAboveGround", order="DESC")
    treeseg = pdal.Filter.litree(min_height=2.0)

    writer = pdal.Writer.las(
        outfile.replace('.geojson', '.laz'),
        extra_dims="HeightAboveGround=float32,ClusterID=int32",
        a_srs=f"EPSG:{to_epsg}",
    )

    pipeline = reader | rng | sort_hag | treeseg | writer
    pipeline.execute()

    arr = pipeline.arrays[0]
    srs = pipeline.srswkt2
    crowns = get_crowns(arr, srs)
    crowns.to_file(outfile, driver='GeoJSON')
    print(f"Saved {len(crowns)} crowns to {outfile}")
    
    return outfile


if __name__ == "__main__":

    PLOTS = 'data/dev/features/plot_features.geojson'
    OUTLIERS = 'data/dev/features/outlier_uuids.csv'
    LAZDIR = '/mnt/data/FESDataRepo/remote_sensing_plots/interim/lidar/' + \
            'accepted_plot_clips/**/hectare_clips/'
    
    plots = gpd.read_file(PLOTS).set_index('uuid')
    outliers = pd.read_csv(OUTLIERS)
    outliers['uuid'] = outliers.outlier_uuid.apply(lambda x: x.split('-')[0])
    outliers = outliers.set_index('uuid')

    splots = plots[~plots.index.isin(outliers.index)]
    splots = splots.sort_index().iloc[0:20]

    laz_files = image_collection(LAZDIR, file_pattern="*.laz")
    uuid_path = [
        {
            'uuid': Path(x).stem.split('-')[0], 
            'year': Path(x).stem.split('_')[-1].split('-')[-1],
            'path': x
        } 
        for x in laz_files
    ]
    df = pd.DataFrame(uuid_path).set_index('uuid')

    splots = df.merge(splots, left_index=True, right_index=True)
    splots['outfile'] = splots.apply(
        lambda row: f'data/dev/processed/lidar/{row.year}/{row.name}_{row.year}_{row.source}_PC.laz', axis=1)

    params = [
        {
            "infile" : row.path,
            "outfile" : row.outfile,
            "to_epsg" : row.epsg,
        }
        for idx, row in splots.iterrows()
    ]

    # multithreaded_execution(classify, params, threads=10)

    # Generate rasters
    params = [
        {
            "infile" : row.outfile,
            "prefix" : Path(row.outfile).stem,
            "bbox" : [row.utm_xmin, row.utm_ymin, row.utm_xmax, row.utm_ymax],
            # "resolution" : 0.5,
        }
        for idx, row in splots.iterrows()
    ]

    # multithreaded_execution(make_rasters, params, threads=10)

    params = [
        {
            "infile" : row.outfile,
            "prefix" : Path(row.outfile).stem,
            "to_epsg" : row.epsg,
            # "resolution" : 0.5,
        }
        for idx, row in splots.iterrows()
    ]
    multithreaded_execution(treeseg, params, threads=10)
    # for param in params:
    #     treeseg(**param)
