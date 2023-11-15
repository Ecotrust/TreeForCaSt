"""
Combine lidar-derived rasters into a single raster.
Copy rasters into a collection in dir processing
Copy crown gejson into a collection in dir processing
"""

# %%
from pathlib import Path
import re
from PIL import Image
import shutil

import numpy as np
import rasterio
from rasterio import MemoryFile
from rio_cogeo.profiles import cog_profiles
from rio_cogeo.cogeo import cog_translate
import matplotlib.pyplot as plt

from gdstools import image_collection, ConfigLoader


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


def generate_mb_rasters(image_dir, out_dir):
    # out_dir = Path('data/dev/processed/')
    rasters = image_collection(image_dir)#'data/dev/interim/lidar-derived/')
    rast_dict = paths_to_dict(rasters, 2)

    PROFILE = {
        'driver': 'GTiff',
        'interleave': 'band',
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
        'compress': 'lzw',
        'nodata': -9999,
        'dtype': rasterio.float32,
        # 'count': 5 # set number of bands
    }

    cog_profile = cog_profiles.get("deflate")

    for y, paths in rast_dict[list(rast_dict.keys())[0]].items():
        uuids = {Path(x).stem.split('_')[0] for x in paths}
        
        for i in uuids:
            sbs = [x for x in paths if Path(x).stem.split('_')[0] == i]
            assert len(sbs) == 3

            chm_r = [x for x in sbs if Path(x).stem.endswith('-CHM')]
            dtm_r = [x for x in sbs if Path(x).stem.endswith('-DTM')]
            int_r = [x for x in sbs if Path(x).stem.endswith('-Intensity')]
            assert len(chm_r) == 1 & len(dtm_r) == 1 & len(int_r) == 1

            with rasterio.open(*chm_r) as src:
                chm = src.read(1)
                chm_trf = src.transform

            with rasterio.open(*dtm_r) as src:
                dtm = src.read(1)
                dtm_trf = src.transform

            with rasterio.open(*int_r) as src:
                int_a = src.read(1)
                int_trf = src.transform

            assert chm_trf == dtm_trf == int_trf

            PROFILE.update(crs=src.crs, transform=chm_trf, width=src.width,
                        height=src.height, count=3)
            
            band_info = {
                'CHM': 'Canopy Height Model',
                'DTM': 'Digital Terrain Model',
                'Intensity': 'Lidar Intensity',
            }

            bands = [chm, dtm, int_a]

            outfile = Path(chm_r[0]).name.replace('-CHM.tif', '-Rast-cog.tif')
            outdir = out_dir / 'lidar-rast' / y
            outdir.mkdir(parents=True, exist_ok=True)

            with MemoryFile() as memfile:
                with memfile.open(**PROFILE) as dst:
                    dst_idx = 1
                    for band, data in zip(band_info.keys(), bands):
                        # output = np.zeros(dst.shape, rasterio.float32)
                        dst.write(data, dst_idx)
                        dst.set_band_description(dst_idx, band_info[band])

                        # Select band to generate preview
                        if dst_idx == 1:
                            cm = plt.get_cmap('gist_earth')
                            norm_out = cm(data / data.max())[:, :, :3] * 255
                            preview = Image.fromarray(
                                norm_out.astype(np.uint8)).convert('RGB')
                            preview.save(
                                outdir / outfile.replace('-cog.tif', '-preview.png'))

                        dst_idx += 1

                    cog_translate(
                        dst,
                        outdir / outfile,
                        cog_profile,
                        in_memory=True,
                        quiet=True
                    )

    return 

def copy_laz(image_dir, out_dir):
    laz_files = image_collection(image_dir, file_pattern='*-Crowns.laz')
    crown_files = image_collection(image_dir, file_pattern='*-Crowns.geojson')
    idx = Path(laz_files[0]).parts.index('lidar-derived') - 1
    laz_dict = paths_to_dict(laz_files, idx)
    crown_dict = paths_to_dict(crown_files, idx)

    out_dir = Path(out_dir) / 'lidar'
    out_dir.mkdir(parents=True, exist_ok=True)

    for y, paths in laz_dict['lidar-derived'].items():
        for path in paths:
            outdir = out_dir / y
            outdir.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, out_dir / y / Path(path).name)

    for y, paths in crown_dict['lidar-derived'].items():
        for path in paths:
            outdir = out_dir / y
            outdir.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, out_dir / y / Path(path).name)

    return


if __name__ == "__main__":

    run_as = 'prod'
    # Load config file
    conf = ConfigLoader(Path(__file__).parent.parent).load()

    if run_as == 'dev':
        DATADIR = Path(conf.DEV_DATADIR).parent
    else:
        DATADIR = Path(conf.DATADIR)

    image_dir = DATADIR / 'interim/lidar-derived/'
    out_dir = DATADIR / 'processed'
    generate_mb_rasters(image_dir, out_dir)
    # copy_laz(image_dir, out_dir)
    
    print('Done!')

# for y, pa print(len(laz_files), len(crown_files))
