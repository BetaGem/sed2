import os
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from multiprocess import Pool
from tqdm import tqdm

from ..utils import *

__all__ = [
    "crop",
    "remove_naninf_image_2dinter",
    "remove_naninf_image_fill",
]


def remove_naninf_image_2dinter(data_image):
    '''
    Interpolate over NaN and Inf values in a 2D image.
    This function is adopted from piXedfit.piXedfit_images 
    (Abdurro\'uf et al. 2021). Used in `crop_image`.
    '''
    from scipy.interpolate import griddata

    rows_nan, cols_nan = np.where((np.isnan(data_image)==True) | (np.isinf(data_image)==True))
    
    if len(rows_nan) > 0:
        y = np.arange(0, data_image.shape[0])
        x = np.arange(0, data_image.shape[1])
        xx, yy = np.meshgrid(x, y)

        rows, cols = np.where((np.isnan(data_image)==False) & (np.isinf(data_image)==False))
        data_image_new = griddata((rows, cols), 
                            data_image[rows, cols], (yy, xx))

        rows_nan, cols_nan = np.where((np.isnan(data_image_new)==True) | (np.isinf(data_image_new)==True))
        
        if len(rows_nan) > 0:
            rows, cols = np.where((np.isnan(data_image_new)==False) & (np.isinf(data_image_new)==False))
            data_image_new[rows_nan, cols_nan] = np.percentile(data_image_new[rows, cols], 50)

        return data_image_new
    else:
        return data_image
      

def remove_naninf_image_fill(data_image, fill_value=0):
    '''
    Fill NaN and Inf values in a 2D image with a specified value.
    This function is adopted from piXedfit.piXedfit_images 
    (Abdurro\'uf et al. 2021). Used in `crop_image`.
    '''
    data_image_new = data_image.copy()
    rows_nan, cols_nan = np.where(np.isfinite(data_image)==False)
    data_image_new[rows_nan, cols_nan] = fill_value
      
    return data_image_new
    


def _crop_band(args):
    '''Crop a single band (worker for multiprocessing).'''
    path, crop_path, galaxy, f, sci_file, var_file, ra, dec, pix_scale, crop_size = args

    dim_y0, dim_x0 = crop_size
    dim_y1 = int(dim_y0 / pix_scale)
    dim_x1 = int(dim_x0 / pix_scale)

    # science image
    hdu = fits.open(path / "image" / sci_file)[0]
    wcs = WCS(hdu.header)
    position = wcs.wcs_world2pix(ra, dec, 1)
    cutout = Cutout2D(hdu.data, wcs=wcs, position=position,
                      size=(dim_y1, dim_x1))
    header = cutout.wcs.to_header()

    # copy relevant header keywords
    for key in hdu.header:
        if 'MAGZP' in key:
            header['MAGZP'] = hdu.header[key]

    fits.writeto(crop_path / f"crop_{galaxy}_{f}.fits",
                 cutout.data, header, overwrite=True)

    # variance image
    hdu = fits.open(path / "image" / var_file)[0]
    wcs = WCS(hdu.header)
    position = wcs.wcs_world2pix(ra, dec, 1)
    cutout = Cutout2D(hdu.data, wcs=wcs, position=position,
                      size=(dim_y1, dim_x1))
    fits.writeto(crop_path / f"crop_var_{galaxy}_{f}.fits",
                 cutout.data, header, overwrite=True)


def crop(workdir, filters, coord, crop_size,
         galaxy='galaxy', n_process=None):
    '''
    This function is adopted from piXedfit.piXedfit_images 
    (Abdurro'uf et al. 2021)

    Parameters
    ----------
    workdir : str
        Path to the working directory.
    filters : list
        List of filters to apply.
    galaxy : str, optional
        Name of the galaxy to crop around.
    crop_size : tuple, optional
        Size of the crop (height, width) in arcsec.
    coord : astropy.coordinates.SkyCoord
        SkyCoord object representing the coordinates of the galaxy.
    n_process : int, optional
        Number of processes to use for cropping. Defaults to the number
        of available CPUs (capped by the number of filters).
    '''
    path = Path(workdir)
    crop_path = path / "cropped"

    sci_img = {f: f"skybgsub_{galaxy}_{f}.fits" for f in filters}
    var_img = {f: f"var_{galaxy}_{f}.fits" for f in filters}
    ra, dec = coord.ra.value, coord.dec.value

    # get band with the largest pixel size
    pix_scales = {}
    for f in filters:
        pix_scales[f] = get_pixel_size(path / "image" / sci_img[f])

    if isinstance(crop_size, int):
          crop_size = (crop_size, crop_size)

    # whether to overwrite existing crops
    if check_output(crop_path / f"crop_{galaxy}_{filters[-1]}.fits"):
        return crop_path

    print("Cropping starts! Crop size: ", crop_size, "arcsec")

    if n_process is None:
        n_process = min(len(filters), os.cpu_count() or 1)

    args_list = [(path, crop_path, galaxy, f, sci_img[f], var_img[f],
                  ra, dec, pix_scales[f], crop_size) for f in filters]

    with Pool(processes=n_process) as pool:
        for _ in tqdm(pool.imap(_crop_band, args_list), total=len(args_list)):
            pass
