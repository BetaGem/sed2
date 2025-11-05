import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D

from ..utils import *
from SparseFit.path import PATH

__all__ = ["remove_naninf_image_2dinterpolation",
           "remove_naninf_image_fill",
           "crop"]


def remove_naninf_image_2dinterpolation(data_image):
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
    


def crop(img_path, sci_img, var_img, filters, robust=False,
         galaxy=None, crop_size=None,
         ra=None, dec=None, crop_path=None):
    '''
    This function is adopted from piXedfit.piXedfit_images 
    (Abdurro'uf et al. 2021)

    Parameters
    ----------
    img_path : str
        Path to the image directory.
    sci_img : dict
        Dictionary of science images.
    var_img : dict
        Dictionary of variance images.
    filters : list
        List of filters to apply.
    robust : bool, optional
        Whether to fill NaN/Inf values.
    galaxy : str, optional
        Name of the galaxy to crop around.
    crop_size : tuple, optional
        Size of the crop (height, width).
    ra : float, optional
        Right ascension of the center of the crop.
    dec : float, optional
        Declination of the center of the crop.
    crop_path : str, optional
        Path to the output directory for cropped images.
    '''

    # get band with the largest pixel size
    pix_scales = np.full_like(filters, np.nan, dtype=float)
    for i, img in enumerate(list(sci_img.values())):
        pix_scales[i] = get_pixel_size(img_path + img)

    target_pix_scale = np.max(pix_scales)
    # target_pix_idx = np.where(np.abs(pix_scales - target_pix_scale) < 1e-2)[0][0]
    
    # load geometry
    if galaxy is not None:
        try:
            coord = SkyCoord.from_name(galaxy)
            ra, dec = coord.ra.value, coord.dec.value
        except:
            if ra is None or dec is None:
                raise ValueError(f"Failed to resolve {galaxy}, RA and DEC should be specified.")
        crop_size = load_ref_info(galaxy)[2]

    if isinstance(crop_size, int):
          crop_size = (crop_size, crop_size)

    # create output directory
    if crop_path is None:
        import os
        from pathlib import Path

        crop_path = Path(img_path).parent / "cropped"
        os.makedirs(crop_path, exist_ok=True)

    # whether to overwrite existing crops
    if check_output(crop_path / f"crop_{sci_img[filters[-1]]}"):
        return crop_path

    print("Cropping starts!")

    for i in range(len(filters)):

        print(f"Processing {filters[i]}...")

        dim_y0, dim_x0 = crop_size
        dim_y1 = int(dim_y0 * target_pix_scale / pix_scales[i] * 1.5)
        dim_x1 = int(dim_x0 * target_pix_scale / pix_scales[i] * 1.5)

        # science image
        hdu = fits.open(img_path + sci_img[filters[i]])[0]
        wcs = WCS(hdu.header)
        position = wcs.wcs_world2pix(ra, dec, 1)
        cutout = Cutout2D(hdu.data, position=position, 
                          size=(dim_y1, dim_x1), wcs=wcs)
        header = cutout.wcs.to_header()
        
        # copy relevant header keywords
        for key in hdu.header.keys():
            if 'MAGZP' in key:
                header['MAGZP'] = hdu.header[key]

        fits.writeto(f"{crop_path}/crop_{sci_img[filters[i]]}", 
                     cutout.data, header, overwrite=True)

        # variance image
        hdu = fits.open(img_path + var_img[filters[i]])[0]
        wcs = WCS(hdu.header)
        position = wcs.wcs_world2pix(ra, dec, 1)
        cutout = Cutout2D(hdu.data, position=position, 
                          size=(dim_y1, dim_x1), wcs=wcs)
        fits.writeto(f"{crop_path}/crop_{var_img[filters[i]]}", 
                     cutout.data, header, overwrite=True)
