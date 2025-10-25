import numpy as np 
import sys, os, warnings
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS 
from astropy.nddata import Cutout2D
from astropy.convolution import convolve_fft
from reproject import reproject_interp
from photutils.psf.matching import resize_psf

from ..utils import *
from ..path import PATH

__all__ = ['load_kernel', 'match_image']


def load_kernel(band1, band2, pix_scale=None, eps=1e-3, gaussian_w4=True):
    '''
    load a kernel for PSF matching from band1 to band2.

    Parameters
    ----------
    band1 : str
        The high-resolution band for the PSF matching.
    band2 : str
        The low-resolution band for the PSF matching.
    pix_scale : float, optional
        The pixel scale to which the kernel should be resized (in arcsec/pixel).
    eps : float, optional
        The tolerance for checking pixel scale difference.
    gaussian_w4 : bool, optional, default=True
        Whether to use a 15 arcsec Gaussian PSF for wise_w4 as in z0MGS (Leroy et al. 2019).

    Returns
    -------
    kernel : 2D numpy array
        The PSF kernel for matching from band1 to band2.
    '''

    if gaussian_w4:
        if band1 == 'wise_w4': band1 = 'gauss15'
        if band2 == 'wise_w4': band2 = 'gauss15'

    kernel_path = f"{PATH}/kernels/kernel_{band1}_to_{band2}.fits.gz"

    if not os.path.exists(kernel_path):
        print(f"Kernel file kernel_{band1}_to_{band2}.fits.gz does not exist.")
        sys.exit(1)
        
    with fits.open(kernel_path) as hdul:
        kernel = hdul[0].data
        pix_scale_0 = WCS(hdul[0].header).proj_plane_pixel_scales()[0].value * 3600

        if pix_scale is not None and abs(pix_scale - pix_scale_0) > eps:
            # Resize the kernel to the new pixel scale
            kernel = resize_psf(kernel, pix_scale_0, pix_scale)

    return kernel


def match_image(img_path, sci_img, var_img, filters,
                ref_band='wise_w4', ref_band_pixscale=None,
                flux_or_sb=None, out_shape=None, prefix='match_', out_path=None):
    '''
    Match PSF of the images to the target filter.
    Match pixel size of the images to the largest.

    Parameters
    ----------
    img_path : str
        The path to the images.
    sci_img: dict
        The science images.
    var_img: dict
        The variance images.
    filters: list
        The list of filters to use for matching.
    ref_band: str
        The target filter to match PSF to.
    ref_band_pixscale: str, optional
        The target filter to match pixel scale to.
    flux_or_sb: dict | str, optional
        The type of flux to use for matching.
    out_shape: tuple, optional
        The shape of the output images.
    prefix: str, optional
        The prefix to add to the output filenames.
    out_path: str, optional
        The path to save the matched images.
    '''
    # handle output directory
    if out_path is None:
        out_path = f"{img_path}/matched/"
    out_path = Path(out_path)
    if not out_path.exists():
        warnings.warn(f"Creating directory: {out_path}")
        out_path.mkdir(parents=True)

    # handle image units
    if flux_or_sb is None:
        flux_or_sb = get_default_units(filters)
    elif isinstance(flux_or_sb, str):
        flux_or_sb = {f: flux_or_sb for f in filters}
    elif not isinstance(flux_or_sb, dict):
        raise TypeError("flux_or_sb must be a dict or a str.")

    # set target PSF to match
    fwhms = get_psf_size(filters)
    target_fwhm = get_psf_size(ref_band)

    # identify low & high resolution filters
    low_res_filt_flg  = fwhms >  target_fwhm
    high_res_filt_flg = fwhms <= target_fwhm

    # set target pixel scale to match
    pix_scales = np.full_like(filters, np.nan, dtype=float)
    for i, img in enumerate(list(sci_img.values())):
        pix_scales[i] = get_pixel_size(img_path + img)

    if ref_band_pixscale is not None:
        target_pix_idx = np.where(np.array(filters) == ref_band_pixscale)[0][0]
        target_pix_scale = pix_scales[target_pix_idx]
    else:
        target_pix_scale = np.max(pix_scales[high_res_filt_flg])
        target_pix_idx = np.where(np.abs(pix_scales - target_pix_scale) < 1e-3)[0][0]

    target_hdu = fits.open(img_path + sci_img[filters[target_pix_idx]])
    target_wcs = WCS(target_hdu[0].header)

    print(f"Matching starts!")
    print(f"Target PSF = {ref_band}")
    print(f"Target pixel scales = {target_pix_scale:.3f} arcsec/pixel.")
    print(f"Output directory: {out_path}")

    # Check for existing matched images
    if check_output(f"{out_path}/{prefix}{sci_img[filters[0]]}"):
        return high_res_filt_flg

    # iterate over filters
    # ------ the main loop ------
    for i, f in enumerate(filters):

        print(f"Processing {f}...")

        # open images
        hdu = fits.open(img_path + sci_img[f])
        var = fits.open(img_path + var_img[f])
        wcs = WCS(hdu[0].header)

        if high_res_filt_flg[i]:

            # load and crop kernel
            kernel = load_kernel(f, ref_band, pix_scale=pix_scales[i])
            kernel = Cutout2D(kernel, position=(kernel.shape[1]//2, 
                                                kernel.shape[0]//2), 
                              size=hdu[0].data.shape[0]//2, mode='trim').data
            # convolve
            psf_match_data = convolve_fft(hdu[0].data, kernel, 
                                          normalize_kernel=True, nan_treatment='fill',
                                          preserve_nan=True, allow_huge=True)
            psf_match_var = convolve_fft(var[0].data, kernel**2 / np.sum(kernel**2), 
                                          normalize_kernel=True, nan_treatment='fill',
                                          preserve_nan=True, allow_huge=True)
            
        elif low_res_filt_flg[i]:
            # keep the original resolution
            psf_match_data = hdu[0].data.copy()
            psf_match_var = var[0].data.copy()

        # match pixel size
        if out_shape is None:
            out_shape = target_hdu[0].data.shape

        img_proj, _ = reproject_interp((psf_match_data, wcs), target_wcs, 
                                       shape_out=out_shape)
        var_proj, _ = reproject_interp((psf_match_var, wcs), target_wcs, 
                                       shape_out=out_shape)
        if flux_or_sb[f] == 'flux':
            img_proj *= (target_pix_scale / pix_scales[i])**2
            var_proj *= (target_pix_scale / pix_scales[i])**2

        # write output
        out_header = target_wcs.to_header()
        out_header['FILTER'] = f
        out_header['MAGZP'] = hdu[0].header.get('MAGZP', -8.9)
        fits.writeto(f"{out_path}/{prefix}{sci_img[f]}", img_proj, out_header, 
                     overwrite=True)
        fits.writeto(f"{out_path}/{prefix}{var_img[f]}", var_proj, out_header, 
                     overwrite=True)
        hdu.close()
        var.close()

    # --- finalize ---
    target_hdu.close()

    return high_res_filt_flg