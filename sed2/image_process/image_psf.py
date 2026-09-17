import os  # noqa: EXE002
import sys
from pathlib import Path

import numpy as np
from astropy.convolution import convolve_fft
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from multiprocess import Pool
from photutils.psf.matching import resize_psf
from reproject import reproject_interp
from tqdm import tqdm

from ..path import PATH
from ..utils import *

__all__ = ['load_kernel', 'match_image']


def load_kernel(band1, band2, pix_scale=None, eps=1e-2, gaussian_w4=True):
    '''
    load a kernel for PSF matching from band1 to band2.
    Replace this function with a more general with your own kernel library if needed.

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

    kernel_path = PATH / "kernels" / f"kernel_{band1}_to_{band2}.fits.gz"

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


def _match_band(args):
    '''Match a single band's PSF (worker for multiprocessing).'''
    path, galaxy, f, is_high_res, pix_scale, ref_band, target_wcs,\
        target_pix_scale, out_shape, unit, out_path = args

    hdu = fits.open(path / "masked" / f"{galaxy}_{f}.fits")
    var = fits.open(path / "masked" / f"var_{galaxy}_{f}.fits")
    wcs = WCS(hdu[0].header)

    if is_high_res:
        # load and crop kernel
        kernel = load_kernel(f, ref_band, pix_scale=pix_scale)
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
    else:
        # keep the original resolution
        psf_match_data = hdu[0].data.copy()
        psf_match_var = var[0].data.copy()

    img_proj, _ = reproject_interp((psf_match_data, wcs), target_wcs,
                                   shape_out=out_shape)
    var_proj, _ = reproject_interp((psf_match_var, wcs), target_wcs,
                                   shape_out=out_shape)
    if unit == 'flux':
        img_proj *= (target_pix_scale / pix_scale)**2
        var_proj *= (target_pix_scale / pix_scale)**2

    out_header = target_wcs.to_header()
    out_header['FILTER'] = f
    out_header['MAGZP'] = hdu[0].header.get('MAGZP', -8.9)
    fits.writeto(out_path / f"{galaxy}_{f}.fits", img_proj, out_header,
                 overwrite=True)
    fits.writeto(out_path / f"var_{galaxy}_{f}.fits", var_proj, out_header,
                 overwrite=True)
    hdu.close()
    var.close()


def match_image(workdir, galaxy, filters,
                ref_band='wise_w4', target_wcs=None, ref_band_pixscale=None,
                flux_or_sb=None, out_shape=None, out_path=None, n_process=None):
    '''
    Match PSF of the images to the target filter.
    Match pixel size of the images to the largest.

    Parameters
    ----------
    workdir : str
        Path to the working directory.
    galaxy : str
        Name of the galaxy to match.
    filters: list
        The list of filters to use for matching.
    ref_band: str
        The target filter to match PSF to.
    target_wcs: astropy.wcs.WCS, optional
        The target WCS to match to.
    ref_band_pixscale: str, optional
        The target filter to match pixel scale to.
    flux_or_sb: dict | str, optional
        The type of flux to use for matching.
    out_shape: tuple, optional
        The shape of the output images.
    out_path: str, optional
        The path to save the matched images.
    n_process: int, optional
        Number of processes to use for matching. Defaults to the number
        of available CPUs (capped by the number of filters).
    '''
    # handle output directory
    path = Path(workdir)
    if out_path is None:
        out_path = path / "matched"
    out_path = Path(out_path)

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
    for i, f in enumerate(filters):
        pix_scales[i] = get_pixel_size(path / "masked" / f"{galaxy}_{f}.fits")

    if ref_band_pixscale is not None:
        target_pix_idx = np.where(np.abs(pix_scales - ref_band_pixscale) < 1e-2)[0][0]
        target_pix_scale = pix_scales[target_pix_idx]
    else:
        target_pix_scale = np.max(pix_scales[high_res_filt_flg])
        target_pix_idx = np.where(np.abs(pix_scales - target_pix_scale) < 1e-2)[0][0]

    target_hdu = fits.open(path / "masked" / f"{galaxy}_{filters[target_pix_idx]}.fits")
    if target_wcs is None:
        target_wcs = WCS(target_hdu[0].header)

    print("Matching starts!")
    print(f" > Target PSF = {ref_band}")
    print(f" > Target pixel scales = {target_pix_scale:.3f} arcsec/pixel.")
    print(f" > Output directory: {out_path}")

    # Check for existing matched images
    if check_output(out_path / f"{galaxy}_{filters[-1]}.fits"):
        return high_res_filt_flg

    if out_shape is None:
        out_shape = target_hdu[0].data.shape

    if n_process is None:
        n_process = min(len(filters), os.cpu_count() or 1)

    args_list = [
        (path, galaxy, f, bool(high_res_filt_flg[i]), pix_scales[i], ref_band,
         target_wcs, target_pix_scale, out_shape, flux_or_sb[f], out_path)
        for i, f in enumerate(filters)
    ]

    with Pool(processes=n_process) as pool:
        for _ in tqdm(pool.imap(_match_band, args_list), total=len(args_list)):
            pass

    # --- finalize ---
    target_hdu.close()

    # write the matched filter list
    np.savetxt(path / "aux" / "filters_highres.txt", 
               np.array(filters)[high_res_filt_flg], fmt="%s")
    np.savetxt(path / "aux" / "filters_lowres.txt", 
               np.array(filters)[low_res_filt_flg], fmt="%s")