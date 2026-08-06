import numpy as np  # noqa: EXE002
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from .utils_file import *
from .utils_filter import *

__all__ = [
    "R_extinction",
    "bin_noise",
    "get_a_b_ellipse",
    "get_ebv",
    "get_pixel_size",
    "save_flux",
]

def get_pixel_size(img_path):
    '''
    Get the pixel size (in arcsec) of an image.
    '''
    with fits.open(img_path) as hdul:
        wcs = WCS(hdul[0].header)
        pix_scale = np.sqrt(wcs.proj_plane_pixel_area().value) * 3600

    return round(pix_scale, 3)


def get_ebv(ra=0, dec=0, coord=None):
    """
    Get the E(B-V) value at the specified coordinates.

    Parameters
    -----------
    ra : float
        Right ascension in degrees.
    dec : float
        Declination in degrees.
    """
    from astroquery.ipac.irsa.irsa_dust import IrsaDust

    if coord is None:
        coord = SkyCoord(ra, dec, unit="deg")
    table = IrsaDust.get_query_table(coord, section='ebv')

    return table['ext SFD ref'][0]


def R_extinction(lamb_Ang, Rv=3.1):
    '''
    Gordon23 Milky Way R(V) dependent extinction model. 
    ref: 2023ApJ...950..86G

    Parameters
    ----------
    lamb_Ang : float
        Wavelength in Angstroms.
    Rv : float
        Av / E(B-V)

    Returns
    -------
    float
        R(lambda) value at the specified wavelength.
    '''
    import astropy.units as u
    from dust_extinction.parameter_averages import G23

    ext_model = G23(Rv=Rv)

    # assume no extinction if wavelength is too large
    if isinstance(lamb_Ang, (float, int)) and lamb_Ang > 3.2e5:
        
        return 0
    
    return ext_model(lamb_Ang * u.Angstrom) * Rv



def bin_noise(img, bin_mask, aper_src=None, mask_src=None, apers=None,
              bootstrap=100, max_overlap=0.1, circle_aper=False):
    '''
    Estimate the uncertainties by randomly sampling the blank sky.

    Parameters
    ----------
    img : 2D array
        The input image data.
    bin_mask : 2D array
        A mask to define the bins for noise estimation.
    aper_src : CircularAperture, optional
        The original aperture to exclude from the noise estimation.
    mask_src : 2D array, optional
        A mask to exclude certain regions from the noise estimation.
    apers : list of EllipticalAperture, optional
        A list to user defined apertures.
    bootstrap : int
        The number of bootstrap samples to draw.
    max_overlap : float
        The maximum allowed overlap between the random apertures.
    circle_aper : bool
        Whether to use circular apertures for random sampling.

    Returns
    -------
    float
        The estimated noise level.
    float
        The total flux of the source.
    '''
    from photutils.aperture import EllipticalAperture
    
    # random positions and angles
    np.random.seed(42)
    xy_rd = np.random.rand(bootstrap, 2) * img.shape[0]
    theta_rd = np.random.rand(bootstrap) * 2 * np.pi

    # exclude sources from noise estimation
    if aper_src is not None:
        p0 = aper_src.positions
        mask = np.zeros_like(img).astype(bool)
        ap_p0 = aper_src.to_mask(method='center').data.astype(bool)
        x0, y0 = int(p0[0]), int(p0[1])
        mask[y0-ap_p0.shape[0]//2+1: y0+1-ap_p0.shape[0]//2+ap_p0.shape[0],
             x0-ap_p0.shape[1]//2+1: x0+1-ap_p0.shape[1]//2+ap_p0.shape[1]] |= ap_p0
        mask[~np.isfinite(img)] = True
    elif mask_src is None: 
        raise TypeError("When `aper_src` is None, `mask_src` must be provided!")
    else: 
        mask = mask_src.copy().astype(bool)

    # calculate shape parameters of the bin
    if np.sum(bin_mask) >= 5 and not circle_aper:
        a, b = get_a_b_ellipse(bin_mask)
    else:
        a = np.sqrt(np.nansum(bin_mask) / np.pi)
        b = a

    # random ellisptical apertures
    if apers is not None:
        aper_all = apers
    else:
        aper_all = []
        for i, p in enumerate(xy_rd):
            try:
                ap_rd = EllipticalAperture(p, a, b, theta=theta_rd[i])
                if i and mask[int(p[1]), int(p[0])]:
                    continue
                if i and ap_rd.do_photometry(mask)[0][0] > ap_rd.area * max_overlap:
                    continue
                ap_p = ap_rd.to_mask(method='center').data.astype(bool)
                x, y = int(p[0]), int(p[1])
                mask[y-ap_p.shape[0]//2+1:y+1-ap_p.shape[0]//2+ap_p.shape[0],
                     x-ap_p.shape[1]//2+1:x+1-ap_p.shape[1]//2+ap_p.shape[1]] |= ap_p
                aper_all.append(ap_rd)
            except: pass

    # get the fluxes in the random apertures
    ap_fluxes = [aper.do_photometry(img)[0] for aper in aper_all]

    img = np.nan_to_num(img, nan=0.0)
    if aper_src is not None: 
        # total flux of the source
        aper_src_flux = aper_src.do_photometry(img)[0][0]  
    else: 
        aper_src_flux = np.nansum(img[mask_src])

    if np.sum(np.isfinite(ap_fluxes)) < 3: 
        return 0, aper_src_flux, []

    return np.nanstd(ap_fluxes, ddof=1), aper_src_flux, aper_all



def get_a_b_ellipse(bin_mask):
    '''
    Approximate the bin with an ellipse.
    '''
    y, x = np.where(bin_mask)
    center_x = np.mean(x)
    center_y = np.mean(y)
    x_centered = x - center_x
    y_centered = y - center_y

    cov = np.cov(x_centered, y_centered)

    eigenvalues, _ = np.linalg.eig(cov)

    a = np.sqrt(max(eigenvalues[0], eigenvalues[1]) * 2)
    b = np.sqrt(min(eigenvalues[0], eigenvalues[1]) * 2)

    return a, b


def save_flux(bin_flux, bin_flux_err, filters, 
              out_path=None, prefix='', overwrite=True):
    '''
    Save the flux and flux error maps to FITS files.
    '''
    import os
    from astropy.table import Table

    n_bin = bin_flux.shape[0]

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if check_output(out_path + f'{prefix}flux_table.fits'):
        return 0 

    flux_table = Table(data=np.hstack((np.indices((n_bin, 1))[0], bin_flux)), 
                       names=np.hstack((['bin_id'], filters)), 
                       dtype=np.hstack(([np.int16], [float]*len(filters))))
    
    flux_table.write(out_path + f'{prefix}flux_table.fits', 
                     overwrite=overwrite)

    elux_table = Table(data=np.hstack((np.indices((n_bin, 1))[0], bin_flux_err)), 
                       names=np.hstack((['bin_id'], filters)), 
                       dtype=np.hstack(([np.int16], [float]*len(filters))))
    
    elux_table.write(out_path + f'{prefix}flux_table_err.fits', 
                     overwrite=overwrite)