import numpy as np
from math import cos, sin, sqrt
from astropy.io import fits

from ..utils import *
from ..path import PATH

__all__ = ['estimate_metallicity', 'plot_metal_map']

def estimate_metallicity(pixbin_path, ref_img_path=None, logMstar=10,
                         R50=None, PA=None, incl=None, gradient=-0.1, Zsun=8.69,
                         out_path='manual_priors.npz'):
    '''
    Estimate nebular metallicity for each spatial bin based on
    the mass-metallicity relation and a radial gradient.
    
    Parameters:
    -----------
    pixbin_path : str
        Path to the pixel binning FITS file.
    ref_img_path : str
        Path to the reference image FITS file for spatial scale.
        If None, will assume 5.5 arcsec/pixel.
    R50 : float
        Half-light radius in arcseconds. 
    logMstar : float
        Estimated log stellar mass in solar masses.
    PA : float
        Position angle of the galaxy in degrees.
    gradient : float
        Radial metallicity gradient in dex/R50. 
        Default is -0.1 (Sanchez et al. 2014).
    incl : float
        Inclination of the galaxy in degrees.
    out_path : str
        Path to save the nebular metallicity array (numpy array).
    '''
    
    # apply MZR in Sanchez et al. 2019 (2019MNRAS.484.3042S)
    Z0 = 8.51 + 0.007 * (logMstar - 11.5) / np.exp(logMstar - 11.5)

    if gradient != 0 and (PA is None or incl is None):
        print("Warning: either position angle 'PA' or inclination 'incl'\
               must be provided to estimate spatially resolved metallicity.")
        PA, incl = 0, 0
    
    PA = np.deg2rad(PA)
    incl = np.deg2rad(incl)

    binmap = fits.open(pixbin_path)[0].data
    pixel_scale = get_pixel_size(ref_img_path)
    Z_map = np.full(binmap.shape, np.nan)
    x_cen, y_cen = (binmap.shape[1] - 1) / 2, (binmap.shape[0] - 1) / 2

    for i in range(binmap.shape[0]):
        for j in range(binmap.shape[1]):
            if binmap[i, j] == 0: continue
            x, y = j - x_cen, i - y_cen
            if incl <= np.deg2rad(75):
                Z_map[i, j] = sqrt((x*sin(PA) - y*cos(PA))**2 +\
                                   (x*cos(PA) + y*sin(PA))**2 / cos(incl)**2)\
                                   * pixel_scale / R50
            else:
                Z_map[i, j] = abs(x*sin(PA) - y*cos(PA)) * pixel_scale / R50

    Z_map = 10**(Z0 + gradient * (Z_map - 1) - Zsun)

    # median metallicity in each bin
    Z_bin_arr = []
    for i in range(np.max(binmap).astype(int)):
        Z_bin = np.median(Z_map[binmap == i + 1])
        Z_bin_arr.append(Z_bin)

    np.savez(out_path, zgas=np.array(Z_bin_arr))


def estimate_eta(logMstar, out_path='manual_priors.npz'):
    '''
    Estimate the eta parameter based on Lin & Kong (2020)
    '''
    import os
    eta = 3.460 - 0.277 * logMstar
    
    if not os.path.exists(out_path):
        np.savez(out_path, eta=eta)
    else:
        npz = np.load(out_path, allow_pickle=True)
        zgas = npz['zgas'] if 'zgas' in npz else None
        np.savez(out_path, zgas=zgas, eta=eta)


def plot_metal_map(prior_path, pixbin_path, **kwargs):
    '''
    Plot the nebular metallicity map.
    
    Parameters:
    -----------
    prior_path : str
        Path to the manual priors (numpy .npz).
    pixbin_path : str
        Path to the pixel binning FITS file.
    '''
    import matplotlib.pyplot as plt

    binmap = fits.open(pixbin_path)[0].data
    nebular_metal = np.load(prior_path)['zgas']

    metal_map = np.zeros_like(binmap, dtype=float)
    for i in range(int(np.max(binmap))):
        metal_map[binmap == i+1] = nebular_metal[i]

    plt.imshow(metal_map, origin='lower', **kwargs)
    plt.colorbar(label=r'Nebular Metallicity ($Z/Z_\odot$)')
    plt.title('Nebular Metallicity Map')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.show()
