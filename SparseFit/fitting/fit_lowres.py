import numpy as np
from math import sqrt
from astropy.io import fits
from astropy.table import Table

from ..utils import *
from .load_sed import BSEDresults
from ..image_process.image_psf import load_kernel
from ..path import PATH

__all__ = ["predict_flux_table", "convolve_flux_map", 
           "predict_flux_map", "new_flux_table"]


def predict_flux_table(highres_flux_path, filters, seds=None,
                       run='_highres', galaxy=None, path_posterior='', manual_prior=None,
                       out_path=None, prefix='pred_', overwrite=True):
    '''
    Create flux table involving all filters. 
    The fluxes of low-resolution filters are predicted using
    the high-resolution SEDs.

    Parameters
    ----------
    highres_flux_path : str
        Path to the high-resolution flux table.
    filters : list
        List of filter names.
    seds : list, optional
        List of BSEDresults objects for each bin.
    galaxy : str, optional
        Name of the galaxy.
    path_posterior : str, optional
        Path to the posterior files.
    manual_prior : str, optional
        Path to the manual prior file.
    out_path : str, optional
        Output path for the predicted flux table.
    prefix : str, optional
        Prefix for the output file name.
    overwrite : bool, optional
        Overwrite existing files.
    '''
    # initialize
    highres_tab = Table.read(highres_flux_path)
    highres_err = Table.read(highres_flux_path.replace(".fits", "_err.fits"))
    n_bin = len(highres_tab)
    highres_filters = highres_tab.colnames[1:]

    # calibration error
    e_cali = calib_error(filters)

    pred_flux = np.full((n_bin, len(filters)), np.nan)
    pred_flux_err = np.full((n_bin, len(filters)), np.nan)

    # create predicted flux table
    for j in range(n_bin):

        # Load SED for the current bin
        if seds is not None:
            sed = seds[j]
        else:
            sed = BSEDresults(galaxy, ID=j, run=run, advanced=True,
                              flux_table=highres_flux_path, manual_prior=manual_prior,
                              path_posterior=path_posterior)

        for i, f in enumerate(filters):
            if f in highres_filters:
                pred_flux[j, i] = highres_tab[f][j]
                pred_flux_err[j, i] = highres_err[f][j]
            else:
                pred_phot = sed.predict_flux(f)
                pred_flux[j, i] = pred_phot[1]
                pred_flux_err[j, i] = sqrt((pred_phot[2] - pred_phot[0])**2 / 4 + (e_cali[i] * pred_flux[j, i])**2)

        # clean up memory
        del sed

    save_flux(pred_flux, pred_flux_err, filters, 
              out_path=out_path, prefix=prefix, overwrite=overwrite)



def convolve_flux_map(pred_map, pixbin_map, pix_scale=None, out_path=None):
    '''
    Convolve the predicted flux map to the low-res PSF.

    Parameters
    ----------
    pred_map : dict
        Dictionary of predicted flux maps for each low-res filter.
    pixbin_map : str
        Path to the pixel bin map.
    pix_scale : float
        Pixel scale in arcsec/pixel.
    out_path : str, optional
        Output path for the convolved maps.

    Returns
    -------
    pred_maps_conv : dict
        Dictionary of convolved predicted flux maps for each low-res filter.
    '''
    from astropy.nddata import Cutout2D
    from astropy.convolution import convolve_fft

    with fits.open(pixbin_map) as hdul:
        header = hdul[0].header

    pred_maps_conv = {}

    for f in pred_map.keys():
        kernel = load_kernel(header['psfband'], f, pix_scale=pix_scale)
        kernel = Cutout2D(kernel, position=(kernel.shape[1]//2, 
                                            kernel.shape[0]//2), 
                          size=pred_map[f][0].shape[0]//2, mode='trim').data

        pred_map_conv = convolve_fft(pred_map[f][0], kernel, 
                                normalize_kernel=True, nan_treatment='fill',
                                preserve_nan=True, allow_huge=True)
        pred_var_conv = convolve_fft(pred_map[f][1]**2, kernel**2 / np.sum(kernel**2), 
                                normalize_kernel=True, nan_treatment='fill',
                                preserve_nan=True, allow_huge=True)
        pred_maps_conv[f] = (pred_map_conv, np.sqrt(pred_var_conv))

    # save convolved maps
    if out_path is not None:
        np.savez(out_path + "pred_map_conv.npz", **pred_maps_conv)
        return out_path + "pred_map_conv.npz"
    else:
        return pred_maps_conv



def predict_flux_map(pred_flux_path, pixbin_map, filters_lowres=None, 
                     out_path=None):
    '''
    Map the predicted flux table to the pixel bin map.

    Parameters
    ----------
    pred_flux_path : str
        Path to the predicted flux table.
    pixbin_map : str
        Path to the pixel bin map.
    filters_lowres : list, optional
        List of low-resolution filter names.
    out_path : str, optional
        Output path for the predicted flux maps.

    Returns
    -------
    pred_maps : dict
        Dictionary of predicted flux maps for each low-res filter.
    '''

    pred_flux_tab = Table.read(pred_flux_path)
    pred_flux_err = Table.read(pred_flux_path.replace(".fits", "_err.fits"))

    with fits.open(pixbin_map) as hdul:
        binmap = hdul[0].data.astype(int)
        header = hdul[0].header
        n_bin = binmap.max()

    # bin areas in pixel
    bin_areas = np.bincount(binmap.ravel())[1:]

    # get filter lists
    if filters_lowres is None:
        filters = pred_flux_tab.colnames[1:]
        filters_highres = [header[key] for key in list(header.keys()) if key.startswith('FIL')]
        filters_lowres = np.setdiff1d(filters, filters_highres)

    print("Low-res filters:\n", filters_lowres)

    # fill predicted maps
    pred_maps = {}
    
    # iterate over low-res filters
    for f in filters_lowres:

        pred_map = np.full_like(binmap, 0, dtype=np.float64)
        pred_err = np.full_like(binmap, 0, dtype=np.float64)

        for i in range(n_bin):
            pred_map[binmap == i + 1] = pred_flux_tab[f][i] / bin_areas[i]
            pred_err[binmap == i + 1] = pred_flux_err[f][i] / sqrt(bin_areas[i])

        pred_maps[f] = (pred_map, pred_err)

    # save predicted maps
    if out_path is not None:
        np.savez(out_path + "pred_map.npz", **pred_maps)
        return out_path + "pred_map.npz"
    else:
        return pred_maps



def new_flux_table(fluxmap_lowres, pixbin_map, pred_flux_path, 
                   iteration=1, pix_scale=None, filters_lowres=None,
                   out_file=None, overwrite=True, save_pred_path=None):
    '''
    Create a scaled high-res flux table.

    Parameters
    ----------
    fluxmap_lowres : str
        Path to the low-res flux map.
    pixbin_map : str
        Path to the pixel bin map.
    pred_flux_path : str
        Path to the predicted flux table.
    iteration : int
        Number of iterations for scaling.
    pix_scale : float
        Pixel scale of the matching kernel in arcsec/pixel.
    out_file : str
        Path to the output flux table.
    overwrite : bool
        Whether to overwrite existing files.
    save_pred_path : str
        Path to save the predicted flux maps.

    Returns
    -------
    pred_map_upd : dict
        Dictionary of updated predicted flux maps for each low-res filter.
    '''
    from tqdm import tqdm
    from copy import deepcopy

    # load data and initialize
    pred_map_upd = {}

    pred_map = predict_flux_map(pred_flux_path, pixbin_map, 
                                filters_lowres=filters_lowres)
    if save_pred_path is not None:
        np.savez(save_pred_path + "pred_map.npz", **pred_map)

    pixbin_hdul = fits.open(pixbin_map)
    binmap = pixbin_hdul[0].data.astype(int)
    bin_ids = binmap.ravel()
    n_bin = binmap.max()

    fluxmap = fits.open(fluxmap_lowres)
    flux_unit = fluxmap[0].header['UNIT']
    filters_lowres = [fluxmap[0].header[k] for k in fluxmap[0].header if k.startswith('FIL')]

    if pred_flux_path is not None:
        pred_flux = deepcopy(Table.read(pred_flux_path))
        pred_err  = deepcopy(Table.read(pred_flux_path.replace(".fits", "_err.fits")))

    for n in tqdm(range(iteration)):
        pred_map_conv = convolve_flux_map(pred_map, pixbin_map, pix_scale=pix_scale)

        for i, f in enumerate(filters_lowres[1:]):

            map_obs = fluxmap[0].data[i+1] * flux_unit
            map_obs[~np.isfinite(map_obs)] = 0

            # e_obs = fluxmap[1].data[i+1] * flux_unit
            pred_map_upd[f] = deepcopy(pred_map[f])

            # observed flux and error
            flux_obs = np.bincount(bin_ids, 
                                   weights=map_obs.ravel())
            # e_flux_obs = np.sqrt(np.bincount(bin_ids, 
            #                                  weights=(e_obs**2).ravel()))

            # convolved predicted flux and error
            flux_conv = np.bincount(bin_ids, 
                                    weights=pred_map_conv[f][0].ravel())
            # e_flux_conv = np.sqrt(np.bincount(bin_ids, 
            #                                   weights=(pred_map_conv[f][1]**2).ravel()))

            # scaling factor (0th = background outside RoI)
            scale = np.ones(n_bin + 1)
            valid = (flux_conv > 0) & (flux_obs > 0)
            scale[valid] = flux_obs[valid] / flux_conv[valid]

            # select bright bins or faint bins
            sb_obs = (flux_obs / np.bincount(bin_ids))[1:]
            bright = sb_obs > np.median(sb_obs)
            target_bins = np.where(bright)[0] if n % 2 else np.where(~bright)[0]

            # update predictions for each bin (vectorized mask)
            # for j in range(n_bin):
            for j in target_bins:

                bin_mask = (binmap == j + 1)
                pred_map_upd[f][0][bin_mask] *= scale[j + 1]
                pred_map_upd[f][1][bin_mask] *= scale[j + 1]

                if pred_flux_path is not None:
                    pred_flux[f][j] *= scale[j + 1]
                    pred_err[f][j]  *= max(1, scale[j + 1])

        if iteration > 1:
            pred_map = pred_map_upd

    if out_file is not None:
        pred_flux.write(out_file, overwrite=overwrite)
        pred_err.write(out_file.replace(".fits", "_err.fits"), overwrite=overwrite)

    if save_pred_path is not None:
        np.savez(save_pred_path + "pred_map_updated.npz", **pred_map_upd)
    else:
        return pred_map_upd
