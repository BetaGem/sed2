import json
import numpy as np

from ..path import PATH

__all__ = ["load_gal_filters", "get_psf_size", "get_filter_trans_curve", 
           "get_filter_waves", "load_ref_info", "file_name", "calib_error", 
           "get_default_units", "get_mirband_idx"]

def load_gal_filters(gal_name):
    '''
    Load the filters for a given galaxy.
    '''
    with open(f"{PATH}/params/filter_used.json") as f:
        _dict = json.load(f)

    return _dict[gal_name]


def get_psf_size(filters):
    '''
    Get the PSF size for a list of filters.

    Parameters
    ----------
    filters : list of str
        The list of filters for which to get the PSF size.

    Returns
    -------
    psf_size : list or float
        The PSF size for the given filters.
    '''
    with open(f"{PATH}/filters/_filter_info.json") as f:
        _dict = json.load(f)

    if isinstance(filters, str):
        return _dict[filters]["psf_fwhm"]
    elif isinstance(filters, list):
        psf_sizes = []
        for i in filters:
            if i in _dict:
                psf_sizes.append(_dict[i]["psf_fwhm"])
            else:
                psf_sizes.append(None)
                print(f"Warning: Filter {i} not found in filter info.")
        return np.array(psf_sizes)
    else:
        raise TypeError("filters should be a string or a list of strings.")


def get_filter_trans_curve(filter_name):
    '''
    Get the transmission curve for a given filter.
    '''
    filter_data = np.loadtxt(f"{PATH}/filters/{filter_name}.par")
    wave, trans = filter_data[:, 0], filter_data[:, 1]
    
    return wave, trans / np.nanmax(trans)


def get_filter_waves(filter_name):
    '''
    Get the pivot & effective wavelengths for a list of filters.

    Parameters
    ----------
    filters : str
        The filter for which to get the pivot & effective wavelengths.

    Returns
    -------
    waves : tuple
        The pivot & effective wavelengths for the given filter.
    '''
    wave, trans = get_filter_trans_curve(filter_name)
    pivot_wave = np.sqrt(np.nansum(wave * trans) / np.nansum(trans / wave))

    effective_wave = np.nansum(wave**2 * trans) / np.nansum(wave * trans)

    return round(pivot_wave, 3), round(effective_wave, 3)


def load_ref_info(gal_name):
    '''
    load reference mask parameters, crop size of a galaxy
    '''
    with open(f"{PATH}/params/band_refs.json") as f:
        _dict = json.load(f)

    return _dict[gal_name]["filter"], _dict[gal_name]["nsigma"], _dict[gal_name]["size"]


def file_name(filter_name):
    '''
    get the corresponding file name of a filter or a list of filters.
    '''
    with open(f"{PATH}/filters/_filter_info.json") as f:
        _dict = json.load(f)

        if isinstance(filter_name, str):
            return _dict[filter_name]["fname"]
        elif isinstance(filter_name, list):
            return np.array([_dict[filt]["fname"] for filt in filter_name])


def calib_error(filter_name):
    '''
    get the calibration errors of a filter or a list of filters.
    '''
    with open(f"{PATH}/filters/_filter_info.json") as f:
        _dict = json.load(f)
        if isinstance(filter_name, str):
            return float(_dict[filter_name]["cali_err"]) / 100
        elif isinstance(filter_name, list):
            return np.array([float(_dict[filt]["cali_err"]) / 100 for filt in filter_name])
        

def get_default_units(filters):
    '''
    Get the default units (flux or surface brightness) for a list of filters.

    Parameters
    ----------
    filters : list of str
        The list of filters for which to get the default units.

    Returns
    -------
    units : list of str
        The default units for the given filters.
    '''
    with open(f"{PATH}/filters/_filter_info.json") as f:
        _dict = json.load(f)

    units = {}
    for filt in filters:
        if filt in _dict:
            units[filt] = _dict[filt].get("default_unit", "flux")
        else:
            units[filt] = "flux"  # default to flux if filter not found
            print(f"Warning: Filter {filt} not found in filter info.")

    return units


def get_mirband_idx(filters):
    '''
    Get the index of MIR bands in the filter list.
    '''
    mir_idx = []
    for f in ['wise_w1', 'wise_w2', 'wise_w3', 'wise_w4',
              'spitzer_irac_36', 'spitzer_irac_45',
              'spitzer_irac_58', 'spitzer_irac_80']:
        idx = np.where(np.array(filters) == f)
        if len(idx[0]): 
            mir_idx.append(idx[0][0])
        else:
            mir_idx.append(-1)

    return mir_idx