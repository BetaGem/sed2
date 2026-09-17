import json  # noqa: EXE002
from pathlib import Path

import numpy as np

from ..path import PATH

__all__ = [
    "calib_error",
    "get_default_units",
    "get_filter_trans_curve",
    "get_filter_waves",
    "get_mirband_idx",
    "get_psf_size",
    "load_filter_ids",
    "load_filters",
    "load_filters_feasts",
    "load_filters_from_txt",
    "load_ref_info",
]


def load_filters_from_txt(filter_file):
    '''
    Load the filter list from a text file.

    Parameters
    ----------
    filter_file : str or Path
        Path to the text file containing the filter list.

    Returns
    -------
    filters : list of str
        Filter names read from the text file.
    '''
    filter_file = Path(filter_file)
    if not filter_file.is_file():
        raise FileNotFoundError(f"Filter file not found: {filter_file}")

    with open(filter_file) as f:
        filters = [line.strip() for line in f if line.strip()]

    return filters


def load_filter_ids(workdir, par_str):
    '''
    Load the filter list from string in the parameter file.
    '''
    try:
        filters = load_filters_from_txt(workdir / "aux" / "filter_list.txt")
    except FileNotFoundError:
        raise FileNotFoundError("Filter list file is required to load filter IDs")

    if par_str is None:
        f_high = load_filters_from_txt(workdir / "aux" / "filters_highres.txt")
        filter_ids = [filters.index(f) for f in f_high]
    elif isinstance(par_str, str):
        filter_ids = np.where([par_str in filters[i] for i in range(len(filters))])[0]
    elif isinstance(par_str, int):
        filter_ids = [par_str]
    elif isinstance(par_str, (tuple, list)):
        if isinstance(par_str[0], int):
            filter_ids = list(par_str)
        elif isinstance(filters[0], str):
            filter_ids = []
            for b in par_str:
                idx = np.where([b in filters[i] for i in range(len(filters))])[0]
                filter_ids.extend(idx)

    return filter_ids


def load_filters_feasts(gal_name):
    '''
    Load the filters for a given galaxy. (this is an old function)
    '''
    with open(PATH / "params" / "filter_used.json") as f:
        _dict = json.load(f)

    return _dict[gal_name]


def _filters_from_images(image_dir, gal_name=None):
    '''
    Derive the filter list from the image file names. The file name must
    contain exactly the filter name (after stripping the extension and the
    galaxy prefix).
    '''
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    filters = []
    for p in sorted(image_dir.iterdir()):
        stem = p.name
        for ext in (".fits.gz", ".fits", ".fit"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break

        # the file name must contain exactly the filter name
        if gal_name is not None and stem.startswith(f"{gal_name}_"):
            stem = stem[len(gal_name) + 1:]

        filters.append(stem)

    return filters


def load_filters(workdir, gal_name=None):
    '''
    Load the complete filter list for a galaxy.

    Filters are read from ``<workdir>/aux/filter_list.txt``. If that file
    does not exist, the list is generated from the FITS file names in
    ``<workdir>/image`` and written back to ``filter_list.txt``. Each filter
    is then checked against the available filters in ``sed2/filters``; an
    error is raised if a filter is missing.

    Parameters
    ----------
    workdir : str or Path
        Galaxy working directory (must contain an ``image`` sub-folder).
    gal_name : str, optional
        Galaxy name, used to strip the prefix from image file names.

    Returns
    -------
    list of str
        Filter names available for this galaxy.
    '''
    workdir = Path(workdir)
    filter_file = workdir / "aux" / "filter_list.txt"

    if filter_file.is_file():
        with open(filter_file) as f:
            filters = [line.strip() for line in f if line.strip()]
    else:
        filters = _filters_from_images(workdir / "image", gal_name)
        with open(filter_file, "w") as f:
            f.write("\n".join(filters) + "\n")

    # check that all filters exist in the filter library
    for f in filters:
        par_file = PATH / "filters" / f"{f}.par"
        if not par_file.is_file():
            raise FileNotFoundError(
                f"Filter '{f}' not found in {PATH / 'filters'} "
                f"({par_file.name} does not exist)."
            )

    # sort the filters according to their wavelengths (pivot wavelength)
    waves = [get_filter_waves(f)[0] for f in filters]
    sorted_indices = np.argsort(waves)
    filters = [filters[i] for i in sorted_indices]

    return filters


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
    with open(PATH / "filters" / "_filter_info.json") as f:
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
                print(f"Warning: Filter {i} info not found, please add in sed2/filters/_filter_info.json.")
        return np.array(psf_sizes)
    else:
        raise TypeError("filters should be a string or a list of strings.")


def get_filter_trans_curve(filter):
    '''
    Get the transmission curve for a given filter.
    '''
    filter_data = np.loadtxt(PATH / "filters" / f"{filter}.par")
    wave, trans = filter_data[:, 0], filter_data[:, 1]
    
    return wave, trans / np.nanmax(trans)


def get_filter_waves(filter):
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
    wave, trans = get_filter_trans_curve(filter)
    pivot_wave = np.sqrt(np.nansum(wave * trans) / np.nansum(trans / wave))

    effective_wave = np.nansum(wave**2 * trans) / np.nansum(wave * trans)

    return round(pivot_wave, 3), round(effective_wave, 3)


def load_ref_info(gal_name):
    '''
    load reference mask parameters, crop size of a galaxy (this is an old function)
    '''
    with open(PATH / "params" / "band_refs.json") as f:
        _dict = json.load(f)

    return _dict[gal_name]["filter"], _dict[gal_name]["nsigma"], _dict[gal_name]["size"]


def calib_error(filter_name):
    '''
    get the calibration errors of a filter or a list of filters.
    '''
    with open(PATH / "filters" / "_filter_info.json") as f:
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
    with open(PATH / "filters" / "_filter_info.json") as f:
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