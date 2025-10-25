import json
from ..path import PATH
from .utils_filter import *

__all__ = ["load_json", "save_json", "check_image", "check_halpha",
           "load_filenames", "select_dict", "plot_stamps", "check_output"]

def check_output(out_path):
    '''
    check whether the output file exists.
    '''
    import os
    
    if os.path.exists(out_path):
        user_input = input("Output files exist. Press 'Y' to overwrite.")
        if user_input.upper() != 'Y':
            print("Exiting...")
            return 1
    return 0


def load_json(path):
    '''
    open a json file
    '''
    with open(path) as f: _dict = json.load(f)
    return _dict


def save_json(dict, path):
    '''
    save a python dictionary to json
    '''
    with open(path, 'w+') as f:
        json.dump(dict, f, indent=4)
        

def check_image(gal_name, filter_name):
    '''
    check whether an image exists.
    '''
    import os
    tarf = file_name(filter_name)
    try:
        for fname in os.listdir(f"{PATH}/data/{gal_name}/masked"):
            if tarf in fname and gal_name in fname and ".fits" in fname: 
                return True
    except: 
        return False


def check_halpha(gal_name):
    import os
    try:
        for fname in os.listdir(f"{PATH}/data/{gal_name}/masked"):
            if "ha" in fname and ".fits" in fname: 
                return True
    except: 
        return False
    

def load_filenames(gal_name, prefix=''):
    '''
    Load the filenames for a given galaxy.
    '''
    filters = load_gal_filters(gal_name)
    sci_img = {}
    var_img = {}
    for filt in filters:
        sci_img[filt] = f"{prefix}skybgsub_{gal_name}_{file_name(filt)}.fits"
        var_img[filt] = f"{prefix}var_{gal_name}_{file_name(filt)}.fits"

    return sci_img, var_img


def select_dict(dict, keys):
    '''
    Select a subset of a dictionary.
    '''
    return {k: dict[k] for k in keys if k in dict}



def plot_stamps(img_paths, filters=None, figsize=(15, 8), log=False,
                vmin=16, vmax=99, out_path=None):
    '''
    plot a set of images.

    Parameters
    ----------
    img_paths : list
        A list of image file paths to plot.
    filters : list, optional
        A list of filters to apply to the images.
    figsize : tuple, optional
        The size of the figure to create.
    log : bool, optional
        Whether to use a logarithmic scale for the images.
    '''
    import numpy as np
    from astropy.io import fits
    import matplotlib.pyplot as plt

    n_img = len(img_paths)
    col, row = 6, (n_img // 6) + 1

    plt.figure(figsize=figsize)

    for i, img_path in enumerate(img_paths):    
        # load image
        hdu = fits.open(img_path)
        img_data = np.log10(hdu[0].data) if log else hdu[0].data
        # create WCS projection
        plt.subplot(row, col, i+1)
        plt.imshow(img_data, 
                   vmin=np.nanpercentile(img_data, vmin),
                   vmax=np.nanpercentile(img_data, vmax),
                   origin='lower', cmap='turbo')
        if filters is not None:
            plt.title(filters[i])
    
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
