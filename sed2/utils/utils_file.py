import json  # noqa: EXE002
import re
from pathlib import Path

from .utils_filter import *

__all__ = [
    "check_halpha",
    "check_image",
    "check_output",
    "load_filenames",
    "load_json",
    "parse_steps",
    "plot_stamps",
    "read_par",
    "resolve_coord",
    "save_json",
    "select_dict",
]


def read_par(par_file):
    """Read a parameter file (key-value format, # for comments)."""
    params = {}
    with open(par_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, val = line.split(None, 1)
            val = val.split("#")[0].strip()
            if val.lower() == "true":
                val = True
            elif val.lower() == "false":
                val = False
            elif val.lower() == "none":
                val = None
            elif "," in val:
                vals = [v.strip() for v in val.split(",")]
                try:
                    val = [int(v) for v in vals]
                except ValueError:
                    val = vals
            else:
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass

            if key == "steps" and isinstance(val, str):
                val = [val]

            params[key] = val
    return params


def parse_steps(steps_str):
    """Parse comma-separated step list into a list of step names."""
    return [s.strip() for s in steps_str.split(",")]


def _set_value(line, value):
    """Replace the value in a key-value line, keeping any trailing comment."""
    match = re.match(r"^(\s*\S+\s+)\S+(\s*#.*)?\s*$", line.rstrip("\n"))
    if not match:
        return line
    comment = match.group(2) or ""
    return f"{match.group(1)}{value}{comment}\n"


def resolve_coord(par_file, params):
    """
    Resolve the galaxy coordinates and update the parameter file.

    If ``ra``/``dec`` are already set in ``params``, they are returned
    as-is. Otherwise the galaxy name is resolved with
    ``SkyCoord.from_name`` and the resolved values are written back into
    the parameter file. An error is raised if the name cannot be resolved
    and no ra/dec are provided.
    """
    from astropy.coordinates import SkyCoord

    ra = params.get("ra")
    dec = params.get("dec")
    if ra is not None and dec is not None:
        return SkyCoord(ra, dec, unit="deg")

    galaxy = params.get("galaxy")
    if galaxy is None:
        raise ValueError("'galaxy' must be set to resolve coordinates.")

    try:
        coord = SkyCoord.from_name(galaxy)
    except Exception as exc:
        raise ValueError(
            f"Failed to resolve '{galaxy}' with SkyCoord and no ra/dec "
            f"provided in {par_file}."
        ) from exc

    ra, dec = coord.ra.value, coord.dec.value

    par_file = Path(par_file)
    with open(par_file) as f:
        lines = f.readlines()
    with open(par_file, "w") as f:
        for line in lines:
            key = line.split(None, 1)[0] if line.split() else ""
            if key == "ra":
                f.write(_set_value(line, f"{ra:.6f}"))
            elif key == "dec":
                f.write(_set_value(line, f"{dec:.6f}"))
            else:
                f.write(line)

    return coord

def check_output(out_path):
    '''
    check whether the output file exists.
    '''
    import os
    
    if os.path.exists(out_path):
        user_input = input("Output files exist. Enter 'Y' to overwrite: ")
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
        

def check_image(path, gal_name, filter_name):
    '''
    check whether an image exists.
    '''
    import os
    tarf = filter_name
    try:
        for fname in os.listdir(Path(path) / "data" / gal_name / "image"):
            if tarf in fname and gal_name in fname and ".fits" in fname: 
                return True
    except FileNotFoundError: 
        return False


def check_halpha(path, gal_name):
    import os
    try:
        for fname in os.listdir(Path(path) / "data" / gal_name / "image"):
            if "ha" in fname and ".fits" in fname: 
                return True
    except FileNotFoundError: 
        return False
    

def load_filenames(gal_name, prefix=''):
    '''
    Load the filenames for a given galaxy.
    '''
    filters = load_filters_feasts(gal_name)
    sci_img = {}
    var_img = {}
    for filt in filters:
        sci_img[filt] = f"{prefix}skybgsub_{gal_name}_{filt}.fits"
        var_img[filt] = f"{prefix}var_{gal_name}_{filt}.fits"

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
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.io import fits

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
    else:
        plt.show()