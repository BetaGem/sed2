import numpy as np 
import bagpipes as pipes
from astropy.table import Table

from ..path import PATH
from ..utils import *

__all__ = ["load_phot", "load_filters", "fit_info", 
           "build_all", "run"]

# functions
# -----------------------------
def load_phot(ID):
    """
    Load photometry for a given spatial bin ID.
    """
    global fini
    ID = int(ID)

    # flux in unit of erg/s/cm^2/Angstrom
    if ID == 9999:
        f_lamb = np.nansum(list(catalog.values()), axis=1)[1:]
        f_lamb_err = np.sqrt(np.nansum(np.square(list(catalog_err.values())), axis=1))[1:]
    else:
        f_lamb = np.array(list(catalog[ID].values()))[1:]
        f_lamb_err = np.array(list(catalog_err[ID].values()))[1:]

    # remove nan values
    fini = np.isfinite(f_lamb) & np.isfinite(f_lamb_err)
    f_lamb = f_lamb[fini]
    f_lamb_err = f_lamb_err[fini]

    # turn this into a 2D array
    photometry = np.array([f_lamb, f_lamb_err]).T

    # Enforce a maximum SNR
    for i in range(len(photometry)):
        
        max_snr = 20
        if ID == 9999 and 'mips' in filters[i]: 
            max_snr = 10

        if photometry[i, 0] / photometry[i, 1] > max_snr:
            photometry[i, 1] = photometry[i, 0] / max_snr

    return photometry


def load_filters(filter_names):
    """
    Create a list of filter paths from sedpy.
    """
    filter_path = f"{PATH}/filters"
    filter_list = [f"{filter_path}/{f}.par" for f in filter_names]

    return np.array(filter_list)[fini]


def fit_info(dust_emission=True, nebular_emission=True, Leja_SFH=True,
             use_halpha=False, nebular_metallicity=None):
    """
    Create the fit instruction dictionary.
    """
    fit_instructions = {}
    fit_instructions["t_bc"] = 0.01
    fit_instructions["redshift"] = z
    _age_myr = (pipes.utils.cosmo.age(fit_instructions["redshift"]).value - \
                pipes.utils.cosmo.age(10).value) * 1e3  # convert to Myr

    # dust attenuation / emission parameters
    dust = {}
    if dust_emission:
        dust["qpah"]  = (0.1, 4.58)
        dust["umin"]  = (0.1, 25.0)
        dust["gamma"] = (1e-4, 0.5)

    dust["type"] = "CF00"
    dust["Av"] = (0., 4.)
    dust["n"] = (0.3, 1.5)
    dust["n_prior"] = "Gaussian"
    dust["n_prior_mu"] = 0.7
    dust["n_prior_sigma"] = 0.3
    # eta parameter based on Wild et al. (2011)
    dust["eta"] = (1., 4.)
    dust["eta_prior"] = "Gaussian"
    dust["eta_prior_mu"] = 2.5
    dust["eta_prior_sigma"] = 0.5 
    fit_instructions["dust"] = dust
    
    if nebular_emission:
        nebular = {}
        nebular['logU'] = -3
        if nebular_metallicity is not None:
            nebular['metallicity'] = nebular_metallicity
        fit_instructions["nebular"] = nebular

    if Leja_SFH:
        continuity = {}
        continuity["massformed"] = (4, 12)
        continuity["metallicity"] = (0.01, 3)
        continuity["metallicity_prior"] = "log_10"
        if use_halpha:
            continuity['bin_edges'] = [0, 10, 30, 100, 300, 
                                       1000, 3000, 7000, _age_myr]
        else:
            continuity['bin_edges'] = [0, 30, 100, 300, 
                                       1000, 3000, 7000, _age_myr]
        for i in range(1, len(continuity["bin_edges"]) - 1):
            continuity["dsfr" + str(i)] = (-5., 5.)
            continuity["dsfr" + str(i) + "_prior"] = "student_t"
        fit_instructions["continuity"] = continuity
    
    return fit_instructions


def build_all(ID, flux_table, filter_list=None, 
              redshift=0.0022, nebular_metal=None):
    """ 
    Build the galaxy and fit instruction objects for Bagpipes.
    Please modify this function for specphotometry fitting.

    Parameters
    ----------
    ID : int
        The ID of the galaxy to fit.
    flux_table : str
        The path to the flux table.
    filter_list : list
        The list of filters to use.
    redshift : float
        The redshift of the galaxy.
    nebular_metal_table : str
        The path to the nebular metallicity table.
        If None, fixed to stellar metallicity.
    Returns
    -------
    galaxy : pipes.galaxy
        The galaxy object.
    fit_inst : dict
        The fit instruction dictionary.
    """
    # define some global variables
    global catalog, catalog_err, filters, fini, z

    z = redshift
    catalog = Table.read(flux_table)
    catalog_err = Table.read(flux_table.replace(".fits", "_err.fits"))
    
    if filter_list is not None:
        filters = filter_list
    else:
        filters = catalog.colnames[1:] # use all filters in the table

    # check if Halpha is in the filter list
    halpha = True if 'ha' in file_name(filters) else False

    # load metallicity info
    if nebular_metal is not None:
        zgas = np.load(nebular_metal)[ID]
    else: zgas = None

    # initialize fini
    fini = np.full_like(filters, True)
    _ = load_phot(ID)  

    filter_paths = load_filters(filters)
    galaxy = pipes.galaxy(ID, load_phot, 
                          filt_list=filter_paths, 
                          spectrum_exists=False, 
                          phot_units='ergscma')
    fit_inst = fit_info(use_halpha=halpha, 
                        nebular_metallicity=zgas)

    return galaxy, fit_inst


def run(ID, flux_table, nebular_metal=None, 
        filter_list=None, redshift=0.0022, 
        run='.', nlive=1000, pool=1, verbose=True):
    '''
    Run Bagpipes SED fitting.
    '''

    galaxy, fit_inst = build_all(ID, flux_table, filter_list, 
                                 redshift, nebular_metal)

    fit = pipes.fit(galaxy, fit_inst, run=run)
    fit.fit(sampler="nautilus", 
            verbose=verbose, n_live=nlive, pool=pool)
