import bagpipes as pipes  # noqa: EXE002
import numpy as np
from astropy.table import Table

from ..path import PATH

__all__ = [
    "build_all",
    "fit_info",
    "run",
]

# Narrow-band H-alpha filters
# TODO: add more filters if needed
HA_FILTERS = {
    "bok_halpha_657",
    "kpno_kp1563",
    "kpno_kp1564",
    "ctio_ct6602",
    "ctio_ct6586",
    "vatt_halpha_658",
}

# functions
# -----------------------------
class _Photometry:
    """
    Callable photometry provider for Bagpipes.

    Encapsulates the flux table and the finite-value mask so that no module
    level state is required: the instance is passed directly to
    ``pipes.galaxy`` and called with the bin ``ID``.
    """
    def __init__(self, flux_table, filter_list=None, redshift=0.0022):
        self.catalog = Table.read(str(flux_table))
        self.catalog_err = Table.read(str(flux_table).replace(".fits", "_err.fits"))
        self.filters = (list(filter_list) if filter_list is not None
                        else list(self.catalog.colnames[1:]))
        self.redshift = redshift
        self.fini = None

    def __call__(self, ID):
        ID = int(ID)

        # flux in unit of erg/s/cm^2/Angstrom
        if ID == 9999:
            f_lamb = np.nansum(list(self.catalog.values()), axis=1)[1:]
            f_lamb_err = np.nansum(list(self.catalog_err.values()), axis=1)[1:]
        else:
            f_lamb = np.array(list(self.catalog[ID].values()))[1:]
            f_lamb_err = np.array(list(self.catalog_err[ID].values()))[1:]

        # remove nan values
        self.fini = np.isfinite(f_lamb) & np.isfinite(f_lamb_err)
        f_lamb = f_lamb[self.fini]
        f_lamb_err = f_lamb_err[self.fini]

        # turn this into a 2D array
        photometry = np.array([f_lamb, f_lamb_err]).T

        # add a constant model uncertainty of 5%
        photometry[:, 1] = np.sqrt(photometry[:, 1]**2
                                   + (0.05 * photometry[:, 0])**2)

        return photometry

    def filter_paths(self):
        """Filter transmission files matching the finite photometry."""
        if self.fini is None:
            raise RuntimeError(
                "Call the photometry provider before requesting filter paths.")
        filter_path = PATH / "filters"
        filter_list = [filter_path / f"{f}.par" for f in self.filters]
        return np.array(filter_list)[self.fini]


def fit_info(dust_emission=True, nebular_emission=True, Leja_SFH=True,
             use_halpha=False, nebular_metallicity=None, dust_ratio=None,
             redshift=0.0022):
    """
    Create the fit instruction dictionary.

    The users should modify this function to change the SED models and priors.
    For more details, please refer to the Bagpipes documentation:
    https://bagpipes.readthedocs.io
    """
    fit_instructions = {}
    fit_instructions["t_bc"] = 0.01
    fit_instructions["redshift"] = redshift
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
    dust["n"] = (0., 1.5)
    dust["n_prior"] = "Gaussian"
    dust["n_prior_mu"] = 0.7
    dust["n_prior_sigma"] = 0.3
    dust["eta"] = (1., 4.)
    dust["eta_prior"] = "Gaussian"
    if dust_ratio is not None:
        dust["eta_prior_mu"] = float(dust_ratio)
    else:
        dust["eta_prior_mu"] = 1 / 0.44
    dust["eta_prior_sigma"] = 0.3

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
        continuity["metallicity"] = (0.02, 2.5)    
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
              redshift=0.0022, manual_prior=None):
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
    manual_prior : str
        The path to the manual prior file

    Returns
    -------
    galaxy : pipes.galaxy
        The galaxy object.
    fit_inst : dict
        The fit instruction dictionary.
    """
    # photometry provider (encapsulates the catalog: no module-level state)
    phot = _Photometry(flux_table, filter_list, redshift)
    filters = phot.filters

    # check if Halpha is in the filter list
    halpha = any(f in HA_FILTERS for f in filters)

    # load priors calibrated from spectroscopic data
    ## you can modify this part to load your own priors
    ## by default, the priors are produces in sed2.binning.pix_priors
    zgas, dust_ratio = None, None

    if manual_prior is not None:
        prior_npz = np.load(manual_prior)
        if 'zgas' in prior_npz:
            try:
                zgas = prior_npz['zgas'][ID]
            except IndexError:
                zgas = prior_npz['zgas'][0]
                # zgas = np.nanmean(prior_npz['zgas'])

        if 'eta' in prior_npz:
            dust_ratio = prior_npz['eta']

    _ = phot(ID)      # populate the finite-value mask
    filter_paths = phot.filter_paths()

    # standard Bagpipes objects
    galaxy = pipes.galaxy(ID, phot, filt_list=filter_paths,
                          spectrum_exists=False, phot_units='ergscma')
    fit_inst = fit_info(use_halpha=halpha,
                        nebular_metallicity=zgas, dust_ratio=dust_ratio,
                        redshift=redshift)

    return galaxy, fit_inst


def run(ID, flux_table, manual_prior=None, 
        filter_list=None, redshift=0.0022, 
        run='.', nlive=1000, pool=1, 
        path='.', verbose=True, **kwargs):
    '''
    Run Bagpipes SED fitting.
    '''

    galaxy, fit_inst = build_all(ID, flux_table, filter_list, 
                                 redshift, manual_prior)

    fit = pipes.fit(galaxy, fit_inst, run=run, path=path)
    fit.fit(sampler="nautilus", 
            verbose=verbose, n_live=nlive, pool=pool, **kwargs)
