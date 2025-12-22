import numpy as np
import matplotlib.pyplot as plt
import bagpipes as pipes

from ..path import PATH
from ..utils.utils_filter import *
from . import fit_bagpipes

__all__ = ["BSEDresults"]


class BSEDresults(object):
    """
    Class to handle SED fitting results from Bagpipes
    (assuming a non-parametric SFH).
    """
    def __init__(self, name, ID, flux_table, manual_prior=None,
                 filter_list=None, redshift=0.0022, distance=0, run='',
                 advanced=False, path_posterior='', n_posterior=1000,
                 save_memory=False):
        """
        Initialize the SEDresults object with the result files.
        """

        script = fit_bagpipes
        self.name = name
        self.ID = ID
        self.run = name + run

        self.galaxy, self.fit_info = script.build_all(ID, flux_table,
                                                      filter_list, redshift,
                                                      manual_prior=manual_prior)
        # TODO: Remove the path when publishing
        self.fit = pipes.fit(self.galaxy, self.fit_info,
                             run=self.run, n_posterior=n_posterior,
                             path=path_posterior)  
        if advanced:
            # WARNING: this is memory intensive
            # The program may crash when loading too many objects in this mode
            self.fit.posterior.get_advanced_quantities()

        if distance > 0:
            self._correct_distance(distance)
            self.distance = distance
        else:
            self.distance = pipes.utils.cosmo.luminosity_distance(redshift).value

        if save_memory:
            self.age_bins = self.fit_info['continuity']['bin_edges']
            self.sfh = self.Leja19_SFH(plot=False)
            # SFH samples consume a lot of memory, delete them
            del self.fit.posterior.samples["sfh"]
            del self.fit.posterior.samples2d

    def _get_sample_percentiles(self, param, perc=(16, 50, 84)):
        '''
        Get the percentiles of the posterior samples for a given parameter.
        '''
        if param in self.fit.posterior.samples:
            return np.percentile(self.fit.posterior.samples[param], perc)
        else:
            raise ValueError(f"Parameter {param} not found in posterior samples.")
        
    def _get_redshift(self):
        """
        Get the redshift of the galaxy.
        """
        if "redshift" in self.fit.fitted_model.params:
            z = self._get_sample_percentiles("redshift")[1]
        else:
            z = self.fit.fitted_model.model_components["redshift"]
        self.z = z
        return z
    
    def _correct_distance(self, distance):
        """
        Correct the distance of the galaxy.
        """
        _z = self._get_redshift()
        # correct the distance
        dist_cosmo = pipes.utils.cosmo.luminosity_distance(_z).value
        self.fit.posterior.samples['stellar_mass'] += 2 * np.log10(distance / dist_cosmo)
        self.fit.posterior.samples['sfr'] *= (distance / dist_cosmo)**2
        self.fit.posterior.samples['sfh'] *= (distance / dist_cosmo)**2

    def corner_plot(self):
        '''
        Plot the corner plot of the parameters.
        '''
        self.fit.plot_corner(save=False, show=True)

    def plot_results(self, plot_range='full', save_path=None, 
                     xmin=None, xmax=None, ymin=None, ymax=None):
        '''
        Plot the results of the SED fitting.
        '''
        # load the posterior samples
        phot = self.fit.galaxy.photometry
        if 'ifs' in self.run:
            spec = self.fit.galaxy.spectrum
        _z = self._get_redshift()

        try:
            wav = self.fit.posterior.model_galaxy.wavelengths * (1. + _z)
        except:
            self.fit.posterior.get_advanced_quantities()
            wav = self.fit.posterior.model_galaxy.wavelengths * (1. + _z)

        spec_post = (np.percentile(self.fit.posterior.samples["spectrum_full"] * wav,
                                  (16, 50, 84), axis=0).T).astype(float)

        phot_post = np.percentile(self.fit.posterior.samples["photometry"] * phot[:, 0],
                                  (16, 50, 84), axis=0).T

        plt.figure(figsize=(6, 4))
        gs = plt.GridSpec(3,1)
        ax1 = plt.subplot(gs[:2])
        ax2 = plt.subplot(gs[2:], sharex=ax1)

        # plot the spectrum and photometry
        ax1.errorbar(phot[:, 0], phot[:, 1] * phot[:, 0],
                     yerr=phot[:, 2] * phot[:, 0],
                     fmt='s', ms=6, lw=1, c='k', mfc='none', label='obs:phot', zorder=10)
        if 'ifs' in self.run:
            ax1.plot(spec[:, 0], 
                     spec[:,1]*spec[:,0]*np.median(self.fit.posterior.samples['calib:0']),
                     lw=1, c='k', label='obs:spec')
        ax1.plot(wav, spec_post[:, 1], 
                 lw=1, c='orange', alpha=0.6, label='model:spec')
        ax1.fill_between(wav, spec_post[:, 0], spec_post[:, 2], 
                         lw=0, fc='orange', alpha=.3)
        ax1.scatter(phot[:, 0], phot_post[:, 1],
                    s=20, c='r', label='model:phot', zorder=5)
        
        ax2.set_xlabel(r'$\lambda~[\AA]$')
        ax1.set_ylabel(r'$\lambda f_\lambda~\mathrm{[erg/s/cm^2]}$')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend()

        # plot residuals
        self.chi2 = self.fit.posterior.samples['chisq_phot'].min() / len(phot)
        ax2.scatter(phot[:, 0], 
                    (phot[:, 1] - phot_post[:, 1] / phot[:, 0]) / phot[:, 2],
                    s=20, c='gray')
        if 'ifs' in self.run:
            ax2.plot(spec[:, 0], 
                     (spec[:, 0] * spec[:, 1] - self.fit.posterior.model_galaxy.spectrum[:, 1]) / spec[:, 2],
                     c='k', lw=0.5, alpha=0.5)
        ax2.axhline(0, lw=1, c='k', ls='--')
        ax2.set_ylim(-8,8)
        ax2.set_ylabel(r"$\chi$")
        ax2.text(0.02, 0.8, rf'$\chi_\nu^2 = {self.chi2:.2f}$', transform=ax2.transAxes)

        if plot_range == 'full':
            xmin, xmax = 1e3, 6e6
        elif plot_range == 'optical':
            xmin, xmax = 3e3, 1e4
        elif plot_range == 'Ha':
            xmin, xmax = 6.5e3, 6.9e3
        elif plot_range == 'Hb':    
            xmin, xmax = 4.8e3, 5.2e3
        elif plot_range == 'user':
            xmin, xmax = xmin, xmax
        if ymin is None or ymax is None:
            ymin = np.min(spec_post[:, 1][(wav > xmin) & (wav < xmax)])
            ymax = np.max(spec_post[:, 1][(wav > xmin) & (wav < xmax)])
        ax1.set_ylim(ymin * 0.8, ymax * 1.2)
        ax1.set_xlim(xmin, xmax)

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()

        
    def Leja19_SFH(self, plot=False,
                   minssfr=-12.5, maxssfr=-8, mint=6.5, maxt=10.2, save_path=None):
        """
        Get the average SFR over the last t Myr.
        """
        post = np.percentile(self.fit.posterior.samples["sfh"], 
                             (16, 50, 84), axis=0).T
        ages = self.fit.posterior.sfh.ages / 1e6  # convert to Myr
        bins = self.fit_info['continuity']['bin_edges']

        SFR_arr = np.full((len(bins)-1, 3), np.nan)
        for i in range( len(bins)-1 ):
            idx = (ages >= bins[i]) & (ages < bins[i+1])
            SFR_arr[i, 0] = np.median(post[idx, 1])
            SFR_arr[i, 1] = np.median(post[idx, 0])
            SFR_arr[i, 2] = np.median(post[idx, 2])  

        if plot:
            ages = np.log10(ages) + 6
            post = np.log10(post)
            mass = self._get_sample_percentiles('stellar_mass')[1]
            
            plt.plot(ages, post[:, 1], c='k', lw=1)
            plt.fill_between(ages, post[:, 0], post[:, 2], alpha=.1, fc='r', lw=0)
            plt.ylim(mass + minssfr, mass + maxssfr)
            plt.xlim(mint, maxt)
            plt.ylabel('log SFR [Msol / yr]')
            plt.xlabel('log lookback time [Gyr]')

            if save_path is not None:
                plt.savefig(save_path, bbox_inches='tight')

        return SFR_arr, bins     
         

    def line_flux(self):
        '''
        Get the line fluxes from the max-likelihood model.
        '''
        max_lh = np.argmax(self.fit.results["lnlike"])
        self.fit.fitted_model._update_model_components(self.fit.results["samples2d"][max_lh, :])
        max_lh_model = pipes.model_galaxy(self.fit.fitted_model.model_components,
                                          filt_list=self.fit.galaxy.filt_list,
                                          spec_wavs=np.arange(3500, 9000, 1))
        line_fluxes = max_lh_model.line_fluxes
        f_ha = line_fluxes["H  1  6562.81A"]
        f_hb = line_fluxes["H  1  4861.33A"]

        return f_ha, f_hb, line_fluxes
    
    
    def model_no_nebem(self):
        """
        Get the stellar continuum from the max-likelihood model.
        """
        import copy
        model = copy.deepcopy(self.fit.fitted_model.model_components)
        del model['nebular']
        model_no_neb = pipes.model_galaxy(model,
                                          filt_list=self.fit.galaxy.filt_list,
                                          spec_wavs=np.arange(3500, 9000, 1))
        return model_no_neb
    
    def chi2(self):
        """
        Get the reduced chi2 of the fit.
        TODO: THIS IS SLOW
        """
        from copy import deepcopy
        temp_sed = deepcopy(self)
        post = temp_sed.fit.posterior
        ndof = temp_sed.fit.galaxy.photometry.shape[0]

        if 'chisq_phot' in post.samples.keys():
            chi2_rd = post.samples['chisq_phot'].min() / ndof
        else:
            temp_sed.fit.posterior.get_advanced_quantities()
            chi2_rd = temp_sed.fit.posterior.samples['chisq_phot'].min() / ndof

        del temp_sed, post
        return chi2_rd


    def dust_mass(self, n_boot=1):
        """
        Estimate dust mass from the samples.
        """
        from copy import deepcopy
        from bagpipes.models.model_galaxy import model_galaxy
        
        sed = deepcopy(self)
        post = sed.fit.posterior
        _z = sed._get_redshift()

        dust_masses = []
        for i in range(n_boot):
            post.fitted_model._update_model_components(post.samples2d[i, :])
            
            model = model_galaxy(post.fitted_model.model_components,
                                 filt_list=post.galaxy.filt_list,
                                 spec_wavs=post.galaxy.spec_wavs,
                                 index_list=post.galaxy.index_list)
            
            wav = model.wavelengths * (1. + _z)
            fir_idx = (wav >= 3e5) & (wav <= 1e7)
            wav = wav[fir_idx]

            dust_model = pipes.models.dust_emission_model.dust_emission(wav)
            dust_spec = dust_model.spectrum(qpah=sed.fit.posterior.samples['dust:qpah'][i],
                                            umin=sed.fit.posterior.samples['dust:umin'][i],
                                            gamma=sed.fit.posterior.samples['dust:gamma'][i])

            # unit conversion (m_sun/ m_H)
            dust_spec *= wav**2 * 1e3 / ((4 * np.pi) * (sed.distance * 3.086e24)**2) * (1.989e33 / 1.6736e-24)
            obs_spec = model.spectrum_full[fir_idx] * wav**2 * 3.3356e4  # Jy
            dust_masses.append(np.median(obs_spec / dust_spec))

        del sed
        return np.log10(np.percentile(dust_masses, (16, 50, 84)))


    def predict_flux(self, filter_name):
        """
        Predict the flux for a given filter using the SED model.
        """
        # load filter transmission curve
        filter_wave, filter_trans = get_filter_trans_curve(filter_name)

        # posterior of the full spectrum
        spec_post = np.percentile(self.fit.posterior.samples["spectrum_full"],
                                  (16, 50, 84), axis=0).T
        spec_wave = self.fit.posterior.model_galaxy.wavelengths * (1 + self._get_redshift())

        # predict the photometry
        phot_pred = np.full((3,), np.nan)
        for i in range(3):
            spec_interp = np.interp(filter_wave, spec_wave, spec_post[:, i])
            phot_pred[i] = np.trapezoid(spec_interp * filter_wave * filter_trans,
                                         filter_wave)
            phot_pred[i] /= np.trapezoid(filter_wave * filter_trans, filter_wave)

        return phot_pred
