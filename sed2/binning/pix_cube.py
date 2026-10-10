import sys  # noqa: EXE002
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import binary_dilation, binary_fill_holes

from ..utils import *

__all__ = ["flux_map", "galaxy_region", "plot_flux_maps", "plot_sample_sed"]


def galaxy_region(workdir, galaxy, roi_bands=None,
                  thresh=3, npixels=50, dilate_iter=3, 
                  plot_region=True, plot_band='galex_fuv',
                  **kwargs):
    '''
    Select galaxy regions from pixel binning.
    ----------
    Parameters
    ----------
    workdir : str
        The path to the working directory.
    galaxy : str
        The name of the galaxy.
    roi_bands : int, list, str, tuple, optional
        The bands to use for selecting the region of interest.
    thresh : float
        The threshold for source detection.
    npixels : int
        The minimum number of pixels for a source.
    dilate_iter : int
        The number of iterations for binary dilation.
    plot_region : bool
        Whether to plot the detected galaxy regions.
    plot_band : str
        The band to plot.
    **kwargs : keyword arguments
        Additional arguments passed to the deblend_sources function.
    '''
    from photutils.segmentation import deblend_sources, detect_sources

    path = Path(workdir)
    if check_output(path / "flux" / "roi.npy"):
        return 

    filters = load_filters_from_txt(workdir / "aux" / "filter_list.txt")
    filter_ids = load_filter_ids(workdir, roi_bands)

    segments_all = []

    for i in filter_ids:

        img = fits.open(path / "matched" / f"{galaxy}_{filters[i]}.fits")[0].data
        var = fits.open(path / "matched" / f"var_{galaxy}_{filters[i]}.fits")[0].data

        # Detect sources in the matched image
        var[~(var > 0)] = +np.inf
        sources = detect_sources(img, threshold=thresh * np.sqrt(var), 
                                 npixels=npixels)
        # Deblend sources
        deblended = deblend_sources(img, sources, 
                                    npixels=npixels, progress_bar=False,
                                    **kwargs)
        # keep the central segment
        segm = deblended.data
        segm[segm != segm[segm.shape[0]//2, segm.shape[1]//2]] = 0

        segments_all.append(segm)

    # Create a combined galaxy region mask
    gal_region = np.zeros_like(segments_all[0])
    for segm in segments_all:
        gal_region[segm > 0] = 1

    # post-processing
    gal_region = binary_dilation(gal_region, 
                                 iterations=dilate_iter, 
                                 structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    gal_region = binary_fill_holes(gal_region).astype(np.int8)

    # plot the results
    if plot_region:
        import matplotlib.pyplot as plt
        img = fits.open(path / "matched" / f"{galaxy}_{plot_band}.fits")[0].data
        plt.figure(figsize=(4, 4))
        plt.imshow(img, origin='lower', cmap='turbo',
                   vmin=np.nanpercentile(img, 16),
                   vmax=np.nanpercentile(img, 99))
        plt.imshow(gal_region, origin='lower', cmap='gray', alpha=0.3)
        plt.title("Region-of-interest (RoI)")
        plt.savefig(path / "plot" / "roi.png", dpi=300, bbox_inches='tight')

    np.save(path / "flux" / "roi.npy", gal_region)
    return gal_region



def flux_map(workdir, galaxy, filters, 
             Ebv=None, coord=None, mag_zp_2mass=None, 
             unit_spire='Jy_per_beam', scale_unit=1e-17, img_unit=None, dp_unit=False,
             ref_band='wise_w4', name_out_fits=None):
    '''
    Calculating the maps of multiband fluxes from the matched images.
    This function is adopted from piXedfit (Abdurro'uf et al. 2021)

    [https://github.com/aabdurrouf/piXedfit]

    Parameters:
    -----------
    workdir : str
        The path to the working directory.
    galaxy : str
        The name of the galaxy.
    filters : list
        The list of filters to use.
    Ebv : float, optional
        The E(B-V) value for dust extinction correction.
    coord : astropy.coordinates.SkyCoord, optional
        The coordinates of the galaxy.
    mag_zp_2mass : list, optional
        The magnitude zero-points for 2MASS.
    unit_spire : str, optional
        The unit for SPIRE images.
    scale_unit : float, optional
        The scale factor for the fluxes.
    img_unit : dict, optional
        The units for the image data.
    dp_unit : bool, optional
        Whether to use DustPedia units (Jy/pixel).
    ref_band : str, optional
        The reference band.
    name_out_fits : str, optional
        The output FITS file name.
    '''

    # calculate Milky Way extinction
    if img_unit is None:
        img_unit = {}
    if Ebv is None:
        if coord is None:
            raise ValueError("Either Ebv or coord must be provided.")
        Ebv = get_ebv(coord=coord)

    gal_region = np.load(workdir / "flux" / "roi.npy")

    wave_piv = np.array([get_filter_waves(f)[1] for f in filters])
    A_lambda = np.array([R_extinction(wave) for wave in wave_piv]) * Ebv
    A_factor = 10**(0.4 * A_lambda)

    # load reference band
    ref_img = fits.open(workdir / "matched" / f"{galaxy}_{ref_band}.fits")
    ref_var = fits.open(workdir / "matched" / f"var_{galaxy}_{ref_band}.fits")
    pix_scale = get_pixel_size(workdir / "matched" / f"{galaxy}_{ref_band}.fits")

    # setup
    n_band = len(filters)
    map_flux     = np.full((n_band, *ref_img[0].data.shape), np.nan)
    map_flux_err = np.full((n_band, *ref_var[0].data.shape), np.nan)

    # per-filter conversion constants
    FLUXZP_2MASS = {"2mass_j": 3.129e-13,
                    "2mass_h": 1.133e-13,
                    "2mass_k": 4.283e-14}
    DN_TO_JY = {"wise_w1": 1.9350e-06, "wise_w2": 2.7048e-06,
                "wise_w3": 2.9045e-06, "wise_w4": 5.2269e-05}
    BEAM_AREA = {"herschel_spire_250": 469.3542,
                 "herschel_spire_350": 831.275,
                 "herschel_spire_500": 1804.3058}
    SDSS = {"sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"}
    SPITZER = {"spitzer_irac_36", "spitzer_irac_45", "spitzer_irac_58",
               "spitzer_irac_80", "spitzer_mips_24", "spitzer_mips_70",
               "spitzer_mips_160"}
    PACS = {"herschel_pacs_70", "herschel_pacs_100", "herschel_pacs_160"}
    SPIRE = {"herschel_spire_250", "herschel_spire_350", "herschel_spire_500"}

    def _conv_factor(filt, wav, mag_zp=None):
        '''Factor converting the native image unit to erg/s/cm^2/Ang.'''
        jy = 1.0e-23 * 2.998e+18 / wav**2          # Jy -> erg/s/cm^2/Ang
        if dp_unit:                                # DustPedia images: Jy/pixel
            return jy
        if filt == 'galex_fuv':
            return 1.40e-15
        if filt == 'galex_nuv':
            return 2.06e-16
        if filt in SDSS:
            return 3.631e-6 * jy
        if filt in FLUXZP_2MASS:
            return FLUXZP_2MASS[filt] * 10**(-0.4 * mag_zp) * 1.0e3
        if filt in SPITZER:
            return 2.35044e-5 * pix_scale**2 * jy
        if filt in ('wise_w1', 'wise_w2', 'wise_w3'):
            return DN_TO_JY[filt] * jy
        if filt == 'wise_w4' or filt in PACS:
            return jy
        if filt in SPIRE:
            if unit_spire == 'Jy_per_beam':
                return pix_scale**2 / BEAM_AREA[filt] * jy
            if unit_spire == 'MJy_per_sr':
                return 2.35044e-5 * pix_scale**2 * jy
            if unit_spire == 'Jy_per_pixel':
                return jy
            print("unit of Herschel images is not recognized!")
            sys.exit()
        if filt[:4] in ('ctio', 'kpno', 'bok_', 'vatt'):
            return jy
        # otherwise the unit must be provided by the user
        unit = img_unit[filt]
        if unit == 'erg/s/cm2/A':
            return 1.0
        if unit == 'Jy':
            return jy
        if unit == 'MJy/sr':
            return 2.35044e-5 * pix_scale**2 * jy
        print(f"unit of filter {filt} is not recognized!")
        sys.exit()

    # pixel coordinates of the galaxy region (constant across bands)
    r, c = np.where(gal_region == 1)

    # iterate over bands
    for i in range(n_band):

        # load science and variance images, apply extinction correction
        with fits.open(workdir / "matched" / f"{galaxy}_{filters[i]}.fits") as sci_hdu, \
                fits.open(workdir / "matched" / f"var_{galaxy}_{filters[i]}.fits") as var_hdu:
            sci_img_data = sci_hdu[0].data.copy() * A_factor[i]
            var_img_data = var_hdu[0].data.copy() * A_factor[i]**2

        # magnitude zero-point of 2MASS images
        mag_zp = None
        if filters[i] in FLUXZP_2MASS:
            if mag_zp_2mass is None:
                with fits.open(workdir / "matched" / f"{galaxy}_{filters[i]}.fits") as hdu:
                    mag_zp = float(hdu[0].header["MAGZP"])
            else:
                mag_zp = mag_zp_2mass[{"2mass_j": 0,
                                       "2mass_h": 1,
                                       "2mass_k": 2}[filters[i]]]

        factor = _conv_factor(filters[i], wave_piv[i], mag_zp)

        map_flux[i][r, c] = sci_img_data[r, c] * factor
        map_flux_err[i][r, c] = np.sqrt(np.abs(var_img_data[r, c])) * factor

    # --- end of flux conversion ---

    # scaling the flux maps:
    map_flux /= scale_unit
    map_flux_err /= scale_unit

    # write to file (the SED cube)
    hdul = fits.HDUList()
    hdr = fits.Header()
    hdr['nfilters'] = n_band

    hdr['unit'] = scale_unit
    hdr['bunit'] = 'erg/s/cm^2/A'
    hdr['GalEBV'] = Ebv
    hdr['struct'] = '(band,y,x)'
    hdr['pixsize'] = pix_scale
    hdr['psfband'] = ref_band

    for i in range(n_band):
        hdr[f"fil{i}"] = filters[i]

    hdul.append(fits.ImageHDU(data=map_flux, header=hdr, name='flux'))
    hdul.append(fits.ImageHDU(map_flux_err, name='flux_err'))
    hdul.append(fits.ImageHDU(gal_region, name='galaxy_region'))
        
    if name_out_fits is None:
        name_out_fits = 'fluxmap.fits'
    hdul.writeto(workdir / "flux" / name_out_fits, overwrite=True)

    return name_out_fits


def plot_flux_maps(cube_path, out_path=None):
    """
    Plot the flux maps from the given SED data cube.
    """
    import matplotlib.pyplot as plt

    with fits.open(cube_path) as hdu:
        header = hdu[0].header
        unit_flux = float(header['unit'])
        flux_map     = hdu['FLUX'].data * unit_flux
        flux_err_map = hdu['FLUX_ERR'].data * unit_flux

    n_band = flux_map.shape[0]

    # flux map
    _fig, axes = plt.subplots(n_band//6 + 1, 6, figsize=(12, n_band / 3))
    for i in range(n_band):
        ax = axes[i // 6, i % 6]
        ax.imshow(np.log10(flux_map[i]), origin='lower', cmap='nipy_spectral')
        ax.set_title(header[f"FIL{i}"])

    plt.tight_layout()
    if out_path is not None:
        plt.savefig(Path(out_path) / "flux_map.png")
    plt.show()

    # snr map
    _fig, axes = plt.subplots(n_band//6 + 1, 6, figsize=(12, n_band / 3))
    for i in range(n_band):
        ax = axes[i // 6, i % 6]
        ax.imshow(np.abs(flux_map[i] / flux_err_map[i]), 
                  origin='lower', cmap='nipy_spectral', vmin=0, vmax=50)
        ax.contour(np.abs(flux_map[i]) / flux_err_map[i], 
                   levels=[5], colors='gold', linewidths=1)
        ax.set_title(header[f"FIL{i}"])

    plt.tight_layout()
    if out_path is not None:
        plt.savefig(Path(out_path) / "snr_map.png")
    plt.show()



def plot_sample_sed(cube_path, x, y):
    """
    Plot the SED for a single pixel from the given SED data cube.
    """
    import matplotlib.pyplot as plt

    with fits.open(cube_path) as hdu:
        header = hdu[0].header
        unit_flux = float(header['unit'])
        flux_map = hdu['FLUX'].data * unit_flux
        flux_err_map = hdu['FLUX_ERR'].data * unit_flux

    n_band = flux_map.shape[0]
    # wavelengths
    wave_piv = np.array([get_filter_waves(header[f'FIL{i}'])[0] for i in range(n_band)])

    # Plot SED for a specific pixel (e.g., pixel at (x, y))
    _fig, ax = plt.subplots(figsize=(7, 3))
    ax.errorbar(wave_piv / 1.0e+4, 
                flux_map[:, y, x] * wave_piv, 
                yerr=flux_err_map[:, y, x] * wave_piv,
                fmt='-o', markersize=2, lw=0.5)
    ax.set_xlabel(r"$\lambda~(\mathrm{\mu m})$")
    ax.set_ylabel(r"$\lambda f_\lambda~\mathrm{(erg/s/cm^2)}$")
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.show()
