import sys
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from scipy.ndimage import binary_fill_holes, binary_dilation

from ..utils import *
from ..path import PATH

__all__ = ["galaxy_region", "flux_map", "plot_flux_maps", "plot_sample_sed"]


def galaxy_region(matched_path, matched_img, matched_var, filter_ids=None,
                  thresh=3, npixels=50, dilate_iter=3, 
                  plot_region=True, plot_idx=0,
                  out_file=None, **kwargs):
    '''
    Select galaxy regions from pixel binning.
    ----------
    Parameters
    ----------
    matched_path : str
        The path to the matched data.
    matched_img : dict
        The paths for matched image data.
    matched_var : dict
        The paths for matched variance data.
    thresh : float
        The threshold for source detection.
    npixels : int
        The minimum number of pixels for a source.
    dilate_iter : int
        The number of iterations for binary dilation.
    plot_region : bool
        Whether to plot the detected galaxy regions.
    plot_idx : int
        The index of band to plot.
    out_file : str
        The path to the output file for the galaxy region mask.
    **kwargs : keyword arguments
        Additional arguments passed to the deblend_sources function.
    '''
    from photutils.segmentation import detect_sources, deblend_sources

    matched_img = list(matched_img.values())
    matched_var = list(matched_var.values())

    if filter_ids is None:
        filter_ids = np.arange(len(matched_img))

    segments_all = []
    for i in filter_ids:

        try: 
            img = fits.open(matched_path + matched_img[i])[0].data
            var = fits.open(matched_path + matched_var[i])[0].data
        except FileNotFoundError:
            # low resolution filters are not available
            continue

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
    gal_region = binary_dilation(gal_region, iterations=dilate_iter, 
                                 structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    gal_region = binary_fill_holes(gal_region).astype(np.int8)

    # plot the results
    if plot_region:
        import matplotlib.pyplot as plt
        img = fits.open(matched_path + matched_img[plot_idx])[0].data
        plt.figure(figsize=(4, 4))
        plt.subplot(111)
        plt.imshow(img, origin='lower', cmap='turbo',
                   vmin=np.nanpercentile(img, 16),
                   vmax=np.nanpercentile(img, 99))
        plt.imshow(gal_region, origin='lower', cmap='gray', alpha=0.3)
        plt.title("Region")
        plt.show()

    if out_file is not None:
        np.save(out_file, gal_region)

    return gal_region



def flux_map(img_path, sci_img, var_img, filters, gal_region, 
             Ebv=None, mag_zp_2mass=None, 
             unit_spire='Jy_per_beam', scale_unit=1e-17, img_unit={}, dp_unit=False,
             ref_band='wise_w4', gname='', name_out_fits=None):
    '''
    Calculating the maps of multiband fluxes from the matched images.
    This function is adopted from piXedfit (Abdurro'uf et al. 2021)

    [https://github.com/aabdurrouf/piXedfit]

    Parameters:
    -----------
    img_path : str
        The path to the image files.
    sci_img : dict
        The paths for science image data.
    var_img : dict
        The paths for variance image data.
    filters : list
        The list of filters to use.
    gal_region : np.ndarray
        The galaxy region mask.
    Ebv : float, optional
        The E(B-V) value for dust extinction correction.
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
    gname : str, optional
        The name of the galaxy.
    ref_band : str, optional
        The reference band.
    name_out_fits : str, optional
        The output FITS file name.
    '''

    # calculate Milky Way extinction
    if Ebv is None:
        coord = SkyCoord.from_name(gname)
        Ebv = get_ebv(coord=coord)

    wave_piv = np.array([get_filter_waves(f)[1] for f in filters])
    A_lambda = np.array([R_extinction(wave) for wave in wave_piv]) * Ebv
    A_factor = 10**(0.4 * A_lambda)

    # load reference band
    ref_img = fits.open(img_path + sci_img[ref_band])
    ref_var = fits.open(img_path + var_img[ref_band])
    pix_scale = get_pixel_size(img_path + sci_img[ref_band])
    

    # setup
    n_band = len(filters)
    map_flux     = np.full((n_band, *ref_img[0].data.shape), np.nan)
    map_flux_err = np.full((n_band, *ref_var[0].data.shape), np.nan)

    # iterate over bands
    for i in range(n_band):

        # load science and variance images
        sci_hdu = fits.open(img_path + sci_img[filters[i]])
        var_hdu = fits.open(img_path + var_img[filters[i]])
        sci_img_data = sci_hdu[0].data
        var_img_data = var_hdu[0].data
        sci_hdu.close()
        var_hdu.close()

        # apply extinction correction
        sci_img_data *= A_factor[i]
        var_img_data *= A_factor[i]**2

        # get pixel coordinates of the galaxy region
        r, c = np.where(gal_region == 1)

        # DustPedia images are in Jy / pixel
        if dp_unit:
            map_flux[i][r,c] = sci_img_data[r,c] * 1e-23 * 2.998e+18 / wave_piv[i]**2
            map_flux_err[i][r,c] = np.sqrt(np.abs(var_img_data[r,c])) * 1e-23 * 2.998e+18 / wave_piv[i]**2
            continue

        # get magnitude zero-point and flux zero-point of 2MASS:
        if '2mass' in filters[i]:
            # get magnitude zero-point, -8.9 for Jy/pixel
            if mag_zp_2mass is None:
                hdu = fits.open(img_path + sci_img[filters[i]])
                MAGZP_2mass = float(hdu[0].header["MAGZP"])
                hdu.close()
            else:
                MAGZP_2mass = mag_zp_2mass[{"2mass_j": 0, 
                                            "2mass_h": 1, 
                                            "2mass_k": 2}[filters[i]]]

            # get flux at magnitude zero-point (Cohen+2003; in W/cm^2/micron)
            FLUXZP_2mass = {"2mass_j": 3.129e-13,
                            "2mass_h": 1.133E-13,
                            "2mass_k": 4.283E-14}[filters[i]]

        # get DN to Jy correction factors for WISE bands
        if 'wise' in filters[i]:
            DN_to_Jy = {"wise_w1": 1.9350e-06,
                        "wise_w2": 2.7048E-06,
                        "wise_w3": 2.9045e-06,
                        "wise_w4": 5.2269E-05}[filters[i]]

        # get beam area of Herschel SPIRE (in arcsec^2)
        if 'herschel_spire' in filters[i]:
            beam_area = {"herschel_spire_250": 469.3542,
                         "herschel_spire_350": 831.275,
                         "herschel_spire_500": 1804.3058}[filters[i]]

        # GALEX/FUV
        if filters[i] == 'galex_fuv':
            map_flux[i][r,c] = sci_img_data[r,c] * 1.40e-15   # in erg/s/cm^2/Ang.
            map_flux_err[i][r,c] = np.sqrt(np.abs(var_img_data[r,c])) * 1.40e-15   # in erg/s/cm^2/Ang.

        # GALEX/NUV
        elif filters[i] == 'galex_nuv':
            map_flux[i][r,c] = sci_img_data[r,c] * 2.06e-16   # in erg/s/cm^2/Ang.
            map_flux_err[i][r,c] = np.sqrt(np.abs(var_img_data[r,c])) * 2.06e-16   # in erg/s/cm^2/Ang.

        # SDSS
        elif filters[i] in ['sdss_u', 'sdss_g', 'sdss_r', 'sdss_i', 'sdss_z']:
            f0 = sci_img_data[r,c] * 3.631e-6          # in Jy
            map_flux[i][r,c] = f0 * 2.998e-5 / wave_piv[i]**2     # in erg/s/cm^2/Ang.

            f0 = np.sqrt(np.abs(var_img_data[r,c])) * 3.631e-6     # in Jy
            map_flux_err[i][r,c] = f0 * 2.998e-5 / wave_piv[i]**2  # in erg/s/cm^2/Ang.

        # 2MASS (the image is in DN)
        elif filters[i] in ['2mass_j', '2mass_h', '2mass_k']:
            # flux
            map_flux[i][r,c] = FLUXZP_2mass * sci_img_data[r,c] * 10**(-0.4 * MAGZP_2mass) * 1.0e+3  # in erg/s/cm^2/Ang.

            # flux error
            map_flux_err[i][r,c] = FLUXZP_2mass * np.sqrt(np.abs(var_img_data[r,c])) * 10**(-0.4 * MAGZP_2mass) * 1.0e+3  # in erg/s/cm^2/Ang.

        # Spitzer: IRAC and MIPS (Spitzer image is in Mjy/sr)
        elif filters[i] in ['spitzer_irac_36', 'spitzer_irac_45', 'spitzer_irac_58', 'spitzer_irac_80', 
                            'spitzer_mips_24', 'spitzer_mips_70', 'spitzer_mips_160']:
            f0 = sci_img_data[r,c] * 2.35044e-5 * pix_scale**2               # in unit of Jy
            map_flux[i][r,c] = f0 * 1.0e-23 * 2.998e+18 / wave_piv[i]**2     # in erg/s/cm^2/Ang.

            f0 = np.sqrt(np.abs(var_img_data[r,c])) * 2.35044e-5 * pix_scale**2   # in unit of Jy
            map_flux_err[i][r,c] = f0 * 1.0e-23 * 2.998e+18 / wave_piv[i]**2      # in erg/s/cm^2/Ang.

        # WISE (image is in DN)
        elif filters[i] in ['wise_w1', 'wise_w2', 'wise_w3']:
            map_flux[i][r,c] = sci_img_data[r,c] * DN_to_Jy * 1.0e-23 * 2.998e+18 / wave_piv[i]**2                      # in erg/s/cm^2/Ang.
            map_flux_err[i][r,c] = np.sqrt(np.abs(var_img_data[r,c])) * DN_to_Jy * 1.0e-23 * 2.998e+18 / wave_piv[i]**2   # in erg/s/cm^2/Ang.

        # !! -- w4 from z0mgs is in Jy -- !!
        elif filters[i] == 'wise_w4':
            map_flux[i][r,c] = sci_img_data[r,c] * 1.0e-23 * 2.998e+18 / wave_piv[i]**2                        # in erg/s/cm^2/Ang.
            map_flux_err[i][r,c] = np.sqrt(np.abs(var_img_data[r,c])) * 1.0e-23 * 2.998e+18 / wave_piv[i]**2   # in erg/s/cm^2/Ang.
            
        # Herschel PACS
        # image is in Jy/pixel or Jy --> this is not surface brightness unit but flux density
        elif filters[i] in ['herschel_pacs_70', 'herschel_pacs_100', 'herschel_pacs_160']:
            map_flux[i][r,c] = sci_img_data[r,c] * 1.0e-23 * 2.998e+18 / wave_piv[i]**2                          # in erg/s/cm^2/Ang.
            map_flux_err[i][r,c] = np.sqrt(np.abs(var_img_data[r,c])) * 1.0e-23 * 2.998e+18 / wave_piv[i]**2     # in erg/s/cm^2/Ang.

        # Herschel SPIRE
        elif filters[i] in ['herschel_spire_250', 'herschel_spire_350', 'herschel_spire_500']:
            if unit_spire == 'Jy_per_beam':
                # image is in Jy/beam -> surface brightness unit
                # Jy/pixel = Jy/beam x beam/arcsec^2 x arcsec^2/pixel  -> flux density
                f0 = sci_img_data[r,c] * pix_scale**2 / beam_area                  # Jy
                map_flux[i][r,c] = f0 * 1.0e-23 * 2.998e+18 / wave_piv[i]**2       # erg/s/cm^2/Ang.

                f0 = np.sqrt(np.abs(var_img_data[r,c])) * pix_scale**2 / beam_area  # Jy
                map_flux_err[i][r,c] = f0 * 1.0e-23 * 2.998e+18 / wave_piv[i]**2    # erg/s/cm^2/Ang.

            # in case the data is in Mjy/sr
            elif unit_spire == 'MJy_per_sr':
                f0 = sci_img_data[r,c] * 2.35044e-5 * pix_scale**2            # Jy
                map_flux[i][r,c] = f0 * 1.0e-23 * 2.998e+18 / wave_piv[i]**2  # erg/s/cm^2/Ang.

                f0 = np.sqrt(np.abs(var_img_data[r,c])) * 2.35044e-5 * pix_scale**2   # Jy
                map_flux_err[i][r,c] = f0 * 1.0e-23 * 2.998e+18 / wave_piv[i]**2      # erg/s/cm^2/Ang.

            # in case the data is in Jy/pixel or Jy
            elif unit_spire == 'Jy_per_pixel':
                map_flux[i][r,c] = sci_img_data[r,c] * 1.0e-23 * 2.998e+18 / wave_piv[i]**2        # in erg/s/cm^2/Ang.
                map_flux_err[i][r,c] = np.sqrt(np.abs(var_img_data[r,c])) * 1.0e-23 * 2.998e+18 / wave_piv[i]**2    # in erg/s/cm^2/Ang.

            else:
                print ("unit of Herschel images is not recognized!")
                sys.exit()
            
        # H-alpha images should be in Jy
        elif filters[i][:4] in ['ctio', 'kpno', 'bok_', 'vatt']:
            map_flux[i][r,c] = sci_img_data[r,c] * 1.0e-23 * 2.998e+18 / wave_piv[i]**2        # in erg/s/cm^2/Ang.
            map_flux_err[i][r,c] = np.sqrt(np.abs(var_img_data[r,c])) * 1.0e-23 * 2.998e+18 / wave_piv[i]**2    # in erg/s/cm^2/Ang.

        # units of other images should be provided in img_unit
        else:
            if img_unit[filters[i]]=='erg/s/cm2/A':
                map_flux[i][r,c] = sci_img_data[r,c]
                map_flux_err[i][r,c] = np.sqrt(np.abs(var_img_data[r,c]))

            elif img_unit[filters[i]]=='Jy':
                map_flux[i][r,c] = sci_img_data[r,c] * 1.0e-23 * 2.998e+18 / wave_piv[i]**2
                map_flux_err[i][r,c] = np.sqrt(np.abs(var_img_data[r,c]))*1.0e-23*2.998e+18/ wave_piv[i]**2

            elif img_unit[filters[i]]=='MJy/sr':
                f0 = sci_img_data[r,c]*2.35044e-5 * pix_scale**2             # in unit of Jy
                map_flux[i][r,c] = f0 * 1.0e-23 * 2.998e+18 / wave_piv[i]**2 # in erg/s/cm^2/Ang.

                f0 = np.sqrt(np.abs(var_img_data[r,c])) * 2.35044e-5 * pix_scale**2   # in unit of Jy
                map_flux_err[i][r,c] = f0 * 1.0e-23 * 2.998e+18 / wave_piv[i]**2      # in erg/s/cm^2/Ang.
            else:
                print (f"unit of filter {filters[i]} is not recognized!")
                sys.exit()

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
    hdul.writeto(img_path + name_out_fits, overwrite=True)

    return name_out_fits


def plot_flux_maps(cube_path, out_path=None):
    """
    Plot the flux maps from the given SED data cube.
    """
    import matplotlib.pyplot as plt

    with fits.open(cube_path) as hdu:
        header = hdu[0].header
        unit_flux = float(header['unit'])
        gal_region   = hdu['GALAXY_REGION'].data
        flux_map     = hdu['FLUX'].data * unit_flux
        flux_err_map = hdu['FLUX_ERR'].data * unit_flux

    n_band = flux_map.shape[0]

    # flux map
    fig, axes = plt.subplots(n_band//6 + 1, 6, figsize=(12, n_band / 3))
    for i in range(n_band):
        ax = axes[i // 6, i % 6]
        ax.imshow(np.log10(flux_map[i]), origin='lower', cmap='nipy_spectral')
        ax.set_title(header[f"FIL{i}"])

    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path + f"flux_map.png")
    plt.show()

    # snr map
    fig, axes = plt.subplots(n_band//6 + 1, 6, figsize=(12, n_band / 3))
    for i in range(n_band):
        ax = axes[i // 6, i % 6]
        ax.imshow(np.abs(flux_map[i] / flux_err_map[i]), 
                  origin='lower', cmap='nipy_spectral', vmin=0, vmax=50)
        ax.contour(np.abs(flux_map[i]) / flux_err_map[i], 
                   levels=[5], colors='gold', linewidths=1)
        ax.set_title(header[f"FIL{i}"])

    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path + f"snr_map.png")
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
    fig, ax = plt.subplots(figsize=(7, 3))
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
