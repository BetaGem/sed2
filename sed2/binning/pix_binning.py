import sys  # noqa: EXE002
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import binary_dilation

from ..utils import *

__all__ = [
    "get_bin_flux",
    "get_neighboring_pixels",
    "pixel_binning",
    "plot_binning",
    "redchi2_two_seds",
]

def redchi2_two_seds(sed1=None, e_sed1=None, sed2=None, e_sed2=None):
    '''
    Calculate the reduced chi-squared statistic for two SEDs
    '''
    for s in [sed1, sed2, e_sed1, e_sed2]:
        if s is None:
            s = []
    top = np.sum(sed2 * sed1 / (e_sed1**2 + e_sed2**2))
    bottom = np.sum(sed1**2 / (e_sed1**2 + e_sed2**2))
    norm = top / bottom

    red_chi2 = np.sum((sed2 - (norm * sed1))**2 / (e_sed1**2 + e_sed2**2)) / len(sed1)

    return red_chi2


def get_neighboring_pixels(arr, dilate_pix):
    '''
    Calculate the neighboring pixels of a 2D region though dilating the region
    by a specified number of pixels and calculate the difference between the 
    original and dilated regions.

    Parameters
    ----------
    arr: 2D array
        The input binary array.
    dilate_pix: int
        The number of pixels to dilate the region.

    Returns
    -------
    2D boolean array
        A boolean array indicating the neighboring pixels.
    '''    
    arr = np.array(arr)
    
    if len(arr.shape) != 2:
        raise ValueError("Input array must be 2D.")
    if arr.dtype != bool:
        raise ValueError("Input array must be of boolean type.")
    dilated = binary_dilation(arr, structure=np.ones((dilate_pix*2+1, 
                                                      dilate_pix*2+1)))
    return dilated & ~arr



def pixel_binning(workdir, ref_band, 
                  snr=None, snr_band=None, out_snr_factor=1.0,
                  Dmin_bin=3.0, r_growth=1, dr_growth=1, 
                  redc_chi2_limit=4.0, grow_percentile=50, dilate_npixels=100,
                  connected=True, verbose=False):
    """
    A modified function for pixel binning based on piXedfit.piXedfit_bin (Abdurro'uf et al. 2021)

    [https://github.com/aabdurrouf/piXedfit]

    :param workdir: str
        Path to the working directory containing the necessary input files.
    :param ref_band: str
        The reference band (filter) for sorting pixels based on the brightness. The central pixel of a bin is the brightest pixel in this reference band.
    :param Dmin_bin: int
        Minimum diameter of a bin in unit of pixel.
    :param snr: int, float, array_like, None
        S/N thresholds in all bands. The length of this array should be the same as the number of bands in snr_band.
        S/N threshold can vary across the filters. If SNR is None, the S/N is set to 5.0
    :param snr_band: int, None
        Index of the band to be used for calculating the S/N. If None, all bands are used.
    :param redc_chi2_limit: float
        A maximum reduced chi-square value for a pair of two SEDs to be considered as having a similar shape.
    :param r_growth: int
        Increment of pixels in each iteration when growing the bins.
    :param dr_growth: int
        Increment of pixels in each iteration when growing the bins.
    :param grow_percentile: float, 0 < grow_percentile < 100
        In each iteration of bin growth, only pixels with flux in the ref_band higher than this percentile will be included.
    :param out_snr_factor: int
        The S/N thresholds for the last bin are multiplied by this factor.
    :param dilate_npixels: int
        Minimum number of pixels for a bin to be dilated.
    :param connected: bool
        If True, pixels in the bin (except the last one) will form a connected region (i.e., no "island")
    """
    path = Path(workdir)
    fits_fluxmap = path / "flux" / "fluxmap.fits"

    out_path = path / "flux" / f"{fits_fluxmap.stem}_pixbin.fits"
    filters = load_filters_from_txt(path / "aux" / "filter_list.txt")
    ref_band = filters.index(ref_band) 

    if check_output(out_path): 
        return 

    # load data
    hdu = fits.open(fits_fluxmap)
    header = hdu[0].header
    gal_region   = hdu['GALAXY_REGION'].data
    map_flux     = hdu['FLUX'].data
    map_flux_err = hdu['FLUX_ERR'].data
    hdu.close()

    # number of filters
    n_band = map_flux.shape[0]
    # transpose from (wave,y,x) -> (y,x,wave)
    map_flux_trans     = np.transpose(map_flux,     axes=(1,2,0))
    map_flux_err_trans = np.transpose(map_flux_err, axes=(1,2,0))

    # find systematic error factor
    # (assume that SED shapes are always similar for the 9 central pixels)
    rows, cols = np.where(gal_region == 1)
    idx = np.unravel_index(map_flux[ref_band][rows, cols].argmax(), 
                           map_flux[ref_band][rows, cols].shape)
    yc, xc = rows[idx[0]], cols[idx[0]]   # peak flux in ref band

    status_add = 1
    factor = 0.01                         # 1% sys. flux error!
    while status_add == 1:
        sed1 = map_flux_trans[yc][xc]
        e_sed1 = np.sqrt((map_flux_err_trans[yc][xc])**2 + (factor*sed1)**2)
        pix_chi2 = []
        for yy in range(yc-1, yc+2):      # original: yc-2:yc+2
            for xx in range(xc-1, xc+2):  # original: xc-2:xc+2
                if yy != yc and xx != xc:
                    sed2 = map_flux_trans[yy][xx]
                    e_sed2 = np.sqrt((map_flux_err_trans[yy][xx])**2 + (factor*sed2)**2)
                    red_chi2 = redchi2_two_seds(sed1=sed1, e_sed1=e_sed1,
                                                sed2=sed2, e_sed2=e_sed2)
                    pix_chi2.append(red_chi2)
        pix_chi2 = np.asarray(pix_chi2)
        if np.median(pix_chi2) <= 2.0:
            status_add = 0

        factor = factor + 0.01            # add sys. error if SED shape is not similar in adjacent pixels

    # apply the factor
    map_flux_err_corr = np.sqrt( map_flux_err**2 + (factor * map_flux)**2 )
    # transpose from (band,y,x) -> (y,x,band)
    map_flux_err_corr_trans = np.transpose(map_flux_err_corr, axes=(1,2,0))

    # get reference band for pixel brightness
    if not (isinstance(ref_band, int) and ref_band < n_band):
        print("Please provide the correct reference band index!")
        sys.exit()

    if snr_band is not None:
        filter_ids = load_filter_ids(workdir, snr_band)
    else:
        filter_ids = np.arange(n_band)

    snr_thresh = np.zeros(n_band)
    if snr is None:
        snr_thresh[filter_ids] = 5.0
    elif isinstance(snr, (int, float)):
        snr_thresh[filter_ids] = snr
    elif isinstance(snr, (list, np.ndarray)):
        if len(snr) == len(filter_ids):
            snr_thresh[filter_ids] = np.asarray(snr)
        else:
            raise ValueError(f"The length of SNR array: {len(snr)} should be\
                              the same as the number of filters: {len(filter_ids)}.")
    snr_thresh[snr_thresh <= 0] = -1e+5

    # shape of the image
    dim_y = gal_region.shape[0]
    dim_x = gal_region.shape[1]

    # initialize
    pixbin_map       = np.zeros((dim_y, dim_x))
    map_bin_flux     = np.zeros((dim_y, dim_x, n_band))
    map_bin_flux_err = np.zeros((dim_y, dim_x, n_band))

    rows, cols = np.where((gal_region==1) & (pixbin_map==0))
    tot_npixs  = len(rows)   # total number of pixels in the region-of-interest

    # ----- the main loop -----
    
    count_bin = 0                         # number of bins finished
    cumul_npixs_in_bin = 0                # number of pixels finished
    
    while len(rows) > 0:
        # central pixel of this bin
        idx = np.unravel_index(map_flux[ref_band][rows, cols].argmax(), map_flux[ref_band][rows, cols].shape)
        bin_y_cent, bin_x_cent = rows[idx[0]], cols[idx[0]]

        # first circle
        # first, do square crop around the circle
        bin_rad = 0.5 * Dmin_bin
        del_dim = bin_rad + 3
        xmin = int(bin_x_cent - del_dim)
        xmax = int(bin_x_cent + del_dim)
        ymin = int(bin_y_cent - del_dim)
        ymax = int(bin_y_cent + del_dim)

        xmin = max(xmin, 0)
        if xmax >= dim_x: xmax = dim_x - 1

        ymin = max(ymin, 0)
        if ymax >= dim_y: ymax = dim_y - 1

        x = np.linspace(xmin, xmax, xmax - xmin + 1)
        y = np.linspace(ymin, ymax, ymax - ymin + 1)
        x_mesh, y_mesh = np.meshgrid(x, y)   # central square

        crop_gal_region = gal_region[ymin : ymax+1, xmin : xmax+1]
        crop_pixbin_map = pixbin_map[ymin : ymax+1, xmin : xmax+1]

        data2D_rad = np.sqrt((x_mesh - bin_x_cent)**2 + (y_mesh - bin_y_cent)**2)
        # (within radius) & (within RoI) & (not occupied by other bins)
        rows1, cols1 = np.where((data2D_rad <= bin_rad) & (crop_gal_region == 1) & (crop_pixbin_map == 0))
        # tranform coords in cropped array to coords in global array
        rows1 = rows1 + ymin
        cols1 = cols1 + xmin

        # get total fluxes within the bin
        tot_bin_flux = np.sum(map_flux_trans[rows1, cols1], axis=0)
        tot_bin_flux_err2 = np.sum(np.square(map_flux_err_trans[rows1, cols1]), axis=0)

        tot_SNR = np.nan_to_num(tot_bin_flux / np.sqrt(tot_bin_flux_err2))  # SNR for all bands
        idx0 = np.where(tot_SNR >= snr_thresh)             # bands that achieve the SNR limit

        if len(idx0[0]) == n_band:                           # if SNR is satisfied in all bands
            # get bin
            count_bin = count_bin + 1
            pixbin_map[rows1, cols1] = count_bin
            map_bin_flux[rows1, cols1] = tot_bin_flux
            map_bin_flux_err[rows1, cols1] = np.sqrt(tot_bin_flux_err2)
            cumul_npixs_in_bin = cumul_npixs_in_bin + len(rows1)

        # if SNR threthold is not satisfied, grow the bin
        else:
            dr_growth_cp = dr_growth
            stat_increase = 1
            cumul_rows = rows1.tolist()
            cumul_cols = cols1.tolist()

            this_bin_map = np.full((dim_y,dim_x), 0)
            this_bin_map[cumul_rows, cumul_cols] = 1
            if verbose:
                print("\n", count_bin + 1, "SNR low, growing bin...")

            while stat_increase == 1:

                # select pixels near the previous region
                neighbor_pix = get_neighboring_pixels(this_bin_map > 0, r_growth)
                rows1, cols1 = np.where(neighbor_pix & (gal_region==1) & (pixbin_map==0))

                # select the bright half of these pixels (bright & within RoI & not occupied by other bins)
                if len(rows1):
                    bright_lim = np.nanpercentile(map_flux[ref_band][rows1, cols1], 
                                                  grow_percentile)
                else:
                    bright_lim = -np.inf
                pix_select = (map_flux[ref_band] > bright_lim) &\
                      (gal_region==1) & (pixbin_map==0)
                
                if connected:
                    rows1, cols1 = np.where(pix_select & neighbor_pix)
                else:
                    rows1, cols1 = np.where(pix_select)

                # check similarity of SED shape (with the central pixel of this bin)
                cent_pix_SED_flux      = np.zeros((dim_y, dim_x, n_band))
                cent_pix_SED_flux_err = np.zeros((dim_y, dim_x, n_band))
                norm0 = np.zeros((n_band, dim_y, dim_x))

                cent_pix_SED_flux[rows1, cols1] = map_flux_trans[bin_y_cent][bin_x_cent]
                cent_pix_SED_flux_err[rows1, cols1] = map_flux_err_corr_trans[bin_y_cent][bin_x_cent]

                top0 = np.nansum(map_flux_trans[rows1, cols1] * cent_pix_SED_flux[rows1, cols1] / (np.square(map_flux_err_corr_trans[rows1, cols1]) + np.square(cent_pix_SED_flux_err[rows1, cols1])), axis=1)
                bottom0 = np.nansum(np.square(cent_pix_SED_flux[rows1, cols1]) / (np.square(map_flux_err_corr_trans[rows1, cols1]) + np.square(cent_pix_SED_flux_err[rows1, cols1])), axis=1)
                for i in range(n_band):
                    norm0[i][rows1, cols1] = top0 / bottom0
                # transpose from (band,y,x) -> (y,x,band)
                norm0_trans = np.transpose(norm0, axes=(1,2,0))
                pix_chi2 = np.nansum(np.square(map_flux_trans[rows1, cols1] - (norm0_trans[rows1, cols1] * cent_pix_SED_flux[rows1, cols1])) / (np.square(map_flux_err_corr_trans[rows1, cols1]) + np.square(cent_pix_SED_flux_err[rows1, cols1])), axis=1)

                idx_sel = np.where((pix_chi2 / n_band) <= redc_chi2_limit)

                # cut, only select pixels with similar SED shape to the central brightest pixel
                rows1_cut = rows1[idx_sel[0]]
                cols1_cut = cols1[idx_sel[0]]

                cumul_rows = cumul_rows + rows1_cut.tolist()
                cumul_cols = cumul_cols + cols1_cut.tolist()
                # avoid double-counting (https://stackoverflow.com/questions/53389236)
                magic_array = np.array(list(set(map(tuple, np.array([cumul_rows, cumul_cols]).T)))).T
                cumul_rows, cumul_cols = list(magic_array[0]), list(magic_array[1])

                # get total fluxes of the updated bin
                tot_bin_flux = tot_bin_flux + np.sum(map_flux_trans[rows1_cut, cols1_cut], axis=0)
                tot_bin_flux_err2 = tot_bin_flux_err2 + np.sum(np.square(map_flux_err_trans[rows1_cut, cols1_cut]), axis=0)

                tot_SNR = np.nan_to_num(tot_bin_flux / np.sqrt(tot_bin_flux_err2))
                idx0 = np.where(tot_SNR >= snr_thresh)

                if len(idx0[0]) == n_band:    # if SNR is high enough
                    # get bin
                    count_bin = count_bin + 1
                    pixbin_map[cumul_rows, cumul_cols] = count_bin
                    cumul_npixs_in_bin = cumul_npixs_in_bin + len(cumul_rows)

                    stat_increase = 0
                    if verbose: 
                        print(count_bin, "SNR satisfied, bin finished.")
                else:
                    # check remaining pixels
                    rows_rest, cols_rest = np.where((gal_region==1) & (pixbin_map==0))
                    tflux = np.sum(map_flux_trans[rows_rest, cols_rest], axis=0)
                    tflux_err2 = np.sum(np.square(map_flux_err_trans[rows_rest, cols_rest]), axis=0)
                    tSNR = np.nan_to_num(tflux / np.sqrt(tflux_err2))
                    tidx = np.where(tSNR >= snr_thresh * out_snr_factor)

                    # if SNR is not high enough even if adding all remaining pixels
                    if len(tidx[0]) < n_band or r_growth > np.sqrt(np.sum(gal_region) / np.pi):   
                        # bin all remaining pixels:
                        count_bin = count_bin + 1
                        pixbin_map[rows_rest, cols_rest] = count_bin
                        cumul_npixs_in_bin = cumul_npixs_in_bin + len(rows_rest)

                        stat_increase = 0
                        if verbose:
                            print(count_bin, "SNR not satisfied, bin finished with all remaining pixels.")
                        break

                    elif (len(cumul_rows) + cumul_npixs_in_bin) == tot_npixs:  # if all pixels have been binned
                        # get bin
                        count_bin = count_bin + 1
                        pixbin_map[cumul_rows, cumul_cols] = count_bin
                        cumul_npixs_in_bin = cumul_npixs_in_bin + len(cumul_rows)

                        stat_increase = 0
                        if verbose:
                            print(count_bin, "all pixels binned, bin finished.")
                        break

                    else:
                        stat_increase = 1

                if verbose:
                    print(count_bin, "fill the bin and try again ...")

                if len(rows1_cut) == 0:
                    r_growth += dr_growth_cp
                    dr_growth_cp += 3
                    if verbose:
                        print(count_bin, "growing radius increased to", r_growth)
                else:
                    this_bin_map[rows1_cut, cols1_cut] = count_bin
                    if verbose:
                        print(count_bin, "growing pixels:", len(rows1_cut), ". this bin map", np.sum(this_bin_map > 0))

        rows, cols = np.where((gal_region==1) & (pixbin_map==0))
        # end of while ...

        sys.stdout.write('\r')
        sys.stdout.write(f'Bins: {count_bin}  => accumulated pixels: {cumul_npixs_in_bin}/{tot_npixs}')
        sys.stdout.flush()
    sys.stdout.write('\n')

    # dilate outer bins
    for i in range(1, count_bin + 1):

        this_bin_map = pixbin_map == i
        if np.sum(this_bin_map) < dilate_npixels:
            continue
        this_bin_map = binary_dilation(this_bin_map,
                            structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
        pixbin_map[(this_bin_map > 0) & (pixbin_map > i)] = i

    # remove empty bins
    print("Removing empty bins ...")
    pixbin_map_clean = np.full((dim_y, dim_x), 0)
    for n, i in enumerate(np.unique(pixbin_map)): 
        pixbin_map_clean[pixbin_map == i] = n
        
    count_bin = int(np.max(pixbin_map_clean))
    pixbin_map = pixbin_map_clean

    # calculate bin flux and flux error maps
    for i in range(1, np.max(pixbin_map) + 1):
        rows, cols = np.where(pixbin_map == i)
        tot_bin_flux = np.sum(map_flux_trans[rows, cols], axis=0)
        tot_bin_flux_err2 = np.sum(np.square(map_flux_err_trans[rows, cols]), axis=0)
        map_bin_flux[rows, cols] = tot_bin_flux
        map_bin_flux_err[rows, cols] = np.sqrt(tot_bin_flux_err2)

    # transpose from (y,x,wave) => (wave,y,x)
    map_bin_flux_trans = np.transpose(map_bin_flux, axes=(2,0,1))
    map_bin_flux_err_trans = np.transpose(map_bin_flux_err, axes=(2,0,1))

    print (f"Number of bins: {count_bin}")

    ## store into FITS file
    hdul = fits.HDUList()
    hdr = fits.Header()
    hdr['nfilters'] = n_band
    hdr['refband'] = ref_band
    hdr['psfband'] = header['psfband']
    for key in ['GalEBV', 'fsamp', 'pixsize', 'unit']:
        if key in header:
            hdr[key] = header[key]

    hdr['nbins'] = count_bin
    hdr['bunit'] = 'erg/s/cm^2/A'
    hdr['struct'] = '(band,y,x)'

    for i in range(n_band):
        str_temp = f'fil{i:d}'
        hdr[str_temp] = header[str_temp]

    hdul.append(fits.ImageHDU(data=pixbin_map, header=hdr, name='bin_map'))
    hdul.append(fits.ImageHDU(map_bin_flux_trans, name='bin_flux'))
    hdul.append(fits.ImageHDU(map_bin_flux_err_trans, name='bin_fluxerr'))
    hdul.writeto(out_path, overwrite=True)

    return out_path


def plot_binning(workdir, show_idx=True, 
                 xlim=None, ylim=None, vmin=0, out_path=None):
    '''
    Plot the pixel binning map.

    Parameters
    ----------
    pixbin_map : str
        The file path to the pixel binning map to plot.
    '''
    import matplotlib.pyplot as plt

    with fits.open(workdir / "flux" / "fluxmap_pixbin.fits") as hdu:
        bin_map = hdu['bin_map'].data.copy()
        n_bin = int(np.max(bin_map))

    bin_map[bin_map == np.max(bin_map)] += 5  # for illustration only

    plt.figure(figsize=(7, 5))
    plt.imshow(bin_map, cmap='nipy_spectral_r', origin='lower', vmin=vmin)
    plt.colorbar(label='Bin Index')
    plt.title('Pixel Binning Map')
    plt.xlabel('X [Pixel]')
    plt.ylabel('Y [Pixel]')

    if xlim is not None: plt.xlim(xlim)
    if ylim is not None: plt.ylim(ylim)

    if show_idx:
        for i in range(n_bin):
            plt.text(np.mean(np.where(bin_map == i + 1)[1])-.5, 
                     np.mean(np.where(bin_map == i + 1)[0])-.5,
                     i, c='w', fontsize=8)
            
    if out_path is not None: 
        plt.savefig(out_path)
    else:
        plt.show()


def get_bin_flux(workdir, galaxy, fluxmap_path=None,
                 bootstrap=100, plot_sed=True, out_path=None):
    '''
    Get the flux values and uncertainties for each bin from the flux map.

    Parameters
    ----------
    workdir : str
        The working directory containing the pixel binning results.
    fluxmap_path : str
        The file path to the flux map FITS file.
    bootstrap : int
        The number of bootstrap samples to use for uncertainty estimation.
    plot_sed : bool
        Whether to plot the SED for each bin.
    out_path : str
        The file path to save the flux tables.
    Returns
    -------
    bin_flux : list
        A list of flux values for each bin.
    bin_flux_err : list
        A list of flux uncertainties for each bin.
    '''
    from tqdm import tqdm
    path = Path(workdir)

    # load binning data
    hdu = fits.open(path / "flux" / "fluxmap_pixbin.fits")
    binmap  = hdu['bin_map'].data
    fluxmap = hdu['bin_flux'].data
    fluxmap_err = hdu['bin_fluxerr'].data
    n_bin = int(np.max(binmap))
    filters_all = [hdu[0].header[k] for k in hdu[0].header if k.startswith('FIL')]

    if fluxmap_path is None:
        fluxmap_path = path / "flux" / "fluxmap.fits"

    fits_fluxmap = fits.open(fluxmap_path)
    unit_flux = float(fits_fluxmap[0].header['UNIT'])
    n_band = int(fits_fluxmap[0].header['NFILTERS'])
    filters = [fits_fluxmap[0].header[f'FIL{i}'] for i in range(n_band)]

    # index of filters to use
    if len(filters_all) > len(filters):
        filter_idx = [filters_all.index(f) for f in filters]
    else:
        filter_idx = np.arange(n_band)

    # wise corrections 
    # https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html
    # w12_vega = np.array([-0.4040, -0.0538, 0.2939, 0.6393, 0.9828, 1.3246, 1.6649, 2.0041])
    # w_fc = np.array([[1.0283, 1.0084, 0.9961, 0.9907, 0.9921, 1, 1.0142, 1.0347],
    #                  [1.0206, 1.0066, 0.9976, 0.9935, 0.9943, 1, 1.0107, 1.0265],
    #                  [1.1344, 1.0088, 0.9393, 0.9169, 0.9373, 1, 1.1081, 1.2687], 
    #                  [1.0142, 1.0013, 0.9934, 0.9905, 0.9926, 1, 1.0130, 1.0319]])
    w_zpc = np.array([0.034, 0.041, -0.030, 0.029])

    # interpolate
    # inter = [CubicSpline(w12_vega, w_fc[i]) for i in range(4)]

    # get band index for corrections
    mir = get_mirband_idx(filters)

    # MIR SED correction
    for i in range(n_bin):
        bin_id = i + 1
        r, c = np.where(binmap == bin_id)
        # [W1 - W2] color
        # w1_flux = fluxmap[mir[0], r[0], c[0]] * unit_flux
        # w2_flux = fluxmap[mir[1], r[0], c[0]] * unit_flux
        # w12 = -2.5 * np.log10(w1_flux*33526**2 / (w2_flux*46028**2)) + 3.339 - 2.699 # AB to vega

        # apply wise color corrections
        # for n, l in enumerate(mir[:4]):
        #     fluxmap[l, r, c] *= inter[n](w12)

        # IRAC corrections
        # https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/46/#_Toc82083698
        cal_IRAC = np.array([0.91, 0.94, 0.66, 0.74])
        for n, l in enumerate(mir[4:]):
            if l < 0: continue
            fluxmap[l, r, c] *= cal_IRAC[n]

    # WISE zero point correction
    for n, l in enumerate(mir[:4]):
        fluxmap[l] *= 10**(-0.4 * w_zpc[n])

    # mask the region of interest
    msk = binmap > 0
    apers = []

    # Calculate background sampling error
    print("Calculating background sampling error ...")
    for i in tqdm(range(n_band)): 

        img = fits.open(path / "matched" / f"{galaxy}_{filters[i]}.fits")[0].data

        # flux conversion unit
        bin_ = binmap == 1
        bin_unit = np.median(fluxmap[i][bin_]) / np.nansum(img[bin_])

        e_last = 0.0

        for j in range(1, n_bin+1):
            bin_ = binmap == j
            aper_ = None if i == 0 else apers[j-1]

            flux_err, _, aper = bin_noise(
                img, bin_mask=bin_, apers=aper_,
                mask_src=msk, bootstrap=bootstrap)
            if flux_err == 0:
                # increase bootstrap samples and retry
                flux_err, _, aper = bin_noise(
                    img, bin_mask=bin_, apers=aper_,
                    mask_src=msk, bootstrap=bootstrap*2, circle_aper=True)
            if flux_err == 0:
                # assume Gaussian noise
                flux_err = e_last * np.sqrt(np.sum(bin_) / np.sum(binmap == j-1))

            e_last = flux_err
            if i == 0: apers.append(aper)

            bkg_flux_err = np.nanmax(fluxmap_err[i][bin_])
            fluxmap_err[i][bin_] = np.sqrt((flux_err*bin_unit)**2 + bkg_flux_err**2)

    # Add calibration error
    e_cali = calib_error(filters)
    for n, fi in enumerate(filter_idx):
        fluxmap_err[fi] = np.sqrt(fluxmap_err[fi]**2 + (fluxmap[fi] * e_cali[n])**2)

    # get bin flux and error
    bin_flux = np.zeros((n_bin, n_band))
    bin_flux_err = np.zeros((n_bin, n_band))

    for j in range(n_bin):
        r, c = np.where(binmap == j + 1)
        bin_flux[j] = fluxmap[filter_idx, r[0], c[0]] * unit_flux
        bin_flux_err[j] = fluxmap_err[filter_idx, r[0], c[0]] * unit_flux

    if plot_sed:
        wave = np.array([get_filter_waves(f)[0] for f in filters])

        import matplotlib.pyplot as plt

        plt.figure(figsize=(8,4))
        for j in np.linspace(0, n_bin-1, 5, dtype=int):
            plt.errorbar(np.array(wave) / 1e+4, 
                         bin_flux[j] * wave,
                         yerr=bin_flux_err[j] * wave, 
                         markersize=6, fmt='o-', alpha=0.4, lw=1)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r"$\lambda~(\mathrm{\mu m})$")
        plt.ylabel(r"$\lambda f_\lambda~\mathrm{(erg/s/cm^2)}$")
        plt.savefig(path / "plot" / f"{galaxy}_sed_sample.png", dpi=300)
        plt.close()

    if out_path is None:
        out_path = path / "flux" 

    save_flux(bin_flux, bin_flux_err, filters, out_path,
              prefix="highres_")

    return bin_flux, bin_flux_err