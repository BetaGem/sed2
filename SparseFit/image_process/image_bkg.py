import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.segmentation import detect_sources, detect_threshold, deblend_sources
from photutils.background import SExtractorBackground, Background2D, MeanBackground, MedianBackground
from scipy.optimize import curve_fit
from scipy import ndimage

from reproject import reproject_interp

def Gaussian(x, A, mu, sigma):
    return A * np.exp(- (x - mu) ** 2 / 2 / sigma**2)


def fit_gaussian_for_image(data, model_percent=95, verbose=True):
    data_flat = data.reshape(-1)
    hist, bin = np.histogram(data_flat[(data_flat < np.nanpercentile(data_flat, model_percent)) &\
                                       (data_flat > np.nanpercentile(data_flat, 10))], bins=100)
    bins = (bin[:-1] + bin[1:]) / 2
    
    popt, pcov = curve_fit(Gaussian, bins, hist, 
                           p0=[np.nanmax(hist), bins[np.argmax(hist)], sigma_clipped_stats(data_flat)[2]], 
                           bounds=([np.nanmax(hist)/2, -np.inf, 0], [np.nanmax(hist)*2, np.inf, np.inf]))
    if verbose:
        plt.scatter(bins, hist, c='k')
        plt.plot(_x:=np.linspace(bins[0], bins[-1], num=100), Gaussian(_x, *popt), 'r--')
        plt.show()

    return popt, pcov


def mask_reproject(ref_mask, header):
    '''
    project the mask in reference band to WCS(header)
    '''
    arr, _ = reproject_interp(ref_mask, WCS(header))
    arr[arr > 0.5] = 1
    
    return arr.astype(int)


def dilate(radius_idx, maskim, psf_px):
    '''
    mask dilation.
    '''
    import scipy.ndimage as ndimage
    
    # ensure boolean array
    mask_bool = maskim.astype(bool)

    radius = max(round(radius_idx * psf_px), 1)
    yy = np.arange(-radius, radius+1)
    xx = yy.reshape((-1, 1))
    struct = (xx**2 + yy**2) <= radius**2

    # Apply dilation directly. 
    # Previously a binary_opening was applied first,
    # which eroded small regions (smaller than the opening kernel).
    dilated = ndimage.binary_dilation(mask_bool, structure=struct)

    # write back to input array in-place if possible
    try:
        maskim[...] = dilated.astype(maskim.dtype)
    except Exception:
        # fallback: return the dilated array
        return dilated.astype(maskim.dtype)


def mask_region_bgmodel(data, thresh=3, model_percent=86, npixels=10, 
                        verbose=False):

    # fit a Gaussian to the histogram of pixel values for source detection
    data = ndimage.gaussian_filter(data, 3)
    popt, pcov = fit_gaussian_for_image(data, verbose=verbose,
                                        model_percent=model_percent)
    
    thresh = popt[1] + popt[2] * thresh  # should be replaced with SEb if works bad
    segm_map = detect_sources(data, threshold=thresh, npixels=npixels)
    if segm_map is None:
        return np.full(data.shape, False), data.shape[0]
    box_max  = np.sqrt( np.sum(segm_map.areas) / np.pi) * 3
    dim_y = segm_map.data.shape[0]
    dim_x = segm_map.data.shape[1]
    mask_region = np.zeros((dim_y, dim_x))

    rows, cols = np.where(segm_map.data > 0)
    mask_region[rows, cols] = 1

    return mask_region, int(box_max)



def reference_mask(path, fits_image, name_out_mask=None,
                   nsigma=3, npixels=200, smooth=0, dilate=5):
    '''
    generate mask for the reference band
    '''
    hdu = fits.open(path + fits_image)[0]
    wcs = WCS(hdu.header)
    data = hdu.data.copy()
    if smooth:
        data = ndimage.gaussian_filter(data, smooth)
    thresh = detect_threshold(data, nsigma=nsigma)
    segm   = detect_sources(data, thresh, npixels=npixels)

    mask = segm.data > 0
    mask = ndimage.binary_dilation(mask, iterations=dilate)

    if name_out_mask is None:
        name_out_mask = path + "ref_mask.fits"
    fits.writeto(name_out_mask, mask.astype(np.int8), hdu.header, overwrite=True)

    plt.figure(figsize=(4, 4))
    plt.imshow(hdu.data, vmax=np.nanpercentile(hdu.data, 95), origin='lower')
    plt.imshow(mask, cmap='gray', alpha=.3, origin='lower')
    plt.savefig(name_out_mask + ".jpg", dpi=300)



def subtract_background(path, fits_image, hdu_idx=0, sigma=3.0, 
                        box_size=None, mask_ref=None, plot_mask=True,
                        mask_thresh=2, npixels=10, 
                        gaussian_noise=True, verbose=False):

    # open the input image:
    try:
        hdu = fits.open(path + fits_image)
    except FileNotFoundError:
        print(fits_image, "not found in path:", path)
        return 0
    data_image = hdu[int(hdu_idx)].data
    header = hdu[int(hdu_idx)].header
    hdu.close()

    # define box size: depending on the dimension of the image:
    dim_x = data_image.shape[1]
    dim_y = data_image.shape[0]

    # source mask
    if gaussian_noise:
        mask_region0, box_max = mask_region_bgmodel(data_image,
                                                    thresh=mask_thresh,
                                                    npixels=npixels,
                                                    verbose=verbose)
    else:
        mean, med, std = sigma_clipped_stats(data_image)
        segm = detect_sources(data_image, threshold=mean + mask_thresh * std, npixels=npixels)
        mask_region0 = (segm.data > 0).astype(int)
        box_max = min(dim_x, dim_y) // 2
    
    if box_size is None:
        box_size = min(dim_x, dim_y) // (min(dim_x, dim_y) // box_max)
        box_size = [box_size, box_size]

    print("box size for background estimation =", box_size)

    if mask_ref is None:
        mask_region1 = mask_region0
    else:
        mask_region = mask_reproject(mask_ref, header)
        if mask_region.shape[0]!=dim_y or mask_region.shape[1]!=dim_x:
            print ("dimension of mask_region should be the same with the dimension of fits_image!")
            sys.exit()
        else:
            mask_region1 = np.full((dim_y,dim_x), 0)
            rows, cols = np.where((mask_region0==1) | (mask_region==1))
            mask_region1[rows, cols] = 1

    # dilate mask by 3 pixels
    dilate(1, mask_region1, 3)

    if plot_mask:
        plt.imshow(data_image, 
                   vmin=np.nanpercentile(data_image, 16),
                   vmax=np.nanpercentile(data_image, 97))
        plt.colorbar()
        plt.imshow(mask_region1, alpha=.3, cmap='gray')
        plt.title("mask for background")
        plt.show()

    # final background
    sigma_clip = SigmaClip(sigma=sigma, maxiters=1,
                           cenfunc='median', stdfunc='std', grow=False)
    if gaussian_noise:
        bkg_estimator = SExtractorBackground()
    else:
        bkg_estimator = MeanBackground()
        
    bkg = Background2D(data_image, 
                       (box_size[0], box_size[1]), 
                       filter_size=3, 
                       mask=mask_region1.astype(bool), 
                       coverage_mask=np.isnan(data_image),
                       sigma_clip = sigma_clip,
                       bkg_estimator=bkg_estimator,
                       exclude_percentile=99)

    skybg_image = bkg.background

    if fits_image.endswith(".gz"):
        fits_image = fits_image[:-3]

    # store to fits file:
    name_out_skybg = path + "skybg_%s" % fits_image
    fits.writeto(name_out_skybg, skybg_image.astype('>f4'), header, overwrite=True)
    print("produce %s" % name_out_skybg)

    # get the background error image:
    skybgrms_image = bkg.background_rms
    
    name_out_skybgrms = path + "var_%s" % fits_image
    fits.writeto(name_out_skybgrms, skybgrms_image.astype('>f4')**2, header, overwrite=True)
    print("produce %s" % name_out_skybgrms)

    # calculate background subtracted image:
    skybgsub_image = data_image - skybg_image
    name_out_skybgsub = path + "skybgsub_%s" % fits_image
    fits.writeto(name_out_skybgsub, skybgsub_image.astype('>f4'), header, overwrite=True)
    print("produce %s" % name_out_skybgsub)

    return 1