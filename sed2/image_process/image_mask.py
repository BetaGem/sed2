import multiprocessing as mp
import os
import subprocess
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.wcs import WCS
from photutils.background import Background2D
from photutils.segmentation import deblend_sources, detect_sources, detect_threshold
from reproject import reproject_interp
from scipy import ndimage

from .. import utils

__all__ = [
    "center_stars",
    "do_mask",
    "gaia_download",
    "get_catalogs",
    "get_high_z_source",
    "get_photometry",
    "get_photoz",
    "get_sweep_photoz",
    "get_tractor_catalog",
    "interp_image",
    "load_circ_mask",
    "mask_one_band_inner",
    "mask_one_band_outer",
    "mask_separate",
    "modify_pixel_val"
]


def _download_tractor_tile(task):
    r, d, step, workdir = task
    url = (
        f"https://www.legacysurvey.org/viewer/ls-dr9/cat.fits?"
        f"ralo={r:.1f}&rahi={r + step:.1f}&declo={d:.1f}&dechi={d + step:.1f}"
    )
    subprocess.run(["wget", "-c", "-O", 
                    workdir / "aux" / f"tractor_{r:.1f}_{d:.1f}.fits", url], check=False)


def get_sweep_photoz(workdir, ralo, rahi, declo, dechi):
    '''
    Get photo-z from the Legacy Survey sweep files (v9.1, Zhou et al. 2023).
    '''
    from itertools import product

    def read_sweep_table(path):
        try:
            print(f"Reading sweep file: {path}")
            return Table.read(path, format='fits')
        except OSError:
            return None

    workdir = Path(workdir)
    if os.path.exists(workdir / "aux" / "sweep_photoz.fits"):
        print("Sweep photo-z catalog already exists. Loading ...")
        return Table.read(workdir / "aux" / "sweep_photoz.fits")
    
    # find relevant sweep files
    rastep = 10
    decstep = 5
    ra_grid = np.arange(ralo // rastep * rastep,
                        (rahi // rastep + 1) * rastep + .1, rastep)
    de_grid = np.arange(declo // decstep * decstep,
                        (dechi // decstep + 1) * decstep + .1, decstep)

    for ra, dec in product(ra_grid[:-1], de_grid[:-1]):
        r1 = rastep * np.floor(ra / rastep).astype(int)
        r2 = rastep * np.ceil(ra / rastep + 1).astype(int)
        d1 = decstep * np.floor(dec / decstep).astype(int)
        d2 = decstep * np.ceil(dec / decstep + 1).astype(int)

        d1str = f'p{d1:03d}' if d1 >= 0 else f'm{abs(d1):03d}'
        d2str = f'p{d2:03d}' if d2 >= 0 else f'm{abs(d2):03d}'
        sweep_name = f"sweep-{r1:03d}{d1str}-{r2:03d}{d2str}-pz.fits"
        local_path = workdir / "aux" / sweep_name

        if os.path.exists(local_path):
            print(f"{sweep_name} already exists in {workdir / 'aux'}.")
            continue

        tmp_north = f"{local_path}.north.tmp"
        tmp_south = f"{local_path}.south.tmp"
        if not os.path.exists(tmp_north) and not os.path.exists(tmp_south):
            print(f"Downloading {sweep_name} ...")
            url_north = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9/north/sweep/9.1-photo-z/" + sweep_name
            url_south = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9/south/sweep/9.1-photo-z/" + sweep_name
            
            print(f"Downloading {sweep_name} from north ...")
            subprocess.run(["wget", "-c", "-O", tmp_north, url_north], check=False)
            print(f"Downloading {sweep_name} from south ...")
            subprocess.run(["wget", "-c", "-O", tmp_south, url_south], check=False)
        else:
            print(f"Using existing temporary files for {sweep_name}.")

        sweep_north = read_sweep_table(tmp_north)
        sweep_south = read_sweep_table(tmp_south)

        if sweep_north is not None and sweep_south is not None:
            sweep_cat = vstack([sweep_north, sweep_south])
        elif sweep_north is not None:
            sweep_cat = sweep_north
        elif sweep_south is not None:
            sweep_cat = sweep_south
        else:
            print(f"Failed to download {sweep_name} from both north and south.")
            for tmp_path in (tmp_north, tmp_south):
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            continue

        tmp_out = f"{local_path}.tmp"
        sweep_cat.write(tmp_out, format='fits', overwrite=True)

        os.replace(tmp_out, local_path)

        for tmp_path in (tmp_north, tmp_south):
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    photoz = Table()
    for filename in os.listdir(workdir / "aux"):
        if "sweep" in filename and filename.endswith(".fits"):
            sweep_path = workdir / "aux" / filename
            tab = Table.read(sweep_path)
            photoz = vstack([photoz, tab])
            os.remove(sweep_path)

    photoz.write(workdir / "aux" / "sweep_photoz.fits", overwrite=True)
    return photoz


def get_tractor_catalog(workdir, ralo, rahi, declo, dechi, 
                        nprocess=1, step=0.1):

    from itertools import product

    if os.path.exists(workdir / "aux" / "tractor.fits"):
        print("Tractor catalog already exists. Loading ...")
        return Table.read(workdir / "aux" / "tractor.fits")

    # devide the region into boxes with interval of step
    ralo = ralo // step * step
    rahi = rahi // step * step + step
    declo = declo // step * step
    dechi = dechi // step * step + step

    iters = list(product(np.arange(ralo, rahi+1e-9, step), 
                         np.arange(declo, dechi+1e-9, step)))

    nprocess = min(nprocess, len(iters))
    with mp.Pool(nprocess) as pool:
        pool.map(_download_tractor_tile, 
                 [(r, d, step, Path(workdir)) for r, d in iters])

    # combine all tractor files into one
    tractor_cat = Table()
    for filename in os.listdir(Path(workdir) / "aux"):
        if filename.startswith("tractor_") and filename.endswith(".fits"):
            tractor_path = Path(workdir) / "aux" / filename
            tab = Table.read(tractor_path)
            tractor_cat = vstack([tractor_cat, tab])
            # remove the individual tractor file to save space
            os.remove(tractor_path)

    tractor_cat.write(Path(workdir) / "aux" / "tractor.fits", overwrite=True)    
    return tractor_cat


def get_photoz(sweep_cat, brickid, objid):

    z_idx = np.where((sweep_cat['BRICKID'] == brickid) & (sweep_cat['OBJID'] == objid))[0]
    if len(z_idx) == 0:
        return 99, 1, 99, 99
    z_mean = np.array(sweep_cat['Z_PHOT_MEAN'])[z_idx]
    z_std  = np.array(sweep_cat['Z_PHOT_STD' ])[z_idx]

    return z_mean, z_std


def get_photometry(tractor_cat, src_index):

    i = src_index

    # flux can be non-positive, yielding NaN magnitudes; suppress the warning
    with np.errstate(divide="ignore", invalid="ignore"):
        mr = 22.5 - 2.5 * np.log10(tractor_cat["flux_r"][i] / tractor_cat["mw_transmission_r"][i])
        mz = 22.5 - 2.5 * np.log10(tractor_cat["flux_z"][i] / tractor_cat["mw_transmission_z"][i])
    return mr, mz


def gaia_download(coord, width=0.5, height=0.5, out_path="."):
    from astroquery.gaia import Gaia

    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
    Gaia.ROW_LIMIT = -1
    
    # check your network connections
    print(f"Downloading Gaia DR3 catalog for {coord.to_string('hmsdms')} with width={width:.4f} deg and height={height:.4f} deg ...")
    gaia_table = Gaia.query_object_async(coord, 
                                         width=width*u.deg, 
                                         height=height*u.deg)
    gaia_table.write(out_path, format="votable", overwrite=True)


def mask_separate(working_dir, bkg_ref_band, galaxy=None,
                  deblend=False, npixels=1000, **kwargs):
    '''
    separate the central mask and outer mask using image in bkg_ref_band.
    '''
    path = Path(working_dir)
    ref_mask = fits.open(path / "aux" / "ref_mask.fits")
    ref_img = fits.open(path / "cropped" / f"crop_{galaxy}_{bkg_ref_band}.fits")
    mask = ndimage.gaussian_filter(ref_mask[0].data, 1)
    mask = mask > 0.5
    # mask_label, _ = ndimage.label(mask)

    segm = detect_sources(mask.astype(int), 0.5, npixels=1)
    if deblend:
        segm = deblend_sources(ref_img[0].data, segm, npixels=npixels, **kwargs)

    mask_label = segm.data
    
    center_pix   = mask.shape[0] // 2, mask.shape[1] // 2
    center_label = mask_label[center_pix[0]][center_pix[1]]
    
    mask_center  = mask_label == center_label

    fits.writeto(path / "aux" / "ref_mask_inner.fits", 
                 mask_center.astype(np.int8), ref_mask[0].header, overwrite=True)


def center_stars(workdir, hdu, seg_center, coord,
                 year_to_gaia=-12, pm_sigma=5):
    '''
    get gaia stars within the inner mask.
    '''
    path = Path(workdir)
    wcs = WCS(hdu.header)
    img_size = np.array(hdu.data.shape) * np.abs(wcs.pixel_scale_matrix.diagonal())

    # file_path = "/home/pku/Astro/FEASTS_SED/data/ancillary/"
    gaia_cat  = 'Gaia3_stars.vot'
    
    if gaia_cat not in os.listdir(path / "aux"):
        gaia_download(coord, 
                      width=img_size[1] / np.cos(np.deg2rad(coord.dec.value)), 
                      height=img_size[0],
                      out_path=path / "aux" / gaia_cat)
    
    gaia_cat = Table.read(path / "aux" / gaia_cat)

    gaia_select = ((np.abs(gaia_cat['pmra']) / gaia_cat['pmra_error'] > pm_sigma) |\
                   (np.abs(gaia_cat['pmdec']) / gaia_cat['pmdec_error'] > pm_sigma)) & (gaia_cat['ra'] > 0)
    stars = gaia_cat[gaia_select]
    
    # correct for proper motion of stars (catalog unit = mas)
    stars['ra']  += year_to_gaia * stars['pmra']  / 1000 / 3600
    stars['dec'] += year_to_gaia * stars['pmdec'] / 1000 / 3600
    
    star_coord = SkyCoord(stars['ra'], stars['dec'], unit='deg')
    star_pixel = WCS(hdu.header).world_to_pixel(star_coord)
    stars['xcentroid'] = star_pixel[0]
    stars['ycentroid'] = star_pixel[1]
    
    # skip sources outside the mask and at image center
    cstars_index = []
    for i in range(len(stars)):
        x = round(stars['xcentroid'][i])
        y = round(stars['ycentroid'][i])
        if x < 0 or y < 0 or x >= hdu.data.shape[1] or y >= hdu.data.shape[0]:
            continue
        if seg_center[y][x] == 0:
            continue
        # _r = 50 if galaxy in ['NGC2903', 'NGC4214', 'NGC4449', 'NGC5194'] else 10
        _r = 10
        if np.sqrt( (x-WCS(hdu.header).pixel_shape[1]/2)**2 + (y-WCS(hdu.header).pixel_shape[0]/2)**2) < _r:
            continue
        if stars['phot_rp_mean_mag'][i] > 19 and stars['phot_bp_mean_mag'][i] > 19: # skip faint stars
            continue
        cstars_index.append(i)

    return stars[cstars_index]
    

def get_high_z_source(hdu, tractor_cat, sweep_cat, seg_center, gal_z=0):
    '''
    get high-z sources from the tractor catalog.
    '''
    tractor_cat_highz = tractor_cat.copy()

    src_pixel = WCS(hdu.header).world_to_pixel(SkyCoord(tractor_cat['ra'], 
                                                        tractor_cat['dec'],
                                                        unit='deg'))
    tractor_cat_highz['xcentroid'] = src_pixel[0]
    tractor_cat_highz['ycentroid'] = src_pixel[1]

    highz_idx = []
    for i in range(len(tractor_cat)):
        # skip sources outside the mask
        x = round(tractor_cat_highz['xcentroid'][i])
        y = round(tractor_cat_highz['ycentroid'][i])
        if x < 0 or y < 0 or x >= hdu.data.shape[1] or y >= hdu.data.shape[0]:
            continue
        if seg_center[y][x] == 0:
            continue
        # skip sources at image center
        if np.sqrt( (x-WCS(hdu.header).pixel_shape[1]/2)**2 + (y-WCS(hdu.header).pixel_shape[0]/2)**2) < 50:
            continue
        
        # get photo-z from sweep catalog
        z_mean, z_std = get_photoz(sweep_cat, 
                                   tractor_cat["brickid"][i], 
                                   tractor_cat["objid"][i])

        # get desi ls-dr9 photometry
        r, z = get_photometry(tractor_cat, i)

        # select possible high-z sources
        if z < 19 and ((z_mean - 3 * z_std > gal_z) or (z_mean > 0.05 and 2. * z_std < z_mean)) and r - z > 0 and z > 12:
            highz_idx.append(i)
    
    return tractor_cat_highz[highz_idx]


def get_catalogs(workdir, galaxy, bkg_ref_band, coord,
                 gal_z=0, year_to_gaia=-12, pm_sigma=5, nprocess=1):
    '''
    get the catalogs center stars and high-z sources.
    '''
    if os.path.exists(Path(workdir) / "aux" / "cstars.ecsv") and \
       os.path.exists(Path(workdir) / "aux" / "csources.ecsv"):
        print("Loading existing catalogs. To regenerate, delete aux/cstars.ecsv and aux/csources.ecsv.")
        cstars = Table.read(Path(workdir) / "aux" / "cstars.ecsv")
        csources = Table.read(Path(workdir) / "aux" / "csources.ecsv")
        return cstars, csources
    
    # load reference image and mask
    path = Path(workdir)
    ref_hdu = fits.open(path / "cropped" / f"crop_{galaxy}_{bkg_ref_band}.fits")[0]
    ref_wcs = WCS(ref_hdu.header)
    seg_center = fits.open(path / "image" / "ref_mask_inner.fits")[0]
    seg_center = reproject_interp((seg_center.data, WCS(seg_center.header)), ref_hdu.header,
                                  return_footprint=False)
    
    # get desi catalogs
    center_pix = np.where(seg_center > 0)
    ralo  = ref_wcs.pixel_to_world_values(np.max(center_pix[0]), 0)[0]
    rahi  = ref_wcs.pixel_to_world_values(np.min(center_pix[0]), 0)[0]
    declo = ref_wcs.pixel_to_world_values(0, np.min(center_pix[1]))[1]
    dechi = ref_wcs.pixel_to_world_values(0, np.max(center_pix[1]))[1]

    # source detection within the central region
    cstars = center_stars(workdir, ref_hdu, seg_center, 
                          coord=coord, 
                          year_to_gaia=year_to_gaia, pm_sigma=pm_sigma)

    try:
        photoz  = get_sweep_photoz(workdir, ralo, rahi, declo, dechi)
        tractor = get_tractor_catalog(workdir, ralo, rahi, declo, dechi, nprocess=nprocess)
        csources = get_high_z_source(ref_hdu, tractor, photoz, seg_center, gal_z)
        print(f"Found {len(cstars)} center stars and {len(csources)} high-z sources.")
    except (RuntimeError, ConnectionError, TimeoutError): 
        print("Warning: Get high-z source from Legacy Survey failed ...")
        csources = Table()

    for name, tab in (("cstars", cstars), ("csources", csources)):
        if len(tab) > 0:
            Table(tab).write(path / "aux" / f"{name}.ecsv", 
                             format="ascii.ecsv", overwrite=True)
        else:
            Table().write(path / "aux" / f"{name}.ecsv", 
                          format="ascii.ecsv", overwrite=True)

    return cstars, csources


def mask_one_band_inner(workdir, galaxy, band, cstars, csources,
                        crop_size=3, crop_size_large=6, thresh=2, fwhm=1, dilate=1.5):
    '''
    generate inner mask for the current band.
    '''
    cur_img = Path(workdir) / "cropped" / f"crop_{galaxy}_{band}.fits"
    cur_hdu = fits.open(cur_img)
    cur_wcs = WCS(cur_hdu[0].header)
    
    mask = np.full(cur_hdu[0].data.shape, 0)
    
    for i in range(len(cstars) + len(csources)):
        # source center
        if i < len(cstars):
            ra, dec = cstars['ra'][i], cstars['dec'][i]
        else:
            j = i - len(cstars)
            ra, dec = csources['ra'][j], csources['dec'][j]
        pixel = cur_wcs.world_to_pixel(SkyCoord(ra, dec, unit='deg'))
        cy, cx = int(pixel[1]), int(pixel[0])
        # skip sources outside the image
        if cx < 0 or cy < 0 or cx >= mask.shape[1] or cy >= mask.shape[0]:
            continue
        # crop region near the source, use larger crop size for very bright stars
        size = crop_size
        if i < len(cstars) and cstars['phot_rp_mean_mag'][i] < 14:
            size = crop_size_large
            if cstars['phot_rp_mean_mag'][i] < 10:
                size *= 2
        # elif i >= len(cstars) and 22.5 - 2.5*np.log10(csources['flux_r'][j]) < 19:
        #     size = crop_size_large
        
        crop_radius = int(size * fwhm)
        crop = cur_hdu[0].data[cy - crop_radius: cy + crop_radius + 1,
                               cx - crop_radius: cx + crop_radius + 1]
        if "wise" not in band:         # wise images are already smoothed enough ......
            crop = ndimage.gaussian_filter(crop, fwhm / 2.235)
        try:
            crop_bkg = Background2D(crop, box_size=crop_radius*2, exclude_percentile=99)
        except ValueError: 
            continue
        back_median, back_std = crop_bkg.background_median, crop_bkg.background_rms_median
        
        if cur_hdu[0].data[cy, cx] < back_median + back_std * thresh:
            continue
        else: 
            # mask central segment
            crop_mask = crop > back_median + back_std * thresh
            crop_mask_label, _ = ndimage.label(crop_mask)
            crop_mask[crop_mask_label != crop_mask_label[crop_radius][crop_radius]] = 0
            mask[cy - crop_radius: cy + crop_radius + 1,
                 cx - crop_radius: cx + crop_radius + 1] = crop_mask

    if dilate > 0:
        mask = ndimage.binary_dilation(mask, iterations=int(fwhm * dilate) )

    return mask


def mask_one_band_outer(workdir, galaxy, band, center_mask, 
                        nsigma=3, fwhm=1, dilate=0, deblend=False, clean=True):

    cur_img = Path(workdir) / "cropped" / f"crop_{galaxy}_{band}.fits"
    cur_hdu = fits.open(cur_img)

    # mask center
    center_mask = center_mask > 0
    temp_mask = center_mask | (np.isnan(cur_hdu[0].data))
    
    # detect outer sources
    data = ndimage.gaussian_filter(cur_hdu[0].data, fwhm / 2.355)
    thresh  = detect_threshold(data, nsigma=nsigma, background=0)
    segm    = detect_sources  (data, thresh, int(fwhm**2), mask=temp_mask)

    if segm is None:             # no outer mask
        return np.full(data.shape, False)
    
    if deblend:
        segm = deblend_sources(data, segm, int(fwhm**2)//3, nlevels=8, contrast=0.1)
    
    # the mask
    mask = segm.data > 0

    if clean:
        from scipy.ndimage import label
        # recover the total mask
        mask_add = mask.astype(np.int8) + center_mask.astype(np.int8)
        labels, _ = label(mask_add)
        center_label = np.nanmedian(labels[center_mask])
        # remove regions connected to the central mask
        mask[labels == center_label] = 0
    
    if dilate > 0:
        mask = ndimage.binary_dilation(mask, iterations=int(fwhm * dilate) )

    return mask


def load_circ_mask(workdir, galaxy, band):
    '''
    load interactive circular masks from file
    '''
    path = Path(workdir)
    try:
        circ_mask = np.load(path / "masked" / f"circ_mask_{galaxy}_{band}.npy").astype(bool)
        return circ_mask
    except FileNotFoundError:
        return False


def interp_image(cur_hdu, mask, region_in, box_size=3):
    '''
    fill the mask with interpolation and noise
    '''
    # inner mask has mask_type == 2
    mask_type = mask.astype(int) * ((region_in > 0.5).astype(int) + 1) 

    img_clean  = cur_hdu.data.copy()
    # fill outer mask with noise
    rms = (np.nanpercentile(img_clean[(region_in == 0) & (~mask)], 84) - \
            np.nanpercentile(img_clean[(region_in == 0) & (~mask)], 16)) / 2
    img_clean[mask_type == 1] = np.random.randn(np.sum(mask_type == 1)) * rms
    # interpolate inner masks
    img_interp = Background2D(img_clean,
                              box_size=box_size,
                              filter_size=1,
                              mask=mask,
                              coverage_mask=np.isnan(img_clean) & (~mask),
                              exclude_percentile=50,
                             )
    img_clean[mask_type == 2] = img_interp.background[mask_type == 2]
    return img_clean


def modify_pixel_val(wcs, catalog):
    '''...'''
    coord = SkyCoord(catalog['ra'], catalog['dec'], unit='deg')
    pixel = wcs.world_to_pixel(coord)
    catalog['xcentroid'] = pixel[0]
    catalog['ycentroid'] = pixel[1]


def do_mask(workdir, galaxy, band, bkg_ref_band, 
            crop_size=3, crop_size_large=6, thresh=1.5, dilate=1.5, 
            thresh_out=3, clean=True, 
            load_mask=False, save_plot=True, verbose=True):
    '''
    generate the inner and outer masks for the current band.
    '''
    path = Path(workdir)

    seg_center = fits.open(path / "aux" / "ref_mask_inner.fits")[0].data
    ref_img = fits.open(path / "cropped" / f"crop_{galaxy}_{bkg_ref_band}.fits")[0]
    cur_img = fits.open(path / "cropped" / f"crop_{galaxy}_{band}.fits")[0]
    cur_var = fits.open(path / "cropped" / f"crop_var_{galaxy}_{band}.fits")[0]
    ref_wcs = WCS(ref_img.header)
    cur_wcs = WCS(cur_img.header)

    seg_cen_proj = reproject_interp((seg_center, ref_wcs), cur_wcs, return_footprint=False)

    psf_asec = utils.get_psf_size(band)
    pix_asec = utils.get_pixel_size(path / "cropped" / f"crop_{galaxy}_{band}.fits")
    fwhm = psf_asec / pix_asec

    # load mask parameters
    try:
        p = utils.load_json(path / "aux" / "mask_param.json")[band]
        crop_size, crop_size_large, thresh, dilate = p
    except (FileNotFoundError, KeyError):
        pass

    # load catalogs (needed for mask generation and pixel centroids)
    cwave = utils.get_filter_waves(band)[0]
    cstars   = Table.read(path / "aux" / "cstars.ecsv")
    csources = Table.read(path / "aux" / "csources.ecsv")

    mask = None
    if load_mask:
        try:
            mask = fits.open(path / "masked" / f"mask_{galaxy}_{band}.fits")[0].data.astype(bool)
            if verbose:
                print(f"loading existing mask for {band}..")
        except FileNotFoundError:
            if verbose:
                print(f"failed to load existing mask for {band}.", end=" ")

    if mask is None:
        if verbose:
            print(f"generating new mask for {band} ...")

        star_cat  = [] if cwave > 3e5 else cstars    # no star mask beyond 30 um
        highz_cat = [] if cwave < 3e3 else csources  # no high-z source mask below 300 A

        # inner and outer masks
        mask_inner = mask_one_band_inner(path, galaxy, band, 
                                         star_cat, highz_cat, 
                                         thresh=thresh, crop_size=crop_size, crop_size_large=crop_size_large,
                                         fwhm=fwhm, dilate=dilate)
        dilate = 0.5 if 'wise' in band else 1.5
        mask_outer = mask_one_band_outer(path, galaxy, band, 
                                         seg_cen_proj, 
                                         nsigma=thresh_out,
                                         fwhm=fwhm, dilate=dilate, clean=clean)

        mask = mask_inner | mask_outer
    

    # add interactive masks
    mask |= load_circ_mask(path, galaxy, band)
    # interp masks
    clean_img = interp_image(cur_img, mask, seg_cen_proj, box_size=3)

    # save the mask and cleaned image
    fits.writeto(path / "masked" / f"mask_{galaxy}_{band}.fits", 
                 mask.astype(np.int8), cur_img.header, overwrite=True)
    fits.writeto(path / "masked" / f"{galaxy}_{band}.fits", 
                 clean_img, cur_img.header, overwrite=True)
    fits.writeto(path / "masked" / f"var_{galaxy}_{band}.fits", 
                 cur_var.data, cur_var.header, overwrite=True)

    modify_pixel_val(cur_wcs, cstars)
    if isinstance(csources, Table): 
        modify_pixel_val(cur_wcs, csources)

    if save_plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,4))
        plt.subplot(121)
        plt.title(band)
        plt.imshow(_t:=np.cbrt(cur_img.data), origin='lower',
                   vmin=np.nanpercentile(_t, 16), 
                   vmax=np.nanpercentile(_t, 99.9))
        plt.imshow(mask, cmap='gray', alpha=.3)
        if isinstance(csources, Table): 
            plt.scatter(csources['xcentroid'], csources['ycentroid'], s=20, ec='r'   , marker='o', fc='none', lw=.2)
        plt.scatter(cstars['xcentroid'], cstars['ycentroid'], s=20, ec='gold', marker='o', fc='none', lw=.2)
        plt.xlim(0, cur_img.data.shape[1])
        plt.ylim(0, cur_img.data.shape[0])
        plt.subplot(122)
        plt.imshow(_t:=np.cbrt(clean_img),
                   vmin=np.nanpercentile(_t, 16), 
                   vmax=np.nanpercentile(_t, 99.9), 
                   origin='lower', cmap='jet')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(path / "plot" / f"mask_{galaxy}_{band}.png", dpi=300)