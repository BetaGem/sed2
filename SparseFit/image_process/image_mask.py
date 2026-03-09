import os
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from photutils.segmentation import deblend_sources

from .image_bkg import *


PATH = "/home/pku/Astro/FEASTS_SED/data"

def mask_separate(mask_ref, img_ref, coord, deblend=False):

    ref_hdu = fits.open(mask_ref)
    ref_wcs = WCS(ref_hdu[0].header)
    mask = ndimage.gaussian_filter(ref_hdu[0].data, 1)
    mask = mask > 0.5
    # mask_label, _ = ndimage.label(mask)

    segm = detect_sources(mask.astype(int), 0.5, npixels=1)
    if deblend:
        segm = deblend_sources(img_ref.data, segm, 1000, nlevels=512, contrast=0.01)

    mask_label = segm.data
    
    center_pix   = ref_wcs.world_to_array_index(coord)
    center_label = mask_label[center_pix[1]][center_pix[0]]
    
    mask_center  = mask_label == center_label
    mask_outer   = (mask_label > 0) & (mask_label != center_label)

    return mask_center, mask_outer


def get_sweep_photoz(ralo, rahi, declo, dechi):
    
    # find relevant sweep files
    rastep = 10
    decstep = 5
    r1 = rastep * np.floor(ralo / rastep).astype(int)
    r2 = rastep * np.ceil (rahi / rastep).astype(int)
    d1 = decstep * np.floor(declo / decstep).astype(int)
    d2 = decstep * np.ceil (dechi / decstep).astype(int)

    sweep_name = f"sweep-{r1:03d}p{d1:03d}-{r2:03d}p{d2:03d}-pz.fits"
    file_path  = "/home/pku/Astro/FEASTS_SED/data/ancillary/"
    print("sweep file:", sweep_name)
    
    if not sweep_name in os.listdir(file_path):
        print("target sweep file not found, downloading ......")
        sn = "north" if dechi > 32 else "south"
        url = f"https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9/{sn}/sweep/9.1-photo-z/" + sweep_name
        os.system(f"wget -O {file_path}{sweep_name} {url}")

    photoz = Table.read(file_path + sweep_name)
    return photoz


def get_tractor_catalog(ralo, rahi, declo, dechi):

    from astropy.table import Table, vstack

    # get brick names
    url = f"https://www.legacysurvey.org/viewer/ls-dr10/cat.fits?ralo={ralo}&rahi={rahi}&declo={declo}&dechi={dechi}"
    desi_cat = fits.open(url)
    desi_cat = Table(desi_cat[1].data)
    bricks = list(set(desi_cat['brickname']))
    print("bricks:", bricks)
    
    file_path  = "/home/pku/Astro/FEASTS_SED/data/ancillary/"

    # download and stack catalogs in different bricks
    desi_cat = []
    for brick in bricks:
        brick_name = f"tractor-{brick}.fits"
        if not brick_name in os.listdir(file_path):
            sn = "north" if dechi > 32 else "south"
            url = f"https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9/{sn}/tractor/{brick[:3]}/{brick_name}"
            os.system(f"wget -O {file_path}{brick_name} {url}")
            
        desi_cat.append( Table.read(file_path + brick_name) )

    return vstack(desi_cat)

    
def get_photoz(sweep_cat, release, brickid, objid):

    # z_idx = np.where((sweep_cat['RELEASE'] == release) & (sweep_cat['BRICKID'] == brickid) & (sweep_cat['OBJID'] == objid))
    z_idx = np.where((sweep_cat['BRICKID'] == brickid) & (sweep_cat['OBJID'] == objid))[0]
    if len(z_idx) == 0:
        return 99, 1, 99, 99
    z_mean = np.array(sweep_cat['Z_PHOT_MEAN'])[z_idx]
    z_std  = np.array(sweep_cat['Z_PHOT_STD' ])[z_idx]
    z_L95  = np.array(sweep_cat['Z_PHOT_L95' ])[z_idx]
    z_U95  = np.array(sweep_cat['Z_PHOT_U95' ])[z_idx]

    return z_mean, z_std, z_L95, z_U95


def get_photometry(tractor_cat, src_index):

    i = src_index

    mg = 22.5 - 2.5 * np.log10(tractor_cat["flux_g"][i] / tractor_cat["mw_transmission_g"][i])
    mr = 22.5 - 2.5 * np.log10(tractor_cat["flux_r"][i] / tractor_cat["mw_transmission_r"][i])
    mz = 22.5 - 2.5 * np.log10(tractor_cat["flux_z"][i] / tractor_cat["mw_transmission_z"][i])
    return mg, mr, mz


def gaia_download(ra, dec, width=0.5, height=0.5, out_path="."):
    from astroquery.gaia import Gaia

    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
    Gaia.ROW_LIMIT = -1
    
    # check your network connections
    gaia_table = Gaia.query_object_async(SkyCoord(ra, dec, unit='deg'), 
                                         width=width*u.deg, 
                                         height=height*u.deg)
    gaia_table.write(out_path, format="votable", overwrite=True)


def center_stars(name, hdu, seg_center, year_to_gaia=-12, coord=None):
    '''
    gaia stars within inner mask
    '''

    if coord is None: coord = SkyCoord.from_name(name)
    gal_ra, gal_dec = coord.ra.value, coord.dec.value

    file_path = "/home/pku/Astro/FEASTS_SED/data/ancillary/"
    gaia_cat  = f'Gaia3_star_{name}.vot'
    
    if not gaia_cat in os.listdir(file_path):
        print("downloading", gaia_cat)
        gaia_download(gal_ra, gal_dec, 
                      width=0.5 / np.cos(gal_dec/57.3), height=0.5,
                      out_path=f"{file_path}{gaia_cat}")
    
    gaia_cat = Table.read(f"{file_path}{gaia_cat}")

    # gaia_select = (gaia_cat['ruwe'] < 1.4*1000) & (gaia_cat['ra'] > 0)
    gaia_select = ((np.abs(gaia_cat['pmra']) / gaia_cat['pmra_error'] > 5) |\
                   (np.abs(gaia_cat['pmdec']) / gaia_cat['pmdec_error'] > 5)) & (gaia_cat['ra'] > 0)
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
        x = int(round(stars['xcentroid'][i]))
        y = int(round(stars['ycentroid'][i]))
        if x < 0 or y < 0 or x >= hdu.data.shape[1] or y >= hdu.data.shape[0]:
            continue
        if seg_center[y][x] == 0:
            continue
        _r = 50 if name in ['NGC2903', 'NGC4214', 'NGC4449', 'NGC5194'] else 10
        if np.sqrt( (x-WCS(hdu.header).pixel_shape[1]/2)**2 + (y-WCS(hdu.header).pixel_shape[0]/2)**2) < _r:
            continue
        if stars['phot_rp_mean_mag'][i] > 18 and stars['phot_bp_mean_mag'][i] > 18: # skip faint stars
            continue
        cstars_index.append(i)

    return stars[cstars_index]
    

def get_high_z_source(hdu, tractor_cat, sweep_cat, seg_center, gal_z=0):
    '''get high-z sources from the tractor catalog.'''
    
    tractor_cat_highz = tractor_cat.copy()

    src_pixel = WCS(hdu.header).world_to_pixel(SkyCoord(tractor_cat['ra'], 
                                                        tractor_cat['dec'],
                                                        unit='deg'))
    tractor_cat_highz['xcentroid'] = src_pixel[0]
    tractor_cat_highz['ycentroid'] = src_pixel[1]

    highz_idx = []
    for i in range(len(tractor_cat)):
        # skip sources outside the mask
        x = int(round(tractor_cat_highz['xcentroid'][i]))
        y = int(round(tractor_cat_highz['ycentroid'][i]))
        if x < 0 or y < 0 or x >= hdu.data.shape[1] or y >= hdu.data.shape[0]:
            continue
        if seg_center[y][x] == 0:
            continue
        if np.sqrt( (x-WCS(hdu.header).pixel_shape[1]/2)**2 + (y-WCS(hdu.header).pixel_shape[0]/2)**2) < 50:
            continue
        
        # get redshift
        z_mean, z_std, z_L95, z_U95 = get_photoz(sweep_cat, 
                                                 tractor_cat["release"][i], 
                                                 tractor_cat["brickid"][i], 
                                                 tractor_cat["objid"][i])
        
        # get desi ls-dr9 photometry
        g, r, z = get_photometry(tractor_cat, i)

        # select possible high-z sources
        if z < 19 and ((z_mean - 3 * z_std > gal_z) or (z_mean > 0.05 and 2. * z_std < z_mean)) and r - z > 0 and z > 12:
            highz_idx.append(i)
    
    return tractor_cat_highz[highz_idx]


def get_catalogs(name, ref_hdu, seg_center, gal_z=0, coord=None, year_to_gaia=-12):
    '''
    use the reference band to do this!
    '''
    # get desi catalogs
    ref_wcs = WCS(ref_hdu.header)
    center_pix = np.where(seg_center > 0)
    ralo  = ref_wcs.pixel_to_world_values(np.max(center_pix[0]), 0)[0]
    rahi  = ref_wcs.pixel_to_world_values(np.min(center_pix[0]), 0)[0]
    declo = ref_wcs.pixel_to_world_values(0, np.min(center_pix[1]))[1]
    dechi = ref_wcs.pixel_to_world_values(0, np.max(center_pix[1]))[1]
    # outliers
    if name == 'NGC3521': declo = 0

    # source detection within the central region
    if coord is None: coord = SkyCoord.from_name(name)
    cstars = center_stars(name, ref_hdu, seg_center, year_to_gaia=year_to_gaia, coord=coord)

    if name in ['NGC7331', 'SexB']:
        csources = []
    else:
        try:
            photoz  = get_sweep_photoz(ralo, rahi, declo, dechi)
            tractor = get_tractor_catalog(ralo, rahi, declo, dechi)
            csources = get_high_z_source(ref_hdu, tractor, photoz, seg_center, gal_z)
        except: 
            # in case of no internet connection
            print("Warning: Get high-z source from Legacy Survey failed ...")
            csources = []

    return cstars, csources


def mask_one_band_inner(name, sci_file, cstars, csources,
                        crop_size=3, crop_size_large=6, thresh=2, fwhm=1, dilate=1.5):
    '''
    generate inner mask for the current band.
    '''
    cur_img = f'{PATH}/{name}/cropped/crop_{sci_file}'
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
        if not "wise" in cur_img:         # wise images are already smoothed enough ......
            crop = ndimage.gaussian_filter(crop, fwhm / 2.235)
        try:
            crop_bkg = Background2D(crop, box_size=crop_radius*2, exclude_percentile=99)
        except: continue
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


def mask_one_band_outer(name, sci_file, center_mask, nsigma=3, fwhm=1, dilate=0, deblend=False, clean=True):

    cur_img = f'{PATH}/{name}/cropped/crop_{sci_file}'
    cur_hdu = fits.open(cur_img)
    cur_wcs = WCS(cur_hdu[0].header)

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


def load_circ_mask(gname, fname):
    '''load interactive circular masks from file'''
    try:
        circ_mask = np.load(f"{PATH}/{gname}/masked/circ_mask_{gname}_{fname}.npy").astype(bool)
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