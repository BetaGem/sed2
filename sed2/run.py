# Pipeline driver for SED2
from pathlib import Path

import multiprocess as mp

from . import binning, fitting, utils
from . import image_process as improc
from .path import PATH


def check_workdir(params):
    """
    Verify and create the working directory layout.

    The ``image`` sub-folder is the input and must already exist (an error
    is raised if missing). The ``cropped``, ``masked``, ``matched`` and
    ``flux`` sub-folders are created if they do not yet exist.

    Parameters
    ----------
    params : dict
        Parameter dict read from the parameter file. Must contain
        ``galaxy`` and ``workdir``.

    Returns
    -------
    Path
        Path to the galaxy working directory.
    """
    galaxy = params.get("galaxy")
    workdir = Path(params.get("workdir"))
    if galaxy is None or workdir is None:
        raise KeyError("'galaxy' and 'workdir' must be set in the parameter file.")

    image_dir = workdir / "image"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Input image directory not found: {image_dir}")

    for sub in ("cropped", "masked", "matched", "flux", "aux", "plot"):
        (workdir / sub).mkdir(parents=True, exist_ok=True)

    return workdir


def _subtract_background_band(args):
    """
    Background subtraction for a single band (worker for multiprocessing).
    """
    workdir, galaxy, filt, parc = args

    mask_ref = workdir / "aux" / "ref_mask.fits"
    if not parc.get("bkg_mask_ref", True):
        mask_ref = None

    improc.image_bkg.subtract_background(
        workdir, f"{galaxy}_{filt}.fits",
        mask_ref=mask_ref,
        box_size=parc["bkg_box_size"],
        mask_thresh=parc["bkg_mask_thresh"],
        npixels=parc["bkg_npixels"],
        gaussian_noise=parc["bkg_gaussian_noise"],
    )


def _do_mask_band(args):
    """
    Source masking for a single band (worker for multiprocessing).
    """
    workdir, galaxy, band, bkg_ref_band, mask_box, mask_box_large,\
          mask_in_thresh, mask_in_dilate, mask_out_thresh, mask_clean = args
    improc.do_mask(workdir, galaxy, 
                   band=band, bkg_ref_band=bkg_ref_band,
                   crop_size=mask_box, crop_size_large=mask_box_large,
                   thresh=mask_in_thresh,
                   dilate=mask_in_dilate,
                    thresh_out=mask_out_thresh, clean=mask_clean)


def run(par_file=None):
    """
    Run the SED2 pipeline using a parameter file.

    Parameters
    ----------
    par_file : str or Path, optional
        Path to the parameter file. If None, ``demo/demo.par`` is used.
    """
    if par_file is None:
        par_file = PATH / "demo" / "demo.par"

    # read every key/value from scratch into one dict.
    params = utils.read_par(Path(par_file))

    workdir = check_workdir(params)
    galaxy = params.get("galaxy", "galaxy")
    nprocess = params.get("n_process", 1)

    params["filters"] = utils.load_filters(workdir, gal_name=galaxy)
    params["coord"] = utils.resolve_coord(par_file, params)
    print(params["coord"])

    print(f"{'='*50}")
    print(f"  SED2: {galaxy}")
    print(f"{'='*50}")
    print(f"  {'Workdir':>15}: {workdir}")
    for key, val in params.items():
        print(f"  {key:>15}: {val}")
    print(f"{'='*50}")

    # -------------------------
    # ---- pipeline begins ----
    # -------------------------
    for step in params.get("steps", []):
        # ---- background subtraction ----
        if step == "bkg":
            print(f"\n>>> Background subtraction [{galaxy}]")
            improc.reference_mask(workdir,
                                  f"{galaxy}_{params['bkg_ref_band']}.fits",
                                  nsigma=params.get("bkg_ref_sigma", 1.5))
            # build per-band arguments
            args_list = []
            for f in params["filters"]:
                parc = params.copy()
                try:
                    p = utils.load_json(workdir / "aux" / "bkg_param.json")[f]
                    parc.update(p)
                except (FileNotFoundError, KeyError):
                    pass
                args_list.append((workdir, galaxy, f, parc))

            with mp.Pool(processes=nprocess) as pool:
                pool.map(_subtract_background_band, args_list)

        # --- image cropping ----
        elif step == "crop":
            print(f"\n>>> Image cropping [{galaxy}]")
            improc.crop(workdir, galaxy=galaxy,
                        crop_size=params.get("crop_size", 60),
                        filters=params["filters"],
                        coord=params["coord"])

        # --- source masking ----
        elif step == "mask":
            print(f"\n>>> Source masking [{galaxy}]")
            improc.reference_mask(workdir, 
                                  f"crop_{galaxy}_{params['bkg_ref_band']}.fits",
                                  type="cropped",
                                  nsigma=params.get("bkg_ref_sigma", 2.0))
            improc.mask_separate(workdir,
                                 bkg_ref_band=params["bkg_ref_band"],
                                 galaxy=galaxy,
                                 deblend=params["mask_deblend"])
            improc.get_catalogs(workdir, galaxy=galaxy,
                                bkg_ref_band=params["bkg_ref_band"], 
                                coord=params["coord"],
                                gal_z=params.get("redshift", 0.0),
                                nprocess=nprocess)

            args_list = [(workdir, galaxy, f, 
                          params["bkg_ref_band"], 
                          params.get("mask_box", 3),
                          params.get("mask_box_large", 6),
                          params.get("mask_in_thresh", 1.5),
                          params.get("mask_in_dilate", 1.5),
                          params.get("mask_out_thresh", 3),
                          params.get("mask_clean", True))
                         for f in params["filters"]]
            
            with mp.Pool(processes=params.get("n_process", 1)) as pool:
                pool.map(_do_mask_band, args_list)

        # --- PSF matching ----
        elif step == "psf":
            print(f"\n>>> PSF matching [{galaxy}]")
            improc.match_image(workdir, galaxy, 
                               filters=params["filters"],
                               flux_or_sb=params["psf_img_unit"], 
                               ref_band=params["psf_ref_band"])
            matched_img = [workdir / "matched" / f"{galaxy}_{f}.fits" for f in params["filters"]]
            utils.plot_stamps(matched_img, 
                              filters=params["filters"],
                              out_path=workdir / "plot" / f"{galaxy}_matched.png")

        # --- get RoI and flux maps ----
        elif step == "roi":
            print(f"\n>>> Flux map construction [{galaxy}]")
            f_high = utils.load_filters_from_txt(workdir / "aux" / "filters_highres.txt")
            f_low  = utils.load_filters_from_txt(workdir / "aux" / "filters_lowres.txt")

            binning.galaxy_region(workdir, galaxy=galaxy,
                                  roi_bands=params["roi_bands"],
                                  thresh=params["roi_thresh"],
                                  dilate_iter=params["roi_dilate"])
            binning.flux_map(workdir, galaxy, filters=f_high,
                             dp_unit=params["dp_unit"],
                             coord=params["coord"],
                             ref_band=params["psf_ref_band"])
            binning.flux_map(workdir, galaxy, filters=f_low,
                             dp_unit=params["dp_unit"],
                             coord=params["coord"],
                             ref_band=params["psf_ref_band"], 
                             unit_spire=params.get("unit_spire", "Jy_per_pixel"),
                             name_out_fits="fluxmap_lowres.fits")

        # binning and flux extraction
        elif step == "bin":
            print(f"\n>>> Pixel binning [{galaxy}]")
            dmin_bin = params.get("bin_npix_min", 3)
            binning.pixel_binning(workdir, 
                                  ref_band=params["bin_ref_band"],
                                  Dmin_bin=dmin_bin,
                                  r_growth=dmin_bin,
                                  snr=params.get("bin_snr_thresh", None),
                                  snr_band=params.get("bin_snr_band", None),
                                  grow_percentile=params.get("bin_percent", 50))
            binning.plot_binning(workdir, show_idx=False, vmin=0,
                                 out_path=workdir / "plot" / "pixbin.png")

            binning.get_bin_flux(workdir, galaxy, bootstrap=10)
                                 
        # SED fitting 
        elif step == "fit":
            print(f"\n>>> SED fitting [{galaxy}] (this can be slow ...)")
            fitting.fit_all(workdir, 
                            flux_table=workdir / "flux" / "highres_flux_table.fits",
                            redshift=params["redshift"],
                            nprocess=nprocess,
                            nlive=params.get("n_live", 1000),
                            manual_prior=params.get("manual_prior", None),
                            run_name=f"{galaxy}_highres",
                            test_mode=params["test_mode"])

        elif step == "2pass":
            print(f"\n>>> SED fitting (2nd-pass) [{galaxy}] (this can be slow ...)")

            path_ft = workdir / "flux" / "flux_table.fits"
            if utils.check_output(path_ft):
                continue

            utils.load_filters_from_txt(workdir / "aux" / "filters_lowres.txt")
            fitting.predict_flux_table(workdir,
                                       filters=params["filters"],
                                       galaxy=galaxy, run="_highres",
                                       manual_prior=params.get("manual_prior", None),
                                       n_processes=nprocess)
            fitting.new_flux_table(workdir, 
                                   out_file=path_ft)
            fitting.fit_all(workdir, 
                            flux_table=path_ft,
                            redshift=params["redshift"],
                            nprocess=nprocess,
                            nlive=params.get("n_live", 1000),
                            manual_prior=params.get("manual_prior", None),
                            run_name=f"{galaxy}_full",
                            test_mode=params["test_mode"])
                
        else:
            print(f"Warning!!! Unknown step: {step}")
