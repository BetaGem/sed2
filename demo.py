# This is a demo script for running SED2

import sys
from pathlib import Path

import sed2

PATH = sed2.path.PATH


def read_par(par_file):
    """Read a parameter file (key-value format, # for comments)."""
    params = {}
    with open(par_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, val = line.split(None, 1)
            val = val.split("#")[0].strip()
            if val.lower() == "true":
                val = True
            elif val.lower() == "false":
                val = False
            elif val.lower() == "none":
                val = None
            else:
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
            params[key] = val
    return params


def parse_steps(steps_str):
    """Parse comma-separated step list into a list of step names."""
    return [s.strip() for s in steps_str.split(",")]


def main():
    par_file = Path(__file__).parent / "demo.par"
    if len(sys.argv) > 1:
        par_file = Path(sys.argv[1])

    params = read_par(par_file)

    galaxy = params.get("galaxy", "NGC0628")
    steps = parse_steps(params.get("steps", ""))
    detect_band = params.get("detect_band", "sdss_r")
    detect_nsigma = params.get("detect_nsigma", 1.0)
    crop_size = params.get("crop_size", 151)
    snr_thresh = params.get("snr_thresh", 3.0)
    npix_min = params.get("npix_min", 30)
    psf_ref = params.get("psf_ref", "spitzer_mips_24")
    fit_mode = params.get("fit_mode", "highres")
    n_process = params.get("n_process", 60)
    out_dir = params.get("out_dir", "data")
    redshift = params.get("redshift", 0.0)
    verbose = params.get("verbose", True)

    data_path = PATH / out_dir / galaxy / "flux"

    print(f"{'='*50}")
    print(f"  SED2 demo: {galaxy}")
    print(f"{'='*50}")
    print(f"  Steps:       {steps}")
    print(f"  Detect band: {detect_band} (nsigma={detect_nsigma})")
    print(f"  Crop size:   {crop_size} px")
    print(f"  SNR thresh:  {snr_thresh}")
    print(f"  Min pixels:  {npix_min}")
    print(f"  PSF ref:     {psf_ref}")
    print(f"  Fit mode:    {fit_mode}")
    print(f"  N processes: {n_process}")
    print(f"  Redshift:    {redshift}")
    print(f"  Output:      {data_path}")
    print(f"{'='*50}")

    # ---- pipeline begins ----
    for step in steps:
        if step == "bkg":
            print(f"\n>>> Background subtraction [{galaxy}]")
            # sf.image_process.image_bkg.subtract_background(...)

        elif step == "crop":
            print(f"\n>>> Image cropping [{galaxy}]")
            # sf.image_process.image_crop.crop(...)

        elif step == "psf":
            print(f"\n>>> PSF matching [{galaxy}]")
            # sf.image_process.image_psf.match_image(...)

        elif step == "mask":
            print(f"\n>>> Source masking [{galaxy}]")
            # sf.image_process.image_mask.mask_one_band_inner(...)
            # sf.image_process.image_mask.mask_one_band_outer(...)

        elif step == "fluxmap":
            print(f"\n>>> Flux map construction [{galaxy}]")
            # region = sf.binning.pix_cube.galaxy_region(...)
            # sf.binning.pix_cube.flux_map(...)

        elif step == "binning":
            print(f"\n>>> Pixel binning [{galaxy}]")
            # sf.binning.pix_binning.pixel_binning(...)
            # sf.binning.pix_binning.get_bin_flux(...)

        elif step == "fit":
            print(f"\n>>> SED fitting [{galaxy}] (mode={fit_mode})")
            # sf.fitting.fit_bagpipes.run(...)

        else:
            print(f"  Unknown step: {step}")


if __name__ == "__main__":
    main()
