import os
import subprocess
from pathlib import Path

import numpy as np
from astropy.table import Table
from tqdm import tqdm

from .fit_bagpipes import run

__all__ = [
    "fit_all",
]

def fit_all(workdir, flux_table, 
            manual_prior=None, redshift=0.0022,
            nprocess=1, nlive=1000, run_name='.',
            test_mode=False):
    """
    Run the SED fitting for all spatial bins using Bagpipes.

    Parameters
    ----------
    workdir : Path
        Path to the working directory.
    flux_table : Path
        Path to the flux table.
    manual_prior : Path, optional
        Path to the manual prior file.
    run_name : str
        Name of the run (e.g., "highres").
    redshift : float
        Redshift of the galaxy.
    nlive : int
        Number of live points for the nested sampling algorithm.
    nprocess : int
        Number of processes to use for the fitting.
    test_mode : bool
        If True, run in test mode (only fit 5 bins).
    """
    path_out  = Path(workdir)
    path_flux = path_out / "flux"
    flux_table = path_flux / flux_table
    table = Table.read(flux_table)

    if not test_mode:

        for i in tqdm(range(len(table))):
            run(ID=i,
                flux_table=flux_table,
                redshift=redshift,
                manual_prior=manual_prior,
                pool=nprocess, nlive=nlive, 
                run=run_name)
    else:
        idx = len(table) // 5 * np.arange(5)
        for i in tqdm(idx):
            run(ID=i,
                flux_table=flux_table,
                redshift=redshift,
                manual_prior=manual_prior,
                pool=nprocess, nlive=nlive, 
                run=run_name)

        current_dir = os.getcwd()
        print(f"Current directory: {current_dir}")
        for i in range(len(idx)):
            end = idx[i + 1] if (i + 1) < len(idx) else len(table)
            for j in range(idx[i] + 1, end):
                src = f"{current_dir}/pipes/posterior/{run_name}/{idx[i]}.h5"
                dst = f"{current_dir}/pipes/posterior/{run_name}/{j}.h5"
                subprocess.run(["cp", src, dst], check=False)