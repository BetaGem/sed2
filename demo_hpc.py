# This is a demo script for running Bagpipes 
# SED fitting on a HPC cluster.

import sys
import multiprocess as mp
import SparseFit as sf
PATH = sf.path.PATH

name = sys.argv[1]
start, end = int(sys.argv[2]), int(sys.argv[3])
itr = sys.argv[4]

f_path = f'{PATH}/data/{name}/flux/'

def worker(ID):
    if itr == 'highres':
        sf.fitting.fit_bagpipes.run(ID=ID, 
                                    flux_table=f"{f_path}/highres_flux_table.fits",
                                    nebular_metal=f"{f_path}/nebular_metallicity.npy",
                                    verbose=True, run=f"{name}_highres", pool=1)
    elif itr == 'full':
        sf.fitting.fit_bagpipes.run(ID=ID, 
                                    flux_table=f"{f_path}/flux_table.fits",
                                    nebular_metal=f"{f_path}/nebular_metallicity.npy",
                                    verbose=True, run=name, pool=1)
    else:
        raise ValueError("argv[4] should be highres or full!")

with mp.Pool(processes=60) as pool:
    pool.map(worker, range(start, end), chunksize=1)