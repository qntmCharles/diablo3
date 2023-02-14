import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py, os
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from functions import get_metadata, read_params, get_grid, g2gf_1d
from scipy import ndimage

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"

##### ---------------------- #####

def get_index(z, griddata):
    return int(np.argmin(np.abs(griddata - z)))

##### ---------------------- #####

# Get dir locations from param file
base_dir, run_dir, save_dir, version = read_params(params_file)

#out_files = [f for f in os.listdir(save_dir) if f.endswith(".h5") and f.startswith("out.")]
out_files = ["end.h5"]
print(out_files)

# Get simulation metadata
md = get_metadata(run_dir, version)

gxf, gyf, gzf, dzf = get_grid(join(save_dir,'grid.h5'), md)
gx, gy, gz, dz = get_grid(join(save_dir,'grid.h5'), md, fractional_grid=False)

volume = (md['LX']/md['Nx'])**2 * dz
print(volume.shape)


print("Complete metadata: ", md)

# Get data
for out_file in out_files:
    with h5py.File(join(save_dir,out_file), 'r') as f:
        print("Keys: %s" % f.keys())
        keys = f['weights_flux_mem'].keys()
        weights = np.array([f['weights_flux_mem'][k] for k in keys])
        times = np.array([f['weights_flux_mem'][k].attrs['Time'] for k in keys])


with h5py.File(join(save_dir,'movie.h5'), 'r') as f:
    print("Keys: %s" % f.keys())
    time_keys = f['th1_xz'].keys()
    weights2 = np.array([f['svd'][k] for k in time_keys])
    flux = np.array([f['td_flux'][k] for k in time_keys])
    NSAMP = len(flux)

for i in range(1,NSAMP):
    flux[i] += flux[i-1]

print(flux[-1])
print(weights[0])
