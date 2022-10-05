# Script loads in 2D slices and produces a movie of the simulation output import numpy as np import h5py, gc
import h5py
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from functions import get_metadata, read_params, get_grid
from scipy import ndimage

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"

var1_key = 'th1_xz'
var2_key = 'td_scatter'

##### ---------------------- #####

def get_index(z, griddata):
    return int(np.argmin(np.abs(griddata - z)))

##### ---------------------- #####

# Get dir locations from param file
base_dir, run_dir, save_dir, version = read_params(params_file)

# Get simulation metadata
md = get_metadata(run_dir, version)

gxf, gyf, gzf, dzf = get_grid(join(save_dir,'grid.h5'), md)
gx, gy, gz, dz = get_grid(join(save_dir,'grid.h5'), md, fractional_grid=False)

print("Complete metadata: ", md)

# Get data
with h5py.File(join(save_dir,"movie.h5"), 'r') as f:
    print("Keys: %s" % f.keys())
    time_keys = list(f[var1_key])
    print(time_keys)
    # Get buoyancy data
    b = np.array([np.array(f[var1_key][t]) for t in time_keys])
    scatter = np.array([np.array(f[var2_key][t]) for t in time_keys])
    NSAMP = len(b)
    times = np.array([f[var1_key][t].attrs['Time'] for t in time_keys])
    f.close()

print(scatter.shape)

for i in range(NSAMP):
    print(times[i])
    im = plt.imshow(scatter[i])
    im.set_clim(0, 0.1)
    plt.show()
