# Script loads in 2D slices and produces a movie of the simulation output import numpy as np import h5py, gc
import h5py
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from functions import get_metadata, read_params, get_grid

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"

##### ---------------------- #####

def get_index(z, griddata):
    return int(np.argmin(np.abs(griddata - z)))

# Get dir locations from param file
base_dir, run_dir, save_dir, version = read_params(params_file)

# Get simulation metadata
md = get_metadata(run_dir, version)

gxf, gyf, gzf, dzf = get_grid(save_dir+'grid.h5', md)
gx, gy, gz, dz = get_grid(save_dir+'grid.h5', md, fractional_grid=False)

X, Y = np.meshgrid(gx, gz)
Xf, Yf = np.meshgrid(gxf, gzf)

print("Complete metadata: ", md)

# Get data
with h5py.File(save_dir+"/movie.h5", 'r') as f:
    print("Keys: %s" % f.keys())
    time_keys = list(f['th1_xy'])
    print(time_keys)
    # Get buoyancy data
    th2_xy = np.array([np.array(f['th1_xz'][t]) for t in time_keys])
    th2_zy = np.array([np.array(f['th1_xz'][t]) for t in time_keys])
    NSAMP = len(th2_xy)
    times = np.array([float(f['th2_xz'][tstep].attrs['Time']) for tstep in time_keys])
    f.close()

print(th2_xy.shape)

ztest = 0.8*md['H']
h = get_index(ztest, gzf)
cols = plt.cm.rainbow(np.linspace(0, 1, NSAMP))

alpha = 0.1
rm = 1.2 * alpha * (ztest-0.05)
cm = 0.015
conc = 2 * cm * np.exp(-0.5*((gxf-md['LX']/2)/rm)**2)

fig = plt.figure()


for i,c in zip(range(NSAMP), cols):
    plt.plot(gxf, th2_xy[i,h,:], color=c, alpha=0.3)

plt.plot(gxf, conc, color='k', linestyle='dashed')
plt.show()
