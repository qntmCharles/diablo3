import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
from functions import get_metadata, read_params, get_grid, g2gf_1d
from scipy import ndimage, interpolate, spatial

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"

##### ---------------------- #####

def get_index(z, griddata):
    return int(np.floor(np.argmin(np.abs(griddata - z))))

##### ---------------------- #####

#d Get dir locations from param file
base_dir, run_dir, save_dir, version = read_params(params_file)

# Get simulation metadata
md = get_metadata(run_dir, version)

gxf, gyf, gzf, dzf = get_grid(join(save_dir,'grid.h5'), md)
gx, gy, gz, dz = get_grid(join(save_dir,'grid.h5'), md, fractional_grid=False)

print("Complete metadata: ", md)

# Get data
with h5py.File(join(save_dir,"movie.h5"), 'r') as f:
    print("Keys: %s" % f.keys())
    time_keys = list(f['th1_xz'])
    print(time_keys)
    # Get buoyancy data
    u = np.array([np.array(f['u_xz'][t]) for t in time_keys])
    u = g2gf_1d(md, u)
    v = np.array([np.array(f['v_xz'][t]) for t in time_keys])
    v = g2gf_1d(md, v)
    w = np.array([np.array(f['w_xz'][t]) for t in time_keys])
    w = g2gf_1d(md, w)
    b = np.array([np.array(f['th1_xz'][t]) for t in time_keys])
    b = g2gf_1d(md, b)

    times = np.array([f['th1_xz'][t].attrs['Time'] for t in time_keys])
    NSAMP = len(times)

w_hat = np.fft.rfft(w, axis=2)
w2hat = np.power(np.abs(w_hat), 2)
w2hat = np.sum(w2hat, axis=1)

k = np.fft.rfftfreq(md['Nx'], d=md['LX']/md['Nx'])

#plt.axvline(np.power(md['RE'], 3/4), color='k', linestyle='--')
plt.xscale('log')
plt.yscale('log')

for i in range(len(times)):
    plt.plot(k, w2hat[i], label = "{0:.2f} s".format(times[i]))

plt.title(save_dir.split('/')[-2])
plt.gca().set_aspect(3/5)
plt.legend()
plt.show()
