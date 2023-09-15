import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py, gc, sys
import numpy as np
from scipy.interpolate import griddata
from scipy import integrate
import scipy.special as sc
import matplotlib
from datetime import datetime
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from os.path import join, isfile
from os import listdir
from functions import get_metadata, read_params, get_grid, g2gf_1d

def get_index(z, griddata):
    return int(np.argmin(np.abs(griddata - z)))

def compute_pdf(data, ref, bins, normalised=False):
    out_bins = [0 for i in range(len(bins)-1)]

    for i in range(len(bins)-1):
        out_bins[i] = np.sum(np.where(np.logical_and(data >= bins[i],
            data < bins[i+1]), ref, 0))

    out_bins = np.array(out_bins)

    if normalised:
        area = integrate.trapezoid(np.abs(out_bins), 0.5*(bins[1:]+bins[:-1]))
        return out_bins/area
    else:
        return out_bins

##### USER-DEFINED PARAMETERS #####
params_file = "params.dat"

save = False

start_idx = 20

##### ----------------------- #####

##### Get directory locations #####
base_dir, run_dir, save_dir, version = read_params(params_file)
print("Run directory: ", run_dir)
print("Save directory: ", save_dir)

##### Get simulation metadata #####
md = get_metadata(run_dir, version)
print("Complete metadata: ",md)

##### Get grid #####
gxf, gyf, gzf, dzf = get_grid(join(run_dir, 'grid.h5'), md)
gx, gy, gz, dz = get_grid(join(run_dir, 'grid.h5'), md, fractional_grid=False)

##### Get data and time indices #####
with h5py.File(join(save_dir,"mean.h5"), 'r') as f:
    print("Mean keys: %s" % f.keys())
    time_keys = list(f['bbins'])
    print(time_keys)
    vol_props = np.array([np.array(f['chi_volprop'][t]) for t in time_keys])
    times = np.array([float(f['chi_volprop'][t].attrs['Time']) for t in time_keys])
    NSAMP = len(times)

    f.close()

times = times[start_idx:]
vol_props = vol_props[start_idx:]

# Define Omega bins
Omega_min = -1e-5
Omega_max = 2e-5
NOmega = 32

dOmega = (Omega_max - Omega_min)/(NOmega - 1)
bins = np.array([Omega_min + i*dOmega for i in range(NOmega)])

plot_bins = 0.5*(bins[1:] + bins[:-1])

vol_props = vol_props[:, :-1]

fig = plt.figure()

vol_props = np.where(vol_props == 0, np.nan, vol_props)

bars1 = plt.bar(plot_bins, vol_props[0], width=dOmega)
bars2 = plt.bar(plot_bins, 1-vol_props[0], width=dOmega, bottom=vol_props[0])

plt.xlim(Omega_min, Omega_max)
plt.ylim(0, 1)

plt.title(times[0])

def animate(step):
    for i, b in enumerate(bars1):
        b.set_height(vol_props[step, i])
    for i, b in enumerate(bars2):
        b.set_height(1-vol_props[step,i])
        b.set_y(vol_props[step,i])

    plt.title(times[step])

anim = animation.FuncAnimation(fig, animate, interval=250, frames=NSAMP-start_idx)

plt.show()
