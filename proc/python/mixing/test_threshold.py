import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from functions import get_metadata, read_params, get_grid, g2gf_1d, get_plotindex, get_index
from scipy import ndimage

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"

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

plot_min = 0.95*md['H']
plot_max = md['LZ']

idx_minf = get_plotindex(plot_min, gzf)-1
idx_maxf = get_plotindex(plot_max, gzf)

idx_min = idx_minf
idx_max = idx_maxf+1

print(idx_min, idx_max)

gx, gy, dz = np.meshgrid(gxf, dz[idx_min:idx_max], gyf, indexing='ij', sparse=True)
volume = (md['LX']/md['Nx'])**2 * dz
print(volume.shape)

print("Complete metadata: ", md)

with h5py.File(join(save_dir, "movie.h5"), 'r') as f:
    print("Keys: %s" % f.keys())
    time_keys = list(f['th1_xz'])
    print(time_keys)

    svd = np.array([f['pvd'][t] for t in time_keys])
    times = np.array([f['pvd'][t].attrs['Time'] for t in time_keys])

svd = np.where(svd == 0, np.nan, svd)

svd_dt = np.gradient(svd, times, axis=0)

print(np.nanmax(svd))
svd_bins = np.linspace(-0.05, 0.05, 31)

step = -1

print("Done getting data.")

fig, ax = plt.subplots(1, 2)

for step in [-20, -10, -5, -4, -3, -2, -1]:
    avgs = []
    avgs2 = []
    for i in range(len(svd_bins)-1):
        avg = np.nanmean(np.where(np.logical_and(svd[step] >= svd_bins[i], svd[step] < svd_bins[i+1]),
                    svd_dt, np.nan))
        avgs.append(2*avg/abs(svd_bins[i] + svd_bins[i+1]))
        avgs2.append(avg)

    ax[0].plot(0.5*(svd_bins[1:] + svd_bins[:-1]), avgs, label=step)
    ax[1].plot(0.5*(svd_bins[1:] + svd_bins[:-1]), avgs2, label=step)

ax[0].set_ylim(-0.2, 0.2)
ax[1].set_ylim(-0.02, 0.02)
ax[1].legend()
ax[0].axhline(0)
ax[1].axhline(0)

plt.show()


fig, ax = plt.subplots(1,2)

ax[0].plot(0.5*(svd_bins[1:] + svd_bins[:-1]), avgs)
ax[0].axhline(0)

with h5py.File(join(save_dir,"end.h5"), 'r') as f:
    print("Keys: %s" % f['Timestep'].keys())

    svd = np.array(f['Timestep']['PVD'])
    svd = svd[:, idx_min:idx_max, :] # restrict to stratified layer
    svd = np.where(svd == -1e9, np.nan, svd)

    field = np.exp(np.array(f['Timestep']["chi"]))
    field = field[:, idx_min:idx_max, :] # restrict to stratified layer
    field = np.where(np.isnan(svd), np.nan, field) # restrict to plume

    avgs = []
    for i in range(len(svd_bins)-1):
        print(i)
        avg = np.nanmean(np.where(np.logical_and(svd >= svd_bins[i], svd < svd_bins[i+1]),
            field, np.nan))
        avgs.append(avg)

    ax[1].plot(0.5*(svd_bins[1:] + svd_bins[:-1]), avgs)
    ax[1].axhline(0)

    f.close()


plt.show()
