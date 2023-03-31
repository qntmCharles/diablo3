import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from functions import get_metadata, read_params, get_grid, g2gf_1d, get_plotindex, get_index, compute_F0
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

##### Non-dimensionalisation #####

F0 = compute_F0(save_dir, md, tstart_ind = 6*4, verbose=False, zbot=0.7, ztop=0.95, plot=False)
N = np.sqrt(md['N2'])

T = np.power(N, -1)
L = np.power(F0, 1/4) * np.power(N, -3/4)
B = L * np.power(T, -2)
V = L * L * L

times /= T

##### ====================== #####

svd = np.where(svd == 0, np.nan, svd)

svd_dt = np.gradient(svd, times, axis=0)

svd_bins = np.linspace(np.nanmin(svd[30:]), np.nanmax(svd[30:]), 100)
bins_plot = 0.5 * (svd_bins[1:] + svd_bins[:-1])

step = -1

print("Done getting data.")

fig = plt.figure(figsize=(12, 3.5))
ax = plt.gca()

N = 10
col = plt.cm.viridis(np.linspace(0, 1, N))
for step, c in zip(range(-N, 0), col):
    avgs = []
    for i in range(len(svd_bins)-1):
        avg = np.nanmean(np.where(np.logical_and(svd[step] >= svd_bins[i], svd[step] < svd_bins[i+1]),
                    svd_dt, np.nan))
        avgs.append(avg)#/bins_plot[i])

    ax.plot(bins_plot, avgs, label=r"$t = {0:.2f}$".format(times[step]), color=c)

ax.set_ylim(-0.1, 0.1)
ax.set_xlim(-0.02, 0.02)

ax.legend(loc='center left', fancybox=True, shadow=True)

ax.set_ylabel(r"$\mathcal{E}$")
ax.set_xlabel(r"$\hat{\Omega}$")

plt.tight_layout()
#plt.savefig('/home/cwp29/Documents/papers/draft/figs/entrainment.png', dpi=300)
#plt.savefig('/home/cwp29/Documents/papers/draft/figs/entrainment.pdf')
plt.show()
