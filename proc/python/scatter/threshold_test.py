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

    pvd = np.array([f['pvd'][t] for t in time_keys])
    vd = np.array([f['td_scatter'][t] for t in time_keys])
    vd_flux = np.array([np.array(f['td_flux'][t]) for t in time_keys])
    times = np.array([f['pvd'][t].attrs['Time'] for t in time_keys])
    NSAMP = len(times)

with h5py.File(save_dir+"/mean.h5", 'r') as f:
    bbins = np.array(f['PVD_bbins']['0001'])
    phibins = np.array(f['PVD_phibins']['0001'])

for i in range(1,NSAMP):
    vd_flux[i] += vd_flux[i-1]

db = bbins[1] - bbins[0]
dphi = phibins[1] - phibins[0]

##### Non-dimensionalisation #####

F0 = compute_F0(save_dir, md, tstart_ind = 6*4, verbose=False, zbot=0.7, ztop=0.95, plot=False)
N = np.sqrt(md['N2'])

T = np.power(N, -1)
L = np.power(F0, 1/4) * np.power(N, -3/4)
B = L * np.power(T, -2)
V = L * L * L

times /= T
db /= B
vd /= V
vd_flux /= V

##### ====================== #####

sim_step = 58

pvd = np.array([(vd[i] - vd_flux[i])/np.nansum(vd_flux[i]) for i in range(NSAMP)])

dwdt = np.gradient(vd, times, axis=0)

pvd_threshs = np.linspace(0, np.max(pvd[sim_step]), 10)

s_int = np.nansum(np.gradient(vd_flux, times, axis=0), axis=(1,2))

#plt.plot(times, s_int)

cols = plt.cm.rainbow(np.linspace(0, 1, NSAMP))
for m,c in zip(pvd_threshs, cols):
    W_int = np.nansum(np.where(pvd > m, vd, np.NaN), axis=(1,2))
    dW_int_dt = np.gradient(W_int, times, axis=0)
    plt.plot(times, dW_int_dt/s_int, label=m)

plt.legend()
plt.show()
