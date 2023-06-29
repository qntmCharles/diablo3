import sys, os
import matplotlib.colors as colors
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from functions import get_metadata, read_params, get_grid, g2gf_1d, get_plotindex, get_index, compute_F0
from scipy import ndimage, interpolate
from skimage import filters, exposure
import numpy.polynomial.polynomial as poly
from scipy.signal import argrelextrema

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
    vd = np.array([f['td_scatter'][t] for t in time_keys])
    S = np.array([np.array(f['td_flux'][t]) for t in time_keys])
    times = np.array([f['pvd'][t].attrs['Time'] for t in time_keys])
    NSAMP = len(times)

Scum = np.copy(S)

M = np.copy(svd)
for i in range(1, NSAMP):
    Scum[i] += Scum[i-1]
    M[i] *= np.sum(Scum[i])

with h5py.File(save_dir+"/mean.h5", 'r') as f:
    bbins = np.array(f['PVD_bbins']['0001'])
    phibins = np.array(f['PVD_phibins']['0001'])

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
M /= V
vd /= V
S /= V
Scum /= V

##### ====================== #####
#plt.figure()
#plt.axhline(0, color='k', linestyle='--')

vd = np.where(vd == 0, np.nan, vd)
M = np.where(np.isnan(vd), np.nan, M)

db = bbins[1] - bbins[0]
dphi = phibins[1] - phibins[0]

sx, sy = np.meshgrid(np.append(bbins-db/2, bbins[-1]+db/2),
        np.append(phibins - dphi/2, phibins[-1] + dphi/2))

t_idxs = range(20, NSAMP-1)
cols = plt.cm.rainbow(np.linspace(0, 1, len(t_idxs)))
threshs = []
for t,c in zip(t_idxs, cols):
    M_lim = np.nanmax(M[t])
    M_bins = np.linspace(0, M_lim, 100)

    dWdt_int = []
    S_int = []

    for m in M_bins[1:]:
        dWdt_int.append(np.nansum(np.where(M[t] < m, (vd[t+1]-vd[t])/md['SAVE_STATS_DT'], 0)))
        S_int.append(np.nansum(np.where(M[t] < m, S[t], 0)))

    dWdt_int = np.array(dWdt_int)
    S_int = np.array(S_int)

    #plt.plot(M_bins[1:], dWdt_int - S_int, label="{0:.2f}".format(times[t]), color=c)

    if np.min(dWdt_int - S_int) < 0:
        f = interpolate.interp1d(dWdt_int-S_int, M_bins[1:])
        thresh = f(0)
        #plt.axvline(f(0), color=c, linestyle=':')
    else:
        thresh = 0

    threshs.append(thresh)

plt.plot(times[t_idxs], threshs)
plt.plot(times[t_idxs], ndimage.uniform_filter1d(threshs, size=20), color='r')
plt.legend()
plt.show()
