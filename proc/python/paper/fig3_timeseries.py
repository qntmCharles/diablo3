import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from os.path import isfile, join
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from functions import get_metadata, get_grid, read_params, get_az_data, get_index, compute_F0
from itertools import groupby
from scipy import integrate, optimize, interpolate
from datetime import datetime

##### USER-DEFINED VARIABLES #####
params_file = "./params.dat"

z_upper = 95
z_lower = 20

eps = 0.02

##################################

base_dir, run_dir, save_dir, version = read_params(params_file)
md = get_metadata(run_dir, version)
gxf, gyf, gzf, dzf = get_grid(join(run_dir, 'grid.h5'), md)
gx, gy, gz, dz = get_grid(save_dir+"/grid.h5", md, fractional_grid=False)
z_coords, x_coords = np.meshgrid(gzf, gxf, indexing='ij', sparse=True)

# Get buoyancy data

X, Y = np.meshgrid(gx, gz)
Xf, Yf = np.meshgrid(gxf, gzf)

# Get data
with h5py.File(save_dir+"/movie.h5", 'r') as f:
    #print("Keys: %s" % f.keys())
    time_keys = list(f['th1_xz'])
    #print(time_keys)
    th1_xz = np.array([np.array(f['th1_xz'][t]) for t in time_keys])
    th2_xz = np.array([np.array(f['th2_xz'][t]) for t in time_keys])
    NSAMP = len(th1_xz)
    times = np.array([float(f['th1_xz'][t].attrs['Time']) for t in time_keys])
    f.close()

##### Non-dimensionalisation #####

F0 = compute_F0(save_dir, md, tstart_ind = 6*4, verbose=False, zbot=0.7, ztop=0.95, plot=False)
N = np.sqrt(md['N2'])

T = np.power(N, -1)
L = np.power(F0, 1/4) * np.power(N, -3/4)
B = L * np.power(T, -2)
U = L / T

gz -= md['H']
gzf -= md['H']
gz /= L
gzf /= L

times /= T

th1_xz /= B

print(F0)
print(md['b0']*md['r0']*md['r0'])

##### ====================== #####

#############################################################################################################

fig = plt.figure(figsize=(7, 4))

tracer_thresh = 5e-4

X_v, Y_v = np.meshgrid(times[:]+md['SAVE_STATS_DT']/2, gz[100:])
Xf_v, Yf_v = np.meshgrid(0.5*(times[1:]+times[:-1]), gzf[100:])

tracer_data_vert = np.where(th2_xz[1:, 100:, int(md['Nx']/2)] >= tracer_thresh,
        th2_xz[1:, 100:, int(md['Nx']/2)], 0)
plume_vert = np.where(tracer_data_vert >= tracer_thresh, 1, 0)

im = plt.pcolormesh(X_v, Y_v, np.swapaxes(tracer_data_vert,0,1), cmap='viridis')
im.set_clim(0,0.1)
print(np.max(tracer_data_vert))
plt.contour(X_v[1:, 1:] + 0.5*md['LX']/md['Nx'],
        Y_v[1:, 1:] + 0.5*md['LY']/md['Ny'],
        np.swapaxes(tracer_data_vert,0,1), levels=[tracer_thresh], colors=['r'],
        linestyles='--')

#Calculate zmax:
heights = []
gzf = gzf[100:]
for i in range(len(plume_vert)):
    stuff = np.where(plume_vert[i] == 1)[0]
    if len(stuff) == 0:
        heights.append(0)
    else:
        heights.append(gzf[np.max(stuff)+1])

zmax = np.max(heights)
zss = np.mean(heights[11 * 4:])

print(zmax, zss)
plt.axhline(zmax, color='white', linewidth=1)
plt.axhline(zss, color='white', linewidth=1)
plt.text(2, zmax + .7, r"$z_{\max}$", fontsize='large', color='w')
plt.text(2, zss - 1.7, r"$z_{ss}$", fontsize='large', color='w')

plt.ylim(gz[100], 0.15/L)
plt.xlim(2/T, 15.125/T)
plt.xlabel("t")
plt.ylabel("z")

plt.tight_layout()
plt.savefig('/home/cwp29/Documents/papers/draft/figs/timeseries.png', dpi=300)
plt.savefig('/home/cwp29/Documents/papers/draft/figs/timeseries.pdf')
plt.show()
