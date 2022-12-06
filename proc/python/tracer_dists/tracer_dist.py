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

##### ----------------------- #####

##### Get directory locations #####
base_dir, run_dir, save_dir, version = read_params(params_file)
print("Run directory: ", run_dir)
print("Save directory: ", save_dir)

##### Get simulation metadata #####
md = get_metadata(run_dir, version)
print("Complete metadata: ",md)

# Calculate turnover time
F = md['b0'] * (md['r0']**2)
tau = md['r0']**(4/3) * F**(-1/3)
print("Non-dimensional turnover time: {0:.04f}".format(tau))

##### Get grid #####
gxf, gyf, gzf, dzf = get_grid(join(run_dir, 'grid.h5'), md)
gx, gy, gz, dz = get_grid(join(run_dir, 'grid.h5'), md, fractional_grid=False)

##### Get data and time indices #####
with h5py.File(join(save_dir,"mean.h5"), 'r') as f:
    print("Mean keys: %s" % f.keys())
    time_keys = list(f['bbins'])
    print(time_keys)
    bbins_file = np.array([np.array(f['bbins'][t]) for t in time_keys])
    source_dists = np.array([np.array(f['tb_source'][t]) for t in time_keys])
    strat_dists = np.array([np.array(f['tb_strat'][t]) for t in time_keys])
    times = np.array([float(f['tb_source'][t].attrs['Time']) for t in time_keys])

    f.close()

with h5py.File(join(save_dir,"movie.h5"), 'r') as f:
    print("Movie keys: %s" % f.keys())
    time_keys = list(f['th1_xz'])
    print(time_keys)
    # Get buoyancy data
    b = np.array([np.array(f['th1_xz'][tstep]) for tstep in time_keys])
    t = np.array([np.array(f['th2_xz'][tstep]) for tstep in time_keys])
    b = g2gf_1d(md, b)
    t = g2gf_1d(md, t)
    NSAMP = len(b)
    #times = np.array([float(tstep)*md['SAVE_MOVIE_DT'] for tstep in time_keys])
    times = np.array([float(f['th1_xz'][tstep].attrs['Time']) for tstep in time_keys])
    f.close()

# Compute time indices
t_inds = list(range(10, NSAMP, 10))
nplot = len(t_inds)
print(t_inds)

tplot = [times[i] for i in t_inds]

print("Plotting at times: ",tplot)

print(bbins_file.shape)

############################################################################################################
# Source tracer vs. buoyancy distribution: calculation and diagnostics
############################################################################################################

# Create buoyancy bins
centreline_b = b[0,:,int(md['Nx']/2)]
sim_bins = np.max(centreline_b[1:]-centreline_b[:-1])

b_min = 0
factor = 8
b_max = np.max(centreline_b)/factor

nbins = int(np.floor((b_max-b_min)/sim_bins))

plot_min = -1
plot_max = -1
for j in range(md['Nz']):
    if gzf[j] < md['H'] and gzf[j+1] > md['H']:
        plot_min = j
    if centreline_b[j-1] <= b_max and centreline_b[j] >= b_max:
        plot_max = j

if plot_min == -1: print("Warning: plot_min miscalculated")
if plot_max == -1: print("Warning: plot_max miscalculated")

bin_end = int(np.where(bbins_file == -1)[1][0])
bbins = bbins_file[0, :bin_end]
bbins_plot = 0.5*(bbins[1:] + bbins[:-1])

source_dists = source_dists[:, :bin_end-1]
strat_dists = strat_dists[:, :bin_end-1]

##### Compute 'source' PDF #####

source_dist_fig = plt.figure()

# First get data we want to process
top = md['H']
depth = md['H']-0.05

top_idx = get_index(top, gzf)
bottom_idx = get_index(top-depth, gzf)

# Averaging period (s)
t_start = 5
t_end = 10
start_idx = get_index(t_start, times)
end_idx = get_index(t_end, times)

cols = plt.cm.rainbow(np.linspace(0, 1, end_idx-start_idx))
for i, c in zip(range(start_idx, end_idx), cols):
    area = integrate.trapezoid(np.abs(source_dists[i-start_idx]), bbins_plot)
    plt.plot(source_dists[i-start_idx]/area, bbins_plot, color=c, alpha=0.5, linestyle=':')


source_dist = np.nanmean(source_dists[start_idx:end_idx, :], axis=0)
area = integrate.trapezoid(np.abs(source_dist), bbins_plot)
source_dist /= area

plt.plot(source_dist, bbins_plot, color='k', linestyle='--')
plt.xlabel("Tracer (normalised)")
plt.ylabel("Buoyancy")

##### Plot buoyancy and tracer fields with source region #####

source_bt_fig, ax = plt.subplots(1,2, figsize=(10, 3))
Xplot, Yplot = np.meshgrid(gx, gz[:plot_max+1])

ax[0].pcolormesh(Xplot, Yplot, np.mean(b[:,:plot_max],axis=0))
ax[1].pcolormesh(Xplot, Yplot, np.mean(t[:,:plot_max],axis=0))

ax[0].set_title("Buoyancy field")
ax[1].set_title("Tracer field")
ax[0].axhline(top, color='r', linestyle='--')
ax[1].axhline(top, color='r', linestyle='--')
ax[0].axhline(top-depth, color='r', linestyle='--')
ax[1].axhline(top-depth, color='r', linestyle='--')
plt.tight_layout()


##### Generate tracer vs. buoyancy heatmap

"""
source_hmap_fig, ax = plt.subplots(1,2, figsize=(15,5))

bt_dists = []
bt_dists_source = []
bt_dists_source_var = []

for i in range(1,NSAMP):
    bt_dists.append(compute_pdf(b[i][plot_min:plot_max], t[i][plot_min:plot_max], bbins, normalised=True))
    bt_dists_source.append(compute_pdf(b_source[i], t_source[i], bbins, normalised=True))

Xhmap, Yhmap = np.meshgrid(np.append(times[1:], times[-1]+md['SAVE_STATS_DT']), bbins)

bt_dists = np.moveaxis(bt_dists, 0, 1)

for i in range(len(bt_dists_source)):
    bt_dists_source_var.append(bt_dists_source[i] - source_dist)

bt_dists_source = np.moveaxis(bt_dists_source, 0, 1)
bt_dists_source_var = np.moveaxis(bt_dists_source_var, 0, 1)


bt_hmap = ax[0].pcolormesh(Xhmap, Yhmap, bt_dists_source, shading='flat')
ax[0].axvline(t_start, color='r', linestyle='--')
ax[0].axvline(t_end, color='r', linestyle='--')
bt_hmap.set_clim(0, 40)
ax[0].set_title("Tracer vs. buoyancy heatmap")
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Buoyancy")

bt_hmap_var = ax[1].pcolormesh(Xhmap, Yhmap, bt_dists_source_var, shading='flat')
ax[1].axvline(t_start, color='r', linestyle='--')
ax[1].axvline(t_end, color='r', linestyle='--')
bt_hmap_var.set_clim(0, 10)
ax[1].set_title("Tracer vs. buoyancy heatmap with source distribution subtracted")
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Buoyancy")

plt.tight_layout()
"""

############################################################################################################
# z_max, z_ss and z_eq calculation
############################################################################################################

############################################################################################################
# Tracer vs. buoyancy distribution plots
############################################################################################################

##### Set up plot #####
t_cont = 0.0005
X, Y = np.meshgrid(gx, gz[plot_min:plot_max+1])
Xf, Yf = np.meshgrid(gxf, gzf[plot_min-1:plot_max+1])
tcols = plt.cm.OrRd(np.linspace(0,1,nplot+1))[1:]

#fig = plt.figure()
fig, ax = plt.subplots(1,2, figsize=(15, 5))

ax[0].pcolormesh(X, Y, b[0][plot_min:plot_max], cmap=plt.cm.get_cmap('jet'), alpha=0.3)
ax[1].plot(source_dist, bbins_plot, color='k', linestyle='--')

for step,c in zip(t_inds, tcols):
    ax[0].contour(Xf, Yf, t[step][plot_min-1:plot_max+1], levels=[t_cont], colors=[c])
    area = integrate.trapezoid(np.abs(strat_dists[step]), bbins_plot)
    ax[1].plot(strat_dists[step]/area, bbins_plot, color=c, label = "t={0:.3f} s".format(times[step]))

ax[1].axvline(0, color='k')

ax[1].set_ylim(b[0][plot_min][0], b[0][plot_max][0])
ax[0].set_ylim(gz[plot_min], gz[plot_max])
plt.xlabel("tracer (arbitrary units, normalised)")
plt.ylabel("buoyancy ($m \, s^{{-2}}$)")
plt.legend()

plt.tight_layout()
plt.show()
