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
from functions import get_metadata, read_params, get_grid

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
#out_file = "out.002241.h5"
out_file = "end.h5"

save = False
use_3d = False

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
if use_3d:
    with h5py.File(join(save_dir, out_file), 'r') as f:
        times = np.array([f['Timestep'].attrs['Time']])
        b_3d = np.array([f['Timestep']['TH1']])
        t_3d = np.array([f['Timestep']['TH2']])

with h5py.File(join(save_dir,"movie.h5"), 'r') as f:
    print("Movie keys: %s" % f.keys())
    time_keys = list(f['th1_xz'])
    print(time_keys)
    # Get buoyancy data
    b = np.array([np.array(f['th1_xz'][tstep]) for tstep in time_keys])
    t = np.array([np.array(f['th2_xz'][tstep]) for tstep in time_keys])
    NSAMP = len(b)
    #times = np.array([float(tstep)*md['SAVE_MOVIE_DT'] for tstep in time_keys])
    times = np.array([float(f['th1_xz'][tstep].attrs['Time']) for tstep in time_keys])
    f.close()

# Compute time indices
nplot = 4
interval = 5

step = np.round(interval/(np.sqrt(md['N2'])*md['SAVE_STATS_DT']))

t_inds = list(map(int,step*np.array(range(1, nplot+1))))

tplot = [times[i] for i in t_inds]

print("Plotting at times: ",tplot)
print("with interval", step*md['SAVE_STATS_DT'])


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

bbins = np.linspace(b_min, b_max, nbins)
bbins_data = centreline_b[plot_min:plot_max]
bbins = bbins_data

bbins_plot = 0.5*(bbins[1:] + bbins[:-1])

##### Compute 'source' PDF #####

source_dist_fig = plt.figure()

# First get data we want to process
top = 0.95*md['H']
depth = 10*md['r0']

top_idx = get_index(top, gzf)
bottom_idx = get_index(top-depth, gzf)

if use_3d:
    b_source = b_3d[:, :, bottom_idx:top_idx, :]
    t_source = t_3d[:, :, bottom_idx:top_idx, :]
else:
    b_source = b[:, bottom_idx:top_idx]
    t_source = t[:, bottom_idx:top_idx]

# Averaging period (s)
if use_3d:
    start_idx = 0
    end_idx = 1
else:
    t_start = 5
    t_end = 10
    start_idx = get_index(t_start, times)
    end_idx = get_index(t_end, times)

source_dist = []
cols = plt.cm.rainbow(np.linspace(0, 1, end_idx-start_idx))
for i, c in zip(range(start_idx, end_idx), cols):
    source_dist.append(compute_pdf(b_source[i], t_source[i], bbins, normalised=True))
    plt.plot(source_dist[i-start_idx], bbins_plot, color=c, alpha=0.5)

source_dist = np.nanmean(source_dist, axis=0)
plt.plot(source_dist, bbins_plot, color='k', linestyle='--')
plt.xlabel("Tracer (normalised)")
plt.ylabel("Buoyancy")

##### Plot buoyancy and tracer fields with source region #####

source_bt_fig, ax = plt.subplots(1,2, figsize=(10, 3))
Xplot, Yplot = np.meshgrid(gx, gz[:plot_max+1])

if use_3d:
    ax[0].pcolormesh(Xplot, Yplot, b_3d[0,int(md['Nx']/2),:plot_max])
    ax[1].pcolormesh(Xplot, Yplot, t_3d[0,int(md['Nx']/2),:plot_max])
else:
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

############################################################################################################
# z_max, z_ss and z_eq calculation
############################################################################################################

############################################################################################################
# Tracer vs. buoyancy distribution plots
############################################################################################################

##### Set up plot #####
t_cont = 0.005
X, Y = np.meshgrid(gx, gz[plot_min:plot_max+1])
Xf, Yf = np.meshgrid(gxf, gzf[plot_min:plot_max])
tcols = plt.cm.OrRd(np.linspace(0,1,nplot+1))[1:]

#fig = plt.figure()
fig, ax = plt.subplots(1,2)

ax[0].pcolormesh(X, Y, b[0][plot_min:plot_max], cmap=plt.cm.get_cmap('jet'), alpha=0.3)
ax[1].plot(source_dist, 0.5*(bbins[1:]+bbins[:-1]), color='k', linestyle='--')

for step,c in zip(t_inds, tcols):
    ax[0].contour(Xf, Yf, t[step][plot_min:plot_max], levels=[t_cont], colors=[c])
    b_pdf = compute_pdf(b[step][plot_min:plot_max], t[step][plot_min:plot_max], bbins, normalised=True)

    ax[1].plot(b_pdf, 0.5*(bbins[1:]+bbins[:-1]), color=c, label = "t={0:.3f} s".format(times[step]))

plt.xlabel("tracer (arbitrary units, normalised)")
plt.ylabel("buoyancy ($m \, s^{{-2}}$)")
plt.legend()
plt.tight_layout()

poster_fig = plt.figure(facecolor=(0.9,0.9,0.9))
ax = plt.gca()

ax.set_facecolor((0.9, 0.9, 0.9))

N = np.sqrt(md['N2'])
F0 = md['b0'] * np.power(md['r0'],2)
t_nd = np.power(N, -1)
b_nd = np.power(N, 5/4) * np.power(F0, 1/4)

plt.plot(source_dist, bbins_plot/b_nd, color='k', linestyle='--', label="pre-penetration")
for step, c in zip(t_inds, tcols):
    b_pdf = compute_pdf(b[step][plot_min:plot_max], t[step][plot_min:plot_max], bbins, normalised=True)
    plt.plot(b_pdf, bbins_plot/b_nd, color=c, label="t={0:.2f}".format(times[step]/t_nd))

#blue = (19/256, 72/256, 158/256)
#ax.tick_params(color=blue, labelcolor=blue)
#for spine in ax.spines.values():
#    spine.set_edgecolor(blue)

#l = plt.legend(facecolor=(0.9,0.9,0.9))
#for text in l.get_texts():
#    text.set_color(blue)

plt.legend(facecolor=(0.9,0.9,0.9))
plt.xlim(0, 54)
plt.ylim(0, 2.6)
plt.ylabel("buoyancy (non-dimensionalised)")
plt.xlabel("tracer (arbitrary units)")
plt.tight_layout()
plt.savefig('/home/cwp29/Documents/posters/issf2/tb_dist_gray.png', dpi=200)
plt.show()
