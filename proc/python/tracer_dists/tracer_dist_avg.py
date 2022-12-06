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
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)

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
dirs = ["run1", "run2", "run3", "run4", "run5", "run6"]

save = False

##### ----------------------- #####

##### Get directory locations #####
base_dir, run_dir, save_dir, version = read_params(params_file)
print("Run directory: ", run_dir)
print("Save directory: ", save_dir)

##### Get simulation metadata #####
md = get_metadata(join(run_dir,dirs[0]), version)
print("Complete metadata: ",md)

# Calculate turnover time
F = md['b0'] * (md['r0']**2)
tau = md['r0']**(4/3) * F**(-1/3)
print("Non-dimensional turnover time: {0:.04f}".format(tau))

##### Get grid #####
gxf, gyf, gzf, dzf = get_grid(join(run_dir, dirs[0], 'grid.h5'), md)
gx, gy, gz, dz = get_grid(join(run_dir, dirs[0], 'grid.h5'), md, fractional_grid=False)

##### Get data and time indices #####

b = []
t = []
for d in dirs:
    with h5py.File(join(save_dir,d,"movie.h5"), 'r') as f:
        time_keys = list(f['th1_xz'])
        # Get buoyancy data
        b.append(g2gf_1d(md,np.array([np.array(f['th1_xz'][tstep]) for tstep in time_keys])))
        t.append(g2gf_1d(md,np.array([np.array(f['th2_xz'][tstep]) for tstep in time_keys])))
        NSAMP = len(b)
        times = np.array([float(f['th1_xz'][tstep].attrs['Time']) for tstep in time_keys])
        print(times)
        f.close()

b = np.array(b)
t = np.array(t)


# Compute time indices
nplot = 5
interval = 8.7

step = np.round(interval/(np.sqrt(md['N2'])*md['SAVE_STATS_DT']))

t_inds = list(map(int,step*np.array(range(1, nplot+1))))

tplot = [times[i] for i in t_inds]

print("Plotting at times: ",tplot)
print("with interval", step*md['SAVE_STATS_DT'])


############################################################################################################
# Source tracer vs. buoyancy distribution: calculation and diagnostics
############################################################################################################

# Create buoyancy bins
centreline_b = b[0,0,:,int(md['Nx']/2)]
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

source_dist_fig = plt.figure(figsize=(8, 5))

# First get data we want to process
top = 1.000*md['H']
depth = 2*md['r0']

top_idx = get_index(top, gzf)
bottom_idx = get_index(top-depth, gzf)

b_source = b[:,:, bottom_idx:top_idx]
t_source = t[:,:, bottom_idx:top_idx]

# Averaging period (s)
t_start = 5
t_end = 10
start_idx = get_index(t_start, times)
end_idx = get_index(t_end, times)

source_dists = []
cols = plt.cm.rainbow(np.linspace(0,1,len(dirs)))
for d,c in zip(range(len(dirs)),cols):
    source_dist = []
    for i in range(start_idx, end_idx):
        source_dist.append(compute_pdf(b_source[d,i], t_source[d,i], bbins, normalised=True))
        #plt.plot(source_dist[i-start_idx], bbins_plot, color=c, alpha=0.5)

    source_dist = np.nanmean(source_dist, axis=0)
    source_dists.append(source_dist)
    plt.plot(source_dist, bbins_plot, color=c, linestyle='--')

source_dist = np.mean(source_dists,axis=0) # get average source dist for all simulations
plt.plot(source_dist, bbins_plot, color='k')
plt.xlabel("Tracer (normalised)")
plt.ylabel("Buoyancy")

############################################################################################################
# Tracer vs. buoyancy distribution plots
############################################################################################################

tracer_total = np.mean(np.sum(t[:,:,plot_min:plot_max], axis=(2,3)),axis=0)
print(tracer_total)

##### Set up plot #####
t_cont = 0.005
X, Y = np.meshgrid(gx, gz[plot_min:plot_max+1])
Xf, Yf = np.meshgrid(gxf, gzf[plot_min:plot_max])
#tcols = plt.cm.OrRd(np.linspace(0,1,nplot+1))[1:]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "blue"])
tcols = cmap(np.linspace(0,1,nplot))

#fig, ax1 = plt.subplots(facecolor=(0.9,0.9,0.9))
fig, ax1 = plt.subplots(figsize=(8,5))

ax2 = ax1.inset_axes([32, 0.08, 28, 0.06], transform=ax1.transData)
ax2.plot(times, tracer_total,color='b')
#ax2.set_facecolor((0.9,0.9,0.9))
ax2.set_xlim(0, times[-1])
ax2.set_ylim(0, tracer_total[-1])
ax2.set_ylabel("total tracer (arbitrary units)")
ax2.set_xlabel("time")

#ax1.set_facecolor((0.9, 0.9, 0.9))

N = np.sqrt(md['N2'])
F0 = md['b0'] * np.power(md['r0'],2)
t_nd = np.power(N, -1)
b_nd = np.power(N, 5/4) * np.power(F0, 1/4)

ax1.plot(source_dist, bbins_plot, color='k', linestyle='--', label="pre-penetration")

for step,c in zip(t_inds, tcols):
    b_pdfs = []
    for d in range(len(dirs)):
        b_pdf = compute_pdf(b[d,step,plot_min:plot_max], t[d,step,plot_min:plot_max], bbins,
            normalised=True)
        b_pdfs.append(b_pdf)
        #plt.plot(b_pdf, bbins_plot/b_nd, color=c, alpha=0.5, linestyle='--')

    ax1.plot(np.mean(b_pdfs,axis=0), bbins_plot, color=c, label = "t={0:.2f}".format(times[step]))

ax1.set_ylabel("buoyancy")
ax1.set_xlabel("tracer (arbitrary units, normalised)")
ax1.set_ylim(0, 0.15)
ax1.set_xlim(0, 62)
#ax1.legend(loc='upper left', facecolor=(0.9,0.9,0.9), bbox_to_anchor=(0.1, 0.98))
ax1.legend(loc='upper left', bbox_to_anchor=(0.05, 0.98))

plt.tight_layout()
#plt.savefig('/home/cwp29/Documents/posters/issf2/tb_dist_gray.png', dpi=200)
plt.savefig('/home/cwp29/Documents/4report/figs/tb_dist.png', dpi=200)
plt.show()
