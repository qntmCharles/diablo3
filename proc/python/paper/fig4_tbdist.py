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
from functions import get_metadata, read_params, get_grid, g2gf_1d, get_index, get_plotindex, compute_F0

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

##### Get grid #####
gxf, gyf, gzf, dzf = get_grid(join(run_dir, 'grid.h5'), md)
gx, gy, gz, dz = get_grid(join(run_dir, 'grid.h5'), md, fractional_grid=False)

##### Get data and time indices #####

with h5py.File(join(save_dir,"mean.h5"), 'r') as f:
    print("Mean keys: %s" % f.keys())
    time_keys = list(f['tb_source'])
    print(time_keys)
    bbins_file = np.array(f['tb_source_bins']['0001'])
    source_dists = np.array([np.array(f['tb_source'][t]) for t in time_keys])
    strat_dists = np.array([np.array(f['tb_strat'][t]) for t in time_keys])
    times = np.array([float(f['tb_source'][t].attrs['Time']) for t in time_keys])
    NSAMP = len(times)

    f.close()

bin_end = int(np.where(bbins_file == -1)[0][0])
source_dists[:, bin_end:] = np.nan
bbins_file[bin_end:] = np.nan
bbins_plot = 0.5*(bbins_file[1:] + bbins_file[:-1])

##### Non-dimensionalisation #####

F0 = compute_F0(save_dir, md, tstart_ind = 6*4, verbose=False, zbot=0.7, ztop=0.95, plot=False)
N = np.sqrt(md['N2'])

T = np.power(N, -1)
L = np.power(F0, 1/4) * np.power(N, -3/4)
B = L * np.power(T, -2)

gx -= md['LX']/2
gxf -= md['LX']/2

gx /= L
gxf /= L

gz /= L
gzf /= L

times /= T

bbins_file /= B
bbins_plot /= B

print(F0)
print(md['b0']*md['r0']*md['r0'])

##### ====================== #####

# Compute time indices
t_inds = list(range(24, NSAMP, 8))
nplot = len(t_inds)
print(t_inds)

tplot = [times[i] for i in t_inds]

print("Plotting at times: ",tplot)

plot_max = 1.6*md['H']
plot_min = 0.95*md['H']

idx_minf = get_plotindex(plot_min, gzf)-1
idx_maxf = get_plotindex(plot_max, gzf)

idx_min = idx_minf
idx_max = idx_maxf+1

############################################################################################################
# Source tracer vs. buoyancy distribution: calculation and diagnostics
############################################################################################################

tstart = 5
tend = 6

start_idx = get_index(tstart, times)
end_idx = get_index(tend, times)

source_dists_avg = []
cols = plt.cm.rainbow(np.linspace(0, 1, end_idx-start_idx))
for i, c in zip(range(start_idx, end_idx), cols):
    area = integrate.trapezoid(source_dists[i,:bin_end-1], bbins_plot[:bin_end-1])
    plt.plot(source_dists[i, :-1]/area, bbins_plot, color=c, alpha=0.5, linestyle=':')
    source_dists_avg.append(source_dists[i, :-1]/area)
    #plt.plot(source_dists[i, :-1], bbins_plot, color=c, alpha=0.5, linestyle=':')

source_dist = np.mean(source_dists_avg, axis=0)

plt.plot(source_dist, bbins_plot, color='k')

plt.xlabel("Tracer (normalised)")
plt.ylabel("Buoyancy")

############################################################################################################
# Tracer vs. buoyancy distribution plots
############################################################################################################

tracer_total = np.sum(strat_dists, axis=1)
print(tracer_total)

##### Set up plot #####
t_cont = 0.005
X, Y = np.meshgrid(gx, gz[idx_min:idx_max])
Xf, Yf = np.meshgrid(gxf, gzf[idx_minf:idx_maxf])
#tcols = plt.cm.OrRd(np.linspace(0,1,nplot+1))[1:]
#cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "blue"])
#tcols = cmap(np.linspace(0,1,nplot))
tcols = plt.cm.viridis(np.linspace(0,1,nplot))

#fig, ax1 = plt.subplots(facecolor=(0.9,0.9,0.9))
fig, ax1 = plt.subplots(figsize=(10,4))

#ax2 = ax1.inset_axes([32, 0.08, 28, 0.06], transform=ax1.transData)
ax2 = ax1.inset_axes([0.55, 0.5, 0.4, 0.45])
ax2.plot(times, tracer_total,color='k')
#ax2.set_facecolor((0.9,0.9,0.9))
ax2.set_xlim(0, times[-1])
ax2.set_ylim(0, tracer_total[-1])
ax2.set_ylabel("total tracer conc.")
ax2.set_xlabel("time")

#ax1.set_facecolor((0.9, 0.9, 0.9))

N = np.sqrt(md['N2'])
F0 = md['b0'] * np.power(md['r0'],2)
t_nd = np.power(N, -1)
b_nd = np.power(N, 5/4) * np.power(F0, 1/4)

ax1.plot(source_dist[1:], bbins_plot[1:], color='k', linestyle='--', label="pre-penetration")

for step,c in zip(t_inds, tcols):
    area = integrate.trapezoid(strat_dists[step,:bin_end-1], bbins_plot[:bin_end-1])
    ax1.plot(strat_dists[step,:-1]/area, bbins_plot, color=c, label = "t={0:.2f}".format(times[step]))

ax1.set_ylabel("buoyancy")
ax1.set_xlabel("tracer concentration (normalised)")
ax1.set_ylim(0, 0.065/B)
ax1.set_xlim(0, 3)
ax1.legend(loc='upper left', bbox_to_anchor=(0.05, 0.98))

plt.tight_layout()
#plt.savefig('/home/cwp29/Documents/papers/draft/figs/tb_dist.png', dpi=300)
#plt.savefig('/home/cwp29/Documents/papers/draft/figs/tb_dist.pdf')
plt.show()
