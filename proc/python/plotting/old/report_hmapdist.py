import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py, gc, sys
import numpy as np
from scipy.interpolate import griddata
from scipy import integrate
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
dirs = ["run1", "run2", "run3", "run4", "run5", "run6"]
hmap_dir = "run4"
save = False

##### ----------------------- #####

##### Get directory locations #####
base_dir, run_dir, save_dir, version = read_params(params_file)
print("Run directory: ", run_dir)
print("Save directory: ", save_dir)

##### Get simulation metadata #####
md = get_metadata(join(run_dir,hmap_dir), version)
print("Complete metadata: ",md)

##### Get grid #####
gxf, gyf, gzf, dzf = get_grid(join(join(run_dir,hmap_dir), 'grid.h5'), md)
gx, gy, gz, dz = get_grid(join(join(run_dir,hmap_dir), 'grid.h5'), md, fractional_grid=False)

z_coords, x_coords = np.meshgrid(gzf, gxf, indexing='ij', sparse=True)


##### Get data #####

with h5py.File(join(save_dir,hmap_dir,"movie.h5"), 'r') as f:
    print("Movie keys: %s" % f.keys())
    time_keys = list(f['th1_xz'])
    print(time_keys)
    # Get buoyancy data
    b_h = np.array([np.array(f['th1_xz'][tstep]) for tstep in time_keys])
    t_h = np.array([np.array(f['th2_xz'][tstep]) for tstep in time_keys])
    b_h = g2gf_1d(md, b_h)
    t_h = g2gf_1d(md, t_h)
    NSAMP = len(b_h)
    #times = np.array([float(tstep)*md['SAVE_MOVIE_DT'] for tstep in time_keys])
    times_h = np.array([float(f['th1_xz'][tstep].attrs['Time']) for tstep in time_keys])
    f.close()

with h5py.File(join(save_dir,hmap_dir,"mean.h5"), 'r') as f:
    print("Mean keys: %s" % f.keys())
    mtime_keys = list(f['thme01'])
    print(mtime_keys)
    # Get data
    bme = np.array([np.array(f['thme01'][tstep]) for tstep in mtime_keys])
    bme = g2gf_1d(md, bme)
    mNSAMP = len(bme)
    assert mNSAMP == NSAMP
    f.close()

b = []
t = []
for d in dirs:
    with h5py.File(join(save_dir,d,"movie.h5"), 'r') as f:
        time_keys = list(f['th1_xz'])
        # Get buoyancy data
        b.append(np.array([np.array(f['th1_xz'][tstep]) for tstep in time_keys]))
        t.append(np.array([np.array(f['th2_xz'][tstep]) for tstep in time_keys]))
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
fig, ax1 = plt.subplots(figsize=(9,4))

ax2 = ax1.inset_axes([32, 0.08, 28, 0.06], transform=ax1.transData)
ax2.plot(times, tracer_total,color='b')
#ax2.set_facecolor((0.9,0.9,0.9))
ax2.set_xlim(0, times[-1])
ax2.set_ylim(0, tracer_total[-1])
ax2.set_ylabel("total tracer")
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
fig.savefig('/home/cwp29/Documents/4report/figs/tb_dist.png', dpi=200)

############################################################################################################
# Tracer vs. buoyancy heatmap
############################################################################################################

# b contains values of buoyancy at each point
# z_coords contains z value at each point
# t contains tracer value at each point
## Aim is to bin tracer values wrt buoyancy

# Create buoyancy bins
b_min = 0
b_max = np.max(b_h[0,:,int(md['Nx']/2)])/5
nbins = 129

check_data = b_h[0,:,int(md['Nx']/2)]
plot_min = -1
plot_max = -1
for j in range(md['Nz']):
    if gzf[j] < md['H'] and gzf[j+1] > md['H']:
        plot_min = j
    if check_data[j-1] <= b_max and check_data[j] >= b_max:
        plot_max = j

if plot_min == -1: print("Warning: plot_min miscalculated")
if plot_max == -1: print("Warning: plot_max miscalculated")

bbins, db = np.linspace(b_min, b_max, nbins, retstep=True)

print(np.append(times_h, times_h[-1]+md['SAVE_STATS_DT']))
print(len(np.append(times_h, times_h[-1]+md['SAVE_STATS_DT'])))
X, Y = np.meshgrid(np.append(times_h, times_h[-1]+md['SAVE_STATS_DT']), bbins)

bt_pdfs = []
zt_pdfs = []
zb_pdfs = []

dz = gzf[1+plot_min:plot_max] - gzf[plot_min:plot_max-1]

for i in range(NSAMP):
    zb_pdfs.append(compute_pdf(gzf[plot_min:plot_max], bme[i][plot_min:plot_max], gzf[plot_min:plot_max],
        ))
    bt_pdfs.append(compute_pdf(b_h[i][plot_min:plot_max], t_h[i][plot_min:plot_max], bbins))
    zt_pdfs.append(compute_pdf(z_coords[plot_min:plot_max], t_h[i][plot_min:plot_max], gzf[plot_min:plot_max],
        ))

bt_pdfs = np.array(bt_pdfs)
zt_pdfs = np.array(zt_pdfs)
zb_pdfs = np.array(zb_pdfs)

print(bt_pdfs.shape)

bt_pdfs = np.swapaxes(bt_pdfs, axis1=1, axis2=0)
zt_pdfs = np.swapaxes(zt_pdfs, axis1=1, axis2=0)

print(bt_pdfs.shape)
times_h = times_h[:80]
bt_pdfs = bt_pdfs[:,:80]
zt_pdfs = zt_pdfs[:,:80]
print(bt_pdfs.shape)

X, Y = np.meshgrid(np.append(times_h, times_h[-1]+md['SAVE_STATS_DT']), gz[plot_min:plot_max])
cols = plt.cm.rainbow(np.linspace(0, 1, NSAMP))
fig = plt.figure()
for pdf,c in zip(zb_pdfs, cols):
    plt.plot(pdf, gzf[plot_min:plot_max-1], color=c)

plt.xlim(0, b_max)
plt.ylim(gzf[plot_min], gzf[plot_max])
plt.xlabel("buoyancy")
plt.ylabel("height")

bfig = plt.figure()
plt.plot(b_h[0,:,0], gzf)

fig, ax = plt.subplots(1,2, figsize=(12, 3.5), constrained_layout=True)

X, Y = np.meshgrid(np.append(times_h, times_h[-1]+md['SAVE_STATS_DT']), bbins)
bt_im = ax[0].pcolormesh(X, Y, bt_pdfs, shading='flat', cmap=plt.cm.get_cmap('viridis'))
X, Y = np.meshgrid(times_h+md['SAVE_STATS_DT']/2, (bbins[1:]+bbins[:-1])/2)
bt_cont = ax[0].contour(X, Y, bt_pdfs, levels=[0.4], colors='r', alpha=0.7)

bt_cbar = plt.colorbar(bt_im, ax=ax[0], label="tracer conc. (arbitary units)")
bt_cbar.add_lines(bt_cont)

X, Y = np.meshgrid(np.append(times_h, times_h[-1]+md['SAVE_STATS_DT']), gz[plot_min:plot_max])
zt_im = ax[1].pcolormesh(X, Y, zt_pdfs, cmap=plt.cm.get_cmap('viridis'))
plt.colorbar(zt_im, ax=ax[1], label="tracer conc. (arbitary units)")

t_step = int(round(5.0 / md['SAVE_STATS_DT'], 0))
for i in range(0, 80, t_step):
    ax[0].axvline(times_h[i], linestyle='--', color='gray', alpha=0.5)
    ax[1].axvline(times_h[i], linestyle='--', color='gray', alpha=0.5)

ax[0].set_title("tracer vs. buoyancy heatmap")
ax[0].set_xlabel("time (s)")
ax[0].set_ylabel(r"buoyancy ($m\,s^{-2}$)")
ax[1].set_title("tracer vs. height heatmap")
ax[1].set_xlabel("time (s)")
ax[1].set_ylabel("height (m)")

#plt.tight_layout()
fig.savefig('/home/cwp29/Documents/essay/figs/hmap.png', dpi=300)
plt.show()
