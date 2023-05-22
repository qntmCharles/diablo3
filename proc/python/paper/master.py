import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits import axes_grid1
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
from functions import get_metadata, read_params, get_grid, g2gf_1d, compute_F0, get_plotindex, get_index
from os.path import join, isfile
from os import listdir

from scipy.interpolate import griddata
from scipy import integrate, ndimage

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"

save = False
show = True

fig_save_dir = '/home/cwp29/Documents/papers/draft/figs/'

##### ---------------------- #####

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Data acquisition
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Get dir locations from param file
base_dir, run_dir, save_dir, version = read_params(params_file)

# Get simulation metadata
md = get_metadata(run_dir, version)

gxf, gyf, gzf, dzf = get_grid(save_dir+"/grid.h5", md)
gx, gy, gz, dz = get_grid(save_dir+"/grid.h5", md, fractional_grid=False)

X, Y = np.meshgrid(gx, gz)
Xf, Yf = np.meshgrid(gxf, gzf)

print("Complete metadata: ", md)

# Get data
with h5py.File(join(save_dir, 'movie.h5'), 'r') as f:
    print("Keys: %s" % f.keys())
    time_keys = list(f['th1_xz'])
    print(time_keys)
    # Get buoyancy data
    th1_xz = np.array([np.array(f['th1_xz'][t]) for t in time_keys])
    th2_xz = np.array([np.array(f['th2_xz'][t]) for t in time_keys])
    NSAMP = len(th1_xz)
    times = np.array([float(f['th1_xz'][t].attrs['Time']) for t in time_keys])

    b_init = np.array(f['th1_xz']['0005'])
    f.close()

with h5py.File(join(save_dir,"mean.h5"), 'r') as f:
    print("Mean keys: %s" % f.keys())
    time_keys = list(f['tb_source'])
    print(time_keys)
    bbins_file = np.array(f['tb_source_bins']['0001'])
    source_dists = np.array([np.array(f['tb_source'][t]) for t in time_keys])
    strat_dists = np.array([np.array(f['tb_strat'][t]) for t in time_keys])
    t_me = np.array([np.array(f['thme02'][t]) for t in time_keys])*md['Ny']*md['Nx']

    new_times = np.array([float(f['tb_source'][t].attrs['Time']) for t in time_keys])
    assert np.array_equal(times, new_times)
    new_NSAMP = len(times)
    assert new_NSAMP == NSAMP

    f.close()

bin_end = int(np.where(bbins_file == -1)[0][0])
source_dists[:, bin_end:] = np.nan
strat_dists_trunc = strat_dists[:, :bin_end]
bbins_file_trunc = bbins_file[:bin_end]
bbins_file[bin_end:] = np.nan
bbins_plot = 0.5*(bbins_file[1:] + bbins_file[:-1])

centreline_b = b_init[:,int(md['Nx']/2)]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Non-dimensionalisation
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

F0 = compute_F0(save_dir, md, tstart_ind = 6*4, verbose=False, zbot=0.7, ztop=0.95, plot=False)
N = np.sqrt(md['N2'])

T = np.power(N, -1)
L = np.power(F0, 1/4) * np.power(N, -3/4)
B = L * np.power(T, -2)
U = L / T

X -= md['LX']/2
X /= L
Y -= md['H']
Y /= L

Xf -= md['LX']/2
Xf /= L
Yf -= md['H']
Yf /= L

gz -= md['H']
gzf -= md['H']
gz /= L
gzf /= L

times /= T

th1_xz /= B

bbins_file /= B
bbins_plot /= B

print("Prescribed forcing: ", F0)
print("Computed forcing: ", md['b0']*md['r0']*md['r0'])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Set-up
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print("Total time steps: %s"%NSAMP)
print("Dimensional times: ",times)

steps = [24, 40, 56]
labels = ["a", "b", "c"]

tracer_thresh = 9e-4

contours_b = np.linspace(0, md['N2']*9*L/B, 16)
print(contours_b)

plot_max = 1.6*md['H']
plot_min = 0.95*md['H']

idx_minf = get_plotindex(plot_min, gzf)-1
idx_maxf = get_plotindex(plot_max, gzf)
idx_min = idx_minf
idx_max = idx_maxf+1

hmap_plot_max = 1.45*md['H']
hmap_plot_min = 0.99*md['H']

hmap_idx_minf = get_plotindex((hmap_plot_min - md['H'])/L, gzf)-1
hmap_idx_maxf = get_plotindex((hmap_plot_max - md['H'])/L, gzf)
hmap_idx_min = hmap_idx_minf
hmap_idx_max = hmap_idx_maxf+1

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Figure 2:
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig2, axs2 = plt.subplots(1,3,figsize=(12, 4), constrained_layout=True)

th1_xz = np.where(th1_xz < 1e-3/B, 0, th1_xz)

plot_env = np.where(th2_xz <= tracer_thresh, th1_xz, np.NaN)
plot_plume = np.where(th2_xz > tracer_thresh, th2_xz, np.NaN)
plot_outline = np.where(th2_xz <= tracer_thresh, 1, 0)

for i in range(len(steps)):
    im_env = axs2[i].contour(Xf, Yf, plot_env[steps[i]], levels=contours_b, cmap='cool', alpha=0.8)
    contours_env = axs2[i].contourf(im_env, levels=contours_b, cmap='cool', alpha=0.8, extend='min')

    im_plume = axs2[i].pcolormesh(X, Y, plot_plume[steps[i]], cmap='viridis')

    contour_interface = axs2[i].contour(Xf, Yf, plot_env[steps[i]], levels=[0], cmap='gray', alpha=0.5)

    axs2[i].set_aspect(1)

    if i == len(steps)-1:
        cb_env = fig2.colorbar(contours_env, ax = axs2[i], location='right', shrink=0.7, label=r"buoyancy")
        cb_plume = fig2.colorbar(im_plume, ax = axs2[i], location='right', shrink=0.7, label="tracer concentration",
                extend='max')
        cb_env.add_lines(contour_interface)


    im_plume.set_clim(0, 5e-2)

    if i == 0:
        axs2[i].set_ylabel("$z$")

    axs2[i].set_xlim(-0.15/L, 0.15/L)
    axs2[i].set_ylim(np.min(Y), 9)

    axs2[i].set_xlabel("$x$")

    axs2[i].set_title("({0}) t = {1:.0f}".format(labels[i], times[steps[i]]))

if save:
    fig2.savefig(join(fig2_save_dir, 'evolution.png'), dpi=300)
    fig2.savefig(join(fig2_save_dir, 'evolution.pdf'))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Figure 3:
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig3 = plt.figure(figsize=(7, 4))

X_v, Y_v = np.meshgrid(times[:]+md['SAVE_STATS_DT']/2, gz[100:])
Xf_v, Yf_v = np.meshgrid(0.5*(times[1:]+times[:-1]), gzf[100:])

tracer_data_vert = np.where(th2_xz[1:, 100:, int(md['Nx']/2)] >= tracer_thresh,
        th2_xz[1:, 100:, int(md['Nx']/2)], 0)
plume_vert = np.where(tracer_data_vert >= tracer_thresh, 1, 0)

im = plt.pcolormesh(X_v, Y_v, np.swapaxes(tracer_data_vert,0,1), cmap='viridis')
im.set_clim(0,0.05)

print(np.max(tracer_data_vert))
cont = plt.contour(X_v[1:, 1:] + 0.5*md['LX']/md['Nx'],
        Y_v[1:, 1:] + 0.5*md['LY']/md['Ny'],
        np.swapaxes(tracer_data_vert,0,1), levels=[tracer_thresh], colors=['r'],
        linestyles='--')

cbar = plt.colorbar(im, extend='max', label='tracer concentration')
cbar.add_lines(cont)

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

if save:
    fig3.savefig(join(fig_save_dir, 'timeseries.png'), dpi=300)
    fig3.savefig(join(fig_save_dir, 'timeseries.pdf'))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Figure 4:
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig4_a = plt.figure()

# Compute time indices
t_inds = list(range(24, NSAMP, 8))
nplot = len(t_inds)
print(t_inds)

# Source tracer vs. buoyancy distribution: calculation and diagnostics
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

source_dist = np.mean(source_dists_avg, axis=0)

plt.plot(source_dist, bbins_plot, color='k')

plt.xlabel("Tracer (normalised)")
plt.ylabel("Buoyancy")

# Tracer vs. buoyancy distribution plots

tracer_total = np.sum(strat_dists, axis=1)

tcols = plt.cm.viridis(np.linspace(0,1,nplot))

fig4, ax4 = plt.subplots(figsize=(10,4))

ax4_inset = ax4.inset_axes([0.55, 0.5, 0.4, 0.45])
ax4_inset.plot(times, tracer_total,color='k')
ax4_inset.set_xlim(0, times[-1])
ax4_inset.set_ylim(0, tracer_total[-1])
ax4_inset.set_ylabel("total tracer conc.")
ax4_inset.set_xlabel("time")

ax4.plot(source_dist[1:], bbins_plot[1:], color='k', linestyle='--', label="pre-penetration")

for step,c in zip(t_inds, tcols):
    area = integrate.trapezoid(strat_dists[step,:bin_end-1], bbins_plot[:bin_end-1])
    ax4.plot(strat_dists[step,:-1]/area, bbins_plot, color=c, label = "t={0:.2f}".format(times[step]))

ax4.set_ylabel("buoyancy")
ax4.set_xlabel("tracer concentration (normalised)")
ax4.set_ylim(0, 0.065/B)
ax4.set_xlim(0, 3)
ax4.legend(loc='upper left', bbox_to_anchor=(0.05, 0.98))

plt.tight_layout()

if save:
    fig4.savefig(join(fig_save_dir, 'tb_dist.png'), dpi=300)
    fig4.savefig(join(fig_save_dir, 'tb_dist.pdf'))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Figure 5: tracer vs. (buoyancy, height) heatmap
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

bt_pdfs = strat_dists_trunc[:,:-1]
zt_pdfs = t_me[:, hmap_idx_minf:hmap_idx_maxf+1]

bt_pdfs = np.swapaxes(bt_pdfs, axis1=1, axis2=0)
zt_pdfs = np.swapaxes(zt_pdfs, axis1=1, axis2=0)

print(bt_pdfs.shape)

Tz, Z = np.meshgrid(np.append(times, times[-1]+md['SAVE_STATS_DT']), gz[hmap_idx_min:hmap_idx_max+1])
Tzf, Zf = np.meshgrid(times+md['SAVE_STATS_DT']/2, gzf[hmap_idx_min:hmap_idx_max])
Tb, Bp = np.meshgrid(np.append(times, times[-1]+md['SAVE_STATS_DT']), bbins_file_trunc)
Tbf, Bf = np.meshgrid(times+md['SAVE_STATS_DT']/2, 0.5*(bbins_file_trunc[1:]+bbins_file_trunc[:-1]))

fig5, ax5 = plt.subplots(1,2, figsize=(12, 3.5), constrained_layout=True)

bt_im = ax5[0].pcolormesh(Tb, Bp, bt_pdfs, shading='flat', cmap='viridis')
zt_im = ax5[1].pcolormesh(Tz, Z, zt_pdfs, shading='flat', cmap='viridis')

print(centreline_b[hmap_idx_min])
print(hmap_idx_min)
ax5[0].set_ylim(centreline_b[hmap_idx_min]/B, centreline_b[hmap_idx_max]/B)
ax5[1].set_ylim((hmap_plot_min-md['H'])/L, (hmap_plot_max-md['H'])/L)

bt_cont = ax5[0].contour(Tbf, Bf, bt_pdfs, levels=[1e1], colors='r', alpha=0.7)
bt_cbar = plt.colorbar(bt_im, ax=ax5[0], label="tracer conc.")
bt_cbar.add_lines(bt_cont)

zt_cbar = plt.colorbar(zt_im, ax=ax5[1], label="tracer conc.")

t_step = int(round(2.0 / md['SAVE_STATS_DT'], 0))
for i in range(0, NSAMP, t_step):
    ax5[0].axvline(times[i], linestyle='--', color='gray', alpha=0.5)
    ax5[1].axvline(times[i], linestyle='--', color='gray', alpha=0.5)

ax5[0].set_xlabel("t")
ax5[0].set_ylabel("buoyancy")
ax5[1].set_xlabel("t")
ax5[1].set_ylabel("z")

ax5[0].set_xlim(4/T, 15/T)
ax5[1].set_xlim(4/T, 15/T)

if save:
    fig5.savefig(join(fig_save_dir, 'hmap.png'), dpi=300)
    fig5.savefig(join(fig_save_dir, 'hmap.pdf'))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if show:
    plt.show()
