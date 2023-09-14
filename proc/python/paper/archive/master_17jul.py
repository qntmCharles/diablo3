import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
from mpl_toolkits import axes_grid1
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from datetime import datetime
from functions import get_metadata, read_params, get_grid, g2gf_1d, compute_F0, get_plotindex, get_index
from os.path import join, isfile
from os import listdir

from scipy.interpolate import griddata, interp1d
from scipy import integrate, ndimage, spatial

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"

save = False
show = not save

fig_save_dir = '/home/cwp29/Documents/papers/conv_pen/draft2/figs'

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
    # Get buoyancy data
    th1_xz = np.array([np.array(f['th1_xz'][t]) for t in time_keys])
    th2_xz = np.array([np.array(f['th2_xz'][t]) for t in time_keys])

    W = np.array([np.array(f['td_scatter'][t]) for t in time_keys])
    S = np.array([np.array(f['td_flux'][t]) for t in time_keys])
    Scum = np.copy(S)
    F_b = np.array([np.array(f['td_vel_1'][t]) for t in time_keys])
    F_phi = np.array([np.array(f['td_vel_2'][t]) for t in time_keys])

    M = np.array([np.array(f['pvd'][t]) for t in time_keys])

    NSAMP = len(th1_xz)
    times = np.array([float(f['th1_xz'][t].attrs['Time']) for t in time_keys])

    b_init = np.array(f['th1_xz']['0005'])
    f.close()

with h5py.File(join(save_dir,"mean.h5"), 'r') as f:
    print("Mean keys: %s" % f.keys())
    time_keys = list(f['tb_source'])
    bbins_file = np.array(f['tb_source_bins']['0001'])
    source_dists = np.array([np.array(f['tb_source'][t]) for t in time_keys])
    strat_dists = np.array([np.array(f['tb_strat'][t]) for t in time_keys])
    t_me = np.array([np.array(f['thme02'][t]) for t in time_keys])*md['Ny']*md['Nx']

    bbins = np.array(f['PVD_bbins']['0001'])
    phibins = np.array(f['PVD_phibins']['0001'])

    mean_times = np.array([float(f['tb_source'][t].attrs['Time']) for t in time_keys])
    assert np.array_equal(times, mean_times)
    mean_NSAMP = len(times)
    assert mean_NSAMP == NSAMP

    f.close()

with open(join(save_dir, "time.dat"), 'r') as f:
    reset_time = float(f.read())
    print("Plume penetration occured at t={0:.4f}".format(reset_time))

    if len(np.argwhere(times == 0)) > 1:
        t0_idx = np.argwhere(times == 0)[1][0]
        t0 = times[t0_idx-1]

        for i in range(t0_idx):
            times[i] -= reset_time
            print(times[i])

for i in range(1, NSAMP):
    Scum[i] += Scum[i-1]
    #M[i] *= np.sum(Scum[i])

bin_end = int(np.where(bbins_file == -1)[0][0])
source_dists[:, bin_end:] = np.nan
strat_dists_trunc = strat_dists[:, :bin_end]
bbins_file_trunc = bbins_file[:bin_end]
bbins_file[bin_end:] = np.nan
bbins_plot = 0.5*(bbins_file[1:] + bbins_file[:-1])

centreline_b = b_init[:,int(md['Nx']/2)]

F_phi_boundary = F_phi[:, 0, :]

bbin_end = int(np.where(bbins == -1)[0][0])
bbins = bbins[1:bbin_end]

phibin_end = int(np.where(phibins == -1)[0][0])
#dphi = phibins[2]-phibins[1]
#for Bin in phibins:
    #print(Bin-dphi/2)
    #print(Bin)
    #print(Bin+dphi/2)
    #input()

phibins = phibins[1:phibin_end]

md['Nb'] = len(bbins)
md['Nphi'] = len(phibins)
print(md['Nb'], md['Nphi'])
print(S.shape)
print(W.shape)

F_phi_boundary = F_phi[:, 0, 1:]

M = M[:, 1:, 1:]
F_b = F_b[:, 1:, 1:]
F_phi = F_phi[:, 1:, 1:]
Scum = Scum[:, 1:, 1:]
S = S[:, 1:, 1:]
W = W[:, 1:, 1:]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Non-dimensionalisation
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

F0 = compute_F0(save_dir, md, tstart_ind = 6*4, verbose=False, zbot=0.7, ztop=0.95, plot=False)
N = np.sqrt(md['N2'])

T = np.power(N, -1)
L = np.power(F0, 1/4) * np.power(N, -3/4)
B = L * np.power(T, -2)
U = L / T
V = L * L * L

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
bbins /= B

md['SAVE_STATS_DT'] /= T

W /= V
S /= V
Scum /= V
M /= V
F_b /= V*B/T
F_phi /= V/T
F_phi_boundary /= V/T

print("Prescribed forcing: ", F0)
print("Computed forcing: ", md['b0']*md['r0']*md['r0'])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Set-up
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print("Total time steps: %s"%NSAMP)
print("Dimensional times: ",times * T)

##### CHANGE DEPENDING ON SIMULATION
bmax_plot = 3.2
phimax_plot = 0.02
factor = 0.7
aspect = factor*bmax_plot/phimax_plot
Omega_thresh = 0.15
#####

steps = [24, 40, 56]
labels = ["a", "b", "c", "d", "e", "f", "g", "h"]

tracer_thresh = 7e-4
tracer_thresh_low = 2e-3

contours_b = np.linspace(0, md['N2']*9*L/B, 16)
contour_lvls_trace = np.linspace(0.01, 0.1, 8)

plot_max = 1.6*md['H']
plot_min = 0.95*md['H']

idx_minf = get_plotindex(plot_min, gzf)-1
idx_maxf = get_plotindex(plot_max, gzf)
idx_min = idx_minf
idx_max = idx_maxf+1

hmap_plot_max = 1.3*md['H']
hmap_plot_min = 0.98*md['H']

hmap_idx_minf = get_plotindex((hmap_plot_min - md['H'])/L, gzf)-1
hmap_idx_maxf = get_plotindex((hmap_plot_max - md['H'])/L, gzf)
hmap_idx_min = hmap_idx_minf
hmap_idx_max = hmap_idx_maxf+1


db = bbins[1] - bbins[0]
dphi = phibins[1] - phibins[0]

sx, sy = np.meshgrid(np.append(bbins-db/2, bbins[-1]+db/2),
        np.append(phibins - dphi/2, phibins[-1] + dphi/2))
sxf, syf = np.meshgrid(bbins, phibins)

th1_xz = np.where(th1_xz < 1e-3/B, 0, th1_xz)

plot_plume = np.where(
        np.logical_or(
            np.logical_and(th2_xz > tracer_thresh_low, Yf < -1),
            np.logical_and(th2_xz > tracer_thresh, Yf >= -1)),
        th2_xz, np.NaN)
plot_plume_b = np.where(
        np.logical_or(
            np.logical_and(th2_xz > tracer_thresh_low, Yf < -1),
            np.logical_and(th2_xz > tracer_thresh, Yf >= -1)),
        th1_xz, np.NaN)
plot_env = np.where(np.logical_and(np.isnan(plot_plume), Yf >= -1), th1_xz, np.NaN)
plot_outline = np.where(np.logical_and(th1_xz > 0, Yf > -1), plot_env, 0)

zmaxs = []
for i in range(len(plot_outline)):
    heights = []
    for j in range(md['Nx']):
        stuff = np.where(plot_outline[i,:,j] == 0)[-1]
        if len(stuff) == 0:
            heights.append(0)
        else:
            heights.append(gzf[max(stuff)])
    zmaxs.append(max(heights))

min_vol = np.power(md['LX']/md['Nx'], 2) * md['LZ']/md['Nz']
min_vol /= V

W = np.where(W == 0, np.NaN, W)
S = np.where(S == 0, np.NaN, S)

# Thresholding
#for d in range(len(steps)):
    #W[steps[d]] = np.where(W[steps[d]] > 50*min_vol * (64/md['Nb'])**2, W[steps[d]], np.NaN)
    #S[steps[d]] = np.where(W[steps[d]] > 50*min_vol * (64/md['Nb'])**2, S[steps[d]], np.NaN)

div_F = np.gradient(F_b, bbins, axis=2) + np.gradient(F_phi, phibins, axis=1)

M = np.where(np.isnan(W), np.NaN, M)
F_b = np.where(np.isnan(W), np.NaN, F_b)
div_F = np.where(np.isnan(W), np.NaN, div_F)
F_phi = np.where(np.isnan(W), np.NaN, F_phi)
Scum = np.where(Scum == 0, np.NaN, Scum)

#colors_pos = plt.cm.coolwarm(np.linspace(0.5, 1, 256))
#colors_neg = plt.cm.coolwarm(np.linspace(0, 0.3, 256))
#all_cols = np.vstack((colors_neg, colors_pos))
#S_map = colors.LinearSegmentedColormap.from_list('', all_cols)
#S_norm = colors.TwoSlopeNorm(vmin = -0.02, vcenter=0, vmax = 0.1)

S_bounds = np.linspace(-0.05,  0.05, 9)
S_norm = colors.BoundaryNorm(boundaries=S_bounds, ncolors=256)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Figure 2: plume cross-section
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig2, axs2 = plt.subplots(1,3,figsize=(8, 4), constrained_layout=True)

for i in range(len(steps)):
    im_env = axs2[i].contour(Xf, Yf, plot_env[steps[i]], levels=contours_b, cmap='cool', alpha=0.8)

    im_plume = axs2[i].pcolormesh(X, Y, plot_plume[steps[i]], cmap='viridis')

    thresh = 1e-3/B
    #contour_interface = axs2[i].contour(Xf, Yf, plot_outline[steps[i]], levels=[thresh], cmap='gray', alpha=0.5)

    axs2[i].set_aspect(1)

    if i == len(steps)-1:
        norm = colors.Normalize(vmin = im_env.cvalues.min(), vmax = im_env.cvalues.max())
        sm = plt.cm.ScalarMappable(norm=norm, cmap=im_env.cmap)
        sm.set_array([])

        #cb_env = fig2.colorbar(sm, ticks=im_env.levels[::2], ax = axs2[i], location='right', shrink=0.7,
            #label=r"buoyancy")
        cb_env = fig2.colorbar(im_env, ax=axs2[i], location='right', shrink=0.7)
        cb_env.set_label("$b$", rotation=0, labelpad=10)
        cb_plume = fig2.colorbar(im_plume, ax = axs2[i], location='right', shrink=0.7, extend='max')
        cb_plume.set_label("$\phi$", rotation = 0, labelpad=10)
        #cb_env.add_lines(contour_interface)

    im_plume.set_clim(0, 5e-2)

    if i == 0:
        axs2[i].set_ylabel("$z$")

    axs2[i].set_xlim(-0.1/L, 0.1/L)
    axs2[i].set_ylim(np.min(Y), 9)

    axs2[i].set_xlabel("$x$")

    axs2[i].set_title("({0}) t = {1:.0f}".format(labels[i], times[steps[i]]))

if save:
    fig2.savefig(join(fig_save_dir, 'evolution.png'), dpi=300)
    fig2.savefig(join(fig_save_dir, 'evolution.pdf'))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Figure 3: tracer concentration cross-section
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig3 = plt.figure(figsize=(8, 4))

X_v, Y_v = np.meshgrid(times[:]+md['SAVE_STATS_DT']/2, gz)
Xf_v, Yf_v = np.meshgrid(0.5*(times[1:]+times[:-1]), gzf)

tracer_data_vert = np.where(th2_xz[1:, :, int(md['Nx']/2)] >= tracer_thresh,
        th2_xz[1:, :, int(md['Nx']/2)], 0)
plume_vert = np.where(tracer_data_vert >= tracer_thresh, 1, 0)

im = plt.pcolormesh(X_v, Y_v, np.swapaxes(tracer_data_vert,0,1), cmap='viridis')
im.set_clim(0,0.05)

cont = plt.contour(X_v[1:, 1:] + 0.5*md['LX']/md['Nx'],
        Y_v[1:, 1:] + 0.5*md['LY']/md['Ny'],
        np.swapaxes(tracer_data_vert,0,1), levels=[tracer_thresh], colors=['r'],
        linestyles='--')

cbar = plt.colorbar(im, extend='max', label='tracer concentration')
cbar.add_lines(cont)

#Calculate zmax:
heights = []
#gzf_trunc = gzf[100:]
for i in range(len(plume_vert)):
    stuff = np.where(plume_vert[i] == 1)[0]
    if len(stuff) == 0:
        heights.append(0)
    else:
        heights.append(gzf[np.max(stuff)+1])

zmax = np.max(heights[:15 * 4])
zss = np.mean(heights[8 * 4:15 * 4])

plt.axhline(zmax, color='white', linewidth=1)
plt.axhline(zss, color='white', linewidth=1)
plt.text(3.5, zmax + .7, r"$z_{\max}$", fontsize='x-large', color='w')
plt.text(3.5, zss - .5 - .7, r"$z_{ss}$", fontsize='x-large', color='w')

plt.ylim(-6, 9)
plt.xlim(times[0]+md['SAVE_STATS_DT']/2, times[-1]/T)

plt.axvline(0, color='w', lw=1, ls=':')

plt.xlabel(r"$t$")
plt.ylabel(r"$z$")

plt.tight_layout()

if save:
    fig3.savefig(join(fig_save_dir, 'timeseries.png'), dpi=300)
    fig3.savefig(join(fig_save_dir, 'timeseries.pdf'))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Figure 4: comparison with source phi(b) distribution
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig4_a = plt.figure()

# Compute time indices
t_inds = list(range(24, NSAMP, 8))
nplot = len(t_inds)

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

fig4, ax4 = plt.subplots(figsize=(6,4))

ax4_inset = ax4.inset_axes([0.55, 0.5, 0.4, 0.45])
ax4_inset.plot(times, tracer_total,color='k')
ax4_inset.set_xlim(times[0], times[-1])
ax4_inset.set_ylim(0, tracer_total[-1])
ax4_inset.set_ylabel("total tracer conc.")
ax4_inset.set_xlabel("time")

ax4.plot(source_dist[1:], bbins_plot[1:], color='k', linestyle='--', label="source")

for step,c in zip(t_inds, tcols):
    area = integrate.trapezoid(strat_dists[step,:bin_end-1], bbins_plot[:bin_end-1])
    ax4.plot(strat_dists[step,:-1]/area, bbins_plot, color=c, label = "t={0:.0f}".format(times[step]))

ax4.set_ylabel("$b$")
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

end_idx = get_index(15, times)+1
times_trunc = times[:end_idx]

bt_pdfs = strat_dists_trunc[:end_idx,:-1]
zt_pdfs = t_me[:end_idx, hmap_idx_minf:hmap_idx_maxf+1]

bt_pdfs = np.swapaxes(bt_pdfs, axis1=1, axis2=0)
zt_pdfs = np.swapaxes(zt_pdfs, axis1=1, axis2=0)

Tz, Z = np.meshgrid(np.append(times_trunc, times_trunc[-1]+md['SAVE_STATS_DT']), gz[hmap_idx_min:hmap_idx_max+1])
Tzf, Zf = np.meshgrid(times_trunc+md['SAVE_STATS_DT']/2, gzf[hmap_idx_minf:hmap_idx_maxf+1])
Tb, Bp = np.meshgrid(np.append(times_trunc, times_trunc[-1]+md['SAVE_STATS_DT']), bbins_file_trunc)
Tbf, Bf = np.meshgrid(times_trunc+md['SAVE_STATS_DT']/2, 0.5*(bbins_file_trunc[1:]+bbins_file_trunc[:-1]))

fig5, ax5 = plt.subplots(1,2, figsize=(8, 2.5), constrained_layout=True)

bt_im = ax5[0].pcolormesh(Tb, Bp, bt_pdfs, shading='flat', cmap='viridis')
zt_im = ax5[1].pcolormesh(Tz, Z, zt_pdfs, shading='flat', cmap='viridis')

ax5[0].set_ylim(centreline_b[hmap_idx_min]/B, centreline_b[hmap_idx_max]/B)
ax5[1].set_ylim((hmap_plot_min-md['H'])/L, (hmap_plot_max-md['H'])/L)

bt_cont = ax5[0].contour(Tbf, Bf, bt_pdfs, levels=[10], colors='r', alpha=0.7)
bt_cbar = plt.colorbar(bt_im, ax=ax5[0], label="tracer conc.", shrink=0.7)
bt_cbar.add_lines(bt_cont)

zt_cont = ax5[1].contour(Tzf, Zf, zt_pdfs, levels=[1], colors='r', alpha=0.7)
zt_cbar = plt.colorbar(zt_im, ax=ax5[1], label="tracer conc.", shrink=0.7)
zt_cbar.add_lines(zt_cont)

#zt_cbar = plt.colorbar(zt_im, ax=ax5[1], label="tracer conc.")

ax5[1].axhline(zmax, color='white', linewidth=1)
ax5[1].axhline(zss, color='white', linewidth=1)
ax5[1].text(3.5, zmax + .2, r"$z_{\max}$", fontsize='large', color='w')
ax5[1].text(3.5, zss - .5, r"$z_{ss}$", fontsize='large', color='w')

t_step = int(round(2.0 / md['SAVE_STATS_DT'], 0))
for i in range(0, NSAMP, t_step):
    ax5[0].axvline(times[i], linestyle='--', color='gray', alpha=0.5)
    ax5[1].axvline(times[i], linestyle='--', color='gray', alpha=0.5)

ax5[0].set_xlabel(r"$t$")
ax5[0].set_ylabel(r"$b$")
ax5[1].set_xlabel(r"$t$")
ax5[1].set_ylabel(r"$z$")

ax5[0].set_xlim(0, times_trunc[-1]/T)
ax5[1].set_xlim(0, times_trunc[-1]/T)

ax5[0].set_title("(a) buoyancy-time")
ax5[1].set_title("(b) height-time")

if save:
    fig5.savefig(join(fig_save_dir, 'hmap.png'), dpi=300)
    fig5.savefig(join(fig_save_dir, 'hmap.pdf'))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Figure 6: volume distribution W and source S evolution
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig6, axs = plt.subplots(3,3,figsize=(10, 6), constrained_layout=True)

for d in range(len(steps)):
    im_b_edge = axs[0, d].contour(Xf, Yf, plot_env[steps[d]], levels = contours_b, cmap='cool', alpha=0.8)
    im_phi = axs[0,d].pcolormesh(X,Y,plot_plume[steps[d]], cmap='viridis')

    im_W = axs[1,d].pcolormesh(sx, sy, W[steps[d]], cmap='plasma')
    axs[1,d].set_aspect(aspect)

    im_S = axs[2,d].pcolormesh(sx, sy, S[steps[d]], cmap='coolwarm', norm=S_norm)
    axs[2,d].set_aspect(aspect)

    im_phi.set_clim(0, 0.05)
    im_W.set_clim(0, 0.6)

    max_height = gzf[np.max(np.argwhere(th2_xz[steps[d]] > tracer_thresh)[:, 0])]

    mask = ~np.isnan(W[steps[d]])

    sx_bl, sy_bl = np.meshgrid(bbins - db/2, phibins - dphi/2)
    sx_br, sy_br = np.meshgrid(bbins + db/2, phibins - dphi/2)
    sx_ul, sy_ul = np.meshgrid(bbins - db/2, phibins + dphi/2)
    sx_ur, sy_ur = np.meshgrid(bbins + db/2, phibins + dphi/2)

    points = np.array(list(zip(sx_bl[mask].flatten(), sy_bl[mask].flatten())))
    points = np.append(points, np.array(list(zip(sx_br[mask].flatten(), sy_br[mask].flatten()))), axis=0)
    points = np.append(points, np.array(list(zip(sx_ul[mask].flatten(), sy_ul[mask].flatten()))), axis=0)
    points = np.append(points, np.array(list(zip(sx_ur[mask].flatten(), sy_ur[mask].flatten()))), axis=0)

    if len(points) > 0:
        hull = spatial.ConvexHull(points)

        envelope = points[hull.simplices]

        flat_envelope = envelope.reshape(len(envelope)*2, 2)
        phi_test =  np.max(flat_envelope[:,1])
        b_test = np.min(flat_envelope[np.argwhere(flat_envelope[:,1] == phi_test), 0])
        #b_test is the buoyancy where phi is max

        envelope = []
        for simplex in flat_envelope:
            if simplex[0] >= b_test or simplex[0] == bbins[0]-db/2:
                envelope.append(simplex)
        envelope = np.array(envelope)

        hull = spatial.ConvexHull(envelope)
        for simplex in hull.simplices:
            axs[1,d].plot(envelope[simplex,0], envelope[simplex,1], 'r--')

        if d == 0:
            axs[1,d].plot([-1,-1], [-2, -2], 'r--', label="convex envelope")
            axs[1,0].legend()

    if d == len(steps)-1:
        cb_W = fig6.colorbar(im_W, ax = axs[1,d], label=r"$W$", shrink=0.7)
        cb_W.set_label("$W$", rotation=0, labelpad=10)
        cb_env = fig6.colorbar(im_b_edge, ax=axs[0,d], location='right', shrink=0.7)
        cb_env.set_label(r"$b$", rotation=0,  labelpad=10)
        cb_plume = fig6.colorbar(im_phi, ax = axs[0,d], location='right', shrink=0.7, extend='max')
        cb_plume.set_label(r"$\phi$", rotation=0, labelpad=10)
        cb_S = fig6.colorbar(im_S, ax=axs[2,d], label=r"$S$", shrink=0.7)
        cb_S.set_label("$S$", rotation=0, labelpad=10)

    axs[1,d].axvline(md['N2']*max_height*L/B, linestyle=':', color='r')

    axs[1,d].set_xlim(bbins[0]-db/2, bmax_plot)
    axs[1,d].set_ylim(phibins[0]-dphi/2, phimax_plot)

    axs[2,d].set_xlim(bbins[0]-db/2, bmax_plot)
    axs[2,d].set_ylim(phibins[0]-dphi/2, phimax_plot)

    if d == 0:
        axs[0,d].set_ylabel(r"$z$", rotation=0, labelpad=10)
        axs[1,d].set_ylabel(r"$\phi$", rotation=0, labelpad=10)
        axs[2,d].set_ylabel(r"$\phi$", rotation=0, labelpad=10)

    axs[2,d].set_xlabel(r"$b$")

    axs[0,d].set_xlabel("$x$")
    axs[0,d].set_aspect(1)
    axs[0,d].set_xlim(-3.25/factor, 3.25/factor)
    axs[0,d].set_ylim(-1, 5.5)
    axs[0,d].set_title("({1}) t = {0:.0f}".format(times[steps[d]], labels[d]))

if save:
    fig6.savefig(join(fig_save_dir, 'W_evol.png'), dpi=300)
    fig6.savefig(join(fig_save_dir, 'W_evol.pdf'))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Figure 7: QSS cumulative mixed volume distribution
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig7, axs7 = plt.subplots(1,2, figsize=(8, 2.1), constrained_layout=True)

# colourmap
colors_red = plt.cm.coolwarm(np.linspace(0.53, 1, 32))
colors_blue = plt.cm.coolwarm(np.linspace(0, 0.47, 32))
all_colors = np.vstack((colors_blue, colors_red))
custom_cmap = colors.LinearSegmentedColormap.from_list("cmap", all_colors)

im_b_edge = axs7[1].contour(Xf, Yf, plot_env[steps[-1]], levels = contours_b, cmap='cool', alpha=0.8)

M_lim = np.nanmax(M[steps[-1]])
im_M_bphi = axs7[0].pcolormesh(sx, sy, M[steps[-1]], cmap=custom_cmap,
        norm=colors.CenteredNorm(halfrange = .6*M_lim))

plot_array = np.copy(plot_plume[steps[-1]])
for i in range(int(md['Nb'])):
    for j in range(int(md['Nphi'])):
        plot_array[np.logical_and(np.logical_and(plot_plume_b[steps[-1]] > bbins[i] - db/2,
        plot_plume_b[steps[-1]] <= bbins[i] + db/2),np.logical_and(plot_plume[steps[-1]] > phibins[j] - dphi/2,
        plot_plume[steps[-1]] <= phibins[j] + dphi/2))] = M[steps[-1],j,i]

im_M_xz = axs7[1].pcolormesh(X,Y, plot_array, cmap=custom_cmap,
        norm=colors.CenteredNorm(halfrange = .6*M_lim))

axs7[0].set_title("(a) $(b,\phi)$-space")
axs7[0].set_ylabel(r"$\phi$", rotation=0, labelpad=10)
axs7[0].set_xlabel(r"$b$")
axs7[0].set_aspect(aspect)

axs7[0].set_xlim(bbins[0]-db/2, bmax_plot)
axs7[0].set_ylim(phibins[0]-dphi/2, phimax_plot)

cbar_M = fig7.colorbar(im_M_bphi, ax=axs7[1], location="right", shrink=0.7)
cbar_M.set_label("$M$", rotation=0, labelpad=10)

cbar_env = fig7.colorbar(im_b_edge, ax=axs7[1], location='right', label=r"$b$", shrink=0.7)
cbar_env.set_label("$b$", rotation=0, labelpad=10)

axs7[1].set_title("(b) physical space")
axs7[1].set_ylabel("$z$", rotation=0, labelpad=10)
axs7[1].set_xlabel("$x$")

axs7[1].set_aspect(1)
axs7[1].set_xlim(-0.2/L, 0.2/L)
axs7[1].set_ylim(-0.6, 5.5)

if save:
    fig7.savefig(join(fig_save_dir, 'M_plot.png'), dpi=300)
    fig7.savefig(join(fig_save_dir, 'M_plot.pdf'))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Figure 8: flux divergence and quiver plot
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig8, axs8 = plt.subplots(1,2, figsize=(8, 3), constrained_layout=True)

M_lim = np.nanmax(M[steps[-1]])
im_M_bphi = axs8[0].pcolormesh(sx, sy, M[steps[-1]], cmap=custom_cmap,
        norm=colors.CenteredNorm(halfrange = .6*M_lim), alpha=0.6)

axs8[0].quiver(sxf[::2,::2], syf[::2, ::2], F_b[steps[-1], ::2, ::2],
        F_phi[steps[-1], ::2, ::2], angles='xy', units='xy', pivot='mid',
        fc='k', ec='k', linewidth=0.1, scale=0.1)

axs8[0].set_xlim(bbins[0]-db/2, bmax_plot)
axs8[0].set_ylim(phibins[0]-dphi/2, phimax_plot)

im_div = axs8[1].pcolormesh(sx, sy, div_F[steps[-1]], cmap='bwr', norm=colors.CenteredNorm())
axs8[1].streamplot(sxf, syf, F_b[steps[-1]], F_phi[steps[-1]], density=1, color='k', linewidth=1,
    arrowstyle='fancy', broken_streamlines=False)
#axs8[1].quiver(sxf[::2,::2], syf[::2, ::2], F_b[steps[-1], ::2, ::2],
        #F_phi[steps[-1], ::2, ::2], angles='xy', units='xy', pivot='mid',
        #fc='k', ec='k', linewidth=0.1, scale=0.1)

axs8[1].set_xlim(bbins[0]-db/2, bmax_plot)
axs8[1].set_ylim(phibins[0]-dphi/2, phimax_plot)

im_div.set_clim(-1e-1, 1e-1)

cb_W = fig8.colorbar(im_M_bphi, ax = axs8[0], shrink=0.7)
cb_W.set_label("$M$", rotation=0, labelpad=10)

cb_div = fig8.colorbar(im_div, ax = axs8[1], shrink=0.7)
cb_div.set_label(r"$\nabla \cdot \mathbf{F}$", rotation=0, labelpad=10)

axs8[0].set_aspect(aspect)
axs8[1].set_aspect(aspect)

axs8[0].set_ylabel(r"$\phi$", rotation=0, labelpad=10)
axs8[0].set_xlabel(r"$b$")
axs8[1].set_xlabel(r"$b$")

axs8[0].set_title(r"(a) $M$ & $\mathbf{F}$")
axs8[1].set_title(r"(b) $\nabla \cdot \mathbf{F}$ & streamlines of $\mathbf{F}$")

if save:
    fig8.savefig(join(fig_save_dir, 'div_plot.png'), dpi=300)
    fig8.savefig(join(fig_save_dir, 'div_plot.pdf'))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Figure 9: calculating threshold for M
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig9 = plt.figure(constrained_layout=True, figsize=(8, 5))
axs9 = fig9.subplot_mosaic("AA;BC")

threshs = [0]

cols = plt.cm.rainbow(np.linspace(0, 1, NSAMP-1))
for i,c in zip(range(NSAMP-1), cols):
    M_bins = np.linspace(0, np.nanmax(M[i]), 200)

    dWdt_int = []
    S_int = []

    for m in M_bins[1:]:
        dWdt_int.append(np.nansum(np.where(M[i] < m, (W[i+1]-W[i])/md['SAVE_STATS_DT'], 0)))
        S_int.append(np.nansum(np.where(M[i] < m, S[i], 0)))

    dWdt_int = np.array(dWdt_int)
    S_int = np.array(S_int)

    axs9["A"].plot(M_bins[1:], ndimage.uniform_filter1d(dWdt_int - S_int, size=20), color=c)

    threshs.append(M_bins[1:][np.argmin(np.abs(ndimage.uniform_filter1d(dWdt_int - S_int, size=20)))])

    #axs9["A"].axvline(threshs[-1], ls=':', color=c)

axs9["A"].axhline(0, color='k')
sm = plt.cm.ScalarMappable(cmap='rainbow', norm=plt.Normalize(vmin=times[0], vmax=times[-1]))
fig9.colorbar(sm, ax=axs9["A"], label=r"$t$", location='right')
axs9["A"].set_xlabel(r"$m$")
axs9["A"].set_ylabel(r"$f(m; t)$")
axs9["A"].set_xlim(0, np.nanmax(M))

threshs = np.array(threshs)
threshs[np.isnan(threshs)] = 0
axs9["B"].plot(times, threshs, color='b', label=r"$m^*$ (calculated)")
axs9["B"].set_xlabel(r"$t$")

smooth_threshs = ndimage.uniform_filter1d(threshs, size=10)
axs9["B"].plot(times, smooth_threshs, color='r', label=r"$\tilde{m}$ (smoothed)")

axs9["B"].axhline(0, color='k')
axs9["B"].legend()

errors = [0]
smooth_ints = [0]
calc_ints = [0]
for i in range(NSAMP-1):
    dWdt_int = np.nansum(np.where(M[i] < threshs[i], (W[i+1]-W[i])/md['SAVE_STATS_DT'], 0))
    S_int = np.nansum(np.where(M[i] < threshs[i], S[i], 0))
    int_calc = dWdt_int - S_int
    calc_ints.append(int_calc)

    dWdt_int = np.nansum(np.where(M[i] < smooth_threshs[i], (W[i+1]-W[i])/md['SAVE_STATS_DT'], 0))
    S_int = np.nansum(np.where(M[i] < smooth_threshs[i], S[i], 0))
    int_smooth = dWdt_int - S_int
    smooth_ints.append(int_smooth)

    errors.append(100*abs((int_smooth - int_calc)/int_calc))

axs9["C"].axhline(0, color='k')
#axs9["C"].plot(times, errors, color='b')
#axs9["C"].set_ylim(0, 100)
axs9["C"].plot(times, calc_ints, color='b', label=r"$f(m^*; t)$")
axs9["C"].plot(times, smooth_ints, color='r', label=r"$f(\tilde{m}; t)$")
axs9["C"].legend()

axs9["C"].set_xlabel(r"$t$")

axs9["B"].set_xlim(times[0], times[-1])
axs9["C"].set_xlim(times[0], times[-1])

if save:
    fig9.savefig(join(fig_save_dir, 'm_plot.png'), dpi=300)
    fig9.savefig(join(fig_save_dir, 'm_plot.pdf'))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Figure 10: partitioned distribution M
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig10, axs10 = plt.subplots(1,2, figsize=(8, 2.5), constrained_layout=True)

# colourmap
colors_red = plt.cm.coolwarm(np.linspace(0.53, 1, 32))
colors_blue = plt.cm.coolwarm(np.linspace(0, 0.47, 32))
all_colors = np.vstack((colors_blue, colors_red))
custom_cmap = colors.LinearSegmentedColormap.from_list("cmap", all_colors)

nodes = [-np.nanmax(M[steps[-1]]), 0, 0, smooth_threshs[steps[-1]], smooth_threshs[steps[-1]], np.nanmax(M[steps[-1]])]
custom_colors = ["blue", plt.cm.Blues(0.5), plt.cm.Greens(0.5), "green", plt.cm.Reds(0.5),
        "red"]
norm = plt.Normalize(min(nodes), max(nodes))
custom_cmap = colors.LinearSegmentedColormap.from_list("", list(zip(map(norm,nodes), custom_colors)))

im_b_edge = axs10[1].contour(Xf, Yf, plot_env[steps[-1]], levels = contours_b, cmap='cool', alpha=0.8)

im_M_bphi = axs10[0].pcolormesh(sx, sy, M[steps[-1]], cmap=custom_cmap, norm=norm)
im_M_xz = axs10[1].pcolormesh(X,Y, plot_array, cmap=custom_cmap, norm = norm)

b_A_com = np.nansum(sxf*np.where(M[steps[-1]] > smooth_threshs[steps[-1]], M[steps[-1]], 0)) / \
    np.nansum(np.where(M[steps[-1]] > smooth_threshs[-1], M[steps[-1]], 0))
phi_A_com = np.nansum(syf*np.where(M[steps[-1]] > smooth_threshs[steps[-1]], M[steps[-1]], 0)) / \
    np.nansum(np.where(M[steps[-1]] > smooth_threshs[-1], M[steps[-1]], 0))

b_U_com = np.nansum(sxf*np.where(M[steps[-1]] < 0, M[steps[-1]], 0)) / \
    np.nansum(np.where(M[steps[-1]] < 0, M[steps[-1]], 0))
phi_U_com = np.nansum(syf*np.where(M[steps[-1]] < 0, M[steps[-1]], 0)) / \
    np.nansum(np.where(M[steps[-1]] < 0, M[steps[-1]], 0))

axs10[0].scatter(b_A_com, phi_A_com, color='red', ec='w', marker='^', s=100)
axs10[0].scatter(b_U_com, phi_U_com, color='blue', ec='w', marker='^', s=100)

axs10[0].set_aspect(aspect)

axs10[0].set_title("(a) $(b,\phi)$-space")
axs10[0].set_ylabel(r"$\phi$", rotation=0, labelpad=10)
axs10[0].set_xlabel(r"$b$")

axs10[0].set_xlim(bbins[0]-db/2, bmax_plot)
axs10[0].set_ylim(phibins[0]-dphi/2, phimax_plot)

cbar_M = fig10.colorbar(im_M_bphi, ax=axs10[0], location="right", shrink=0.7)
cbar_M.set_label("$M$", rotation=0, labelpad=10)

cbar_env = fig10.colorbar(im_b_edge, ax=axs10[1], location='right', label=r"$b$", shrink=0.7)
cbar_env.set_label("$b$", rotation=0, labelpad=10)

axs10[1].set_title("(b) physical space")
axs10[1].set_ylabel("$z$", rotation=0, labelpad=10)
axs10[1].set_xlabel("$x$")

axs10[1].set_aspect(1)
axs10[1].set_xlim(-0.15/L, 0.15/L)
axs10[1].set_ylim(-0.6, 5.5)

if save:
    fig10.savefig(join(fig_save_dir, 'M_thresh_plot.png'), dpi=300)
    fig10.savefig(join(fig_save_dir, 'M_thresh_plot.pdf'))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Figure 11: volumes & entrainment
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig11, axs11 = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)

total_vol = np.nansum(W*db*dphi, axis=(1,2))
input_vol = np.nansum(Scum*db*dphi, axis=(1,2))
axs11[0].plot(times, total_vol, color='k', label=r"Full plume")
axs11[0].plot(times, input_vol, color='k', linestyle='dashed',
    label=r"Plume input")

W_undiluted = np.where(M <= 0, W, np.nan)
W_mixing = np.array([np.where(np.logical_and(M[i] > 0, M[i] <= smooth_threshs[i]),
                        W[i], np.nan) for i in range(NSAMP)])
W_mixed = np.array([np.where(M[i] > smooth_threshs[i], W[i], np.nan) for i in range(NSAMP)])
print(W_mixing.shape)

W_undiluted *= db*dphi
W_mixing *= db*dphi
W_mixed *= db*dphi

U_vol = np.nansum(W_undiluted, axis=(1,2))
T_vol = np.nansum(W_mixing, axis=(1,2))
A_vol = np.nansum(W_mixed, axis=(1,2))

source_vol = np.nansum(np.where(S > 0, W*db*dphi, 0), axis=(1,2))
cum_source_vol = np.nansum(np.where(S > 0, Scum*db*dphi, 0), axis=(1,2))
axs11[0].plot(times, source_vol, color='purple')
axs11[0].plot(times, cum_source_vol, color='orange')

axs11[0].plot(times, U_vol, color='b', label=r"$V(\mathcal{U})$")
axs11[0].plot(times, T_vol, color='g', label=r"$V(\mathcal{T})$")
axs11[0].plot(times, A_vol, color='r', label=r"$V(\mathcal{A})$")

axs11[0].set_xlabel(r"$t$")
axs11[0].set_ylabel("Volume")
axs11[0].legend()

axs11[1].plot(times, total_vol - input_vol, label=r"$E$", color='k')

A_ent_vol = []
T_ent_vol = []
U_ent_vol = []

for i in range(NSAMP):
    A_ent_vol.append(
            np.sum(np.where(M[i, 0, :] > smooth_threshs[i],
                F_phi_boundary[i]*md['SAVE_STATS_DT']*db, 0)))
    T_ent_vol.append(
            np.sum(np.where(np.logical_and(M[i, 0, :] <= smooth_threshs[i], M[i, 0, :] > 0),
                F_phi_boundary[i]*md['SAVE_STATS_DT']*db, 0)))
    U_ent_vol.append(np.sum(np.where(M[i, 0, :] <= 0, F_phi_boundary[i]*md['SAVE_STATS_DT']*db, 0)))

A_ent_vol = np.array(A_ent_vol)
T_ent_vol = np.array(T_ent_vol)
U_ent_vol = np.array(U_ent_vol)

for i in range(1,NSAMP):
    A_ent_vol[i] += A_ent_vol[i-1]
    T_ent_vol[i] += T_ent_vol[i-1]
    U_ent_vol[i] += U_ent_vol[i-1]

axs11[1].plot(times, U_ent_vol, color='b', label=r"$E(\mathcal{U})$")
axs11[1].plot(times, T_ent_vol, color='g', label=r"$E(\mathcal{T})$")
axs11[1].plot(times, A_ent_vol, color='r', label=r"$E(\mathcal{A})$")
axs11[1].plot(times, A_ent_vol + T_ent_vol + U_ent_vol, color='k', linestyle='--')

A_ent_vol = []
T_ent_vol = []
U_ent_vol = []
for i in range(NSAMP):
    A_ent_vol.append(
            np.sum(np.where(M[i, 0, :] > smooth_threshs[i],
                F_phi[i, 0, :]*md['SAVE_STATS_DT']*db, 0)))
    T_ent_vol.append(
            np.sum(np.where(np.logical_and(M[i, 0, :] <= smooth_threshs[i], M[i, 0, :] > 0),
                F_phi[i, 0, :]*md['SAVE_STATS_DT']*db, 0)))
    U_ent_vol.append(np.sum(np.where(M[i, 0, :] <= 0, F_phi[i, 0, :]*md['SAVE_STATS_DT']*db, 0)))

A_ent_vol = np.array(A_ent_vol)
T_ent_vol = np.array(T_ent_vol)
U_ent_vol = np.array(U_ent_vol)

for i in range(1,NSAMP):
    A_ent_vol[i] += A_ent_vol[i-1]
    T_ent_vol[i] += T_ent_vol[i-1]
    U_ent_vol[i] += U_ent_vol[i-1]

axs11[1].plot(times, A_ent_vol, color='r', alpha=0.5)
axs11[1].plot(times, T_ent_vol, color='g', alpha=0.5)
axs11[1].plot(times, U_ent_vol, color='b', alpha=0.5)
axs11[1].plot(times, A_ent_vol + T_ent_vol + U_ent_vol, color='k', linestyle='--', alpha=0.5)

axs11[1].set_xlabel(r"$t$")
axs11[1].set_ylabel("Entrained volume")
axs11[1].legend()

axs11[0].set_title("(a)")
axs11[1].set_title("(b)")

if save:
    fig11.savefig(join(fig_save_dir, 'vol_plot.png'), dpi=300)
    fig11.savefig(join(fig_save_dir, 'vol_plot.pdf'))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if show:
    plt.show()
