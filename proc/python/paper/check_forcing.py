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
md['kappa'] = md['nu']/0.7

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

    b_xy = np.array([np.array(f['th1_xy'][t]) for t in time_keys])
    phi_xy = np.array([np.array(f['th2_xy'][t]) for t in time_keys])
    w_xy = np.array([np.array(f['w_xy'][t]) for t in time_keys])

    W = np.array([np.array(f['td_scatter'][t]) for t in time_keys])
    S = np.array([np.array(f['td_flux'][t]) for t in time_keys])
    Scum = np.copy(S)
    F_b = np.array([np.array(f['td_vel_1'][t]) for t in time_keys])
    F_phi = np.array([np.array(f['td_vel_2'][t]) for t in time_keys])

    b_dot = np.array([np.array(f['th_forcing1_xz'][t]) for t in time_keys])
    phi_dot = np.array([np.array(f['th_forcing2_xz'][t]) for t in time_keys])
    b_dot_s4 = np.array([np.array(f['diff_th1_xz'][t]) for t in time_keys])
    phi_dot_s4 = np.array([np.array(f['diff_th2_xz'][t]) for t in time_keys])

    M = np.array([np.array(f['pvd'][t]) for t in time_keys])
    F_phi_int = np.array([np.array(f['Ent_phi_flux_int'][t]) for t in time_keys])
    F_phi_check = np.array([np.array(f['Ent_phi_flux_rec'][t]) for t in time_keys])

    Re_b = np.array([np.array(f['Re_b_xz'][t]) for t in time_keys])
    eps = np.array([np.array(f['tked_xz'][t]) for t in time_keys])
    chi = np.array([np.array(f['chi1_xz'][t]) for t in time_keys])

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

    #pre 19jul_3
    #F_phi_int = np.array([np.array(f['F_phi_int'][t]) for t in time_keys])

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

for i in range(1, NSAMP):
    Scum[i] += Scum[i-1]
    F_phi_int[i] += F_phi_int[i-1]
    #F_phi_check[i] += F_phi_check[i-1]

#F_phi_check *= md['SAVE_STATS_DT']

bin_end = int(np.where(bbins_file == -1)[0][0])
source_dists[:, bin_end:] = np.nan
strat_dists_trunc = strat_dists[:, :bin_end]
bbins_file_trunc = bbins_file[:bin_end]
bbins_file[bin_end:] = np.nan
bbins_plot = 0.5*(bbins_file[1:] + bbins_file[:-1])

centreline_b = b_init[:,int(md['Nx']/2)]

F_phi_boundary = F_phi_int

#bbin_end = int(np.where(bbins == -1)[0][0])
#bbins = bbins[1:bbin_end]

#phibin_end = int(np.where(phibins == -1)[0][0])
#phibins = phibins[1:phibin_end]

#md['Nb'] = len(bbins)
#md['Nphi'] = len(phibins)

#F_phi_boundary = F_phi[:, 0, 1:]

#M = M[:, 1:, 1:]
#F_b = F_b[:, 1:, 1:]
#F_phi = F_phi[:, 1:, 1:]
#Scum = Scum[:, 1:, 1:]
#S = S[:, 1:, 1:]
#W = W[:, 1:, 1:]

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
F_phi_boundary /= V

print("Prescribed forcing: ", md['b0']*md['r0']*md['r0'])
print("Computed forcing: ", F0)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Set-up
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print("Total time steps: %s"%NSAMP)
print("Dimensional times: ",times * T)

##### CHANGE DEPENDING ON SIMULATION

x_max = 0.1/L

bmax_plot = 3.2
phimax_plot = 0.02
factor = 0.7
aspect = factor*bmax_plot/phimax_plot

#####

steps = [24, 40, 50]
labels = ["a", "b", "c", "d", "e", "f", "g", "h"]

tracer_thresh = 7e-4
tracer_thresh_low = 2e-3

contours_b = np.linspace(0, md['N2']*9*L/B, 16)
contour_lvls_trace = np.linspace(0.01, 0.1, 8)

plot_max = (1.6*md['H'] - md['H'])/L
plot_min = (0.95*md['H'] - md['H'])/L

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
axs7[1].set_xlim(-x_max, x_max)
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
axs9["A"].set_title(r"$f(m; t)$")

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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Figure 11: volumes & entrainment
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig11, axs11 = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)

total_vol = np.nansum(W, axis=(1,2))
input_vol = np.nansum(Scum, axis=(1,2))
axs11[0].plot(times, total_vol, color='k', label=r"Full plume")
axs11[0].plot(times, input_vol, color='k', linestyle='dashed',
    label=r"Plume input")

W_undiluted = np.where(M <= 0, W, np.nan)
W_mixing = np.array([np.where(np.logical_and(M[i] > 0, M[i] <= smooth_threshs[i]),
                        W[i], np.nan) for i in range(NSAMP)])
W_mixed = np.array([np.where(M[i] > smooth_threshs[i], W[i], np.nan) for i in range(NSAMP)])

U_vol = np.nansum(W_undiluted, axis=(1,2))
T_vol = np.nansum(W_mixing, axis=(1,2))
A_vol = np.nansum(W_mixed, axis=(1,2))

source_vol = np.nansum(np.where(S > 0, W, 0), axis=(1,2))
cum_source_vol = np.nansum(np.where(S > 0, Scum, 0), axis=(1,2))

#axs11[0].plot(times, source_vol, color='purple')
#axs11[0].plot(times, cum_source_vol, color='orange')

if len(np.argwhere(cum_source_vol[t0_idx+1:] > 2*source_vol[t0_idx+1:])) > 0:
    axs11[0].axvline(times[t0_idx+1+np.min(np.argwhere(cum_source_vol[t0_idx+1:] > 2*source_vol[t0_idx+1:]))],
        color='k', linestyle=':')
    axs11[1].axvline(times[t0_idx+1+np.min(np.argwhere(cum_source_vol[t0_idx+1:] > 2*source_vol[t0_idx+1:]))],
        color='k', linestyle=':')

axs11[0].plot(times, U_vol, color='b', label=r"$V(\mathcal{U})$")
axs11[0].plot(times, T_vol, color='g', label=r"$V(\mathcal{T})$")
axs11[0].plot(times, A_vol, color='r', label=r"$V(\mathcal{A})$")

axs11[0].set_xlabel(r"$t$")
axs11[0].set_ylabel("Volume")
axs11[0].legend()

##########################################################################
# Input volume test
##########################################################################

w_xy = np.where(b_xy > 0, w_xy, 0)
w_xy = np.where(phi_xy > 5e-4, w_xy, 0)
w_xy = np.where(w_xy > 0, w_xy, 0)

input_vol_test = np.sum(w_xy, axis=(1,2))

input_vol_test *= (md['LX']/md['Nx']) * (md['LY']/md['Ny']) * md['SAVE_STATS_DT']
input_vol_test /= V

#axs11[0].plot(times, input_vol, label="TEST", color='pink')

##########################################################################

axs11[1].plot(times, total_vol - input_vol, label=r"$E$", color='k')

A_ent_vol = []
T_ent_vol = []
U_ent_vol = []
sum_vol = []
sum_vol2 = []

phi_min = 5e-4

for i in range(NSAMP):
    E = -(F_phi_boundary[i, 1] - F_phi_boundary[i, 0])/phi_min

    #E *= 3/2 why does this work?!?!

    sum_vol.append(E[0])

    T_idx = np.argwhere(np.logical_and(M[i, 0, :] <= smooth_threshs[i], M[i, 0, :] > 0))
    if len(T_idx) == 0:
        T_ent_vol.append(0)
    else:
        T_ent_vol.append(E[np.min(T_idx)])

    U_ent_vol.append(E[0] - E[1])

    A_ent_vol.append(E[1] - T_ent_vol[-1])


sum_vol = np.array(sum_vol)

axs11[1].plot(times, U_ent_vol, color='b', label=r"$E(\mathcal{U})$")
axs11[1].plot(times, T_ent_vol, color='g', label=r"$E(\mathcal{T})$")
axs11[1].plot(times, A_ent_vol, color='r', label=r"$E(\mathcal{A})$")
#axs11[1].plot(times, sum_vol * (total_vol[-1] - input_vol[-1])/sum_vol[-1], color='r', linestyle='--')
axs11[1].plot(times, sum_vol, color='r', linestyle='--')



axs11[1].set_xlabel(r"$t$")
axs11[1].set_ylabel("Entrained volume")
axs11[1].legend()

axs11[0].set_title("(a) Volume decomposition")
axs11[1].set_title("(b) Entrainment decomposition")

if save:
    fig11.savefig(join(fig_save_dir, 'vol_plot.png'), dpi=300)
    fig11.savefig(join(fig_save_dir, 'vol_plot.pdf'))

# for checking that code is calculating something reasonable...
"""
plt.figure()
for i in range(30, len(times)):
    print(F_phi_check[i])
    plt.plot(bbins, F_phi_check[i])
    print(F_phi[i, 0, :])
    plt.plot(bbins, F_phi[i, 0, :])

    plt.xlim(bbins[0]-db/2, bmax_plot)
    plt.show()
"""

print(np.mean(b_dot[steps[-1]]))
print(np.mean(b_dot_s4[steps[-1]]))

plt.figure()
fig, ax = plt.subplots(1, 2)
ax[0].pcolormesh(X, Y, b_dot_s4[steps[-1]]-b_dot[steps[-1]], cmap='bwr', norm=colors.CenteredNorm())
ax[1].pcolormesh(X, Y, phi_dot_s4[steps[-1]]-phi_dot[steps[-1]], cmap='bwr', norm=colors.CenteredNorm())

plt.show()

