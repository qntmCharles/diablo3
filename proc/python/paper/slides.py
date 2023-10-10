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

save = True
show = not save

fig_save_dir = '/home/cwp29/Documents/talks/ucl/'

dbdphi = 45.7

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

    M = np.array([np.array(f['pvd'][t]) for t in time_keys])
    boundary_flux = np.array([np.array(f['Ent_phi_flux_int'][t]) for t in time_keys])

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
    #boundary_flux[i] += boundary_flux[i-1]

bin_end = int(np.where(bbins_file == -1)[0][0])
source_dists[:, bin_end:] = np.nan
strat_dists_trunc = strat_dists[:, :bin_end]
bbins_file_trunc = bbins_file[:bin_end]
bbins_file[bin_end:] = np.nan
bbins_plot = 0.5*(bbins_file[1:] + bbins_file[:-1])

centreline_b = b_init[:,int(md['Nx']/2)]

#bbin_end = int(np.where(bbins == -1)[0][0])
#bbins = bbins[1:bbin_end]

#phibin_end = int(np.where(phibins == -1)[0][0])
#phibins = phibins[1:phibin_end]

#md['Nb'] = len(bbins)
#md['Nphi'] = len(phibins)


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
boundary_flux /= V

print("Prescribed forcing: ", md['b0']*md['r0']*md['r0'])
print("Computed forcing: ", F0)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Set-up
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print("Total time steps: %s"%NSAMP)
print("Dimensional times: ",times * T)

##### CHANGE DEPENDING ON SIMULATION

x_max = 0.15/L

bmax_plot = 4
phimax_plot = 0.045
factor = 0.7
aspect = factor*bmax_plot/phimax_plot

#####

steps = [16, 32, -1]
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

hmap_plot_max = 1.45*md['H']
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

W_thresh = np.where(W < 1e-3, np.nan, W)

div_F = np.gradient(F_b, bbins, axis=2) + np.gradient(F_phi, phibins, axis=1)

M_nonthresh = np.where(np.isnan(W), np.NaN, M)
M = np.where(np.isnan(W_thresh), np.NaN, M)
F_b = np.where(np.isnan(W_thresh), np.NaN, F_b)
div_F = np.where(np.isnan(W_thresh), np.NaN, div_F)
F_phi = np.where(np.isnan(W_thresh), np.NaN, F_phi)
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

fig2, axs2 = plt.subplots(1,3, figsize=(10, 3.3), constrained_layout=True)

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

    axs2[i].set_xlim(-x_max, x_max)
    axs2[i].set_ylim(-6, 7)

    axs2[i].set_xlabel("$x$")

    axs2[i].set_title("({0}) t = {1:.0f}".format(labels[i], times[steps[i]]))

if save:
    fig2.savefig(join(fig_save_dir, 'evolution.png'), dpi=300)
    fig2.savefig(join(fig_save_dir, 'evolution.pdf'))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Figure 6: volume distribution W and source S evolution
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig6, axs = plt.subplots(2,3,figsize=(12, 5), constrained_layout=True)

W_thresh = np.where(W < 1e-3, np.nan, W)

for d in range(len(steps)):
    im_b_edge = axs[0, d].contour(Xf, Yf, plot_env[steps[d]], levels = contours_b, cmap='cool', alpha=0.8)
    im_phi = axs[0,d].pcolormesh(X,Y,plot_plume[steps[d]], cmap='viridis')

    im_W = axs[1,d].pcolormesh(sx, sy, W_thresh[steps[d]], cmap='plasma')
    axs[1,d].set_aspect(aspect)

    im_phi.set_clim(0, 0.05)
    im_W.set_clim(0, 0.25)

    max_height = gzf[np.max(np.argwhere(th2_xz[steps[d]] > tracer_thresh)[:, 0])]

    mask = ~np.isnan(W_thresh[steps[d]])

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
            if simplex[0] >= b_test or simplex[0] == bbins[0]-db/2 or simplex[1] == phibins[0]-dphi/2:
                envelope.append(simplex)
        envelope = np.array(envelope)

        hull = spatial.ConvexHull(envelope)
        for simplex in hull.simplices:
            axs[1,d].plot(envelope[simplex,0], envelope[simplex,1], 'r--')

        if d == 0:
            axs[1,d].plot([-1,-1], [-2, -2], 'r--', label="convex envelope")

    if d == len(steps)-1:
        cb_W = fig6.colorbar(im_W, ax = axs[1,d], label=r"$W$", shrink=0.7)
        cb_W.set_label("$W$", rotation=0, labelpad=10)
        cb_env = fig6.colorbar(im_b_edge, ax=axs[0,d], location='right', shrink=0.7)
        cb_env.set_label(r"$b$", rotation=0,  labelpad=10)
        cb_plume = fig6.colorbar(im_phi, ax = axs[0,d], location='right', shrink=0.7, extend='max')
        cb_plume.set_label(r"$\phi$", rotation=0, labelpad=10)

    axs[1,d].axvline(md['N2']*max_height*L/B, linestyle=':', color='r', label=r"$b(z_{\max})$")
    axs[1,0].legend(loc='upper right')

    axs[1,d].set_xlim(bbins[0]-db/2, bmax_plot)
    axs[1,d].set_ylim(phibins[0]-dphi/2, phimax_plot)

    if d == 0:
        axs[0,d].set_ylabel(r"$z$", rotation=0, labelpad=10)
        axs[1,d].set_ylabel(r"$\phi$", rotation=0, labelpad=10)

    axs[1,d].set_xlabel(r"$b$")
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

fig8, axs8 = plt.subplots(1,2, figsize=(10, 4), constrained_layout=True)

im_W = axs8[0].pcolormesh(sx, sy, W_thresh[steps[-1]], cmap='plasma', alpha=0.5)
axs8[0].set_aspect(aspect)
im_W.set_clim(0, 0.25)

fn = 4 # filter_num
axs8[0].quiver(sxf[::fn,::fn], syf[::fn, ::fn], F_b[steps[-1], ::fn, ::fn],
        F_phi[steps[-1], ::fn, ::fn], angles='xy', units='xy', pivot='mid',
        fc='k', ec='k', linewidth=0.1, scale=0.1)

axs8[0].set_xlim(bbins[0]-db/2, bmax_plot)
axs8[0].set_ylim(phibins[0]-dphi/2, phimax_plot)

axs8[0].set_aspect(aspect)

axs8[0].set_ylabel(r"$\phi$", rotation=0, labelpad=10)
axs8[0].set_xlabel(r"$b$")

axs8[0].set_title(r"(a) $W$ & $\mathbf{F}$")
axs8[1].set_title(r"(b) Source distribution $S$")

im_S = axs8[1].pcolormesh(sx, sy, S[steps[-1]], cmap='coolwarm', norm=S_norm)
axs8[1].set_aspect(aspect)
axs8[1].set_xlim(bbins[0]-db/2, bmax_plot)
axs8[1].set_ylim(phibins[0]-dphi/2, phimax_plot)

axs8[1].set_xlabel(r"$b$")

cb_S = fig6.colorbar(im_S, ax=axs8[1], label=r"$S$", shrink=0.7)
cb_S.set_label("$S$", rotation=0, labelpad=10)

axs8[1].set_ylabel(r"$\phi$", rotation=0, labelpad=10)

if save:
    fig8.savefig(join(fig_save_dir, 'div_plot.png'), dpi=300)
    fig8.savefig(join(fig_save_dir, 'div_plot.pdf'))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Calculating threshold for M
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
threshs = [0]

for i in range(NSAMP-1):
    M_bins = np.linspace(0, np.nanmax(M[i]), 200)

    divF_int = []

    for m in M_bins[1:]:
        divF_int.append(np.nansum(np.where(np.logical_and(sxf > 45.7*syf,np.logical_and(M[i]>=0, M[i]<m)), div_F[i], 0)))

    divF_int = np.array(divF_int)

    threshs.append(M_bins[1:][np.argmin(np.abs(ndimage.uniform_filter1d(divF_int, size=20)))])

threshs = np.array(threshs)
threshs[np.isnan(threshs)] = 0

smooth_threshs = ndimage.uniform_filter1d(threshs, size=10)

errors = [0]
smooth_ints = [0]
calc_ints = [0]

for i in range(NSAMP-1):
    divF_int = np.nansum(np.where(np.logical_and(sxf > dbdphi*syf,np.logical_and(M[i]>=0, M[i]<threshs[i])), div_F[i], 0))
    int_calc = divF_int
    calc_ints.append(int_calc)

    divF_int = np.nansum(np.where(np.logical_and(sxf > dbdphi*syf,np.logical_and(M[i]>=0, M[i]<smooth_threshs[i])), div_F[i], 0))
    int_smooth = divF_int
    smooth_ints.append(int_smooth)

    errors.append(100*abs((int_smooth - int_calc)/int_calc))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Figure 10: partitioned distribution M
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig10, axs10 = plt.subplots(1,2, figsize=(8, 2.1), constrained_layout=True)

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

cbar_M = fig10.colorbar(im_M_bphi, ax=axs10[1], location="right", shrink=0.7)
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

source_vol = np.nansum(np.where(Scum > 0, W, 0), axis=(1,2))
U_vol = np.nansum(np.where(M <=0, W, np.nan), axis=(1,2))
total_vol = np.nansum(W, axis=(1,2))

axs11[0].plot(times, source_vol, color='b', ls = '--', label=r"$V(\mathcal{S})$")
axs11[0].plot(times, U_vol, color='b', label=r"$V(\mathcal{U})$")
axs11[0].plot(times, total_vol, color='k', label=r"Total plume volume")

axs11[0].set_title("(a) Identifying QSS")

axs11[0].legend()
axs11[0].set_xlabel(r"$t$")

cum_source_vol = np.nansum(np.where(S > 0, Scum, 0), axis=(1,2))
#axs11[0].plot(times, source_vol, color='purple')
#axs11[0].plot(times, cum_source_vol, color='orange')

if len(np.argwhere(cum_source_vol[t0_idx+1:] > 2*source_vol[t0_idx+1:])) > 0:
    t_QSS = times[t0_idx+1+np.min(np.argwhere(cum_source_vol[t0_idx+1:] > 2*source_vol[t0_idx+1:]))]
else:
    t_QSS = times[35]

idx_QSS = get_index(times, t_QSS)

total_vol = np.nansum(W, axis=(1,2))
input_vol = np.nansum(Scum, axis=(1,2))
axs11[1].plot(times, total_vol, color='k', label=r"Full plume")
axs11[1].plot(times, input_vol, color='k', linestyle='dashed',
    label=r"Plume input")

W_undiluted = np.where(M <= 0, W, np.nan)
W_mixing = np.array([np.where(np.logical_and(M[i] > 0, M[i] <= smooth_threshs[i]),
                        W[i], np.nan) for i in range(NSAMP)])
W_mixed = np.array([np.where(M[i] > smooth_threshs[i], W[i], np.nan) for i in range(NSAMP)])

U_vol = np.nansum(W_undiluted, axis=(1,2))
T_vol = np.nansum(W_mixing, axis=(1,2))
A_vol = np.nansum(W_mixed, axis=(1,2))

axs11[0].set_xlim(times[0], times[-1])
axs11[1].set_xlim(times[0], times[-1])
axs11[0].set_ylim(0, 260)
axs11[1].set_ylim(0, 260)

axs11[0].axvline(t_QSS, color='k', linestyle=':')
axs11[1].axvline(t_QSS, color='k', linestyle=':')

axs11[1].plot(times[idx_QSS:], U_vol[idx_QSS:], color='b', label=r"$V(\mathcal{U})$")
axs11[1].plot(times[idx_QSS:], T_vol[idx_QSS:], color='g', label=r"$V(\mathcal{T})$")
axs11[1].plot(times[idx_QSS:], A_vol[idx_QSS:], color='r', label=r"$V(\mathcal{A})$")

axs11[1].set_xlabel(r"$t$")
axs11[1].set_title("(b) Volume decomposition")
axs11[1].set_ylabel("Volume")
axs11[1].legend()

if save:
    fig11.savefig(join(fig_save_dir, 'qss_volume.png'), dpi=300)
    fig11.savefig(join(fig_save_dir, 'qss_volume.pdf'))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Figure 11.5: entrainment & entrainment rate
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig11, axs11 = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)

axs11[0].axvline(t_QSS, color='k', linestyle=':')
axs11[1].axvline(t_QSS, color='k', linestyle=':')

A_ent_vol = []
T_ent_vol = []
U_ent_vol = []
sum_vol = []

phi_min = 5e-4

for i in range(NSAMP):
    if dphi > phi_min:
        E = -(boundary_flux[i, 1] - boundary_flux[i, 0])/(2*phi_min)
    else:
        E = -(boundary_flux[i, 1] - boundary_flux[i, 0])/(2*dphi)

    sum_vol.append(E[0])

    #if times[i] >= t_QSS:
    T_idx = np.argwhere(np.logical_and(M[i, 0, :] <= smooth_threshs[i], M[i, 0, :] > 0))
    T_idx = T_idx[T_idx > 30]
    if len(T_idx) == 0:
        T_ent_vol.append(0)
    else:
        T_ent_vol.append(E[np.min(T_idx)])
        print(times[i], bbins[np.min(T_idx)])

    A_ent_vol.append(E[1] - T_ent_vol[-1])
    U_ent_vol.append(E[0] - E[1])

    #else:
        #T_ent_vol.append(0)
        #A_ent_vol.append(0)
        #U_ent_vol.append(0)

sum_vol = np.array(sum_vol)
T_ent_vol = np.array(T_ent_vol)
U_ent_vol = np.array(U_ent_vol)
A_ent_vol = np.array(A_ent_vol)

for i in range(1, NSAMP):
    sum_vol[i] += sum_vol[i-1]
    T_ent_vol[i] += T_ent_vol[i-1]
    A_ent_vol[i] += A_ent_vol[i-1]
    U_ent_vol[i] += U_ent_vol[i-1]

U_ent_rate = np.gradient(U_ent_vol, times, axis=0)/U_vol
T_ent_rate = np.gradient(T_ent_vol, times, axis=0)/T_vol
A_ent_rate = np.gradient(A_ent_vol, times, axis=0)/A_vol

U_ent_rate_early = np.where(times <= t_QSS, U_ent_rate, np.nan)
T_ent_rate_early = np.where(times <= t_QSS, T_ent_rate, np.nan)
A_ent_rate_early = np.where(times <= t_QSS, A_ent_rate, np.nan)
U_ent_rate = np.where(times < t_QSS, np.nan, U_ent_rate)
T_ent_rate = np.where(times < t_QSS, np.nan, T_ent_rate)
A_ent_rate = np.where(times < t_QSS, np.nan, A_ent_rate)

# mask before QSS
U_ent_vol_early = np.where(times <= t_QSS, U_ent_vol, np.nan)
T_ent_vol_early = np.where(times <= t_QSS, T_ent_vol, np.nan)
A_ent_vol_early = np.where(times <= t_QSS, A_ent_vol, np.nan)
U_ent_vol = np.where(times < t_QSS, np.nan, U_ent_vol)
T_ent_vol = np.where(times < t_QSS, np.nan, T_ent_vol)
A_ent_vol = np.where(times < t_QSS, np.nan, A_ent_vol)

axs11[0].plot(times, U_ent_vol_early, color='b', alpha = 0.4)
axs11[0].plot(times, T_ent_vol_early, color='g', alpha = 0.4)
axs11[0].plot(times, A_ent_vol_early, color='r', alpha = 0.4)

axs11[0].plot(times, U_ent_vol, color='b', label=r"$E(\mathcal{U})$")
axs11[0].plot(times, T_ent_vol, color='g', label=r"$E(\mathcal{T})$")
axs11[0].plot(times, A_ent_vol, color='r', label=r"$E(\mathcal{A})$")

#axs11[0].plot(times, sum_vol, color='k', label=r"$E$")
axs11[0].plot(times, total_vol - input_vol, label=r"$E$", color='k')

axs11[1].plot(times, U_ent_rate_early, color='b', alpha=0.4)
axs11[1].plot(times, T_ent_rate_early, color='g', alpha=0.4)
axs11[1].plot(times, A_ent_rate_early, color='r', alpha=0.4)

axs11[1].plot(times, U_ent_rate, color='b', label=r"$\dot{E}(\mathcal{U})/V(\mathcal{U})$")
axs11[1].plot(times, T_ent_rate, color='g', label=r"$\dot{E}(\mathcal{T})/V(\mathcal{T})$")
axs11[1].plot(times, A_ent_rate, color='r', label=r"$\dot{E}(\mathcal{A})/V(\mathcal{A})$")
#axs11[1].plot(times, sum_vol, color='r', linestyle='--')

axs11[1].set_title("(b) Entrainment rate")
axs11[1].set_ylim(0, 3)
axs11[1].set_xlim(0, times[-1])
axs11[1].set_xlabel(r"$t$")
axs11[1].set_ylabel("Normalised volume flux")
axs11[1].legend()

axs11[0].set_title("(a) Entrained volume")
axs11[0].set_xlim(0, times[-1])
axs11[0].set_ylim(0, 200)
axs11[0].set_xlabel(r"$t$")
axs11[0].set_ylabel("Volume")
axs11[0].legend()

if save:
    fig11.savefig(join(fig_save_dir, 'entrainment.png'), dpi=300)
    fig11.savefig(join(fig_save_dir, 'entrainment.pdf'))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Figure 12: entrainment profile
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig12, axs12 = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

ent_vols = []
vols = np.nansum(W[t0_idx:], axis=1)

cols = plt.cm.rainbow(np.linspace(0, 1, (NSAMP-t0_idx)//4))
for i in range(t0_idx, NSAMP):
    E = -(boundary_flux[i, 1] - boundary_flux[i, 0])/(phi_min)
    ent_vols.append([E[j] - E[j+1] for j in range(len(E)-1)])
    #axs12.plot(ent_vol, 0.5*(bbins[1:] + bbins[:-1]), color=c, label=r"$t={0:.0f}$".format(times[i]))

ent_vols = np.array(ent_vols)
for i in range(1, len(ent_vols)):
    ent_vols[i] += ent_vols[i-1]

ent_vols = np.gradient(ent_vols, times[t0_idx:], axis=0)

for i, c in zip(range(0, len(ent_vols), 4), cols):
    axs12[0].plot(ent_vols[i], 0.5*(bbins[1:] + bbins[:-1]), color=c, label="$t={0:.0f}$".format(times[t0_idx+i]))
    axs12[1].plot(ent_vols[i]/vols[i, :-1], 0.5*(bbins[1:] + bbins[:-1]), color=c,
            label="$t={0:.0f}$".format(times[t0_idx+i]))

axs12[0].set_title("(a) Entrained volume flux")
axs12[0].set_ylabel("$b$")
axs12[0].set_xlabel("Volume flux")
axs12[0].set_xlim(0, 0.9)
axs12[0].set_ylim(bbins[0]-db/2, bmax_plot)
axs12[0].legend()

axs12[1].set_title("(b) Entrainment rate")
axs12[1].set_ylabel("$b$")
axs12[1].set_xlabel("Normalised volume flux")
axs12[1].set_xlim(0, 30)
axs12[1].set_ylim(bbins[0]-db/2, bmax_plot)
#axs12[1].legend()

if save:
    fig12.savefig(join(fig_save_dir, 'entrainment_profile.png'), dpi=300)
    fig12.savefig(join(fig_save_dir, 'entrainment_profile.pdf'))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Figure 12: mixing diagnostics cross sections
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig12, axs12 = plt.subplots(2, 2, figsize=(8, 4), constrained_layout=True)

N2 = np.gradient(th1_xz, gzf, axis=1)

N2 /= np.power(T, -2)
Re_b = np.exp(Re_b)
eps = np.exp(eps)
eps /= (np.power(L, 2) / np.power(T, 3))
chi /= (np.power(L, 2) / np.power(T, 3))

eps = np.where(th2_xz >= tracer_thresh, eps, np.NaN)
chi = np.where(th2_xz >= tracer_thresh, chi, np.NaN)
N2 = np.where(th2_xz >= tracer_thresh, N2, np.NaN)
Re_b = np.where(th2_xz >= tracer_thresh, Re_b, np.NaN)

for single_ax in axs12.ravel():
    single_ax.set_aspect(1)
    single_ax.set_xlabel(r"$x$")
    single_ax.set_ylabel(r"$z$")
    single_ax.contour(Xf, Yf, plot_env[steps[-1]], levels = contours_b, cmap='cool', alpha=0.8)
    single_ax.set_xlim(-x_max, x_max)
    single_ax.set_ylim(-1, 5.5)

eps_im = axs12[0,0].pcolormesh(X, Y, eps[steps[-1]], cmap='hot_r')
eps_contour_t = axs12[0,0].contour(Xf, Yf, th2_xz[steps[-1]], levels=[tracer_thresh], colors='green', linestyles='--')
eps_cb = fig12.colorbar(eps_im, ax=axs12[0,0], label=r"$\varepsilon$")
eps_im.set_clim(0, 100)

chi_im = axs12[0,1].pcolormesh(X, Y, chi[steps[-1]], cmap='hot_r')
#chi_contour_b = axs12[0,1].contour(Xf, Yf, b[steps[-1]], levels=contour_lvls_b, colors='darkturquoise', alpha=0.7)
chi_contour_t = axs12[0,1].contour(Xf, Yf, th2_xz[steps[-1]], levels=[tracer_thresh], colors='green', linestyles='--')

chi_cb = fig12.colorbar(chi_im, ax=axs12[0,1], label=r"$\chi$")
chi_im.set_clim(0, 0.1)

Reb_im = axs12[1,0].pcolormesh(X, Y, Re_b[steps[-1]], cmap='hot_r')
#Reb_contour_b = axs12[1,0].contour(Xf, Yf, b[steps[-1]], levels=contour_lvls_b, colors='darkturquoise', alpha=0.5)
Reb_contour_t = axs12[1,0].contour(Xf, Yf, th2_xz[steps[-1]], levels=[tracer_thresh], colors='green', linestyles='--')

Reb_cb = fig12.colorbar(Reb_im, ax=axs12[1,0], label=r"$\mathrm{Re}_b$")
#Reb_im.set_clim(-1, 5)
Reb_im.set_clim(0, 20)

N2_im = axs12[1,1].pcolormesh(X, Y, N2[steps[-1]], cmap='bwr')
#N2_contour_b = axs12[1,1].contour(Xf, Yf, b[steps[-1]], levels=contour_lvls_b, colors='grey', alpha=0.5)
N2_contour_t = axs12[1,1].contour(Xf, Yf, th2_xz[steps[-1]], levels=[tracer_thresh], colors='green', linestyles='--')

N2_cb = fig12.colorbar(N2_im, ax=axs12[1,1], label=r"$\partial_z b$")
N2_im.set_clim(-2, 2)

if save:
    fig12.savefig(join(fig_save_dir, 'cross_sections.png'), dpi=300)
    fig12.savefig(join(fig_save_dir, 'cross_sections.pdf'))

plt.show()
