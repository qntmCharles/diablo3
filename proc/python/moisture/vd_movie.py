import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.animation as animation
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
from functions import get_metadata, read_params, get_grid, g2gf_1d, compute_F0, get_plotindex, get_index
from scipy import ndimage, interpolate, spatial

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"
convex_hull = True

vapour = True

##### ---------------------- #####

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
    b = np.array([np.array(f['th1_xz'][t]) for t in time_keys])
    phi = np.array([np.array(f['th2_xz'][t]) for t in time_keys])

    if vapour:
        W = np.array([np.array(f['b_phiv_W'][t]) for t in time_keys])
        Scum = np.array([np.array(f['b_phiv_S'][t]) for t in time_keys])
        F_b = np.array([np.array(f['b_phiv_F1'][t]) for t in time_keys])
        F_phi = np.array([np.array(f['b_phiv_F2'][t]) for t in time_keys])

        M = np.array([np.array(f['b_phiv_M'][t]) for t in time_keys])
    else:
        W = np.array([np.array(f['b_phic_W'][t]) for t in time_keys])
        Scum = np.array([np.array(f['b_phic_S'][t]) for t in time_keys])
        F_b = np.array([np.array(f['b_phic_F1'][t]) for t in time_keys])
        F_phi = np.array([np.array(f['b_phic_F3'][t]) for t in time_keys])

        M = np.array([np.array(f['b_phic_M'][t]) for t in time_keys])

    NSAMP = len(b)
    times = np.array([float(f['th1_xz'][t].attrs['Time']) for t in time_keys])

    f.close()

with h5py.File(join(save_dir,"mean.h5"), 'r') as f:
    print("Mean keys: %s" % f.keys())
    time_keys = list(f['tb_source'])

    bbins = np.array(f['PVD_bbins']['0001'])
    if vapour:
        phibins = np.array(f['PVD_phivbins']['0001'])
    else:
        phibins = np.array(f['PVD_phicbins']['0001'])

with open(join(save_dir, "time.dat"), 'r') as f:
    reset_time = float(f.read())
    print("Plume penetration occured at t={0:.4f}".format(reset_time))

    if len(np.argwhere(times == 0)) > 1:
        t0_idx = np.argwhere(times == 0)[1][0]
        t0 = times[t0_idx-1]

        for i in range(t0_idx):
            times[i] -= reset_time

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Set-up
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

F0 = compute_F0(save_dir, md, tstart_ind = 6*4, verbose=False, zbot=0.7, ztop=0.95, plot=False)
N = np.sqrt(md['N2'])
T = np.power(N, -1)
L = np.power(F0, 1/4) * np.power(N, -3/4)

for i in range(1, NSAMP):
    Scum[i] += Scum[i-1]

S = np.gradient(Scum, md['SAVE_STATS_DT'], axis=0)
div_F = np.gradient(F_b, bbins, axis=2) + np.gradient(F_phi, phibins, axis=1)

W = np.where(W == 0, np.NaN, W)
S = np.where(S == 0, np.NaN, S)
M = np.where(M == 0, np.NaN, M)
F_b = np.where(F_b == 0, np.NaN, F_b)
F_phi = np.where(F_phi == 0, np.NaN, F_phi)
M = np.where(M == 0, np.NaN, M)
Scum = np.where(Scum == 0, np.NaN, Scum)
div_F = np.where(div_F == 0, np.NaN, div_F)

tracer_thresh = 1e-5
tracer_thresh_low = 1e-4
plot_plume = np.where(
        np.logical_or(
            np.logical_and(phi > tracer_thresh_low, Yf < md['H']-L),
            np.logical_and(phi > tracer_thresh, Yf >= md['H']-L)),
        phi, np.NaN)
plot_env = np.where(np.logical_and(np.isnan(plot_plume), Yf >= md['H']-L), b, np.NaN)

contours_b = np.linspace(0, np.max(b), 20)[1:]
b = np.where(b < 1e-4, 0, b)

db = bbins[1] - bbins[0]
dphi = phibins[1] - phibins[0]

sx, sy = np.meshgrid(np.append(bbins-db/2, bbins[-1]+db/2),
        np.append(phibins - dphi/2, phibins[-1] + dphi/2))
sxf, syf = np.meshgrid(bbins, phibins)

S_bounds = np.linspace(-1e-6,  1e-6, 9)
S_norm = colors.BoundaryNorm(boundaries=S_bounds, ncolors=256)

colors_red = plt.cm.coolwarm(np.linspace(0.53, 1, 32))
colors_blue = plt.cm.coolwarm(np.linspace(0, 0.47, 32))
all_colors = np.vstack((colors_blue, colors_red))
custom_cmap = colors.LinearSegmentedColormap.from_list("cmap", all_colors)

bmax_plot = 2e-2#bbins[-1]
phimax_plot = phibins[-1]

contours_phi = np.linspace(phibins[0] - dphi/2, 0.2*phimax_plot, 10)
mid_tracer_thresh = contours_phi[2]

tracer_data_vert = np.where(phi[:, :, int(md['Nx']/2)] >= mid_tracer_thresh,
        phi[:, :, int(md['Nx']/2)], 0)
plume_vert = np.where(tracer_data_vert >= mid_tracer_thresh, 1, 0)
heights = []
for i in range(len(plume_vert)):
    stuff = np.where(plume_vert[i] == 1)[0]
    if len(stuff) == 0:
        heights.append(0)
    else:
        heights.append(gzf[np.max(stuff)+1])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plotting
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def decorate(fig, axs, step):
    fig.suptitle("t = {0:.0f} s".format(times[step]))

    for a in axs.ravel()[1:]:
        a.set_xlim(bbins[0]-db/2, bmax_plot)
        a.set_ylim(phibins[0]-dphi/2, phimax_plot)

        a.set_xlabel(r"$b$")
        if vapour:
            a.set_ylabel(r"$\phi_v$")
        else:
            a.set_ylabel(r"$\phi_c$")

    axs[0,0].set_xlim(0.2, 0.4)
    axs[0,0].set_ylim(md['H']-L, md['H']+5.5*L)
    axs[0,0].set_xlabel(r"$x$")
    axs[0,0].set_ylabel(r"$z$")

    im_phi.set_clim(phibins[0]-dphi/2, phimax_plot)
    im_W.set_clim(0, 1e-5)
    im_W_F.set_clim(0, 1e-5)

def decorate_cb(fig, axs, step):
    cb_W = fig.colorbar(im_W, ax=axs[0,1], location='right', shrink=0.8)
    cb_W.set_label("$W$", rotation=0, labelpad=5)

    cb_env = fig.colorbar(im_b_edge, ax=axs[0,0], location='right', shrink=0.8)
    cb_env.set_label(r"$b$", rotation=0, labelpad=5)

    cb_phi = fig.colorbar(cont_phi, ax=axs[0,0], location='right', shrink=0.8)
    cb_phi.set_label(r"$\phi$", rotation=0, labelpad=5)

    cb_plume = fig.colorbar(im_phi, ax = axs[0,0], location='right', shrink=0.8, extend='max')
    if vapour:
        cb_plume.set_label(r"$\phi_v$", rotation=0, labelpad=5)
    else:
        cb_plume.set_label(r"$\phi_c$", rotation=0, labelpad=5)

    cb_Scum = fig.colorbar(im_Scum, ax=axs[1,0], location='right', shrink=0.8)
    cb_Scum.set_label("cumulative $S$", rotation=0, labelpad=5)

    cb_S = fig.colorbar(im_S, ax=axs[1,1], location='right', shrink=0.8)
    cb_S.set_label("$S$", rotation=0, labelpad=5)

    cb_M = fig.colorbar(im_M, ax=axs[1,2], location="right", shrink=0.8)
    cb_M.set_label("$M$", rotation=0, labelpad=5)

    cb_WF = fig.colorbar(im_W, ax = axs[0,2], location='right', shrink=0.8)
    cb_WF.set_label("$W$", rotation=0, labelpad=5)

def plots(fig, axs, step):
    global im_b_edge, im_phi, im_W, im_Scum, im_S, im_M, im_W_F, cont_phi

    im_b_edge = axs[0,0].contour(Xf, Yf, plot_env[step], levels=contours_b, cmap='cool', alpha=0.8)
    im_phi = axs[0,0].pcolormesh(X, Y, plot_plume[step], cmap='viridis', alpha=0.5)
    cont_phi = axs[0,0].contour(Xf, Yf, plot_plume[step], cmap='viridis', levels=contours_phi, ls='--')

    im_W = axs[0,1].pcolormesh(sx, sy, W[step], cmap='plasma')
    im_Scum = axs[1,0].pcolormesh(sx, sy, Scum[step], cmap='coolwarm', norm=S_norm)
    im_S = axs[1,1].pcolormesh(sx, sy, S[step], cmap='coolwarm', norm=S_norm)

    M_lim = np.nanmax(M[step])
    im_M = axs[1,2].pcolormesh(sx, sy, M[step], cmap=custom_cmap,
            norm=colors.CenteredNorm(halfrange = .6*M_lim))

    im_W_F = axs[0,2].pcolormesh(sx, sy, W[step], cmap='plasma', alpha=0.5)

    fn = 2 # filter_num
    im_F = axs[0,2].quiver(sxf[::fn,::fn], syf[::fn, ::fn], F_b[step, ::fn, ::fn],
            F_phi[step, ::fn, ::fn], angles='xy', units='xy', pivot='mid',
            fc='k', ec='k', linewidth=0.1)

    Nz = 10
    cols = plt.cm.rainbow(np.linspace(0,1,Nz))
    for z,c in zip(np.linspace(0.15, 0.3, Nz),cols):
        bs = np.linspace(bbins[0], bmax_plot, 100)

        axs[0,1].plot(bs, md['q0'] * np.exp(md['alpha'] * (bs - md['beta'] * z)), color=c,
            label=r"$z={0:.2f} \, m$".format(z), alpha=0.5)
        axs[1,0].plot(bs, md['q0'] * np.exp(md['alpha'] * (bs - md['beta'] * z)), color=c,
            label=r"$z={0:.2f} \, m$".format(z), alpha=0.5)
        axs[1,1].plot(bs, md['q0'] * np.exp(md['alpha'] * (bs - md['beta'] * z)), color=c,
            label=r"$z={0:.2f} \, m$".format(z), alpha=0.5)

    axs[0,1].plot(bs, md['q0']*np.exp(md['alpha'] * (bs - md['beta']*heights[step])), color='k', ls=':')

    axs[1,0].legend(loc='right')

    decorate(fig, axs, step)

while True:
    fig, axs = plt.subplots(2, 3, figsize=(18, 8), constrained_layout=True)

    plots(fig, axs, -1)
    decorate_cb(fig, axs, -1)

    def animate(step):
        for a in axs.ravel():
            a.clear()

        plots(fig, axs, step)
        decorate(fig, axs, step)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=4, bitrate=-1)

    anim = animation.FuncAnimation(fig, animate, interval=250, frames=list(range(t0_idx, NSAMP)), repeat=False)
    now = datetime.now()
    #anim.save(save_dir+'scatter_%s.mp4'%now.strftime("%d-%m-%Y"),writer=writer)
    plt.show()
