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

x_max = 0.1/L

bmax_plot = 3.7
phimax_plot = 0.04
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
# QSS cumulative mixed volume distribution
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plot_array = np.copy(plot_plume[steps[-1]])
for i in range(int(md['Nb'])):
    for j in range(int(md['Nphi'])):
        plot_array[np.logical_and(np.logical_and(plot_plume_b[steps[-1]] > bbins[i] - db/2,
        plot_plume_b[steps[-1]] <= bbins[i] + db/2),np.logical_and(plot_plume[steps[-1]] > phibins[j] - dphi/2,
        plot_plume[steps[-1]] <= phibins[j] + dphi/2))] = M[steps[-1],j,i]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Calculating threshold for M
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

threshs = [0]

for i in range(NSAMP-1):
    M_bins = np.linspace(0, np.nanmax(M[i]), 200)

    dWdt_int = []
    S_int = []

    for m in M_bins[1:]:
        dWdt_int.append(np.nansum(np.where(M[i] < m, (W[i+1]-W[i])/md['SAVE_STATS_DT'], 0)))
        S_int.append(np.nansum(np.where(M[i] < m, S[i], 0)))

    dWdt_int = np.array(dWdt_int)
    S_int = np.array(S_int)

    threshs.append(M_bins[1:][np.argmin(np.abs(ndimage.uniform_filter1d(dWdt_int - S_int, size=20)))])

threshs = np.array(threshs)
threshs[np.isnan(threshs)] = 0

smooth_threshs = ndimage.uniform_filter1d(threshs, size=10)

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
# Figure 10: partitioned distribution M
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# colourmap
colors_red = plt.cm.coolwarm(np.linspace(0.53, 1, 32))
colors_blue = plt.cm.coolwarm(np.linspace(0, 0.47, 32))
all_colors = np.vstack((colors_blue, colors_red))
custom_cmap = colors.LinearSegmentedColormap.from_list("cmap", all_colors)

def plots(fig, axs, step):
    global im_M_bphi, im_M_xz, im_b_edge

    plot_array = np.copy(plot_plume[step])
    for i in range(int(md['Nb'])):
        for j in range(int(md['Nphi'])):
            plot_array[np.logical_and(np.logical_and(plot_plume_b[step] > bbins[i] - db/2,
            plot_plume_b[step] <= bbins[i] + db/2),np.logical_and(plot_plume[step] > phibins[j] - dphi/2,
            plot_plume[step] <= phibins[j] + dphi/2))] = M[step,j,i]

    if np.count_nonzero(~np.isnan(M[step])) > 0:
        nodes = [-np.nanmax(M[step]), 0, 0, smooth_threshs[step], smooth_threshs[step], np.nanmax(M[step])]
    else:
        nodes = [0, 0, 0, smooth_threshs[step], smooth_threshs[step], smooth_threshs[step]]
    custom_colors = ["blue", plt.cm.Blues(0.5), plt.cm.Greens(0.5), "green", plt.cm.Reds(0.5),
            "red"]
    norm = plt.Normalize(min(nodes), max(nodes))
    custom_cmap = colors.LinearSegmentedColormap.from_list("", list(zip(map(norm,nodes), custom_colors)))

    im_b_edge = axs[1].contour(Xf, Yf, plot_env[step], levels = contours_b, cmap='cool', alpha=0.8)

    im_M_bphi = axs[0].pcolormesh(sx, sy, M[step], cmap=custom_cmap, norm=norm)
    im_M_xz = axs[1].pcolormesh(X,Y, plot_array, cmap=custom_cmap, norm = norm)

    b_A_com = np.nansum(sxf*np.where(M[step] > smooth_threshs[step], M[step], 0)) / \
        np.nansum(np.where(M[step] > smooth_threshs[step], M[step], 0))
    phi_A_com = np.nansum(syf*np.where(M[step] > smooth_threshs[step], M[step], 0)) / \
        np.nansum(np.where(M[step] > smooth_threshs[step], M[step], 0))

    b_U_com = np.nansum(sxf*np.where(M[step] < 0, M[step], 0)) / \
        np.nansum(np.where(M[step] < 0, M[step], 0))
    phi_U_com = np.nansum(syf*np.where(M[step] < 0, M[step], 0)) / \
        np.nansum(np.where(M[step] < 0, M[step], 0))

    axs[0].scatter(b_A_com, phi_A_com, color='red', ec='w', marker='^', s=100)
    axs[0].scatter(b_U_com, phi_U_com, color='blue', ec='w', marker='^', s=100)

def decorate(fig, axs, step):
    axs[0].set_title("(a) $(b,\phi)$-space")
    axs[0].set_ylabel(r"$\phi$", rotation=0, labelpad=10)
    axs[0].set_xlabel(r"$b$")

    axs[0].set_aspect(aspect)
    axs[0].set_xlim(bbins[0]-db/2, bmax_plot)
    axs[0].set_ylim(phibins[0]-dphi/2, phimax_plot)

    axs[1].set_title("(b) physical space")
    axs[1].set_ylabel("$z$", rotation=0, labelpad=10)
    axs[1].set_xlabel("$x$")

    axs[1].set_aspect(1)
    axs[1].set_xlim(-0.15/L, 0.15/L)
    axs[1].set_ylim(-0.6, 5.5)

def decorate_cb(fig, axs, step):
    cbar_M = fig.colorbar(im_M_bphi, ax=axs[0], location="right", shrink=0.7)
    cbar_M.set_label("$M$", rotation=0, labelpad=10)

    cbar_env = fig.colorbar(im_b_edge, ax=axs[1], location='right', label=r"$b$", shrink=0.7)
    cbar_env.set_label("$b$", rotation=0, labelpad=10)

fig, axs = plt.subplots(1,2, figsize=(8, 2.5), constrained_layout=True)

plots(fig, axs, -1)
decorate_cb(fig, axs, -1)

def animate(step):
    for a in axs.ravel():
        a.clear()

    plots(fig, axs, step)
    decorate(fig, axs, step)

anim = animation.FuncAnimation(fig, animate, interval=250, frames=list(range(t0_idx, NSAMP)), repeat=False)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=4, bitrate=1800)
anim.save('/home/cwp29/diablo3/proc/python/scatter/pvd_movie.mp4',writer=writer)
