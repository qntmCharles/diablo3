import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
from matplotlib.collections import LineCollection
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

fig_save_dir = '/home/cwp29/Documents/papers/conv_pen/draft3/figs'

dbdphi = 13.2

b_thresh = 5e-3

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

    W = np.array([np.array(f['td_scatter'][t]) for t in time_keys])
    S = np.array([np.array(f['td_flux'][t]) for t in time_keys])
    Scum = np.copy(S)
    F_b = np.array([np.array(f['td_vel_1'][t]) for t in time_keys])
    F_phi = np.array([np.array(f['td_vel_2'][t]) for t in time_keys])

    Ent = np.copy(F_phi)

    M = np.array([np.array(f['pvd'][t]) for t in time_keys])
    boundary_flux = np.array([np.array(f['Ent_phi_flux_int'][t]) for t in time_keys])
    boundary_F_int = np.array([np.array(f['boundary_F_int'][t]) for t in time_keys])
    boundary_F = np.array([np.array(f['Fphi_boundary'][t]) for t in time_keys])

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
    total_tracer = np.array([f['total_th2'][t][0] for t in time_keys])
    t_me = np.array([np.array(f['thme02'][t]) for t in time_keys])*md['Ny']*md['Nx']

    bbins = np.array(f['PVD_bbins']['0001'])
    phibins = np.array(f['PVD_phibins']['0001'])
    print(phibins[1]-phibins[0] < 5e-4)

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

bin_end = int(np.where(bbins_file == -1)[0][0])
source_dists[:, bin_end:] = np.nan
strat_dists_trunc = strat_dists[:, :bin_end]
bbins_file_trunc = bbins_file[:bin_end]
bbins_file[bin_end:] = np.nan
bbins_plot = 0.5*(bbins_file[1:] + bbins_file[:-1])

centreline_b = b_init[:,int(md['Nx']/2)]

print(th2_xz.shape)
centreline_phi = np.mean(np.mean(th2_xz[:,:,int(md['Nx']/2)-1:int(md['Nx']/2)+2], axis=2), axis=0)
centreline_phi = np.mean(th2_xz[:,:,int(md['Nx']/2)], axis=0)
phi0 = np.max(centreline_phi)
#plt.figure()
#plt.plot(centreline_phi, gzf)
#plt.axhline(md['Lyc'])
#plt.axvline(phi0)

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

th2_xz /= phi0
phibins /= phi0

md['SAVE_STATS_DT'] /= T

W /= V
S /= V
Scum /= V
M /= V
F_b /= V*B/T
F_phi /= V*phi0/T
boundary_flux /= V

print("Prescribed forcing: ", md['b0']*md['r0']*md['r0'])
print("Computed forcing: ", F0)

print("Re, Pr:", np.sqrt(F0)/(md['nu']*np.power(md['N2'], 1/4)), md['nu']/md['kappa'])
print("Non-dimensional bottom of simulation:", md['H']/L)
print("Non-dimensional width of simulation:", md['LX']/L)
print("Non-dimensional forcing parameters:", md['Lyc']/L, md['Lyp']/L, md['r0']/L)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Set-up
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print("Total time steps: %s"%NSAMP)
print("Dimensional times: ",times * T)

##### CHANGE DEPENDING ON SIMULATION

x_max = 0.15/L

bmax_plot = 4
phimax_plot = 0.2
factor = 0.7
aspect = factor*bmax_plot/phimax_plot

#####

steps = [16, 32, -1]
labels = ["a", "b", "c", "d", "e", "f", "g", "h"]

tracer_thresh = 1e-2#5e-3 #7e-4/phi0
tracer_thresh_low = 1e-2#5e-3 #2e-3/phi0

contours_b = np.linspace(0, md['N2']*9*L/B, 16)
contours_b_short = contours_b[:10]
contour_lvls_trace = np.linspace(0.01, 0.1, 8)
contour_lvls_trace /= phi0

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

th1_xz_o = np.copy(th1_xz)
th1_xz = np.where(th1_xz < b_thresh/B, 0, th1_xz)

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

min_vol = 1.1*np.power(md['LX']/md['Nx'], 2) * md['LZ']/md['Nz']
min_vol /= V
# NO THRESHOLDING
min_vol = 0

W = np.where(W == 0, np.NaN, W)
S = np.where(S == 0, np.NaN, S)

#W_thresh = np.where(W < 1e-3, np.nan, W)
W_thresh = np.where(W < min_vol, np.nan, W)

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


while True:
    fig, axs = plt.subplots(1, 3)

    plt_z1 = float(input("enter z min: "))
    plt_z2 = float(input("enter z max: "))
    phi_min = phibins[0]-dphi/2
    plt_times = range(get_index(9.75, times), get_index(10, times))

    cols = plt.cm.rainbow(np.linspace(0, 1, len(plt_times)))

    r_norm = plt.Normalize(0, 0.2*md['LX']/2)

    axs[2].plot(13.12*np.linspace(-0.01, 0.15, 100), np.linspace(-0.01, 0.15, 100), color='k')
    axs[2].pcolormesh(sx, sy, W[plt_times[-1]], cmap='jet', alpha=0.5)

    for plt_z in gzf[get_index(plt_z1, gzf):get_index(plt_z2, gzf)+1]:
        for i,c in zip(plt_times, cols):
            plt_b = th1_xz_o[i, get_index(plt_z, gzf), :]
            plt_phi = th2_xz[i, get_index(plt_z, gzf), :]

            points_b = np.array([gxf, plt_b]).T.reshape(-1,1,2)
            segments_b = np.concatenate([points_b[:-1], points_b[1:]], axis=1)
            lc_b = LineCollection(segments_b, cmap='rainbow', norm=r_norm, alpha=0.5)
            lc_b.set_array(np.abs(gxf-md['LX']/2))
            lc_b.set_linewidth(2)
            line_phi = axs[0].add_collection(lc_b)

            points_phi = np.array([gxf, plt_phi]).T.reshape(-1,1,2)
            segments_phi = np.concatenate([points_phi[:-1], points_phi[1:]], axis=1)
            lc_phi = LineCollection(segments_phi, cmap='rainbow', norm=r_norm, alpha=0.5)
            lc_phi.set_array(np.abs(gxf-md['LX']/2))
            lc_phi.set_linewidth(2)
            line_b = axs[1].add_collection(lc_phi)

            axs[2].scatter(np.where(plt_phi > phi_min, plt_b, np.nan),
                            np.where(plt_phi > phi_min, plt_phi, np.nan),
                    c=np.abs(gxf-md['LX']/2), norm=r_norm, cmap=plt.cm.rainbow)
            axs[2].scatter(np.where(plt_phi <= phi_min, plt_b, np.nan),
                            np.where(plt_phi <= phi_min, plt_phi, np.nan),
                    c=np.abs(gxf-md['LX']/2), norm=r_norm, cmap=plt.cm.rainbow, marker='x')

            mask = plt_b < 13.12*plt_phi
            axs[0].scatter(gxf[mask], plt_b[mask], color='k', marker='x')
            axs[1].scatter(gxf[mask], plt_phi[mask], color='k', marker='x')
            axs[2].scatter(plt_b[mask], plt_phi[mask], color='k', marker='x')

    axs[0].set_xlim(0, 0.6)
    axs[1].set_xlim(0, 0.6)

    axs[2].set_xlim(0, 4)
    axs[2].set_ylim(0, 0.2)
    #axs[2].set_ylim(
    axs[1].axhline(phi_min)
    plt.show()
