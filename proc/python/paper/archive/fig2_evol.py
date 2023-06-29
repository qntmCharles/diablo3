import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
from mpl_toolkits import axes_grid1
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
from functions import get_metadata, read_params, get_grid, g2gf_1d, compute_F0
from scipy import ndimage

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"

##### ---------------------- #####

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
with h5py.File(save_dir+"/movie.h5", 'r') as f:
    print("Keys: %s" % f.keys())
    time_keys = list(f['th1_xz'])
    print(time_keys)
    # Get buoyancy data
    th1_xz = np.array([np.array(f['th1_xz'][t]) for t in time_keys])
    th2_xz = np.array([np.array(f['th2_xz'][t]) for t in time_keys])
    NSAMP = len(th1_xz)
    times = np.array([float(f['th1_xz'][t].attrs['Time']) for t in time_keys])
    f.close()

##### Non-dimensionalisation #####

F0 = compute_F0(save_dir, md, tstart_ind = 6*4, verbose=False, zbot=0.7, ztop=0.95, plot=False)
N = np.sqrt(md['N2'])

T = np.power(N, -1)
L = np.power(F0, 1/4) * np.power(N, -3/4)
B = L * np.power(T, -2)

X -= md['LX']/2
X /= L
Y -= md['H']
Y /= L

Xf -= md['LX']/2
Xf /= L
Yf -= md['H']
Yf /= L

times /= T

th1_xz /= B

print(F0)
print(md['b0']*md['r0']*md['r0'])

##### ====================== #####

print("Total time steps: %s"%NSAMP)
print("Dimensional times: ",times)

step1 = 24
step2 = 40
step3 = 56

th1_xz = np.where(th1_xz <= 1e-3/B, 0, th1_xz)

tracer_thresh = 7e-4
tracer_thresh_low = 2e-3

plot_plume = np.where(
        np.logical_or(
            np.logical_and(th2_xz > tracer_thresh_low, Yf < -1),
            np.logical_and(th2_xz > tracer_thresh, Yf >= -1)),
        th2_xz, np.NaN)
plot_env = np.where(np.logical_and(np.isnan(plot_plume), Yf >= -1), th1_xz, np.NaN)

print("Setting up data arrays...")
fig, axs = plt.subplots(1,3,figsize=(12, 4), constrained_layout=True)

contours_b = np.linspace(0, md['N2']*9*L/B, 16)
print(contours_b)
contour_lvls_trace = np.linspace(0.01, 0.1, 8)

print("Setting up initial plot...")
#im0_w = axs[0].pcolormesh(X,Y,plot_env[step1], cmap='cool')
#im1_w = axs[1].pcolormesh(X,Y,plot_env[step2], cmap='cool')
#im2_w = axs[2].pcolormesh(X,Y,plot_env[step3], cmap='cool')

im0_w_edge = axs[0].contour(Xf, Yf, plot_env[step1], levels = contours_b, cmap='cool', alpha=0.8)
#im0_w = axs[0].contourf(im0_w_edge, levels=contours_b, cmap='cool', alpha=0.8, extend='min')

im1_w_edge = axs[1].contour(Xf, Yf, plot_env[step2], levels = contours_b, cmap='cool', alpha=0.8)
#im1_w = axs[1].contourf(im1_w_edge, levels=contours_b, cmap='cool', alpha=0.8, extend='min')

im2_w_edge = axs[2].contour(Xf, Yf, plot_env[step3], levels = contours_b, cmap='cool', alpha=0.8)
#im2_w = axs[2].contourf(im2_w_edge, levels=contours_b, cmap='cool', alpha=0.8, extend='min')

im0_t = axs[0].pcolormesh(X,Y,plot_plume[step1], cmap='viridis')
im1_t = axs[1].pcolormesh(X,Y,plot_plume[step2], cmap='viridis')
im2_t = axs[2].pcolormesh(X,Y,plot_plume[step3], cmap='viridis')

plot_outline = np.where(np.logical_and(th1_xz > 0, Yf > -1), plot_env, 0)
thresh = 1e-3/B
strat_cont = axs[0].contour(Xf, Yf, plot_outline[step1], levels=[thresh], cmap='gray', alpha=0.5)
strat_cont = axs[1].contour(Xf, Yf, plot_outline[step2], levels=[thresh], cmap='gray', alpha=0.5)
strat_cont = axs[2].contour(Xf, Yf, plot_outline[step3], levels=[thresh], cmap='gray', alpha=0.5)

#col = plt.cm.viridis(np.linspace(0,1, 2))[0]
#outline0 = axs[0].contour(Xf, Yf, plot_outline[step1], levels=[0.5], colors=[col],alpha=0.7)
#outline1 = axs[1].contour(Xf, Yf, plot_outline[step2], levels=[0.5], colors='white',alpha=0.7)
#outline2 = axs[2].contour(Xf, Yf, plot_outline[step3], levels=[0.5], colors='white',alpha=0.7)

axs[0].set_aspect(1)
axs[1].set_aspect(1)
axs[2].set_aspect(1)

norm = colors.Normalize(vmin = im0_w_edge.cvalues.min(), vmax = im0_w_edge.cvalues.max())
sm = plt.cm.ScalarMappable(norm=norm, cmap=im0_w_edge.cmap)
sm.set_array([])

#cb_waves = fig.colorbar(im0_w, ax = axs[2], location='right', shrink=0.7, label=r"buoyancy")
cb_waves = fig.colorbar(sm, ticks=im0_w_edge.levels[::2], ax = axs[2], location='right', shrink=0.7,
    label=r"buoyancy")
cb_plume = fig.colorbar(im0_t, ax = axs[2], location='right', shrink=0.7, label="tracer concentration",
        extend='max')
cb_waves.add_lines(strat_cont)


#im0_w.set_clim(0, md['N2']*(0.35-md['H'])/B)
im0_t.set_clim(0, 5e-2)
#im1_w.set_clim(0, md['N2']*(0.35-md['H']))
im1_t.set_clim(0, 5e-2)
#im2_w.set_clim(0, md['N2']*(0.35-md['H']))
im2_t.set_clim(0, 5e-2)


for ax in axs:
    ax.set_xlim(-0.15/L, 0.15/L)
    ax.set_ylim(0, 0.35/L)

axs[0].set_xlabel("$x$")
axs[0].set_ylabel("$z$")
axs[0].set_title("(a) t = {0:.2f}".format(times[step1]))
axs[0].set_xlim(-0.15/L, 0.15/L)
axs[0].set_ylim(np.min(Y), 9)

axs[1].set_xlabel("$x$")
axs[1].set_title("(b) t = {0:.2f}".format(times[step2]))
axs[1].set_xlim(-0.15/L, 0.15/L)
axs[1].set_ylim(np.min(Y), 9)

axs[2].set_xlabel("$x$")
axs[2].set_title("(c) t = {0:.2f}".format(times[step3]))
axs[2].set_xlim(-0.15/L, 0.15/L)
axs[2].set_ylim(np.min(Y), 9)

#plt.savefig('/home/cwp29/Documents/papers/draft/figs/evolution.pdf')
plt.savefig('/home/cwp29/Documents/papers/draft/figs/evolution.png', dpi=300)
plt.show()
