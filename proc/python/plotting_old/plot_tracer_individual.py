# Script loads in 2D slices and produces a movie of the simulation output import numpy as np import h5py, gc
import h5py
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits import axes_grid1
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
from functions import get_metadata, read_params, get_grid, g2gf_1d
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
    th1_xz = g2gf_1d(th1_xz)
    th2_xz = g2gf_1d(th2_xz)
    NSAMP = len(th1_xz)
    times = np.array([float(f['th1_xz'][t].attrs['Time']) for t in time_keys])
    f.close()

print("Total time steps: %s"%NSAMP)
print("Dimensional times: ",times)

step1 = 24
step2 = 32
step3 = 60

for i in range(NSAMP):
    th1_xz[i] = ndimage.gaussian_filter(th1_xz[i], 2)

N2_t = np.gradient(np.gradient(th1_xz, gzf, axis=1), times, axis=0)

plot_waves = np.where(np.logical_and(th2_xz < 1e-3, np.logical_and(th1_xz > 1e-3, Yf>0.9*md['H'])), N2_t, np.NaN)
plot_plume = np.where(np.logical_or(th2_xz > 1e-3, np.logical_or(th1_xz < 1e-3, Yf<0.9*md['H'])), th2_xz, np.NaN)
plot_outline = np.where(np.logical_and(th2_xz > 1e-3, th1_xz > 1e-3), 1, 0)

print("Setting up data arrays...")
fig, axs = plt.subplots(1,3,figsize=(12, 4), constrained_layout=True)

contour_lvls_b = np.linspace(0.01, np.max(th1_xz[0]), 30)
contour_lvls_trace = np.linspace(0.01, 0.1, 8)

print("Setting up initial plot...")
im0_w = axs[0].pcolormesh(X,Y,plot_waves[step1], cmap='bwr')
im0_t = axs[0].pcolormesh(X,Y,plot_plume[step1], cmap='jet')
im1_w = axs[1].pcolormesh(X,Y,plot_waves[step2], cmap='bwr')
im1_t = axs[1].pcolormesh(X,Y,plot_plume[step2], cmap='jet')
im2_w = axs[2].pcolormesh(X,Y,plot_waves[step3], cmap='bwr')
im2_t = axs[2].pcolormesh(X,Y,plot_plume[step3], cmap='jet')

outline0 = axs[0].contour(Xf, Yf, plot_outline[step1], levels=[1], colors='white',alpha=0.7)
outline1 = axs[1].contour(Xf, Yf, plot_outline[step2], levels=[1], colors='white',alpha=0.7)
outline2 = axs[2].contour(Xf, Yf, plot_outline[step3], levels=[1], colors='white',alpha=0.7)

axs[0].set_aspect(1)
axs[1].set_aspect(1)
axs[2].set_aspect(1)

cb_waves = fig.colorbar(im0_w, ax = axs[2], location='right', shrink=0.7, label="$\\partial_t N^2$")
cb_plume = fig.colorbar(im0_t, ax = axs[2], location='right', shrink=0.7, label="tracer")


im0_w.set_clim(-0.2,0.2)
im0_t.set_clim(0, 5e-2)
im1_w.set_clim(-0.2,0.2)
im1_t.set_clim(0, 5e-2)
im2_w.set_clim(-0.2,0.2)
im2_t.set_clim(0, 5e-2)

axs[0].set_xlabel("$x$ (m)")
axs[0].set_ylabel("$z$ (m)")
axs[0].set_title("(a) $t = {0:.2f} s$".format(times[step1]))

axs[1].set_xlabel("$x$ (m)")
axs[1].set_ylabel("$z$ (m)")
axs[1].set_title("(b) $t = {0:.2f} s$".format(times[step2]))

axs[2].set_xlabel("$x$ (m)")
axs[2].set_ylabel("$z$ (m)")
axs[2].set_title("(c) $t = {0:.2f} s$".format(times[step3]))

now = datetime.now()
#plt.savefig('/home/cwp29/Documents/4report/figs/evolution.png', dpi=200)
plt.show()
