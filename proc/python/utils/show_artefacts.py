import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits import axes_grid1
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
from functions import get_metadata, read_params, get_grid, g2gf_1d
from scipy import ndimage, spatial

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
    print("Movie keys: %s" % f.keys())
    time_keys = list(f['th1_xz'])
    print(time_keys)
    # Get buoyancy data
    th1_xz = np.array([np.array(f['th1_xz'][t]) for t in time_keys])
    th2_xz = np.array([np.array(f['th2_xz'][t]) for t in time_keys])
    vd = np.array([np.array(f['td_scatter'][t]) for t in time_keys])
    vd_flux = np.array([np.array(f['td_flux'][t]) for t in time_keys])

    NSAMP = len(th1_xz)
    times = np.array([float(f['th1_xz'][t].attrs['Time']) for t in time_keys])
    f.close()

vd = np.where(vd == 0, np.NaN, vd)

with h5py.File(save_dir+"/mean.h5", 'r') as f:
    print("Mean keys: %s" % f.keys())
    bbins = np.array(f['PVD_bbins']['0001'])
    phibins = np.array(f['PVD_phibins']['0001'])

print(bbins)
print(phibins)


sx, sy = np.meshgrid(bbins, phibins)

fig, ax = plt.subplots(1,3, figsize=(18, 6))

start_step = 16
start_step = 12 * 4
times = times[start_step:]
th1_xz = th1_xz[start_step:]
th2_xz = th2_xz[start_step:]
vd = vd[start_step:]

phi_test = th2_xz[0]
b_test = th1_xz[0]
vd_test = vd[0]

im1 = ax[0].pcolormesh(Xf, Yf, np.where(np.logical_and(Yf > 0.95*md['H'], np.logical_and(phi_test >
    b_test+0.0015,
    phi_test < 0.015)), phi_test, np.NaN), cmap='viridis')
im2 = ax[0].pcolormesh(Xf, Yf, np.where(phi_test > 5e-4, phi_test, np.NaN), alpha=0.5)
im1.set_clim(0, 0.015)
im2.set_clim(0, 0.015)
plt.colorbar(im1,ax = ax[0])

im1 = ax[1].pcolormesh(Xf, Yf, np.where(np.logical_and(Yf > 0.95*md['H'], np.logical_and(phi_test >
    b_test+0.0015,
    phi_test < 0.015)), b_test, np.NaN), cmap='viridis')
im2 = ax[1].pcolormesh(Xf, Yf, np.where(phi_test > 5e-4, b_test, np.NaN), alpha=0.5)
plt.colorbar(im1,ax = ax[1])
im1.set_clim(0, 0.015)
im2.set_clim(0, 0.015)

vd_im1 = ax[2].pcolormesh(sx, sy, np.where(np.logical_and(sy > sx + 0.0025, sy < 0.015), vd_test, np.NaN),
        cmap='plasma')
vd_im2 = ax[2].pcolormesh(sx, sy, np.where(np.logical_or(sy <= sx + 0.0025, sy >= 0.015), vd_test, np.NaN),
        cmap='plasma', alpha=0.5)
vd_im1.set_clim(0, 9e-6)
vd_im2.set_clim(0, 9e-6)

ax[0].set_ylim(0.2, 0.27)
ax[0].set_xlim(0.25, 0.35)

ax[1].set_ylim(0.2, 0.27)
ax[1].set_xlim(0.25, 0.35)

fig.suptitle(times[0])

plt.show()

def animate(step):
    phi_test = th2_xz[step]
    b_test = th1_xz[step]
    vd_test = vd[step]

    for a in ax: a.clear()

    im1 = ax[0].pcolormesh(Xf, Yf, np.where(np.logical_and(Yf > 0.95*md['H'], np.logical_and(phi_test >
        b_test+0.0015,
        phi_test < 0.015)), phi_test, np.NaN), cmap='viridis')
    im2 = ax[0].pcolormesh(Xf, Yf, np.where(phi_test > 5e-4, phi_test, np.NaN), alpha=0.5)
    im1.set_clim(0, 0.015)
    im2.set_clim(0, 0.015)

    im1 = ax[1].pcolormesh(Xf, Yf, np.where(np.logical_and(Yf > 0.95*md['H'], np.logical_and(phi_test >
        b_test+0.0015,
        phi_test < 0.015)), b_test, np.NaN), cmap='viridis')
    im2 = ax[1].pcolormesh(Xf, Yf, np.where(phi_test > 5e-4, b_test, np.NaN), alpha=0.5)
    im1.set_clim(0, 0.015)
    im2.set_clim(0, 0.015)

    vd_im1 = ax[2].pcolormesh(sx, sy, np.where(np.logical_and(sy > sx + 0.0025, sy < 0.015), vd_test, np.NaN),
            cmap='plasma')
    vd_im2 = ax[2].pcolormesh(sx, sy, np.where(np.logical_or(sy <= sx + 0.0025, sy >= 0.015), vd_test, np.NaN),
            cmap='plasma', alpha=0.5)
    vd_im1.set_clim(0, 9e-6)
    vd_im2.set_clim(0, 9e-6)

    ax[0].set_ylim(0.2, 0.27)
    ax[0].set_xlim(0.25, 0.35)

    ax[1].set_ylim(0.2, 0.27)
    ax[1].set_xlim(0.25, 0.35)

    fig.suptitle(times[step])

anim = animation.FuncAnimation(fig, animate, interval=250, frames=NSAMP-16, repeat=True)

plt.show()
