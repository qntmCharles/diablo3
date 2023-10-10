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

from scipy.interpolate import griddata, interp1d, interpn
from scipy import integrate, ndimage, spatial

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"

save = True
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

print("Complete metadata: ", md)

# Get data
with h5py.File(join(save_dir, 'az_stats.h5'), 'r') as f:
    time_keys = list(f['u_az'].keys())

    # (u,v,w,b,p,phi) data
    u_az = np.array([np.array(f['u_az'][t]) for t in time_keys])
    v_az = np.array([np.array(f['v_az'][t]) for t in time_keys])
    w_az = np.array([np.array(f['w_az'][t]) for t in time_keys])
    b_az = np.array([np.array(f['b_az'][t]) for t in time_keys])
    phi_az = np.array([np.array(f['th_az'][t]) for t in time_keys])

    times = np.array([float(f['u_az'][t].attrs['Time']) for t in time_keys])
    NSAMP = len(times)

    f.close()

with open(join(save_dir, "time.dat"), 'r') as f:
    reset_time = float(f.read())
    print("Plume penetration occured at t={0:.4f}".format(reset_time))

    if len(np.argwhere(times == 0)) > 1:
        t0_idx = np.argwhere(times == 0)[1][0]
        t0 = times[t0_idx-1]

        for i in range(t0_idx):
            times[i] -= reset_time

r_0 = md['r0']
print(r_0)
print(md['H']/r_0)
dr = md['LX']/md['Nx']
nbins = int(md['Nx']/2)
r_bins = np.array([r*dr for r in range(0, nbins+1)])
r_points = np.array([0.5*(r_bins[i]+r_bins[i+1]) for i in range(nbins)])

X, Y = np.meshgrid(r_bins, gz)
Xf, Yf = np.meshgrid(r_points, gzf)

contours_b = np.linspace(1e-3, md['N2']*(md['LZ']-md['H']), 16)

vort_az = np.gradient(u_az, gzf, axis=1) - np.gradient(w_az, r_points, axis=2)

b_env = md['N2'] * (Yf - md['H'])
b_env[Yf < md['H']] = 0
var1 = b_az - b_env
var3 = phi_az
var2 = vort_az

Xfi, Yfi = np.meshgrid(r_points, np.linspace(gzf[0], gzf[-1], len(gzf)))

#steps = [t0_idx+8, t0_idx+12, t0_idx+16]
steps = [t0_idx+16, t0_idx+24, t0_idx+32]
fn = 1 # filter number

plt.plot(times, w_az[:, get_index(0.3, gzf), get_index(0, r_points)])
plt.show()

fig, axs = plt.subplots(len(steps), 2, figsize=(15, 10))

for i in range(len(steps)):
    ui_az = griddata((Xf.flatten(), Yf.flatten()), u_az[steps[i]].flatten(), (Xfi, Yfi))
    wi_az = griddata((Xf.flatten(), Yf.flatten()), w_az[steps[i]].flatten(), (Xfi, Yfi))

    speed = np.sqrt(ui_az**2, wi_az**2)

    phi_plot = axs[i, 0].pcolormesh(X, Y, var3[steps[i]], cmap='YlOrBr')
    b_cont_pos = axs[i, 0].contour(Xf, Yf, var1[steps[i]], levels=np.linspace(1e-4, 1e-2, 6), colors='k')
    b_cont_neg = axs[i, 0].contour(Xf, Yf, var1[steps[i]], levels=np.linspace(-1e-2, 1e-4, 6), colors='k',
        linestyles='--')

    vort_plot = axs[i, 1].pcolormesh(X, Y, var2[steps[i]], cmap='bwr', norm=colors.CenteredNorm())

    qplot = axs[i, 1].streamplot(Xfi[::fn, ::fn], Yfi[::fn, ::fn], ui_az[::fn, ::fn], wi_az[::fn, ::fn],
            color='k')

    vort_plot.set_clim(-1, 1)
    phi_plot.set_clim(0, 1e-2)

    axs[i,0].set_title("$t = {0:.2f} s$".format(times[steps[i]]))
    axs[i,0].set_ylabel("$z$")
    axs[i,1].set_ylabel("$z$")
    axs[i,0].set_xlabel("$r$")
    axs[i,1].set_xlabel("$r$")

    axs[i,0].set_ylim(0.15, 0.4)
    axs[i,0].set_xlim(0, md['LX']/2)
    axs[i,1].set_ylim(0.15, 0.4)
    axs[i,1].set_xlim(0, md['LX']/2)

    axs[i,0].set_aspect(1)
    axs[i,1].set_aspect(1)

plt.tight_layout()
plt.show()
