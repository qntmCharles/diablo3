import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
from os.path import join, isfile
import h5py
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from functions import get_metadata, read_params, get_grid, g2gf_1d

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"

##### ---------------------- #####

# Get dir locations from param file
base_dir, run_dir, save_dir, version = read_params(params_file)

# Get simulation metadata
md = get_metadata(run_dir, version)

gxf, gyf, gzf, dzf = get_grid(save_dir+'grid.h5', md)
gx, gy, gz, dz = get_grid(save_dir+'grid.h5', md, fractional_grid=False)

X, Y = np.meshgrid(gx, gz)
Xf, Yf = np.meshgrid(gxf, gzf)

r_0 = md['r0']
dr = md['LX']/md['Nx']
nbins = int(md['Nx']/2)
r_bins = np.array([r*dr for r in range(0, nbins+1)])
r_points = np.array([0.5*(r_bins[i]+r_bins[i+1]) for i in range(nbins)])

Xa, Ya = np.meshgrid(r_bins, gz)
Xaf, Yaf = np.meshgrid(r_points, gzf)

print("Complete metadata: ", md)

# Get data
with h5py.File(save_dir+"/movie.h5", 'r') as f:
    print("Keys: %s" % f.keys())
    time_keys = list(f['th1_xy'])
    print(time_keys)
    # Get buoyancy data
    b = np.array([np.array(f['th1_xz'][t]) for t in time_keys])
    phi_v = np.array([np.array(f['th2_xz'][t]) for t in time_keys])
    phi_c = np.array([np.array(f['th3_xz'][t]) for t in time_keys])

    phi_c_horiz = np.array([np.array(f['th3_xy'][t]) for t in time_keys])

    u = np.array([np.array(f['u_xz'][t]) for t in time_keys])
    w = np.array([np.array(f['w_xz'][t]) for t in time_keys])

    NSAMP = len(b)
    times = np.array([float(f['th1_xz'][tstep].attrs['Time']) for tstep in time_keys])
    f.close()

with h5py.File(save_dir+"/az_stats.h5", 'r') as f:
    print("Keys: %s" % f.keys())
    time_keys = list(f['u_az'])
    b_az = np.array([np.array(f['th_az'][t]) for t in time_keys])

    f.close()

with open(join(save_dir, "time.dat"), 'r') as f:
    reset_time = float(f.read())
    print("Plume penetration occured at t={0:.4f}".format(reset_time))

    if len(np.argwhere(times == 0)) > 1:
        t0_idx = np.argwhere(times == 0)[1][0]
        t0 = times[t0_idx-1]

        for i in range(t0_idx):
            times[i] -= reset_time

##### ENVIRONMENTAL VARIABLES #####

alpha = md['alpha']
beta = md['beta']
q_0 = md['q0']
T0 = 300 # K

b_env = md['N2'] * (Yf - md['H'])
b_env[gzf < md['H']] = 0

T = 1340 * (b_env - beta * Yf)

phi_sat = q_0 * np.exp(alpha/1340 * T)

theta = 1340 * b

##### ---------------------- #####

r = phi_v/phi_sat
contours_b = np.linspace(10, 150, 15)
fn = 8

steps = [t0_idx+8, t0_idx+16, t0_idx+24]

##### ---------------------- #####

fig_xz, ax_xz = plt.subplots(1,len(steps), constrained_layout=True, figsize=(12,3))

fig, ax = plt.subplots(1, 4, constrained_layout=True)

for i in range(len(steps)):
    wvmr_im = ax_xz[i].pcolormesh(X, Y, r[steps[i]], cmap='YlGnBu')
    wvmr_im.set_clim(0, 5)
    theta_contour = ax_xz[i].contour(Xf, Yf, theta[steps[i]], colors='r', levels=contours_b)
    cloud_contour = ax_xz[i].contour(Xf, Yf, phi_c[steps[i]], colors='k', levels=[5e-4])
    flow = ax_xz[i].quiver(Xf[::fn, ::fn], Yf[::fn, ::fn], u[steps[i], ::fn, ::fn], w[-1, ::fn, ::fn])

    ax_xz[i].set_xlim(0.2, 0.4)
    ax_xz[i].set_ylim(0.9*md['H'], 1.5*md['H'])
    ax_xz[i].set_aspect(1)

    ax_xz[i].set_title("t = {0:.2f}s".format(times[steps[i]]))

    if i == 0:
        wvmr_cbar = fig_xz.colorbar(wvmr_im, label="wvmr", ax=ax_xz[:], location='bottom', shrink=0.7)

    ew = []
    cloud = np.where(b_az[steps[i]] < 1e-3, 0, 1)
    for i in range(md['Nz']):
        ew.append(np.pi*np.power(r_points[np.min(np.argwhere(cloud[i] < 1))],2))

    ax[0].plot(ew, gzf)


plt.show()
