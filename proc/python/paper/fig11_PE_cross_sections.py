import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
from functions import get_metadata, read_params, get_grid, get_az_data, g2gf_1d, compute_F0
from scipy import ndimage

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"
save = True

##### ---------------------- #####

def get_index(z, griddata):
    return int(np.argmin(np.abs(griddata - z)))

##### ---------------------- #####

# Get dir locations from param file
base_dir, run_dir, save_dir, version = read_params(params_file)

# Get simulation metadata
md = get_metadata(run_dir, version)

gxf, gyf, gzf, dzf = get_grid(join(save_dir,'grid.h5'), md)
gx, gy, gz, dz = get_grid(join(save_dir,'grid.h5'), md, fractional_grid=False)


print("Complete metadata: ", md)

# Get data
with h5py.File(join(save_dir,"movie.h5"), 'r') as f:
    print("Keys: %s" % f.keys())
    time_keys = list(f['th1_xz'])
    print(time_keys)
    # Get buoyancy data
    Re_b = np.array([np.array(f['Re_b_xz'][t]) for t in time_keys])
    eps = np.array([np.array(f['tked_xz'][t]) for t in time_keys])
    chi = np.array([np.array(f['chi1_xz'][t]) for t in time_keys])
    b = np.array([np.array(f['th1_xz'][t]) for t in time_keys])
    t = np.array([np.array(f['th2_xz'][t]) for t in time_keys])
    w = np.array([np.array(f['w_xz'][t]) for t in time_keys])

    NSAMP = len(time_keys)
    times = np.array([f['th1_xz'][t].attrs['Time'] for t in time_keys])
    f.close()

_,z_coords,_ = np.meshgrid(times, gzf, gxf, indexing='ij', sparse=True)

Jb = (b - b[0]) * w

X, Y = np.meshgrid(gx, gz)
Xf, Yf = np.meshgrid(gxf, gzf)

N2 = np.gradient(b, gzf, axis=1)

print("Total time steps: %s"%NSAMP)
print("Dimensional times: ",times)

print("Setting up data arrays...")
""" ----------------------------------------------------------------------------------------------------- """

step = 58

#non-dim
F0 = compute_F0(save_dir, md, tstart_ind = 6*4, verbose=False, zbot=0.7, ztop=0.95, plot=False)
N = np.sqrt(md['N2'])

T = np.power(N, -1)
L = np.power(F0, 1/4) * np.power(N, -3/4)
B = L * np.power(T, -2)
V = L * L * L

gz /= L
gzf /= L

X -= md['LX']/2
Xf -= md['LX']/2
Y -= md['H']
Yf -= md['H']

X /= L
Xf /= L

Y /= L
Yf /= L

times /= T
N2 /= np.power(T, -2)
b /= B

eps = np.exp(eps)
eps /= (np.power(L, 2) / np.power(T, 3))
chi /= (np.power(L, 2) / np.power(T, 3))
Jb /= (np.power(L, 2) / np.power(T, 3))
#chi = np.log10(chi)
#eps = np.log10(eps)


print(F0)
print(md['b0']*md['r0']*md['r0'])

contours_b = np.linspace(0, 0.1/B, 11)
contour_lvls_t = [1e-3]

tracer_thresh = 5e-4
plot_env = np.where(t <= tracer_thresh, b, np.NaN)
plot_plume = np.where(t >= tracer_thresh, t, np.NaN)
plot_plume_b = np.where(t >= tracer_thresh, b, np.NaN)
plot_outline = np.where(t <= tracer_thresh, 1, 0)

eps = np.where(t >= tracer_thresh, eps, np.NaN)
chi = np.where(t >= tracer_thresh, chi, np.NaN)
N2 = np.where(t >= tracer_thresh, N2, np.NaN)
Jb = np.where(t >= tracer_thresh, Jb, np.NaN)

""" ----------------------------------------------------------------------------------------------------- """

fig, ax = plt.subplots(2,2, figsize=(12, 5), constrained_layout=True)

for single_ax in ax.ravel():
    single_ax.set_aspect(1)
    single_ax.set_xlabel("$x$")
    single_ax.set_ylabel("$z$")
    single_ax.set_ylim(-0.5, 5)
    single_ax.set_xlim(-6, 6)
    im_contour_b = single_ax.contour(Xf, Yf, plot_env[step], levels = contours_b, cmap='cool', alpha=0.8)
    im_fill_b = single_ax.contourf(im_contour_b, levels=contours_b, cmap='cool', alpha=0.8, extend='min')

""" ----------------------------------------------------------------------------------------------------- """

eps_im = ax[0,0].pcolormesh(X, Y, eps[step], cmap='hot_r')
eps_contour_t = ax[0,0].contour(Xf, Yf, t[step], levels=[tracer_thresh], colors='green', linestyles='--')
eps_cb = plt.colorbar(eps_im, ax=ax[0,0], label=r"$\varepsilon$")
eps_im.set_clim(0, 100)

chi_im = ax[0,1].pcolormesh(X, Y, chi[step], cmap='hot_r')
#chi_contour_b = ax[0,1].contour(Xf, Yf, b[step], levels=contour_lvls_b, colors='darkturquoise', alpha=0.7)
chi_contour_t = ax[0,1].contour(Xf, Yf, t[step], levels=[tracer_thresh], colors='green', linestyles='--')

chi_cb = plt.colorbar(chi_im, ax=ax[0,1], label=r"$\chi$")
chi_im.set_clim(0, 0.1)

Jb_im = ax[1,0].pcolormesh(X, Y, Jb[step], cmap='bwr')
#Jb_contour_b = ax[1,0].contour(Xf, Yf, b[step], levels=contour_lvls_b, colors='darkturquoise', alpha=0.5)
Jb_contour_t = ax[1,0].contour(Xf, Yf, t[step], levels=[tracer_thresh], colors='green', linestyles='--')

Jb_cb = plt.colorbar(Jb_im, ax=ax[1,0], label=r"$J_b$")
Jb_im.set_clim(-2, 2)

N2_im = ax[1,1].pcolormesh(X, Y, N2[step], cmap='bwr')
#N2_contour_b = ax[1,1].contour(Xf, Yf, b[step], levels=contour_lvls_b, colors='grey', alpha=0.5)
N2_contour_t = ax[1,1].contour(Xf, Yf, t[step], levels=[tracer_thresh], colors='green', linestyles='--')

N2_cb = plt.colorbar(N2_im, ax=ax[1,1], label=r"$\partial_z b$")
N2_im.set_clim(-10, 10)

""" ----------------------------------------------------------------------------------------------------- """
now = datetime.now()

#plt.savefig(join(save_dir, 'mixing_t%s_%s.pdf'%(step,now.strftime("%d-%m-%Y-%H"))), dpi=300)
plt.savefig('/home/cwp29/Documents/papers/draft/figs/cross_sections.pdf')
plt.savefig('/home/cwp29/Documents/papers/draft/figs/cross_sections.png', dpi=300)
plt.show()
