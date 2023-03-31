import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
from functions import get_metadata, read_params, get_grid, get_az_data, g2gf_1d
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
    Ri = np.array([np.array(f['Ri_xz'][t]) for t in time_keys])
    eps = np.array([np.array(f['tke_diss_xz'][t]) for t in time_keys])
    e = np.array([np.array(f['diapycvel1_xz'][t]) for t in time_keys])
    chi = np.array([np.array(f['chi1_xz'][t]) for t in time_keys])
    B = np.array([np.array(f['B_xz'][t]) for t in time_keys])
    nu_t = np.array([np.array(f['nu_t_xz'][t]) for t in time_keys])
    b = np.array([np.array(f['th1_xz'][t]) for t in time_keys])
    t = np.array([np.array(f['th2_xz'][t]) for t in time_keys])
    u = np.array([np.array(f['u_xz'][t]) for t in time_keys])
    v = np.array([np.array(f['v_xz'][t]) for t in time_keys])
    w = np.array([np.array(f['w_xz'][t]) for t in time_keys])

    NSAMP = len(b)
    times = np.array([f['th1_xz'][t].attrs['Time'] for t in time_keys])
    f.close()

# Get az data
with h5py.File(join(save_dir,"az_stats.h5"), 'r') as f:
    print("Keys: %s" % f.keys())
    time_keys = list(f['u_az'].keys())
    wbar = np.array([np.array(f['w_az'][t]) for t in time_keys])
    bbar = np.array([np.array(f['b_az'][t]) for t in time_keys])

wbar2 = np.concatenate((np.flip(wbar,axis=2), wbar), axis=2)
wfluc = w-wbar2

bbar2 = np.concatenate((np.flip(bbar,axis=2), bbar), axis=2)
bfluc = b-bbar2

_,z_coords,_ = np.meshgrid(times, gzf, gxf, indexing='ij', sparse=True)

trac = t
b_env = b[0]

plot_max = 0.15 + md['H']
plot_min = 0.9 * md['H']

idx_max = get_index(plot_max, gz)
idx_min = get_index(plot_min, gz)

idx_maxf = get_index(plot_max, gzf)
idx_minf = get_index(plot_min, gzf)

X, Y = np.meshgrid(gx, gz[idx_min:idx_max+1])
Xf, Yf = np.meshgrid(gxf, gzf[idx_minf:idx_maxf])

u_z = np.gradient(u, gzf, axis=1)
v_z = np.gradient(v, gzf, axis=1)
w_z = np.gradient(w, gzf, axis=1)
N2 = np.gradient(b, gzf, axis=1)

# Restrict to penetration region
chi = chi[:, idx_min:idx_max, :]
Re_b = Re_b[:, idx_min:idx_max, :]
eps = eps[:, idx_min:idx_max, :]
e = e[:, idx_min:idx_max, :]
nu_t = nu_t[:, idx_min:idx_max, :]
b = b[:, idx_minf:idx_maxf, :]
t = t[:, idx_minf:idx_maxf, :]
Ri = Ri[:, idx_min:idx_max, :]
B = B[:, idx_min:idx_max, :]
trac = trac[:, idx_min:idx_max, :]

chi = np.log(chi)

print("Total time steps: %s"%NSAMP)
print("Dimensional times: ",times)

print("Setting up data arrays...")
contour_lvls_b = [1e-2, 5e-2, 1e-1, 1.5e-1, 2e-1, 2.5e-1]
contour_lvls_t = [1e-3]

""" ----------------------------------------------------------------------------------------------------- """

step = NSAMP-1
if step >= NSAMP:
    step = int(input("Chosen timestep out of simulation range. Pick again: "))

""" ----------------------------------------------------------------------------------------------------- """

eps_fig = plt.figure()
eps_ax = plt.gca()

bg_eps_im = eps_ax.pcolormesh(X, Y, np.where(t[step] <= 1e-3, eps[step], np.nan), alpha=0.5, cmap='hot_r')
eps_im = eps_ax.pcolormesh(X, Y, np.where(t[step] > 1e-3, eps[step], np.nan), cmap='hot_r')
eps_contour_b = eps_ax.contour(Xf, Yf, b[step], levels=contour_lvls_b, colors='darkturquoise', alpha=0.7)
eps_contour_t = eps_ax.contour(Xf, Yf, t[step], levels=contour_lvls_t, colors='green', linestyles='--')

eps_divider = make_axes_locatable(eps_ax)
eps_cax = eps_divider.append_axes("right", size="5%", pad=0.05)
eps_cb = plt.colorbar(eps_im, cax=eps_cax)
eps_im.set_clim(-25, -5)
bg_eps_im.set_clim(-25, -5)

eps_ax.set_aspect(1)
eps_ax.set_xlabel("$x\, (m)$")
eps_ax.set_ylabel("$z\, (m)$")
eps_ax.set_ylim(gz[idx_min], gz[idx_max])
eps_ax.set_xlim(0.2, 0.4)
eps_ax.set_title("TKE dissipation rate $\\varepsilon$ (log scale)")

""" ----------------------------------------------------------------------------------------------------- """

Ri_fig = plt.figure()
Ri_ax = plt.gca()

bg_Ri_im = Ri_ax.pcolormesh(X, Y, np.where(t[step] <= 1e-3, Ri[step], np.nan), alpha=0.5, cmap='hot')
Ri_im = Ri_ax.pcolormesh(X, Y, np.where(t[step] > 1e-3, Ri[step], np.nan), cmap='hot')
Ri_contour_b = Ri_ax.contour(Xf, Yf, b[step], levels=contour_lvls_b, colors='darkturquoise', alpha=0.7)
Ri_contour_t = Ri_ax.contour(Xf, Yf, t[step], levels=contour_lvls_t, colors='green', linestyles='--')

Ri_divider = make_axes_locatable(Ri_ax)
Ri_cax = Ri_divider.append_axes("right", size="5%", pad=0.05)
Ri_cb = plt.colorbar(Ri_im, cax=Ri_cax)
Ri_im.set_clim(0, 0.25)
bg_Ri_im.set_clim(0, 0.25)

Ri_ax.set_aspect(1)
Ri_ax.set_xlabel("$x\, (m)$")
Ri_ax.set_ylabel("$z\, (m)$")
Ri_ax.set_ylim(gz[idx_min], gz[idx_max])
Ri_ax.set_xlim(0.2, 0.4)
Ri_ax.set_title("Local Richardson number $Ri$")

""" ----------------------------------------------------------------------------------------------------- """

Re_b_fig = plt.figure()
Re_b_ax = plt.gca()

bg_Re_b_im = Re_b_ax.pcolormesh(X, Y, np.where(t[step] <= 1e-3, Re_b[step], np.nan), alpha=0.5, cmap='hot_r')
Re_b_im = Re_b_ax.pcolormesh(X, Y, np.where(t[step] > 1e-3, Re_b[step], np.nan), cmap='hot_r')
Re_b_contour_b = Re_b_ax.contour(Xf, Yf, b[step], levels=contour_lvls_b, colors='darkturquoise', alpha=0.7)
Re_b_contour_t = Re_b_ax.contour(Xf, Yf, t[step], levels=contour_lvls_t, colors='green', linestyles='--')

Re_b_divider = make_axes_locatable(Re_b_ax)
Re_b_cax = Re_b_divider.append_axes("right", size="5%", pad=0.05)
Re_b_cb = plt.colorbar(Re_b_im, cax=Re_b_cax)
Re_b_im.set_clim(-3, 15)
bg_Re_b_im.set_clim(-3, 15)

Re_b_ax.set_aspect(1)
Re_b_ax.set_xlabel("$x\, (m)$")
Re_b_ax.set_ylabel("$z\, (m)$")
Re_b_ax.set_ylim(gz[idx_min], gz[idx_max])
Re_b_ax.set_xlim(0.2, 0.4)
Re_b_ax.set_title("Buoyancy Reynolds number $\\mathrm{{Re}}_b$ (log scale)")

""" ----------------------------------------------------------------------------------------------------- """

chi_fig = plt.figure()
chi_ax = plt.gca()

bg_chi_im = chi_ax.pcolormesh(X, Y, np.where(t[step] <= 1e-3, chi[step], np.nan), alpha=0.5, cmap='hot_r')
chi_im = chi_ax.pcolormesh(X, Y, np.where(t[step] > 1e-3, chi[step], np.nan), cmap='hot_r')
chi_contour_b = chi_ax.contour(Xf, Yf, b[step], levels=contour_lvls_b, colors='darkturquoise', alpha=0.5)
chi_contour_t = chi_ax.contour(Xf, Yf, t[step], levels=contour_lvls_t, colors='green', linestyles='--')

chi_divider = make_axes_locatable(chi_ax)
chi_cax = chi_divider.append_axes("right", size="5%", pad=0.05)
chi_cb = plt.colorbar(chi_im, cax=chi_cax)
chi_im.set_clim(-22.5, -12.5)
bg_chi_im.set_clim(-22.5, -12.5)

chi_ax.set_aspect(1)
chi_ax.set_xlabel("$x\, (m)$")
chi_ax.set_ylabel("$z\, (m)$")
chi_ax.set_ylim(gz[idx_min], gz[idx_max])
chi_ax.set_xlim(0.2, 0.4)
chi_ax.set_title("Thermal variance dissipation rate $\\chi$ (log scale)")

""" ----------------------------------------------------------------------------------------------------- """
now = datetime.now()

chi_fig.savefig('/home/cwp29/Documents/talks/atmos_group/figs/chi.png', dpi=200)
eps_fig.savefig('/home/cwp29/Documents/talks/atmos_group/figs/eps.png', dpi=200)
Re_b_fig.savefig('/home/cwp29/Documents/talks/atmos_group/figs/Re_b.png', dpi=200)
Ri_fig.savefig('/home/cwp29/Documents/talks/atmos_group/figs/Ri.png', dpi=200)
plt.show()
