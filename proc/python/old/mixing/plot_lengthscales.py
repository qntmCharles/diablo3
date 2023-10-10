import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
from functions import get_metadata, read_params, get_grid, get_az_data, g2gf_1d, get_plotindex
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
    eps = np.array([np.array(f['tke_xz'][t]) for t in time_keys])
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

bbar = ndimage.uniform_filter1d(b, size=4, mode='mirror')
bfluc = b - bbar

bfluc2 = ndimage.uniform_filter1d(np.power(b, 2), size=4, mode='mirror') - np.power(bbar, 2)

_,z_coords,_ = np.meshgrid(times, gzf, gxf, indexing='ij', sparse=True)

trac = t
b_env = b[0]

plot_max = 1.5 * md['H']
plot_min = 0.9 * md['H']

idx_minf = get_plotindex(plot_min, gzf)-1
idx_maxf = get_plotindex(plot_max, gzf)+1

idx_min = idx_minf
idx_max = idx_maxf+1

print(idx_min, idx_max)

print(plot_min, gz[idx_min], gzf[idx_minf])

print(plot_max, gz[idx_max], gzf[idx_maxf])

X, Y = np.meshgrid(gx, gz[idx_min:idx_max+1])
Xf, Yf = np.meshgrid(gxf, gzf[idx_minf:idx_maxf+1])

u_z = np.gradient(u, gzf, axis=1)
v_z = np.gradient(v, gzf, axis=1)
w_z = np.gradient(w, gzf, axis=1)
N2 = np.gradient(b, gzf, axis=1)

eps = np.exp(eps) # output is log'd

L_ozmidov = np.sqrt(eps / np.power(md['N2'], 3/2))

L_corrsin = np.sqrt(eps / np.power(np.power(u_z, 2) + np.power(v_z, 2), 3/2))

L_kolmogorov = np.power(np.power(md['NU_RUN'] + nu_t, 3) / eps, 1/4)

L_ellison = np.sqrt(bfluc2)/np.abs(np.gradient(bbar, gzf, axis=1))

L_taylor = np.sqrt((np.power(u, 2) + np.power(v, 2))/(np.power(u_z,2) + np.power(v_z, 2)))

# Restrict to penetration region
L_ozmidov = L_ozmidov[:, idx_min:idx_max, :]
L_corrsin = L_corrsin[:, idx_min:idx_max, :]
L_kolmogorov = L_kolmogorov[:, idx_min:idx_max, :]
L_ellison = L_ellison[:, idx_min:idx_max, :]
L_taylor = L_taylor[:, idx_min:idx_max, :]
b = b[:, idx_min:idx_max, :]
t = t[:, idx_min:idx_max, :]

L_ozmidov = np.log(L_ozmidov)
L_corrsin = np.log(L_corrsin)
L_kolmogorov = np.log(L_kolmogorov)
L_ellison = np.log(L_ellison)
L_taylor = np.log(L_taylor)

print("Total time steps: %s"%NSAMP)
print("Dimensional times: ",times)

print("Setting up data arrays...")
contour_lvls_b = [1e-2, 5e-2, 1e-1, 1.5e-1, 2e-1, 2.5e-1]
contour_lvls_t = [1e-3]

""" ----------------------------------------------------------------------------------------------------- """

step = NSAMP - 1
if step >= NSAMP:
    step = int(input("Chosen timestep out of simulation range. Pick again: "))

""" ----------------------------------------------------------------------------------------------------- """

fig, ax = plt.subplots(2,3, figsize=(18, 7), constrained_layout=True)

for single_ax in ax.ravel():
    single_ax.set_aspect(1)
    single_ax.set_xlabel("$x\, (m)$")
    single_ax.set_ylabel("$z\, (m)$")
    single_ax.set_ylim(gz[idx_min], gz[idx_max])
    single_ax.axhline(md['Lyc']+md['Lyp'],color='grey', linestyle=':')
    single_ax.set_xlim(0.2, 0.4)

""" ----------------------------------------------------------------------------------------------------- """

ozmidov_im = ax[0,0].pcolormesh(X, Y, L_ozmidov[step], cmap='viridis')
ozmidov_contour_b = ax[0,0].contour(Xf, Yf, b[step], levels=contour_lvls_b, colors='white', alpha=0.7)
ozmidov_contour_t = ax[0,0].contour(Xf, Yf, t[step], levels=contour_lvls_t, colors='r', linestyles='--')

ozmidov_divider = make_axes_locatable(ax[0,0])
ozmidov_cax = ozmidov_divider.append_axes("right", size="5%", pad=0.05)
ozmidov_cb = plt.colorbar(ozmidov_im, cax=ozmidov_cax)
ax[0,0].set_title("ozmidov")
ozmidov_im.set_clim(-12, -4)

corrsin_im = ax[0,1].pcolormesh(X, Y, L_corrsin[step], cmap='viridis')
corrsin_contour_b = ax[0,1].contour(Xf, Yf, b[step], levels=contour_lvls_b, colors='white', alpha=0.7)
corrsin_contour_t = ax[0,1].contour(Xf, Yf, t[step], levels=contour_lvls_t, colors='r', linestyles='--')

corrsin_divider = make_axes_locatable(ax[0,1])
corrsin_cax = corrsin_divider.append_axes("right", size="5%", pad=0.05)
corrsin_cb = plt.colorbar(corrsin_im, cax=corrsin_cax)
ax[0,1].set_title("corrsin")
corrsin_im.set_clim(-12, -4)

ellison_im = ax[0,2].pcolormesh(X, Y, L_ellison[step], cmap='viridis')
ellison_contour_b = ax[0,2].contour(Xf, Yf, b[step], levels=contour_lvls_b, colors='white', alpha=0.7)
ellison_contour_t = ax[0,2].contour(Xf, Yf, t[step], levels=contour_lvls_t, colors='r', linestyles='--')

ellison_divider = make_axes_locatable(ax[0,2])
ellison_cax = ellison_divider.append_axes("right", size="5%", pad=0.05)
ellison_cb = plt.colorbar(ellison_im, cax=ellison_cax)
ax[0,2].set_title("ellison")
#ellison_im.set_clim(0, 0.01)

taylor_im = ax[1,0].pcolormesh(X, Y, L_taylor[step], cmap='viridis')
taylor_contour_b = ax[1,0].contour(Xf, Yf, b[step], levels=contour_lvls_b, colors='white', alpha=0.7)
taylor_contour_t = ax[1,0].contour(Xf, Yf, t[step], levels=contour_lvls_t, colors='r', linestyles='--')

taylor_divider = make_axes_locatable(ax[1,0])
taylor_cax = taylor_divider.append_axes("right", size="5%", pad=0.05)
taylor_cb = plt.colorbar(taylor_im, cax=taylor_cax)
ax[1,0].set_title("taylor")
#taylor_im.set_clim(0, 0.05)

kolmogorov_im = ax[1,1].pcolormesh(X, Y, L_kolmogorov[step], cmap='viridis')
kolmogorov_contour_b = ax[1,1].contour(Xf, Yf, b[step], levels=contour_lvls_b, colors='white', alpha=0.7)
kolmogorov_contour_t = ax[1,1].contour(Xf, Yf, t[step], levels=contour_lvls_t, colors='r', linestyles='--')

kolmogorov_divider = make_axes_locatable(ax[1,1])
kolmogorov_cax = kolmogorov_divider.append_axes("right", size="5%", pad=0.05)
kolmogorov_cb = plt.colorbar(kolmogorov_im, cax=kolmogorov_cax)
ax[1,1].set_title("kolmogorov")
#kolmogorov_im.set_clim(0, 1e-3)

fig.delaxes(ax[1,2])

plt.show()
