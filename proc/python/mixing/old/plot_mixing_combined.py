import h5py
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
from functions import get_metadata, read_params, get_grid, g2gf_1d
from scipy import ndimage

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"
save = False

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
    eps = np.array([np.array(f['epsilon_xz'][t]) for t in time_keys])
    kappa = np.array([np.array(f['kappa_t1_xz'][t]) for t in time_keys])
    nu_t = np.array([np.array(f['nu_t_xz'][t]) for t in time_keys])
    diapycvel = np.array([np.array(f['diapycvel1_xz'][t]) for t in time_keys])
    chi = np.array([np.array(f['chi1_xz'][t]) for t in time_keys])
    b = np.array([np.array(f['th1_xz'][t]) for t in time_keys])
    t = np.array([np.array(f['th2_xz'][t]) for t in time_keys])
    u = np.array([np.array(f['u_xz'][t]) for t in time_keys])
    v = np.array([np.array(f['v_xz'][t]) for t in time_keys])
    w = np.array([np.array(f['w_xz'][t]) for t in time_keys])

    eps = g2gf_1d(md, eps)
    kappa = g2gf_1d(md, kappa)
    nu_t = g2gf_1d(md, nu_t)
    diapycvel = g2gf_1d(md, diapycvel)
    chi = g2gf_1d(md, chi)
    b = g2gf_1d(md, b)
    t = g2gf_1d(md, t)
    u = g2gf_1d(md, u)
    v = g2gf_1d(md, v)
    w = g2gf_1d(md, w)

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
N2 = np.gradient(b, gzf, axis=1)

Ri = 2*np.where(np.logical_and(z_coords < md['H'], b<1e-3), np.inf, N2)/(np.power(u_z, 2) + np.power(v_z, 2))

e = kappa * diapycvel
B = bfluc * wfluc
eps *= (md['nu'] + nu_t)/md['nu']
#Re_b = eps/((md['nu'] + nu_t)*np.abs(np.where(np.logical_and(z_coords < md['H'], b<1e-3), 1e5, N2)))
Re_b = eps/((md['nu'] + nu_t)*np.abs(np.where(np.logical_and(z_coords < md['H'], b<1e-3), 1e5, md['N2'])))
#Re_b = eps/((md['nu'] + nu_t)*np.abs(N2))
Re_b = np.log(Re_b)

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


# Adjustment for sgs effects
eps = np.log(eps)

for i in range(NSAMP):
    eps[i] = ndimage.gaussian_filter(eps[i], 0.7)
    e[i] = ndimage.gaussian_filter(e[i], 0.7)

print("Total time steps: %s"%NSAMP)
print("Dimensional times: ",times)

print("Setting up data arrays...")
contour_lvls_b = [1e-2, 5e-2, 1e-1, 1.5e-1, 2e-1, 2.5e-1]
contour_lvls_t = [1e-3]

""" ----------------------------------------------------------------------------------------------------- """

fig, ax = plt.subplots(3,2, figsize=(16, 8))
fig.suptitle("time = 0.00 s")

ax[0,0].set_aspect(1)
ax[1,0].set_aspect(1)
ax[2,0].set_aspect(1)
ax[0,1].set_aspect(1)
ax[1,1].set_aspect(1)
ax[2,1].set_aspect(1)

ax[0,0].set_xlabel("$x$")
ax[0,0].set_ylabel("$z$")
ax[0,0].set_ylim(gz[idx_min], gz[idx_max])
ax[0,0].axhline(md['Lyc']+md['Lyp'],color='grey', linestyle=':')

ax[1,0].set_xlabel("$x$")
ax[1,0].set_ylabel("$z$")
ax[1,0].set_ylim(gz[idx_min], gz[idx_max])
ax[1,0].axhline(md['Lyc']+md['Lyp'],color='grey', linestyle=':')

ax[2,0].set_xlabel("$x$")
ax[2,0].set_ylabel("$z$")
ax[2,0].set_ylim(gz[idx_min], gz[idx_max])
ax[2,0].axhline(md['Lyc']+md['Lyp'],color='grey', linestyle=':')

ax[0,1].set_xlabel("$x$")
ax[0,1].set_ylabel("$z$")
ax[0,1].set_ylim(gz[idx_min], gz[idx_max])
ax[0,1].axhline(md['Lyc']+md['Lyp'],color='grey', linestyle=':')

ax[1,1].set_xlabel("$x$")
ax[1,1].set_ylabel("$z$")
ax[1,1].set_ylim(gz[idx_min], gz[idx_max])
ax[1,1].axhline(md['Lyc']+md['Lyp'],color='grey', linestyle=':')

ax[2,1].set_xlabel("$x$")
ax[2,1].set_ylabel("$z$")
ax[2,1].set_ylim(gz[idx_min], gz[idx_max])
ax[2,1].axhline(md['Lyc']+md['Lyp'],color='grey', linestyle=':')

""" ----------------------------------------------------------------------------------------------------- """

e_im = ax[0,0].pcolormesh(X, Y, e[-1], cmap='bwr')
e_contour_b = ax[0,0].contour(Xf, Yf, b[-1], levels=contour_lvls_b, colors='grey', alpha=0.5)
e_contour_t = ax[0,0].contour(Xf, Yf, t[-1], levels=contour_lvls_t, colors='green', linestyles='--')

e_divider = make_axes_locatable(ax[0,0])
e_cax = e_divider.append_axes("right", size="5%", pad=0.05)
e_cb = plt.colorbar(e_im, cax=e_cax)
e_max = 0.2*np.max(np.abs(e[-1]))
e_im.set_clim(-e_max, e_max)

ax[0,0].set_title("Diapycnal flux $\\kappa_t \\nabla^2 b$")

B_im = ax[1,0].pcolormesh(X, Y, B[-1], cmap='bwr')
B_contour_b = ax[1,0].contour(Xf, Yf, b[-1], levels=contour_lvls_b, colors='grey', alpha=0.5)
B_contour_t = ax[1,0].contour(Xf, Yf, t[-1], levels=contour_lvls_t, colors='green', linestyles='--')

B_divider = make_axes_locatable(ax[1,0])
B_cax = B_divider.append_axes("right", size="5%", pad=0.05)
B_cb = plt.colorbar(B_im, cax=B_cax)
B_max = 0.6*np.max(np.abs(B[-1]))
B_im.set_clim(-B_max, B_max)

ax[1,0].set_title("Vertical turbulent density flux $\\langle b'w'\\rangle$")

eps_im = ax[2,0].pcolormesh(X, Y, eps[-1], cmap='hot_r')
eps_contour_b = ax[2,0].contour(Xf, Yf, b[-1], levels=contour_lvls_b, colors='darkturquoise', alpha=0.7)
eps_contour_t = ax[2,0].contour(Xf, Yf, t[-1], levels=contour_lvls_t, colors='green', linestyles='--')

eps_divider = make_axes_locatable(ax[2,0])
eps_cax = eps_divider.append_axes("right", size="5%", pad=0.05)
eps_cb = plt.colorbar(eps_im, cax=eps_cax)
eps_im.set_clim(-20, -5)

ax[2,0].set_title("TKE dissipation rate $\\varepsilon$")

chi_im = ax[0,1].pcolormesh(X, Y, chi[-1], cmap='hot_r')
chi_contour_b = ax[0,1].contour(Xf, Yf, b[-1], levels=contour_lvls_b, colors='darkturquoise', alpha=0.7)
chi_contour_t = ax[0,1].contour(Xf, Yf, t[-1], levels=contour_lvls_t, colors='green', linestyles='--')

chi_divider = make_axes_locatable(ax[0,1])
chi_cax = chi_divider.append_axes("right", size="5%", pad=0.05)
chi_cb = plt.colorbar(chi_im, cax=chi_cax)
chi_max = 0.2*np.max(chi[-1])
chi_im.set_clim(0, chi_max)

ax[0,1].set_title("Thermal variance dissipation rate $\\chi$")

Ri_im = ax[1,1].pcolormesh(X, Y, Ri[-1], cmap='hot')
Ri_contour_b = ax[1,1].contour(Xf, Yf, b[-1], levels=contour_lvls_b, colors='darkturquoise', alpha=0.7)
Ri_contour_t = ax[1,1].contour(Xf, Yf, t[-1], levels=contour_lvls_t, colors='green', linestyles='--')

Ri_divider = make_axes_locatable(ax[1,1])
Ri_cax = Ri_divider.append_axes("right", size="5%", pad=0.05)
Ri_cb = plt.colorbar(Ri_im, cax=Ri_cax)
Ri_im.set_clim(0, 0.25)

ax[1,1].set_title("Richardson number $Ri$")

"""
trac_im = ax[2,1].pcolormesh(X, Y, trac[-1], cmap='jet')
trac_contour_b = ax[2,1].contour(Xf, Yf, b[-1], levels=contour_lvls_b, colors='grey', alpha=0.5)
trac_contour_t = ax[2,1].contour(Xf, Yf, t[-1], levels=contour_lvls_t, colors='red', linestyles='--')

trac_divider = make_axes_locatable(ax[2,1])
trac_cax = trac_divider.append_axes("right", size="5%", pad=0.05)
trac_cb = plt.colorbar(trac_im, cax=trac_cax)
trac_im.set_clim(0, 0.04)

ax[2,1].set_title("Tracer")
"""

Re_b_im = ax[2,1].pcolormesh(X, Y, Re_b[-1], cmap='hot_r')
Re_b_contour_b = ax[2,1].contour(Xf, Yf, b[-1], levels=contour_lvls_b, colors='darkturquoise', alpha=0.7)
Re_b_contour_t = ax[2,1].contour(Xf, Yf, t[-1], levels=contour_lvls_t, colors='green', linestyles='--')

Re_b_divider = make_axes_locatable(ax[2,1])
Re_b_cax = Re_b_divider.append_axes("right", size="5%", pad=0.05)
Re_b_cb = plt.colorbar(Re_b_im, cax=Re_b_cax)
Re_b_max = 0.9*np.max(np.abs(Re_b[-1]))
Re_b_im.set_clim(0, Re_b_max)

ax[2,1].set_title("Buoyancy Reynolds number $\\mathrm{{Re}}_b$ (log scale)")


""" ----------------------------------------------------------------------------------------------------- """

plt.tight_layout()

def animate(step):
    global e_contour_b, e_contour_t, B_contour_b, B_contour_t, eps_contour_b, eps_contour_t
    global chi_contour_b, chi_contour_t, Ri_contour_b, Ri_contour_t, Re_b_contour_b, Re_b_contour_t

    e_im.set_array(e[step].ravel())
    B_im.set_array(B[step].ravel())
    eps_im.set_array(eps[step].ravel())
    chi_im.set_array(chi[step].ravel())
    Ri_im.set_array(Ri[step].ravel())
    Re_b_im.set_array(Re_b[step].ravel())

    fig.suptitle("time = {0:.2f} s".format(times[step]))

    for coll in e_contour_t.collections:
        coll.remove()
    for coll in e_contour_b.collections:
        coll.remove()
    for coll in B_contour_t.collections:
        coll.remove()
    for coll in B_contour_b.collections:
        coll.remove()
    for coll in eps_contour_t.collections:
        coll.remove()
    for coll in eps_contour_b.collections:
        coll.remove()
    for coll in chi_contour_t.collections:
        coll.remove()
    for coll in chi_contour_b.collections:
        coll.remove()
    for coll in Ri_contour_t.collections:
        coll.remove()
    for coll in Ri_contour_b.collections:
        coll.remove()
    for coll in Re_b_contour_t.collections:
        coll.remove()
    for coll in Re_b_contour_b.collections:
        coll.remove()

    e_contour_b = ax[0,0].contour(Xf, Yf, b[step], levels=contour_lvls_b, colors='grey', alpha=0.5)
    e_contour_t = ax[0,0].contour(Xf, Yf, t[step], levels=contour_lvls_t, colors='green', linestyles='--')
    B_contour_b = ax[1,0].contour(Xf, Yf, b[step], levels=contour_lvls_b, colors='grey', alpha=0.5)
    B_contour_t = ax[1,0].contour(Xf, Yf, t[step], levels=contour_lvls_t, colors='green', linestyles='--')
    eps_contour_b = ax[2,0].contour(Xf, Yf, b[step], levels=contour_lvls_b, colors='grey', alpha=0.5)
    eps_contour_t = ax[2,0].contour(Xf, Yf, t[step], levels=contour_lvls_t, colors='red', linestyles='--')
    chi_contour_b = ax[0,1].contour(Xf, Yf, b[step], levels=contour_lvls_b, colors='grey', alpha=0.5)
    chi_contour_t = ax[0,1].contour(Xf, Yf, t[step], levels=contour_lvls_t, colors='g', linestyles='--')
    Ri_contour_b = ax[1,1].contour(Xf, Yf, b[step], levels=contour_lvls_b, colors='grey', alpha=0.5)
    Ri_contour_t = ax[1,1].contour(Xf, Yf, t[step], levels=contour_lvls_t, colors='green', linestyles='--')
    Re_b_contour_b = ax[2,1].contour(Xf, Yf, b[step], levels=contour_lvls_b, colors='grey', alpha=0.5)
    Re_b_contour_t = ax[2,1].contour(Xf, Yf, t[step], levels=contour_lvls_t, colors='red', linestyles='--')

    return e_im, B_im, eps_im, chi_im, Ri_im, Re_b_im,

print("Initialising mp4 writer...")
Writer = animation.writers['ffmpeg']
writer = Writer(fps=4, bitrate=1800)

print("Starting plot...")
anim = animation.FuncAnimation(fig, animate, interval=250, frames=NSAMP)
now = datetime.now()

if save:
    anim.save(save_dir+'mixing_%s.mp4'%now.strftime("%d-%m-%Y"),writer=writer,dpi=200)
else:
    plt.show()
