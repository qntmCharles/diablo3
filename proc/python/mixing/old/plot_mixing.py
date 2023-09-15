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
    print("Movie keys: %s picture" % f.keys())
    time_keys = list(f['th1_xz'])
    print(time_keys)
    # Get buoyancy data
    eps = np.array([np.array(f['epsilon_xz'][t]) for t in time_keys])
    kappa = np.array([np.array(f['kappa_t1_xz'][t]) for t in time_keys])
    nu_t = np.array([np.array(f['nu_t_xz'][t]) for t in time_keys])
    diapycvel = np.array([np.array(f['diapycvel1_xz'][t]) for t in time_keys])
    #diapycvel2 = np.array([np.array(f['diapycvel_lhs1_xz'][t]) for t in time_keys])
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

"""
with h5py.File(join(save_dir,"mean.h5"), 'r') as f:
    print("Mean keys: %s picture" % f.keys())
    mtime_keys = list(f['bbins'])
    kappa_net = np.array([f['kappa1_net'][t] for t in time_keys])
    kappa_sgs = np.array([f['kappa_sgs'][t] for t in time_keys])
    nu_sgs = np.array([f['nu_sgs'][t] for t in time_keys])
print(kappa_net.shape)
print(kappa_sgs.shape)

kappa_sgs_mean = np.average(kappa_sgs, weights=dz, axis=1)
kappa_infer = (md['nu'] + np.average(nu_sgs, weights=dz, axis=1))/0.7

fig, ax = plt.subplots(1,2)
ax[0].plot(times, kappa_net, color='b', label="$\\kappa_{{net}}$ (Penney)")
ax[0].plot(times, kappa_sgs_mean, color='r', linestyle='--', label="$\\kappa_t$")
ax[0].plot(times, kappa_infer, color='g', linestyle=':', label="$(\\nu + \\nu_t)/Pr$")
ax[0].legend()
ax[0].set_title("Diffusivities")

for i in range(len(kappa_net)):
    kappa_net[i] /= kappa_sgs_mean[i]
    kappa_infer[i] /= kappa_sgs_mean[i]


ax[1].plot(times, kappa_net, color='b')
ax[1].axhline(1, color='r', linestyle='--')
ax[1].plot(times, kappa_infer, color='g', linestyle=':')
ax[1].set_title("Normalised by sgs diffusivty")

plt.tight_layout()
plt.show()
"""

_,z_coords,_ = np.meshgrid(times, gzf, gxf, indexing='ij', sparse=True)

b_env = b[0]

plot_max = 0.2 + md['H']
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
B = (b-b_env)*w
eps *= (md['nu'] + nu_t)/md['nu']
Re_b = eps/((md['nu'] + nu_t)*md['N2'])

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
#contour_lvls_t = [1e-4, 1e-3, 1e-2]

""" ----------------------------------------------------------------------------------------------------- """

""" ----------------------------------------------------------------------------------------------------- """

e_fig = plt.figure(figsize=(10,4))
e_fig.suptitle("time = 0.00 s")
e_ax = plt.gca()
e_ax.set_aspect(1)

e_im = plt.pcolormesh(X, Y, e[-1], cmap='bwr')

e_contour_b = plt.contour(Xf, Yf, b[-1], levels=contour_lvls_b, colors='grey', alpha=0.5)
e_contour_t = plt.contour(Xf, Yf, t[-1], levels=contour_lvls_t, colors='green', linestyles='--')

plt.axhline(md['Lyc']+md['Lyp'],color='grey', linestyle=':')
plt.ylim(gz[idx_min], gz[idx_max])

e_divider = make_axes_locatable(e_ax)
e_cax = e_divider.append_axes("right", size="5%", pad=0.05)
e_cb = plt.colorbar(e_im, cax=e_cax)
e_max = 0.6* np.max(np.abs(e[-1]))
e_im.set_clim(-e_max, e_max)

e_ax.set_xlabel("$x$")
e_ax.set_xlabel("$z$")

e_ax.set_title("Diapycnal flux $\\kappa_t \\nabla^2 b$")
plt.tight_layout()

""" ----------------------------------------------------------------------------------------------------- """

B_fig = plt.figure(figsize=(10,4))
B_fig.suptitle("time = 0.00 s")
B_ax = plt.gca()
B_ax.set_aspect(1)

B_im = plt.pcolormesh(X, Y, B[-1], cmap='bwr')

B_contour_b = plt.contour(Xf, Yf, b[-1], levels=contour_lvls_b, colors='grey', alpha=0.5)
B_contour_t = plt.contour(Xf, Yf, t[-1], levels=contour_lvls_t, colors='green', linestyles='--')

plt.axhline(md['Lyc']+md['Lyp'],color='grey', linestyle=':')
plt.ylim(gz[idx_min], gz[idx_max])

B_divider = make_axes_locatable(B_ax)
B_cax = B_divider.append_axes("right", size="5%", pad=0.05)
B_cb = plt.colorbar(B_im, cax=B_cax)
B_max = np.max(np.abs(B[-1]))
B_im.set_clim(-B_max, B_max)

B_ax.set_xlabel("$x$")
B_ax.set_xlabel("$z$")

B_ax.set_title("Vertical turbulent density flux $\\langle b'w'\\rangle$")
plt.tight_layout()

""" ----------------------------------------------------------------------------------------------------- """

eps_fig = plt.figure(figsize=(10,4))
eps_fig.suptitle("time = 0.00 s")
eps_ax = plt.gca()
eps_ax.set_aspect(1)

eps_im = plt.pcolormesh(X, Y, eps[-1], cmap='jet')

eps_contour_b = plt.contour(Xf, Yf, b[-1], levels=contour_lvls_b, colors='grey', alpha=0.5)
eps_contour_t = plt.contour(Xf, Yf, t[-1], levels=contour_lvls_t, colors='red', linestyles='--')

plt.axhline(md['Lyc']+md['Lyp'],color='grey', linestyle=':')
plt.ylim(gz[idx_min], gz[idx_max])

eps_divider = make_axes_locatable(eps_ax)
eps_cax = eps_divider.append_axes("right", size="5%", pad=0.05)
eps_cb = plt.colorbar(eps_im, cax=eps_cax)

eps_im.set_clim(-20, -5)

eps_ax.set_xlabel("$x$")
eps_ax.set_xlabel("$z$")

eps_ax.set_title("TKE dissipation rate $\\varepsilon$")
plt.tight_layout()

""" ----------------------------------------------------------------------------------------------------- """

chi_fig = plt.figure(figsize=(10,4))
chi_fig.suptitle("time = 0.00 s")
chi_ax = plt.gca()
chi_ax.set_aspect(1)

chi_im = plt.pcolormesh(X, Y, chi[-1], cmap='hot_r')

chi_contour_b = plt.contour(Xf, Yf, b[-1], levels=contour_lvls_b, colors='grey', alpha=0.5)
chi_contour_t = plt.contour(Xf, Yf, t[-1], levels=contour_lvls_t, colors='green', linestyles='--')

plt.axhline(md['Lyc']+md['Lyp'],color='grey', linestyle=':')
plt.ylim(gz[idx_min], gz[idx_max])

chi_divider = make_axes_locatable(chi_ax)
chi_cax = chi_divider.append_axes("right", size="5%", pad=0.05)
chi_cb = plt.colorbar(chi_im, cax=chi_cax)

chi_ax.set_xlabel("$x$")
chi_ax.set_xlabel("$z$")

chi_ax.set_title("Thermal variance dissipation rate $\\chi$")
plt.tight_layout()

""" ----------------------------------------------------------------------------------------------------- """

Re_b_fig = plt.figure(figsize=(10,4))
Re_b_fig.suptitle("time = 0.00 s")
Re_b_ax = plt.gca()
Re_b_ax.set_aspect(1)

Re_b_im = plt.pcolormesh(X, Y, Re_b[-1], cmap='hot_r')

Re_b_contour_b = plt.contour(Xf, Yf, b[-1], levels=contour_lvls_b, colors='grey', alpha=0.5)
Re_b_contour_t = plt.contour(Xf, Yf, t[-1], levels=contour_lvls_t, colors='green', linestyles='--')

plt.axhline(md['Lyc']+md['Lyp'],color='grey', linestyle=':')
plt.ylim(gz[idx_min], gz[idx_max])

Re_b_divider = make_axes_locatable(Re_b_ax)
Re_b_cax = Re_b_divider.append_axes("right", size="5%", pad=0.05)
Re_b_cb = plt.colorbar(Re_b_im, cax=Re_b_cax)
Re_b_max = np.max(np.abs(Re_b[-1]))
Re_b_im.set_clim(0, 1e2)

Re_b_ax.set_xlabel("$x$")
Re_b_ax.set_xlabel("$z$")

Re_b_ax.set_title("Buoyancy Reynolds number $Re_b = \\varepsilon/\\nu N^2$")
plt.tight_layout()

""" ----------------------------------------------------------------------------------------------------- """

Ri_fig = plt.figure(figsize=(10,4))
Ri_fig.suptitle("time = 0.00 s")
Ri_ax = plt.gca()
Ri_ax.set_aspect(1)

Ri_im = plt.pcolormesh(X, Y, Ri[-1], cmap='hot')

Ri_contour_b = plt.contour(Xf, Yf, b[-1], levels=contour_lvls_b, colors='grey', alpha=0.5)
Ri_contour_t = plt.contour(Xf, Yf, t[-1], levels=contour_lvls_t, colors='green', linestyles='--')

plt.axhline(md['Lyc']+md['Lyp'],color='grey', linestyle=':')
plt.ylim(gz[idx_min], gz[idx_max])

Ri_divider = make_axes_locatable(Ri_ax)
Ri_cax = Ri_divider.append_axes("right", size="5%", pad=0.05)
Ri_cb = plt.colorbar(Ri_im, cax=Ri_cax)
Ri_im.set_clim(0, 0.25)

print((plot_max-plot_min)/md['LX'])

Ri_ax.set_xlabel("$x$")
Ri_ax.set_ylabel("$z$")
Ri_ax.set_title("Richardson number $Ri$")
plt.tight_layout()

def eps_animate(step):
    global eps_contour_b, eps_contour_t

    eps_im.set_array(eps[step].ravel())

    eps_fig.suptitle("time = {0:.2f} s".format(times[step]))

    for coll in eps_contour_t.collections:
        coll.remove()
    for coll in eps_contour_b.collections:
        coll.remove()

    eps_contour_b = eps_ax.contour(Xf, Yf, b[step], levels=contour_lvls_b, colors='grey', alpha=0.5)
    eps_contour_t = eps_ax.contour(Xf, Yf, t[step], levels=contour_lvls_t, colors='red', linestyles='--')

    return eps_im,

def chi_animate(step):
    global chi_contour_b, chi_contour_t

    chi_im.set_array(chi[step].ravel())

    chi_fig.suptitle("time = {0:.2f} s".format(times[step]))

    for coll in chi_contour_t.collections:
        coll.remove()
    for coll in chi_contour_b.collections:
        coll.remove()

    chi_contour_b = chi_ax.contour(Xf, Yf, b[step], levels=contour_lvls_b, colors='grey', alpha=0.5)
    chi_contour_t = chi_ax.contour(Xf, Yf, t[step], levels=contour_lvls_t, colors='g', linestyles='--')

    return chi_im,

def Re_b_animate(step):
    global Re_b_contour_b, Re_b_contour_t

    Re_b_im.set_array(Re_b[step].ravel())

    Re_b_fig.suptitle("time = {0:.2f} s".format(times[step]))

    for coll in Re_b_contour_t.collections:
        coll.remove()
    for coll in Re_b_contour_b.collections:
        coll.remove()

    Re_b_contour_b = Re_b_ax.contour(Xf, Yf, b[step], levels=contour_lvls_b, colors='grey', alpha=0.5)
    Re_b_contour_t = Re_b_ax.contour(Xf, Yf, t[step], levels=contour_lvls_t, colors='g', linestyles='--')

    return Re_b_im,

def B_animate(step):
    global B_contour_b, B_contour_t

    B_im.set_array(B[step].ravel())

    B_fig.suptitle("time = {0:.2f} s".format(times[step]))

    for coll in B_contour_t.collections:
        coll.remove()
    for coll in B_contour_b.collections:
        coll.remove()

    B_contour_b = B_ax.contour(Xf, Yf, b[step], levels=contour_lvls_b, colors='grey', alpha=0.5)
    B_contour_t = B_ax.contour(Xf, Yf, t[step], levels=contour_lvls_t, colors='g', linestyles='--')

    return B_im,

def Ri_animate(step):
    global Ri_contour_b, Ri_contour_t

    Ri_im.set_array(Ri[step].ravel())

    Ri_fig.suptitle("time = {0:.2f} s".format(times[step]))

    for coll in Ri_contour_t.collections:
        coll.remove()
    for coll in Ri_contour_b.collections:
        coll.remove()

    Ri_contour_b = Ri_ax.contour(Xf, Yf, b[step], levels=contour_lvls_b, colors='grey', alpha=0.5)
    Ri_contour_t = Ri_ax.contour(Xf, Yf, t[step], levels=contour_lvls_t, colors='g', linestyles='--')

    return Ri_im,

def e_animate(step):
    global e_contour_b, e_contour_t

    e_im.set_array(e[step].ravel())

    e_fig.suptitle("time = {0:.2f} s".format(times[step]))

    for coll in e_contour_t.collections:
        coll.remove()
    for coll in e_contour_b.collections:
        coll.remove()

    e_contour_b = e_ax.contour(Xf, Yf, b[step], levels=contour_lvls_b, colors='grey', alpha=0.5)
    e_contour_t = e_ax.contour(Xf, Yf, t[step], levels=contour_lvls_t, colors='g', linestyles='--')

    return e_im,

print("Initialising mp4 writer...")
Writer = animation.writers['ffmpeg']
writer = Writer(fps=4, bitrate=1800)

print("Starting plot...")
eps_anim = animation.FuncAnimation(eps_fig, eps_animate, interval=250, frames=NSAMP)
chi_anim = animation.FuncAnimation(chi_fig, chi_animate, interval=250, frames=NSAMP)
Re_b_anim = animation.FuncAnimation(Re_b_fig, Re_b_animate, interval=250, frames=NSAMP)
Ri_anim = animation.FuncAnimation(Ri_fig, Ri_animate, interval=250, frames=NSAMP)
B_anim = animation.FuncAnimation(B_fig, B_animate, interval=250, frames=NSAMP)
e_anim = animation.FuncAnimation(e_fig, e_animate, interval=250, frames=NSAMP)
now = datetime.now()

if save:
    e_anim.save(save_dir+'diapycvel_%s.mp4'%now.strftime("%d-%m-%Y"),writer=writer,dpi=200)
    eps_anim.save(save_dir+'tkediss_%s.mp4'%now.strftime("%d-%m-%Y"),writer=writer,dpi=200)
    chi_anim.save(save_dir+'thermdiss_%s.mp4'%now.strftime("%d-%m-%Y"),writer=writer,dpi=200)
    Re_b_anim.save(save_dir+'buoyreynolds_%s.mp4'%now.strftime("%d-%m-%Y"),writer=writer,dpi=200)
    Ri_anim.save(save_dir+'ri_%s.mp4'%now.strftime("%d-%m-%Y"),writer=writer,dpi=200)
    B_anim.save(save_dir+'vertbflux_%s.mp4'%now.strftime("%d-%m-%Y"),writer=writer,dpi=200)
else:
    plt.show()
