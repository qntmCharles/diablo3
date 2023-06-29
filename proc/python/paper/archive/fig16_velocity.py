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
    vd_b_vel = np.array([np.array(f['td_vel_1'][t]) for t in time_keys])
    vd_phi_vel = np.array([np.array(f['td_vel_2'][t]) for t in time_keys])

    pvd = np.array([np.array(f['pvd'][t]) for t in time_keys])

    NSAMP = len(th1_xz)
    times = np.array([float(f['th1_xz'][t].attrs['Time']) for t in time_keys])
    f.close()

with h5py.File(save_dir+"/mean.h5", 'r') as f:
    vd_vol = np.array([np.array(f['td_scatter_vol'][t]) for t in time_keys])

    f.close()

for i in range(1,NSAMP):
    vd_flux[i] += vd_flux[i-1]

##### Non-dimensionalisation #####

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
X /= L
Xf /= L

Y -= md['H']
Yf -= md['H']
Y /= L
Yf /= L

times /= T

th1_xz /= B
#vd /= V
#vd_flux /= V
vd_b_vel /= (B/T)
vd_phi_vel /= 1/T

print(F0)
print(md['b0']*md['r0']*md['r0'])

##### ====================== #####

vd = np.where(vd == 0, np.NaN, vd)
pvd = np.where(pvd == 0, np.NaN, pvd)

with h5py.File(save_dir+"/mean.h5", 'r') as f:
    print("Mean keys: %s" % f.keys())
    bbins = np.array(f['PVD_bbins']['0001'])
    phibins = np.array(f['PVD_phibins']['0001'])

bbins /= B
db = bbins[1] - bbins[0]
dphi = phibins[1] - phibins[0]

phi_min = phibins[0] - dphi/2
phi_max = phibins[-1] - dphi/2

b_min = bbins[0] - db/2
b_max = bbins[-1] - db/2

sx, sy = np.meshgrid(bbins, phibins)

print("Total time steps: %s"%NSAMP)
print("Dimensional times: ",times * N)

steps = [24, 40, 56]

th1_xz = np.where(th1_xz < 1e-3/B, 0, th1_xz)

tracer_thresh = 5e-4
plot_env = np.where(th2_xz <= tracer_thresh, th1_xz, np.NaN)
plot_plume = np.where(th2_xz >= tracer_thresh, th2_xz, np.NaN)
plot_plume_b = np.where(th2_xz >= tracer_thresh, th1_xz, np.NaN)
plot_outline = np.where(th2_xz <= tracer_thresh, 1, 0)

zmaxs = []
for i in range(len(plot_outline)):
    heights = []
    for j in range(md['Nx']):
        stuff = np.where(plot_outline[i,:,j] == 0)[-1]
        if len(stuff) == 0:
            heights.append(0)
        else:
            heights.append(gzf[max(stuff)])
    zmaxs.append(max(heights))

print(zmaxs)

print("Setting up data arrays...")
fig, axs = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)

contours_b = np.linspace(0, md['N2']*9*L/B, 16)
contour_lvls_trace = np.linspace(0.01, 0.1, 8)

print("Setting up initial plot...")

#min_vol = np.power(md['LX']/md['Nx'], 2) * md['LZ']/md['Nz']
#min_vol /= V
#min_vol = 0
#print(min_vol)

labels = ["a", "b", "c", "d", "e"]

for d in range(len(steps)):
    Omega_thresh = 1.5e-3
    nodes = [-np.nanmax(pvd[steps[d]]), 0, 0, Omega_thresh, Omega_thresh, np.nanmax(pvd[steps[d]])]
    custom_colors = ["blue", plt.cm.Blues(0.5), plt.cm.Greens(0.5), "green", plt.cm.Reds(0.5),
            "red"]
    norm = plt.Normalize(min(nodes), max(nodes))
    custom_cmap = colors.LinearSegmentedColormap.from_list("", list(zip(map(norm,nodes), custom_colors)))

    im_b_edge = axs[d, 0].contour(Xf, Yf, plot_env[steps[d]], levels = contours_b, cmap='cool', alpha=0.8)
    im_b = axs[d, 0].contourf(im_b_edge, levels=contours_b, cmap='cool', alpha=0.8, extend='min')
    #im_t = axs[d, 0].pcolormesh(X,Y,plot_plume[steps[d]], cmap='viridis')
    #im_t.set_clim(0, 0.05 * F0/8e-8)

    plot_array = np.copy(plot_plume[steps[d]])
    for i in range(int(md['Nb'])):
        for j in range(int(md['Nphi'])):
            plot_array[np.logical_and(np.logical_and(plot_plume_b[steps[d]] > bbins[i] - db/2,
            plot_plume_b[steps[d]] <= bbins[i] + db/2),np.logical_and(plot_plume[steps[d]] > phibins[j] - dphi/2,
            plot_plume[steps[d]] <= phibins[j] + dphi/2))] = pvd[steps[d], j,i]

    im_pvd = axs[d, 0].pcolormesh(X,Y, plot_array, cmap=custom_cmap, norm=norm)
    im_pvd_bphi = axs[d, 1].pcolormesh(sx, sy, pvd[steps[d]], cmap=custom_cmap, norm=norm)

    col = plt.cm.viridis(np.linspace(0,1, 2))[0]
    outline = axs[d, 0].contour(Xf, Yf, plot_outline[steps[d]], levels=[0.5], colors=[col], alpha=0.7,
            linewidths=[0.7])

    im_scatter = axs[d, 2].pcolormesh(sx, sy, vd[steps[d]]/vd_vol[steps[d]], cmap='plasma')
    im_scatter_2 = axs[d, 3].pcolormesh(sx, sy, vd[steps[d]]/vd_vol[steps[d]], cmap='plasma')

    com_thresh = Omega_thresh
    b_com = np.nansum(sx * np.where(pvd[steps[d]] > com_thresh, pvd[steps[d]], 0)) / np.nansum(
            np.where(pvd[steps[d]] > com_thresh, pvd[steps[d]], 0))
    phi_com = np.nansum(sy * np.where(pvd[steps[d]] > com_thresh, pvd[steps[d]], 0)) / np.nansum(
            np.where(pvd[steps[d]] > com_thresh, pvd[steps[d]], 0))
    axs[d, 1].scatter(b_com, phi_com, color='red', edgecolor='white', marker='^', s=100)
    axs[d, 3].scatter(b_com, phi_com, color='red', edgecolor='white', marker='^', s=100)

    b_com = np.nansum(sx * np.where(pvd[steps[d]] <= 0, pvd[steps[d]], 0)) / np.nansum(
            np.where(pvd[steps[d]] <= 0, pvd[steps[d]], 0))
    phi_com = np.nansum(sy * np.where(pvd[steps[d]] <= 0, pvd[steps[d]], 0)) / np.nansum(
            np.where(pvd[steps[d]] <= 0, pvd[steps[d]], 0))
    axs[d, 1].scatter(b_com, phi_com, color='blue', edgecolor='white', marker='^', s=100)
    axs[d, 3].scatter(b_com, phi_com, color='blue', edgecolor='white', marker='^', s=100)

    # plot b, phi velocity
    b_vel = vd_vol[steps[d]]*vd_b_vel[steps[d]]/vd[steps[d]]
    phi_vel = vd_vol[steps[d]]*vd_phi_vel[steps[d]]/vd[steps[d]]
    axs[d, 2].quiver(sx[::2,::2], sy[::2, ::2], b_vel[::2, ::2],
            phi_vel[::2, ::2], pvd[steps[d], ::2, ::2],
            angles='xy', units='xy', scale=1, cmap=custom_cmap, norm=norm)

    # plot b, phi streamlines
    stream = axs[d, 3].streamplot(sx, sy, b_vel, phi_vel,
        color=pvd[steps[d]], cmap=custom_cmap, norm=norm, density=5)

    strat_cont = axs[d, 0].contour(Xf, Yf, plot_env[steps[d]], levels=[0], colors=['gray'], alpha=0.5)

    # Row label
    pad = 5
    axs[d, 0].annotate("({1}) t = {0:.0f}".format(times[steps[d]], labels[d]), xy=(0, 0.5),
            xytext=(-axs[d,0].yaxis.labelpad - pad, 0), xycoords = axs[d,0].yaxis.label,
            rotation = 90, textcoords='offset points', size='large', ha='right', va='center')

    # PVD spatial plot axs[d, 0]
    axs[d, 0].set_xlabel("$x$")
    axs[d, 0].set_ylabel("$z$")

    axs[d, 0].set_xlim(-0.1/L, 0.1/L)
    axs[d, 0].set_ylim(-0.6, 5.5)
    axs[d, 0].set_aspect(1)

    # PVD b-phi plot axs[d, 1]
    axs[d, 1].set_xlabel("$b$")
    axs[d, 1].set_ylabel("$\phi$")

    axs[d, 1].set_xlim(b_min, 4)#b_max)
    axs[d, 1].set_ylim(phi_min, phi_max)

    # VD with b-phi velocity axs[d, 2]
    axs[d, 2].set_xlabel("$b$")
    axs[d, 2].set_ylabel("$\phi$")

    axs[d, 2].set_xlim(b_min, 4)#b_max)
    axs[d, 2].set_ylim(phi_min, phi_max)

    # VD with b-phi streamlines axs[d, 3]
    axs[d, 3].set_xlabel("$b$")
    axs[d, 3].set_ylabel("$\phi$")

    axs[d, 3].set_xlim(b_min, 4)#b_max)
    axs[d, 3].set_ylim(phi_min, phi_max)

    if d == len(steps)-1:
        fig.colorbar(im_b, ax = axs[d, 0], label="buoyancy", location='bottom')

        fig.colorbar(im_pvd, ax = axs[d, 1], label=r"$\hat{\Omega}$", location='bottom')

        fig.colorbar(im_scatter, ax = axs[d, 2], label=r"$W$", location='bottom')

        #axs[d, 2].legend()
        #cb_vd = plt.colorbar(im_scatter, ax = axs[d, 2], label=r"W")
        #cb_waves = fig.colorbar(im_b, ax = axs[d, 0], location='right', shrink=0.7,
            #label="buoyancy")
        #cb_pvd = fig.colorbar(im_pvd, ax=axs[0, d], label=r"$\hat{\Omega}$")
        #cb_waves.add_lines(strat_cont)

fig = plt.figure()

Omega_thresh = 2e-3
nodes = [-np.nanmax(pvd[steps[-1]]), 0, 0, Omega_thresh, Omega_thresh, np.nanmax(pvd[steps[-1]])]
custom_colors = ["blue", plt.cm.Blues(0.5), plt.cm.Greens(0.5), "green", plt.cm.Reds(0.5),
        "red"]
norm = plt.Normalize(min(nodes), max(nodes))
custom_cmap = colors.LinearSegmentedColormap.from_list("", list(zip(map(norm,nodes), custom_colors)))

#vd[steps[-1]] = np.where(vd[steps[-1]] > 50*min_vol, vd[steps[-1]], np.NaN)
im_scatter = plt.pcolormesh(sx, sy, vd[steps[-1]]/vd_vol[steps[-1]], cmap='plasma')

com_thresh = Omega_thresh
b_com = np.nansum(sx * np.where(pvd[steps[-1]] > com_thresh, pvd[steps[-1]], 0)) / np.nansum(
        np.where(pvd[steps[-1]] > com_thresh, pvd[steps[-1]], 0))
phi_com = np.nansum(sy * np.where(pvd[steps[-1]] > com_thresh, pvd[steps[-1]], 0)) / np.nansum(
        np.where(pvd[steps[-1]] > com_thresh, pvd[steps[-1]], 0))
plt.scatter(b_com, phi_com, color='red', edgecolor='white', marker='^', s=100)

b_com = np.nansum(sx * np.where(pvd[steps[-1]] <= 0, pvd[steps[-1]], 0)) / np.nansum(
        np.where(pvd[steps[-1]] <= 0, pvd[steps[-1]], 0))
phi_com = np.nansum(sy * np.where(pvd[steps[-1]] <= 0, pvd[steps[-1]], 0)) / np.nansum(
        np.where(pvd[steps[-1]] <= 0, pvd[steps[-1]], 0))
plt.scatter(b_com, phi_com, color='blue', edgecolor='white', marker='^', s=100)

stream = plt.streamplot(sx, sy, vd_vol[steps[-1]]*vd_b_vel[steps[-1]]/vd[steps[-1]],
        vd_vol[steps[-1]]*vd_phi_vel[steps[-1]]/vd[steps[-1]],
        color=pvd[steps[-1]], cmap=custom_cmap, norm=norm, density=10)

plt.legend()

cb_vd = plt.colorbar(im_scatter, label=r"$\hat{W}$")
#cb_vel = plt.colorbar(qq, label=r"$\hat{\Omega}$")

plt.xlabel("buoyancy")
plt.ylabel("tracer conc.")
plt.xlim(b_min, b_max)
plt.ylim(phi_min, phi_max)
plt.tight_layout()

plt.show()
