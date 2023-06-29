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
    vd_b_vol = np.array([np.array(f['td_vel_1_vol'][t]) for t in time_keys])
    vd_phi_vol = np.array([np.array(f['td_vel_2_vol'][t]) for t in time_keys])

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

step = 56

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

contours_b = np.linspace(0, md['N2']*9*L/B, 16)
contour_lvls_trace = np.linspace(0.01, 0.1, 8)

#min_vol = np.power(md['LX']/md['Nx'], 2) * md['LZ']/md['Nz']
#min_vol /= V
#min_vol = 0
#print(min_vol)

Omega_thresh = 1.5e-3
nodes = [-np.nanmax(pvd[step]), 0, 0, Omega_thresh, Omega_thresh, np.nanmax(pvd[step])]
custom_colors = ["blue", plt.cm.Blues(0.5), plt.cm.Greens(0.5), "green", plt.cm.Reds(0.5),
        "red"]
norm = plt.Normalize(min(nodes), max(nodes))
custom_cmap = colors.LinearSegmentedColormap.from_list("", list(zip(map(norm,nodes), custom_colors)))

plot_array = np.copy(plot_plume[step])
for i in range(int(md['Nb'])):
    for j in range(int(md['Nphi'])):
        plot_array[np.logical_and(np.logical_and(plot_plume_b[step] > bbins[i] - db/2,
        plot_plume_b[step] <= bbins[i] + db/2),np.logical_and(plot_plume[step] > phibins[j] - dphi/2,
        plot_plume[step] <= phibins[j] + dphi/2))] = pvd[step, j,i]

com_thresh = Omega_thresh
blue_b_com = np.nansum(sx * np.where(pvd[step] > com_thresh, pvd[step], 0)) / np.nansum(
        np.where(pvd[step] > com_thresh, pvd[step], 0))
blue_phi_com = np.nansum(sy * np.where(pvd[step] > com_thresh, pvd[step], 0)) / np.nansum(
        np.where(pvd[step] > com_thresh, pvd[step], 0))
red_b_com = np.nansum(sx * np.where(pvd[step] <= 0, pvd[step], 0)) / np.nansum(
        np.where(pvd[step] <= 0, pvd[step], 0))
red_phi_com = np.nansum(sy * np.where(pvd[step] <= 0, pvd[step], 0)) / np.nansum(
        np.where(pvd[step] <= 0, pvd[step], 0))

# calculate budget terms
b_vel = vd_vol[step]*vd_b_vel[step]/(vd[step] * vd_b_vol[step])
phi_vel = vd_vol[step]*vd_phi_vel[step]/(vd[step] * vd_phi_vol[step])
What = vd[step]/vd_vol[step]

div_uw = np.gradient(b_vel*What, bbins, axis=1) + \
        np.gradient(phi_vel*What, phibins, axis=0)

mag_uw = np.sqrt(np.power(b_vel*What, 2) + np.power(phi_vel*What, 2))

dWhat_dt = np.gradient(np.array([vd[i]/vd_vol[i] for i in range(NSAMP)]), md['SAVE_STATS_DT'], axis=0)[step]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plt.figure()
plt.title(r"Background $\hat{W}$, quiver $\mathbf{u}$")
plt.pcolormesh(sx, sy, vd[step]/vd_vol[step], cmap='plasma')
plt.quiver(sx[::2,::2], sy[::2, ::2], b_vel[::2, ::2],
        phi_vel[::2, ::2], pvd[step, ::2, ::2],
        angles='xy', units='xy', cmap=custom_cmap, norm=norm)

plt.xlabel("$b$")
plt.ylabel("$\phi$")
plt.xlim(b_min, 4)
plt.ylim(phi_min, phi_max)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plt.figure()
plt.title(r"Background $\hat{W}$, quiver $\hat{W}\mathbf{u}$")
plt.pcolormesh(sx, sy, vd[step]/vd_vol[step], cmap='plasma')
plt.quiver(sx[::2,::2], sy[::2, ::2], b_vel[::2, ::2]*What[::2, ::2],
        What[::2, ::2]*phi_vel[::2, ::2], pvd[step, ::2, ::2],
        angles='xy', units='xy', cmap=custom_cmap, norm=norm)

plt.xlabel("$b$")
plt.ylabel("$\phi$")
plt.xlim(b_min, 4)
plt.ylim(phi_min, phi_max)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plt.figure()
plt.title(r"Magnitude of $\hat{W}\mathbf{u}$")
plt.pcolormesh(sx, sy, mag_uw, cmap='plasma')

plt.xlabel("$b$")
plt.ylabel("$\phi$")
plt.xlim(b_min, 4)
plt.ylim(phi_min, phi_max)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plt.figure()
plt.title(r"Magnitude of $\hat{W}\mathbf{u}$ with streamlines of $\mathbf{u}$")
plt.pcolormesh(sx, sy, mag_uw, cmap='plasma')
plt.streamplot(sx, sy, b_vel, phi_vel, color=pvd[step], cmap=custom_cmap, norm=norm, density=5)

plt.xlabel("$b$")
plt.ylabel("$\phi$")
plt.xlim(b_min, 4)
plt.ylim(phi_min, phi_max)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plt.figure()
plt.title(r"Divergence of $\hat{W}\mathbf{u}$ with streamlines of $\mathbf{u}$")
im_div = plt.pcolormesh(sx, sy, div_uw, cmap='PuOr')
plt.streamplot(sx, sy, b_vel, phi_vel, color=pvd[step], cmap=custom_cmap, norm=norm, density=5)

plt.xlabel("$b$")
plt.ylabel("$\phi$")
plt.xlim(b_min, 4)
plt.ylim(phi_min, phi_max)

plt.colorbar(im_div)
im_div.set_clim(-2e-3, 2e-3)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plt.figure()
plt.title(r"$\partial_T \hat{W}$")
im_dt = plt.pcolormesh(sx, sy, dWhat_dt, cmap='PuOr', norm=colors.CenteredNorm())
#plt.streamplot(sx, sy, b_vel, phi_vel, color=pvd[step], cmap=custom_cmap, norm=norm, density=5)

plt.xlabel("$b$")
plt.ylabel("$\phi$")
plt.xlim(b_min, 4)
plt.ylim(phi_min, phi_max)

plt.colorbar(im_dt)
im_dt.set_clim(-1e-3, 1e-3)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plt.show()
