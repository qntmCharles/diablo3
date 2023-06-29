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
    pvd = np.array([np.array(f['pvd'][t]) for t in time_keys])

    vd = np.array([np.array(f['td_scatter'][t]) for t in time_keys])
    vd_flux = np.array([np.array(f['td_flux'][t]) for t in time_keys])

    NSAMP = len(th1_xz)
    times = np.array([float(f['th1_xz'][t].attrs['Time']) for t in time_keys])
    f.close()

with h5py.File(save_dir+"/mean.h5", 'r') as f:
    bbins = np.array(f['PVD_bbins']['0001'])
    phibins = np.array(f['PVD_phibins']['0001'])

for i in range(1,NSAMP):
    vd_flux[i] += vd_flux[i-1]

db = bbins[1] - bbins[0]
dphi = phibins[1] - phibins[0]

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
Y /= L
Yf -= md['H']
Yf /= L

times /= T
vd /= V
vd_flux /= V

th1_xz /= B
bbins /= B
db  /= B

print(F0)
print(md['b0']*md['r0']*md['r0'])

##### ====================== #####

pvd = np.where(pvd == 0, np.NaN, pvd)
#vd = np.where(vd == 0, np.NaN, vd)
#vd_flux = np.where(vd_flux == 0, np.NaN, vd_flux)

with h5py.File(save_dir+"/mean.h5", 'r') as f:
    print("Mean keys: %s" % f.keys())
    bbins = np.array(f['PVD_bbins']['0001'])
    phibins = np.array(f['PVD_phibins']['0001'])

bbins /= B

sx, sy = np.meshgrid(bbins, phibins)

print("Total time steps: %s"%NSAMP)
print("Dimensional times: ",times)

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

print(zmaxs)

print("Setting up data arrays...")
fig, axs = plt.subplots(1,2, figsize=(12, 3), height_ratios=[1], constrained_layout=True)

contours_b = np.linspace(0, md['N2']*9*L/B, 16)
contour_lvls_trace = np.linspace(0.01, 0.1, 8)

print("Setting up initial plot...")

min_vol = np.power(md['LX']/md['Nx'], 2) * md['LZ']/md['Nz']
min_vol /= V
print(min_vol)

# thresholding to remove artefacts
vd = np.where(vd < 50 * min_vol, 0, vd)
vd_flux = np.where(vd_flux < 50 * min_vol, 0, vd_flux)

#colourmap
cvals = [-1, 0, 1]
custom_colors = ["b","g","r"]
norm=plt.Normalize(min(cvals),max(cvals))
tuples = list(zip(map(norm,cvals), custom_colors))
custom_cmap = colors.LinearSegmentedColormap.from_list("", tuples)

#############################################################################################################
Omega_thresh = 5e-3
#############################################################################################################

nodes = [-np.nanmax(pvd[step]), 0, 0, Omega_thresh, Omega_thresh, np.nanmax(pvd[step])]
custom_colors = ["blue", plt.cm.Blues(0.5), plt.cm.Greens(0.5), "green", plt.cm.Reds(0.5),
        "red"]
norm = plt.Normalize(min(nodes), max(nodes))
custom_cmap = colors.LinearSegmentedColormap.from_list("", list(zip(map(norm,nodes), custom_colors)))


im_b_edge = axs[1].contour(Xf, Yf, plot_env[step], levels = contours_b, cmap='cool', alpha=0.8)
im_b = axs[1].contourf(im_b_edge, levels=contours_b, cmap='cool', alpha=0.8, extend='min')

col = plt.cm.viridis(np.linspace(0,1,2))[0]
outline = axs[1].contour(Xf, Yf, plot_outline[step], levels=[0.5], colors=[col], alpha=0.7,
        linewidths=[0.7])

diff = pvd[step] - ((vd[step] - vd_flux[step])/np.nansum(vd_flux[step]))
print(np.nanmean(diff))
diff = ((vd[step] - vd_flux[step])/np.nansum(vd_flux[step]))
diff = np.where(diff == 0, np.nan, diff)

im_scatter = axs[0].pcolormesh(sx, sy, diff, cmap=custom_cmap, norm=norm)
        #norm=colors.CenteredNorm(halfrange = 0.05*np.nanmax(np.abs(pvd))))

plot_array = np.copy(plot_plume[step])
for i in range(int(md['Nb'])):
    for j in range(int(md['Nphi'])):
        plot_array[np.logical_and(np.logical_and(plot_plume_b[step] > bbins[i] - db/2,
        plot_plume_b[step] <= bbins[i] + db/2),np.logical_and(plot_plume[step] > phibins[j] - dphi/2,
        plot_plume[step] <= phibins[j] + dphi/2))] = diff[j,i]
im_t = axs[1].pcolormesh(X,Y,plot_array, cmap=custom_cmap, norm=norm)
        #norm=colors.CenteredNorm(halfrange = 0.05*np.nanmax(np.abs(pvd))))

# add CoM triangles

com_thresh = Omega_thresh
b_com = np.nansum(sx * np.where(diff > com_thresh, diff, 0)) / np.nansum(np.where(diff > com_thresh, diff, 0))
phi_com = np.nansum(sy * np.where(diff > com_thresh, diff, 0)) / np.nansum(np.where(diff > com_thresh, diff, 0))
#axs[0].scatter(b_com, phi_com, color='red', edgecolor='white', marker='^', s=100)

#b_com = np.nansum(sx * np.where(np.logical_and(diff > 0, diff <= Omega_thresh), diff, 0)) / \
        #np.nansum(np.where(np.logical_and(diff > 0, diff <= Omega_thresh), diff, 0))
#phi_com = np.nansum(sy * np.where(np.logical_and(diff > 0, diff <= Omega_thresh), diff, 0)) / \
        #np.nansum(np.where(np.logical_and(diff > 0, diff <= Omega_thresh), diff, 0))
#axs[0].scatter(b_com, phi_com, color='green', edgecolor='white', marker='^', s=100)

b_com = np.nansum(sx * np.where(diff <= 0, diff, 0)) / np.nansum(np.where(diff <= 0, diff, 0))
phi_com = np.nansum(sy * np.where(diff <= 0, diff, 0)) / np.nansum(np.where(diff <= 0, diff, 0))
#axs[0].scatter(b_com, phi_com, color='blue', edgecolor='white', marker='^', s=100)

cax_pvd = axs[0].inset_axes([1.05, 0.1, 0.05, 0.8])
fig.colorbar(im_scatter, ax=axs[0], cax=cax_pvd, label=r"$\hat{\Omega}$")

cax_b = axs[1].inset_axes([1.05, 0.1, 0.05, 0.8])
b_cbar = fig.colorbar(im_b, ax=axs[1], cax=cax_b, label="buoyancy")
strat_cont = axs[1].contour(Xf, Yf, plot_env[step], levels=[0], colors=['gray'], alpha=0.5)
b_cbar.add_lines(strat_cont)

axs[1].set_ylabel("$z$")
axs[0].set_ylabel("tracer conc.")

axs[1].set_aspect(1)
axs[0].set_aspect(bbins[-1]/phibins[-1] * 0.8)
axs[0].set_xlabel("buoyancy")

axs[1].set_xlim(-0.1/L, 0.1/L)
axs[1].set_ylim(-0.6, 5.5)

axs[1].set_xlabel("$x$")

#plt.savefig('/home/cwp29/Documents/papers/draft/figs/pvd_thresh.pdf')
#plt.savefig('/home/cwp29/Documents/papers/draft/figs/pvd_thresh.png', dpi=300)

plt.show()
