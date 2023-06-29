import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py
import numpy as np
from matplotlib import pyplot as plt
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

    NSAMP = len(th1_xz)
    times = np.array([float(f['th1_xz'][t].attrs['Time']) for t in time_keys])
    f.close()

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
vd /= V
vd_flux /= V

print(F0)
print(md['b0']*md['r0']*md['r0'])

##### ====================== #####

vd = np.where(vd == 0, np.NaN, vd)

with h5py.File(save_dir+"/mean.h5", 'r') as f:
    print("Mean keys: %s" % f.keys())
    bbins = np.array(f['PVD_bbins']['0001'])
    phibins = np.array(f['PVD_phibins']['0001'])

bbins /= B


sx, sy = np.meshgrid(bbins, phibins)

print("Total time steps: %s"%NSAMP)
print("Dimensional times: ",times * N)

steps = [24, 40, 56]

th1_xz = np.where(th1_xz < 1e-3/B, 0, th1_xz)

tracer_thresh = 5e-4
plot_env = np.where(th2_xz <= tracer_thresh, th1_xz, np.NaN)
plot_plume = np.where(th2_xz >= tracer_thresh, th2_xz, np.NaN)
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
fig, axs = plt.subplots(2,3,figsize=(12, 5), constrained_layout=True)

contours_b = np.linspace(0, md['N2']*9*L/B, 16)
contour_lvls_trace = np.linspace(0.01, 0.1, 8)

print("Setting up initial plot...")

min_vol = np.power(md['LX']/md['Nx'], 2) * md['LZ']/md['Nz']
min_vol /= V

min_vol *= (64/len(bbins))**2
print(min_vol)

labels = ["a", "b", "c", "d", "e"]

for d in range(len(steps)):
    im_b_edge = axs[0, d].contour(Xf, Yf, plot_env[steps[d]], levels = contours_b, cmap='cool', alpha=0.8)
    im_b = axs[0,d].contourf(im_b_edge, levels=contours_b, cmap='cool', alpha=0.8, extend='min')
    im_t = axs[0,d].pcolormesh(X,Y,plot_plume[steps[d]], cmap='viridis')

    im_t.set_clim(0, 0.05 * F0/8e-8)

    col = plt.cm.viridis(np.linspace(0,1, 2))[0]
    outline = axs[0,d].contour(Xf, Yf, plot_outline[steps[d]], levels=[0.5], colors=[col], alpha=0.7,
            linewidths=[0.7])

    #vd[steps[d]] = np.where(vd[steps[d]] > 50*min_vol, vd[steps[d]], np.NaN)
    im_scatter = axs[1,d].pcolormesh(sx, sy, vd[steps[d]], cmap='plasma')

    #im_scatter.set_clim(0, 0.6)

    axs[0,d].set_aspect(1)

    sx_nan = sx[~np.isnan(vd[steps[d]]+vd_flux[steps[d]+1])].flatten()
    sy_nan = sy[~np.isnan(vd[steps[d]]+vd_flux[steps[d]+1])].flatten()
    points = np.array(list(zip(sx_nan,sy_nan)))


    print(points)
    input()
    if len(points) > 0:
        points = np.append(points, np.array([[md['N2']*(zmaxs[steps[d]]*L-md['H'])/B, phibins[0]]]), axis=0)
        hull = spatial.ConvexHull(points)

        axs[1,d].plot(points[hull.simplices[0],0], points[hull.simplices[0],1], 'r--', label="convex envelope")
        for simplex in hull.simplices:
            axs[1,d].plot(points[simplex,0], points[simplex,1], 'r--')

    strat_cont = axs[0, d].contour(Xf, Yf, plot_env[steps[d]], levels=[0], colors=['gray'], alpha=0.5)
    if d == len(steps)-1:
        axs[1,d].legend()
        cb_vd = plt.colorbar(im_scatter, ax = axs[1,d], label=r"W")
        cb_waves = fig.colorbar(im_b, ax = axs[0,d], location='right', shrink=0.7,
            label="buoyancy")
        cb_plume = fig.colorbar(im_t, ax = axs[0,d], location='right', shrink=0.7, label="tracer concentration",
            extend='max')
        cb_waves.add_lines(strat_cont)
    if d == 0:
        axs[0,d].set_ylabel("$z$")
        axs[1,d].set_ylabel("tracer conc.")

    axs[1,d].set_xlabel("buoyancy")

    axs[0,d].set_xlim(-0.1/L, 0.1/L)
    axs[0,d].set_ylim(-0.6, 5.5)

    axs[0,d].set_xlabel("$x$")
    axs[0,d].set_title("({1}) t = {0:.2f}".format(times[steps[d]], labels[d]))


#plt.savefig('/home/cwp29/Documents/papers/draft/figs/vd_evol.pdf')
#plt.savefig('/home/cwp29/Documents/papers/draft/figs/vd_evol.png', dpi=300)
plt.show()
