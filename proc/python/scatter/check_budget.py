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
from functions import get_metadata, read_params, get_grid, g2gf_1d, compute_F0, get_index
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
vd /= V
#vd_flux /= V
vd_b_vel /= V*(B/T)
vd_phi_vel /= V/T

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

What = np.array([vd[i]/vd_vol[i] for i in range(NSAMP)])
#What = vd

div = []
dWhat_dt = []
for step in range(get_index(10, times), len(times)):
    #b_vel = vd_b_vel[step]/What[step]
    #phi_vel = vd_phi_vel[step]/What[step]

    b_vel = (vd_b_vel[step] / vd_b_vol[step]) / What[step]
    phi_vel = (vd_phi_vel[step] / vd_phi_vol[step]) / What[step]

    vx = b_vel * What[step]
    vy = phi_vel * What[step]

    dvx_dx = np.gradient(vx, db, axis=0)
    dvy_dy = np.gradient(vy, dphi, axis=1)

    div.append(dvx_dx + dvy_dy)

    dWhat_dt.append(np.gradient(What, md['SAVE_STATS_DT'], axis=0)[step])

div = np.array(div)
dWhat_dt = np.array(dWhat_dt)

print(div.shape)
print(dWhat_dt.shape)

for i in range(int(md['Nb'])):
    for j in range(int(md['Nphi'])):
        if not np.isnan(What[-1][i,j]):
            plt.plot(dWhat_dt[:, i,j], times[get_index(10, times):], color='b',
                label=r"$\partial W/\partial t$")
            plt.plot(-div[:, i,j], times[get_index(10, times):], color='r',
                    label=r"$-\nabla \cdot (u_W W)$")
            #plt.plot(dWhat_dt[:, i, j] + div[:, i,j], times[get_index(10, times):], color='k',
                    #label=r"$0$?")

            plt.legend()
            plt.show()

plt.show()
