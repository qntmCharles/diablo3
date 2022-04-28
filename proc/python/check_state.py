import h5py, bisect, time, gc, sys
from os import listdir
from os.path import isfile, join
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from functions import get_metadata, get_grid, read_params, get_az_data
from scipy import integrate, optimize, interpolate
from scipy.ndimage.filters import uniform_filter1d
from matplotlib import cm as cm
from itertools import groupby
from mpl_toolkits.axes_grid1 import make_axes_locatable

##### USER-DEFINED PARAMETERS #####
params_file = "./params.dat"

z_upper = 70 # non-dim, scaled by r_0
z_lower = 20

eps = 0.02
nplots = 5

##### ----------------------- #####

def truncate(var, points, ref_var, ref, trunc_indices):
    # Truncates var at location where ref_var matches ref, computing only at trunc_indices
    # ASSUMES ref_var is decreasing with increasing index
    # Points are the positions of each entry of var
    var_new = np.zeros(shape=(var.shape[0], var.shape[1]+2))
    for i in trunc_indices:
        # Calculate index up to which var is unchanged, after which set to 0
        trunc_idx = max(np.where(ref_var[i,:] > ref[i])[0])
        var_new[i, :trunc_idx+1] = var[i, :trunc_idx+1]

        # Calculate exact point to interpolate
        f = interpolate.interp1d(ref_var[i], points)
        trunc_pt = f(ref[i])

        # Create interpolation function for var
        f = interpolate.interp1d(points, var[i])
        var_new[i, trunc_idx+1] = f(trunc_pt)

    return var_new


##### Get directory locations #####
base_dir, run_dir, save_dir, version = read_params(params_file)
print("Run directory: ", run_dir)
print("Save directory: ", save_dir)

##### Get simulation metadata #####
md = get_metadata(run_dir, version)
print("Complete metadata: ",md)

##### Get grid #####
gxf, gyf, gzf, dz = get_grid(join(run_dir, 'grid.h5'), md)
gzfp = np.flip(gzf)

r_0 = md['r0']
dr = md['LX']/md['Nx']
nbins = int(md['Nx']/2)
r_bins = np.array([r*dr for r in range(0, nbins+1)])
r_points = np.array([0.5*(r_bins[i]+r_bins[i+1]) for i in range(nbins)])

##### Get azimuthal data #####
############################ TODO: check this works okay with very large data files
data = get_az_data(join(save_dir,'az_stats.h5'), md)

ubar = data['u']
vbar = data['v']
wbar = data['w']
bbar = data['b']
pbar = data['p']
print("Data has shape ", bbar.shape)

fig1 = plt.figure()
with h5py.File(save_dir+'az_stats.h5', 'r') as f:
    b = f['b_az']
    time_keys = list(b.keys())
    w = np.array([f['w_az'][t] for t in time_keys])

    cols = plt.cm.rainbow(np.linspace(1,0,len(time_keys)))
    for t,c in zip(time_keys,cols):
        data = b[t][()]
        plt.loglog(data[:,0],gzf,color=c,alpha=0.5)

##### Identify indices where plume is well defined #####

# find indices where centreline velocity is positive
valid_indices = [j for j in range(wbar.shape[0]) if wbar[j,0] > 0]
cont_valid_indices = list(set(list(sum(max([list(y) for i, y in groupby(zip(valid_indices,
    valid_indices[1:]), key = lambda x: (x[1]-x[0]) == 1)], key=len),()))))
if cont_valid_indices[-1] == md['Nz']-1:
    cont_valid_indices.remove(md['Nz']-1)

# check where rtrunc is, if at the edge of domain then remove index from list
plume_indices = []
z_lower_ind = int(min(np.where(gzfp <= z_upper*r_0)[0]))
z_upper_ind = int(max(np.where(z_lower*r_0 <= gzfp)[0]))
for j in cont_valid_indices:
    wtrunc = 0.02*wbar[j, 0]
    rtrunc = max(np.where(wbar[j,:] > wtrunc)[0])
    if (z_lower_ind <= j <= z_upper_ind) and rtrunc != md['Nx']/2-1:
        plume_indices.append(j)

# This array contains the largest continuous run of indices which have a positive centreline velocity
# and valid truncation radius, at z levels between z_lower and z_upper
cont_plume_indices = list(set(list(sum(max([list(y) for i, y in groupby(zip(plume_indices,
    plume_indices[1:]), key = lambda x: (x[1]-x[0]) == 1)], key=len),()))))

##### Truncate data at radius where wbar(r) = eps*wbar(0) #####

r_d = []
wbar_trunc = np.zeros(shape=(wbar.shape[0],wbar.shape[1]+2))
r_integrate = np.zeros(shape=(wbar.shape[0],wbar.shape[1]+2))
for i in plume_indices:
    w_trunc = eps*wbar[i,0]
    r_trunc = max(np.where(wbar[i,:] > wtrunc)[0])

    wbar_trunc[i, :rtrunc+1] = wbar[i, :rtrunc+1]
    wbar_trunc[i, rtrunc+1] = wtrunc

    f = interpolate.interp1d(wbar[i], r_points)
    r_d.append(f(wtrunc))
    r_integrate[i, :rtrunc+1] = r_points[:rtrunc+1]
    r_integrate[i, rtrunc+1] = f(wtrunc)
    r_integrate[i, rtrunc+2] = f(wtrunc)

bbar_trunc = truncate(bbar, r_points, wbar, eps*wbar[:,0], plume_indices)

##### Calculate integral quantities and profile coefficients #####

Q = 2*integrate.trapezoid(wbar_trunc*r_integrate, r_integrate, axis=1)
M = 2*integrate.trapezoid(wbar_trunc*wbar_trunc*r_integrate, r_integrate, axis=1)
F = 2*integrate.trapezoid(wbar_trunc*bbar_trunc*r_integrate, r_integrate, axis=1)
B = 2*integrate.trapezoid(bbar_trunc*r_integrate, r_integrate, axis=1)

r_m = Q/np.sqrt(M)
b_m = B/(r_m*r_m)

plt.loglog(2*b_m, gzf, label="$B/r_m^2$",color='r')
plt.loglog(bbar[:,0], gzf, color='k', linestyle='--', label="b centreline")
fig1.legend()

# Plot sample gradients over b profiles
x_range = plt.gca().get_xlim()
x_split = np.log((x_range[1]-x_range[0])*2)
y_range = plt.gca().get_ylim()
print(x_range)
for i in range(-10,10):
    plt.loglog(np.exp(x_split*i+np.log(gzf[-1]))*np.power(gzf, -5/3), gzf, alpha=0.7, color='grey')

plt.xlim(np.min(bbar[:,0])-0.1*(np.max(bbar[:,0])-np.min(bbar[:,0])),1.5*np.max(bbar[:,0]))
plt.ylim(gzf[0]+0.2*(gzf[-1]-gzf[0]),gzf[-1])
plt.show()

thresh = 3e-1

zs = np.linspace(0,md['LZ'],nplots+2)[1:-1]
z_idxs = [int(zs[i]*md['Nz']/md['LZ']) for i in range(nplots)]
print(zs)

fig2, ax = plt.subplots(2,1,figsize=(15,10))
fig2.suptitle("Vertical velocity steadiness")
cols = plt.cm.rainbow(np.linspace(0,1,nplots))

ts = np.array([md['SAVE_STATS_DT']*(float(t)-1) for t in time_keys])

for i,c in zip(range(nplots),cols):
    print(w.shape)
    data = w[:,z_idxs[i], 0]

    try:
        start = min(np.argwhere(np.abs(data) > thresh))[0]
    except:
        start = 0

    data_trunc = data[start:]
    N_trunc = len(data_trunc)
    ts_trunc = ts[start:]

    w_avg = []
    N = 10
    for j in range(1, N_trunc+1):
        w_avg.append(sum(data_trunc[:j])/j)

    w_running = uniform_filter1d(data, size=N, mode='nearest')

    ax[0].axvline((start+1)*md['SAVE_STATS_DT'], color=c)

    ax[0].plot(ts, data, color=c, alpha=0.5, linestyle=":",
            label="$\overline{{w}}$ at z/r_0={0:.1f}".format(zs[i]/r_0))

    ax[0].plot(ts_trunc, w_avg, color=c,
            label="time average $\overline{{w}}$ at z/r_0={0:.1f}".format(zs[i]/r_0))

    ax[0].plot(ts, w_running, color=c, linestyle="--")

    dwbardt_tavg = np.gradient(w_avg, ts_trunc)
    dwbardt_run = np.gradient(w_running, ts)

    ax[1].axvline((start+1)*md['SAVE_STATS_DT'], color=c)
    ax[1].plot(ts_trunc, dwbardt_tavg, color=c,
        label="$\partial_t$ (time average $\overline{{w}}$) at z/r0={0:.1f}".format(zs[i]/r_0))
    ax[1].plot(ts, dwbardt_run, color=c, linestyle='--',
        label="$\partial_t$ (running average $\overline{{w}}$) at z/r0={0:.1f}".format(zs[i]/r_0))

ax[0].set_xlim(min(ts),max(ts))
ax[0].legend()

ax[1].axhline(0,color='k',alpha=0.5,linestyle='--')
ax[1].set_xlim(min(ts),max(ts))
ax[1].legend()

plt.tight_layout()
plt.show()
