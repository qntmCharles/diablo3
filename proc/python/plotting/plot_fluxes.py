import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from itertools import groupby
from datetime import datetime
from scipy import integrate, optimize, interpolate
from functions import get_metadata, get_grid, read_params, get_az_data

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"
eps = 0.02

##### ---------------------- #####

def get_index(z, griddata):
    return int(np.argmin(np.abs(griddata - z)))

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

##### ---------------------- #####

# Get dir locations from param file
base_dir, run_dir, save_dir, version = read_params(params_file)

# Get simulation metadata
md = get_metadata(run_dir, version)

gxf, gyf, gzf, dzf = get_grid(save_dir+'grid.h5', md)
gx, gy, gz, dz = get_grid(save_dir+'grid.h5', md, fractional_grid=False)

X, Y = np.meshgrid(gx, gz)
Xf, Yf = np.meshgrid(gxf, gzf)

r_0 = md['r0']
N = np.sqrt(md['N2'])
dr = md['LX']/md['Nx']
z_upper = md['H']/r_0
z_lower = (md['Lyc']+md['Lyp'])/r_0

nbins = int(md['Nx']/2)
r_bins = np.array([r*dr for r in range(0, nbins+1)])
r_points = np.array([0.5*(r_bins[i]+r_bins[i+1]) for i in range(nbins)])

print("Complete metadata: ", md)

##### ---------------------- #####

# Get data
with h5py.File(save_dir+"/movie.h5", 'r') as f:
    print("Keys: %s" % f.keys())
    time_keys = list(f['th1_xy'])
    print(time_keys)
    # Get buoyancy data
    th2_xy = np.array([np.array(f['th2_xz'][t]) for t in time_keys])
    th2_zy = np.array([np.array(f['th2_xz'][t]) for t in time_keys])
    NSAMP = len(th2_xy)
    times = np.array([float(f['th2_xz'][tstep].attrs['Time']) for tstep in time_keys])
    f.close()

# Compute time indices
nplot = 2
interval = 80

F = md['b0'] * (md['r0']**2)
tau = md['r0']**(4/3) * F**(-1/3)
step = np.round(interval*tau / md['SAVE_STATS_DT'])

t_inds = list(map(int,step*np.array(range(1, nplot+1))))
tplot = [times[i] for i in t_inds]
print("Plotting at times: ",tplot)
print("with interval", step*md['SAVE_STATS_DT'])

##### ---------------------- #####

with h5py.File(save_dir+'az_stats.h5', 'r') as f:
    tkeys = list(f['w_az'].keys())
    w = np.array([np.array(f['w_az'][t]) for t in tkeys])
    b = np.array([np.array(f['b_az'][t]) for t in tkeys])
    NSAMP = len(w)

wbars = []
bbars = []

avg_ind = 5
for i in t_inds:
    wbars.append(np.mean(w[i-avg_ind:i+avg_ind], axis=0))
    bbars.append(np.mean(b[i-avg_ind:i+avg_ind], axis=0))

Q_full = []
M_full = []
F_full = []
Q = []
M = []
F = []

for i in range(len(wbars)):
    wbar = wbars[i]
    bbar = bbars[i]

    # find indices where centreline velocity is positive
    valid_indices = [j for j in range(wbar.shape[0]) if wbar[j,0] > 0]
    cont_valid_indices = list(set(list(sum(max([list(y) for i, y in groupby(zip(valid_indices,
        valid_indices[1:]), key = lambda x: (x[1]-x[0]) == 1)], key=len),()))))
    if cont_valid_indices[-1] == md['Nz']-1:
        cont_valid_indices.remove(md['Nz']-1)

    # check where rtrunc is, if at the edge of domain then remove index from list
    plume_indices = []
    z_upper_ind = int(max(np.where(gzf <= z_upper*r_0)[0]))
    z_lower_ind = int(min(np.where(z_lower*r_0 <= gzf)[0]))
    for j in cont_valid_indices:
        wtrunc = 0.02*wbar[j, 0]
        rtrunc = max(np.where(wbar[j,:] > wtrunc)[0])
        if (z_lower_ind <= j <= z_upper_ind) and rtrunc != md['Nx']/2-1:
            plume_indices.append(j)

    # This array contains the largest continuous run of indices which have a positive centreline velocity
    # and valid truncation radius, at z levels between z_lower and z_upper
    plume_indices = list(set(list(sum(max([list(y) for i, y in groupby(zip(plume_indices,
        plume_indices[1:]), key = lambda x: (x[1]-x[0]) == 1)], key=len),()))))

    Q_full.append(2*integrate.trapezoid(wbar*r_points, r_points, axis=1))
    M_full.append(2*integrate.trapezoid(wbar*wbar*r_points, r_points, axis=1))
    F_full.append(2*integrate.trapezoid(wbar*bbar*r_points, r_points, axis=1))

    wbar_trunc = np.zeros(shape=(wbar.shape[0],wbar.shape[1]+2))
    r_integrate = np.zeros(shape=(wbar.shape[0],wbar.shape[1]+2))
    for i in cont_valid_indices:
        w_trunc = eps*wbar[i,0]
        try:
            r_trunc = max(np.where(wbar[i,:] > wtrunc)[0])
        except ValueError:
            cont_valid_indices.remove(i)
            continue

        wbar_trunc[i, :rtrunc+1] = wbar[i, :rtrunc+1]
        wbar_trunc[i, rtrunc+1] = wtrunc

        f = interpolate.interp1d(wbar[i], r_points)
        r_integrate[i, :rtrunc+1] = r_points[:rtrunc+1]
        r_integrate[i, rtrunc+1] = f(wtrunc)
        r_integrate[i, rtrunc+2] = f(wtrunc)

    bbar_trunc = truncate(bbar, r_points, wbar, eps*wbar[:,0], cont_valid_indices)

    Q.append(2*integrate.trapezoid(wbar_trunc*r_integrate, r_integrate, axis=1))
    M.append(2*integrate.trapezoid(wbar_trunc*wbar_trunc*r_integrate, r_integrate, axis=1))
    F.append(2*integrate.trapezoid(wbar_trunc*bbar_trunc*r_integrate, r_integrate, axis=1))

F = np.array(F)

F0 = np.mean(F[:, z_lower_ind:z_upper_ind])

alpha = 0.1
factor1 = 1.2 * alpha
factor2 = 0.9 * alpha * F0

Q_theory = factor1 * np.power(factor2, 1/3) * np.power(gzf, 5/3)

Q_factor = np.power(F0, 3/4) * np.power(N, -5/4)
M_factor = F0/N
F_factor = F0
z_factor = np.power(F0, 1/4) * np.power(N, -3/4)

fig, ax = plt.subplots(1,3)

cols = plt.cm.rainbow(np.linspace(0, 1, len(Q)))
for i,c in zip(range(len(Q)), cols):
    ax[0].plot(Q[i]/Q_factor, gzf/z_factor, color=c)
    ax[1].plot(M[i]/M_factor, gzf/z_factor, color=c)
    ax[2].plot(F[i]/F_factor, gzf/z_factor, color=c, label="t = {0:.2f}".format(tplot[i],2))

ax[0].plot(Q_theory/Q_factor, gzf/z_factor, color='k', linestyle='--', label="Theory")

ax[2].legend()

ax[0].set_yscale('log')
ax[0].set_xscale('log')

plt.show()
