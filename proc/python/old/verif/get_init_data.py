import h5py, bisect, time, gc, sys
from os import listdir
from os.path import isfile, join
import numpy as np
from math import sqrt
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from functions import get_metadata, get_grid, read_params, get_az_data
from scipy import integrate, optimize, interpolate
from matplotlib import cm as cm
from itertools import groupby
from mpl_toolkits.axes_grid1 import make_axes_locatable

##### USER-DEFINED PARAMETERS #####
params_file = "./params.dat"
#save_loc = "/home/cwp29/Documents/plume_project/figs/mvr/"
save = False
title = False
show = True

z_upper = 95 # non-dim, scaled by r_0
z_lower = 20

zstar = 0.18

eps = 0.02

matplotlib.rcParams.update({'axes.labelsize': 'large'})

##### ----------------------- #####

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

ufluc2bar = data['ufluc2']
uflucvflucbar = data['uflucvfluc']
uflucwflucbar = data['uflucwfluc']
uflucbflucbar = data['uflucbfluc']
vfluc2bar = data['vfluc2']
vflucwflucbar = data['vflucwfluc']
wfluc2bar = data['wfluc2']
wflucbflucbar = data['wflucbfluc']
bfluc2bar = data['bfluc2']

dwbardz = np.gradient(wbar, gzf, axis=0)
dwbardr = np.gradient(wbar, dr, axis=1)

dpbardz = np.gradient(pbar, gzf, axis=0)
dpbardr = np.gradient(pbar, dr, axis=1)

##### Identify indices where plume is well defined #####

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

print(plume_indices)

##### Truncate data at radius where wbar(r) = eps*wbar(0) #####

Q_full = 2*integrate.trapezoid(wbar*r_points, r_points, axis=1)
M_full = 2*integrate.trapezoid(wbar*wbar*r_points, r_points, axis=1)
F_full = 2*integrate.trapezoid(wbar*bbar*r_points, r_points, axis=1)
B_full = 2*integrate.trapezoid(bbar*r_points, r_points, axis=1)

r_d = []
wbar_trunc = np.zeros(shape=(wbar.shape[0],wbar.shape[1]+2))
r_integrate = np.zeros(shape=(wbar.shape[0],wbar.shape[1]+2))
for i in cont_valid_indices:
    w_trunc = eps*wbar[i,0]
    r_trunc = max(np.where(wbar[i,:] > wtrunc)[0])

    wbar_trunc[i, :rtrunc+1] = wbar[i, :rtrunc+1]
    wbar_trunc[i, rtrunc+1] = wtrunc

    f = interpolate.interp1d(wbar[i], r_points)
    r_d.append(f(wtrunc))
    r_integrate[i, :rtrunc+1] = r_points[:rtrunc+1]
    r_integrate[i, rtrunc+1] = f(wtrunc)
    r_integrate[i, rtrunc+2] = f(wtrunc)

bbar_trunc = truncate(bbar, r_points, wbar, eps*wbar[:,0], cont_valid_indices)
pbar_trunc = truncate(pbar, r_points, wbar, eps*wbar[:,0], cont_valid_indices)
wfluc2bar_trunc = truncate(wfluc2bar, r_points, wbar, eps*wbar[:,0], cont_valid_indices)
ufluc2bar_trunc = truncate(ufluc2bar, r_points, wbar, eps*wbar[:,0], cont_valid_indices)
vfluc2bar_trunc = truncate(vfluc2bar, r_points, wbar, eps*wbar[:,0], cont_valid_indices)
uflucwflucbar_trunc = truncate(uflucwflucbar, r_points, wbar, eps*wbar[:,0], cont_valid_indices)
wflucbflucbar_trunc = truncate(wflucbflucbar, r_points, wbar, eps*wbar[:,0], cont_valid_indices)
dwbardz_trunc = truncate(dwbardz, r_points, wbar, eps*wbar[:,0], cont_valid_indices)
dwbardr_trunc = truncate(dwbardr, r_points, wbar, eps*wbar[:,0], cont_valid_indices)

##### Calculate integral quantities and profile coefficients #####

Q = 2*integrate.trapezoid(wbar_trunc*r_integrate, r_integrate, axis=1)
M = 2*integrate.trapezoid(wbar_trunc*wbar_trunc*r_integrate, r_integrate, axis=1)
F = 2*integrate.trapezoid(wbar_trunc*bbar_trunc*r_integrate, r_integrate, axis=1)
B = 2*integrate.trapezoid(bbar_trunc*r_integrate, r_integrate, axis=1)

print("Initial data at z = ", zstar)
Qf = interpolate.interp1d(gzf, Q)
print("Q:", Qf(zstar))
Mf = interpolate.interp1d(gzf, M)
print("M:", Mf(zstar))
Ff = interpolate.interp1d(gzf, F)
print("F:", Ff(zstar))

r_m = Q/np.sqrt(M)
w_m = M/Q
b_m = B/(r_m*r_m)

theta_m = b_m * Q / F
theta_f = 2/(w_m * b_m * r_m * r_m)*integrate.trapezoid(wflucbflucbar_trunc*r_integrate, r_integrate, axis=1)
theta_g = theta_m + theta_f

beta_f = 2/(w_m*w_m*r_m*r_m)*integrate.trapezoid(wfluc2bar_trunc*r_integrate, r_integrate, axis=1)
beta_p = 2/(w_m*w_m*r_m*r_m)*integrate.trapezoid(pbar_trunc*r_integrate, r_integrate, axis=1)
beta_g = 1 + beta_f + beta_p

beta_g_avg = np.mean(beta_g[plume_indices])
theta_m_avg = np.mean(theta_m[plume_indices])
theta_g_avg = np.mean(theta_g[plume_indices])

print("================ Mean profile coefficients ===============")
print("Beta (g)")
print(beta_g_avg)
print("Theta (m, g)")
print(theta_m_avg, theta_g_avg)

##### Estimate alpha_p (plume entrainment coefficient) #####
analytic_r = lambda z,a,z0: 6/5 * a * (z-z0)
z_comp = np.array([gzf[i] for i in plume_indices])
r_comp = np.array([r_m[i] for i in plume_indices])

popt, _ = optimize.curve_fit(analytic_r, z_comp, r_comp)
alpha_p = popt[0]
z_virt = popt[1]
print(z_virt, alpha_p)

##### Calculate average profiles over r/r_m and turbulent nu, D #####
z_min = plume_indices[-1]
n_regrid = max(np.where(wbar[z_min,:]>0.02*wbar[z_min,0])[0])*2
print("Regridding onto %s grid points..."%n_regrid)
r_regrid = np.linspace(r_points[0]/r_m[z_min],2,n_regrid)[1:]

w_regrid = np.array([interpolate.griddata(r_points/r_m[i], wbar[i]/w_m[i], r_regrid) for i in \
    plume_indices])
w_bar = np.mean(w_regrid, axis=0)

b_regrid = np.array([interpolate.griddata(r_points/r_m[i], bbar[i]/b_m[i], r_regrid) for i in \
    plume_indices])
b_bar = np.mean(b_regrid, axis=0)

##### ----------------------- #####

fig0, axs = plt.subplots(1,3,figsize=(10, 6))
if title: fig0.suptitle("Plume integral quantities Q, M, F")
axs[0].plot(Q, gzf/r_0, label="Thresholded", color='b', linestyle='--')
axs[1].plot(M, gzf/r_0, label="Thresholded", color='b', linestyle='--')
axs[2].plot(F, gzf/r_0, label="Thresholded", color='b', linestyle='--')

analytic_Q = lambda z,a,z0: a*(z-z0)**(5/3)
analytic_M = lambda z,a,z0: a*(z-z0)**(4/3)
analytic_F = lambda z,a: a

poptQ, _ = optimize.curve_fit(analytic_Q, gzf[plume_indices], Q[plume_indices])
poptM, _ = optimize.curve_fit(analytic_M, gzf[plume_indices], M[plume_indices])
poptF, _ = optimize.curve_fit(analytic_F, gzf[plume_indices], F[plume_indices])

axs[0].axvline(Qf(zstar), linestyle='--', color='grey')
axs[1].axvline(Mf(zstar), linestyle='--', color='grey')
axs[2].axvline(Ff(zstar), linestyle='--', color='grey')

axs[0].plot(poptQ[0]*np.power(gzf-poptQ[1],5/3), gzf/r_0, color='r')
axs[1].plot(poptM[0]*np.power(gzf-poptM[1],4/3), gzf/r_0, color='r')
axs[2].axvline(poptF[0], color='r')

axs[0].plot(Q_full, gzf/r_0,label="Full",color='b')
axs[1].plot(M_full, gzf/r_0,label="Full",color='b')
axs[2].plot(F_full, gzf/r_0,label="Full",color='b')
axs[0].set_xlim(0,1.1*max(max(Q),max(Q_full)))
axs[0].set_ylim(0,md['LZ']/r_0)
axs[1].set_xlim(0,1.1*max(max(M),max(M_full)))
axs[1].set_ylim(0,md['LZ']/r_0)
axs[2].set_xlim(0,1.1*max(max(F),max(F_full)))
axs[2].set_ylim(0,md['LZ']/r_0)
axs[0].set_ylabel("$z/r_0$")
axs[1].set_ylabel("$z/r_0$")
axs[2].set_ylabel("$z/r_0$")
nticks = 5
ticks = np.linspace(0,md['LZ']/r_0,nticks)
axs[0].set_yticks(ticks)
axs[0].set_yticklabels([str(int(i)) for i in ticks])
axs[1].set_yticks(ticks)
axs[1].set_yticklabels([str(int(i)) for i in ticks])
axs[2].set_yticks(ticks)
axs[2].set_yticklabels([str(int(i)) for i in ticks])
axs[0].legend()
#axs[1].legend()
#axs[2].legend()
axs[0].set_title("Q, volume flux")
axs[1].set_title("M, momentum flux")
axs[2].set_title("F, buoyancy flux")
axs[0].grid(color='gray',alpha=0.5)
axs[1].grid(color='gray',alpha=0.5)
axs[2].grid(color='gray',alpha=0.5)
plt.tight_layout()

plt.show()
