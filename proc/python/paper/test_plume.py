import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py, bisect, time, gc, sys
from os import listdir
from os.path import isfile, join
import numpy as np
from math import sqrt
import matplotlib.patheffects as pe
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from functions import get_metadata, get_grid, read_params, get_az_data, compute_F0, get_index
from scipy import integrate, optimize, interpolate
from matplotlib import cm as cm
from itertools import groupby
from mpl_toolkits.axes_grid1 import make_axes_locatable

##### USER-DEFINED PARAMETERS #####
params_file = "./params.dat"
save_loc = "/home/cwp29/Documents/papers/conv_pen/draft3/figs/"
save = False
title = False
show = not save


width = 12

tstart = 7.5

eps = 0.02

matplotlib.rcParams.update({'axes.labelsize': 'large'})

##### ----------------------- #####

def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))

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
        if ref[i] < np.min(ref_var[i]):
            trunc_pt = f(np.min(ref_var[i]))
        else:
            trunc_pt = f(ref[i])

        # Create interpolation function for var
        f = interpolate.interp1d(points, var[i])
        var_new[i, trunc_idx+1] = f(trunc_pt)

    return var_new


##### Get directory locations #####
base_dir, run_dir, save_dir, version = read_params(params_file)
#base_dir = '/home/cwp29/diablo3/plume_fawcett/'
#save_dir = '/home/cwp29/diablo3/plume_fawcett/'
#run_dir = '/home/cwp29/diablo3/plume_fawcett/'
print("Run directory: ", run_dir)
print("Save directory: ", save_dir)

##### Get simulation metadata #####
md = get_metadata(run_dir, version)
print("Complete metadata: ",md)

tstart_idx = int(tstart // md['SAVE_STATS_DT'])
z_upper = 0.9 * md['H']/md['r0'] # non-dim, scaled by r_0
z_lower = 0.5 * md['H']/md['r0']

##### Get grid #####
gxf, gyf, gzf, dz = get_grid(join(run_dir, 'grid.h5'), md)
gzfp = np.flip(gzf)

r_0 = md['r0']
print(r_0)
print(md['H']/r_0)
dr = md['LX']/md['Nx']
nbins = int(md['Nx']/2)
r_bins = np.array([r*dr for r in range(0, nbins+1)])
r_points = np.array([0.5*(r_bins[i]+r_bins[i+1]) for i in range(nbins)])

print("##### F0 #####")
F_0 = compute_F0(save_dir, md, tstart_ind = tstart_idx, verbose=False)
print(F_0)
L = np.power(F_0, 1/4) * np.power(md['N2'], -3/8)

##### Get azimuthal data #####
data = get_az_data(join(save_dir,'az_stats.h5'), md, tstart_ind= tstart_idx)

ubar = data['u']
vbar = data['v']
wbar = data['w']
bbar = data['b']
pbar = data['p']
phibar = data['th']

ufluc2bar = data['ufluc2']
uflucvflucbar = data['uflucvfluc']
uflucwflucbar = data['uflucwfluc']
uflucbflucbar = data['uflucbfluc']
uflucphiflucbar = data['uflucphifluc']
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

print([gzf[i] for i in plume_indices])

##### Truncate data at radius where wbar(r) = eps*wbar(0) #####

Q_full = 2*integrate.trapezoid(wbar*r_points, r_points, axis=1)
M_full = 2*integrate.trapezoid(wbar*wbar*r_points, r_points, axis=1)
F_full = 2*integrate.trapezoid(wbar*bbar*r_points, r_points, axis=1)
B_full = 2*integrate.trapezoid(bbar*r_points, r_points, axis=1)

r_d = []
wbar_trunc = np.zeros(shape=(wbar.shape[0],wbar.shape[1]+2))
r_integrate = np.zeros(shape=(wbar.shape[0],wbar.shape[1]+2))
to_remove = []
for i in cont_valid_indices:
    wtrunc = eps*wbar[i,0]
    try:
        rtrunc = ranges(np.where(wbar[i,:] > wtrunc)[0])[0][-1]

        wbar_trunc[i, :rtrunc+1] = wbar[i, :rtrunc+1]
        wbar_trunc[i, rtrunc+1] = wtrunc

        # Edge cases
        if rtrunc == int(md['Nx']/2)-1:
            wtrunc = wbar[i,-1]
        if rtrunc == 0:
            wtrunc = wbar[i, 0]

        wbar_interp = wbar[i, rtrunc:rtrunc+2]
        r_interp = r_points[rtrunc:rtrunc+2]

        f = interpolate.interp1d(wbar_interp, r_interp)
        r_d.append(f(wtrunc))
        r_integrate[i, :rtrunc+1] = r_points[:rtrunc+1]
        r_integrate[i, rtrunc+1] = f(wtrunc)
        r_integrate[i, rtrunc+2] = f(wtrunc)

    except ValueError:
        print(rtrunc)
        print(wbar_interp)
        print(r_interp)

        fig = plt.figure()
        plt.plot(r_points, wbar[i])
        plt.axhline(wtrunc)
        plt.show()

        to_remove.append(i)
        continue

for i in to_remove:
    cont_valid_indices.remove(i)

bbar_trunc = truncate(bbar, r_points, wbar, eps*wbar[:,0], cont_valid_indices)
phibar_trunc = truncate(phibar, r_points, wbar, eps*wbar[:,0], cont_valid_indices)
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
Fphi = 2*integrate.trapezoid(wbar_trunc*phibar_trunc*r_integrate, r_integrate, axis=1)
B = 2*integrate.trapezoid(bbar_trunc*r_integrate, r_integrate, axis=1)

r_m = Q/np.sqrt(M)
w_m = M/Q
b_m = B/(r_m*r_m)
theta_m = b_m * Q / F
theta_f = 2/(w_m * b_m * r_m * r_m)*integrate.trapezoid(wflucbflucbar_trunc*r_integrate, r_integrate, axis=1)
theta_g = theta_m + theta_f

delta_f = 4/(w_m*w_m*w_m*r_m)*integrate.trapezoid(wfluc2bar_trunc*dwbardz_trunc*r_integrate,r_integrate,axis=1)
delta_p = 4/(w_m*w_m*w_m*r_m)*integrate.trapezoid(pbar_trunc*dwbardz_trunc*r_integrate,r_integrate,axis=1)
delta_m = 4/(w_m*w_m*w_m*r_m)*integrate.trapezoid(uflucwflucbar_trunc*dwbardr_trunc*r_integrate,r_integrate,axis=1)
delta_g = delta_m + delta_p + delta_f

gamma_f = 4/(w_m*w_m*w_m*r_m*r_m)*integrate.trapezoid(wbar_trunc*wfluc2bar_trunc*r_integrate, r_integrate,axis=1)
gamma_p = 4/(w_m*w_m*w_m*r_m*r_m)*integrate.trapezoid(wbar_trunc*pbar_trunc*r_integrate,r_integrate)
gamma_m = 2/(w_m*w_m*w_m*r_m*r_m)*integrate.trapezoid(np.power(wbar_trunc,3)*r_integrate, r_integrate, axis=1)
gamma_g = gamma_m + gamma_p + gamma_f

beta_f = 2/(w_m*w_m*r_m*r_m)*integrate.trapezoid(wfluc2bar_trunc*r_integrate, r_integrate, axis=1)
beta_u = 2/(w_m*w_m*r_m*r_m)*integrate.trapezoid(ufluc2bar_trunc*r_integrate, r_integrate, axis=1)
beta_v = 2/(w_m*w_m*r_m*r_m)*integrate.trapezoid(vfluc2bar_trunc*r_integrate, r_integrate, axis=1)
beta_p = 2/(w_m*w_m*r_m*r_m)*integrate.trapezoid(pbar_trunc*r_integrate, r_integrate, axis=1)
beta_m = 1
beta_g = 1 + beta_f + beta_p

##### Estimate alpha_p (plume entrainment coefficient) #####
analytic_r = lambda z,a,z0: 6/5 * a * (z-z0)
z_comp = np.array([gzf[i] for i in plume_indices])
r_comp = np.array([r_m[i] for i in plume_indices])

popt, _ = optimize.curve_fit(analytic_r, z_comp, r_comp)
alpha_p = popt[0]
z_virt = popt[1]
print(z_virt, alpha_p)

Gamma = 5*F*np.power(Q,2)/(8*alpha_p*beta_g*theta_m*np.power(M,5/2))

##### Calculate alpha (entrainment coefficient) analytically #####
dQdz = np.gradient(Q[cont_valid_indices], gzf[cont_valid_indices])
alpha = 0.5*np.power(M[cont_valid_indices],-1/2)*dQdz

dQdz_cont = np.gradient(Q[plume_indices], gzf[plume_indices])
alpha_cont = 0.5*np.power(M[plume_indices],-1/2)*dQdz_cont

alpha_avg = np.nanmean(np.where(np.logical_and(gzf[cont_valid_indices]/r_0 <= z_upper,
    gzf[cont_valid_indices]/r_0 >= z_lower), alpha, np.nan))

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

uu_regrid = np.array([interpolate.griddata(r_points/r_m[i], ufluc2bar[i]/(w_m[i]*w_m[i]), r_regrid) for i in \
    plume_indices])
uu = np.mean(uu_regrid, axis=0)

uv_regrid = np.array([interpolate.griddata(r_points/r_m[i], uflucvflucbar[i]/(w_m[i]*w_m[i]), r_regrid) for i in \
    plume_indices])
uv = np.mean(uv_regrid, axis=0)

uw_regrid = np.array([interpolate.griddata(r_points/r_m[i], uflucwflucbar[i]/(w_m[i]*w_m[i]), r_regrid) for i in \
    plume_indices])
uw = np.mean(uw_regrid, axis=0)

vv_regrid = np.array([interpolate.griddata(r_points/r_m[i], vfluc2bar[i]/(w_m[i]*w_m[i]), r_regrid) for i in \
    plume_indices])
vv = np.mean(vv_regrid, axis=0)

vw_regrid = np.array([interpolate.griddata(r_points/r_m[i], vflucwflucbar[i]/(w_m[i]*w_m[i]), r_regrid) for i in \
    plume_indices])
vw = np.mean(vw_regrid, axis=0)

ww_regrid = np.array([interpolate.griddata(r_points/r_m[i], wfluc2bar[i]/(w_m[i]*w_m[i]), r_regrid) for i in \
    plume_indices])
ww = np.mean(ww_regrid, axis=0)

uw_regrid = np.array([interpolate.griddata(r_points/r_m[i], uflucwflucbar[i]/(w_m[i]*w_m[i]), r_regrid) for i in \
    plume_indices])
uw = np.mean(uw_regrid, axis=0)

ub_regrid = np.array([interpolate.griddata(r_points/r_m[i], uflucbflucbar[i]/(w_m[i]*b_m[i]), r_regrid) for i in \
    plume_indices])
ub = np.mean(ub_regrid, axis=0)

##### Compute invariants of anisotropy tensor ######
e = uu + vv + ww
b_ij = np.array([
    [uu/e-1/3, uv/e, uw/e],
    [uv/e, vv/e-1/3, vw/e],
    [uw/e, vw/e, ww/e-1/3]
    ])
b_ij = np.moveaxis(b_ij, -1, 0)

b2 = np.linalg.matrix_power(b_ij, 2)
b3 = np.linalg.matrix_power(b_ij, 3)

eta = np.sqrt(np.trace(b2,axis1=1,axis2=2)/6)
xi = np.cbrt(np.trace(b3,axis1=1,axis2=2)/6)

''' PLOTTING '''
# Set up colourbar for height z
cols = plt.cm.rainbow(np.linspace(0,1,len(plume_indices)))
sm = plt.cm.ScalarMappable(cmap='rainbow',
        norm=plt.Normalize(vmin=np.min(gzf[plume_indices])/r_0,vmax=np.max(gzf[plume_indices])/r_0))

##### ----------------------- #####

dr_m = np.zeros(shape=(md['Nz']))
dw_m = np.zeros(shape=(md['Nz']))
db_m = np.zeros(shape=(md['Nz']))

#F_0 = md['b0']*r_0*r_0
alphae = md['alpha_e']
zvirt = -r_0 / (1.2 * alphae)
for j in range(md['Nz']):
    dr_m[j] = 1.2 * alphae * (gzf[j]-zvirt)
    dw_m[j] = (0.9 * alphae * F_0)**(1/3) * (gzf[j] - zvirt)**(2/3) / dr_m[j]
    db_m[j] = F_0/(dr_m[j]**2 * dw_m[j])

fig, ax = plt.subplots(1,3)
ax[0].plot(dr_m, gzf, linestyle=':', color='b')
ax[0].plot(r_m, gzf, color='b')
ax[1].plot(dw_m, gzf, linestyle=':', color='b')
ax[1].plot(w_m, gzf, color='b')
ax[2].plot(db_m, gzf, linestyle=':', color='b')
ax[2].plot(b_m, gzf, color='b')

##### ----------------------- #####
# Non-dim
N = np.sqrt(md['N2'])

T = np.power(N, -1)
L = np.power(F_0, 1/4) * np.power(N, -3/4)
B = L * np.power(T, -2)
U = L / T

factor = 0.6

Z = (gzf - md['H'])/L



fig0, axs = plt.subplots(1,4,figsize=(8, 3), sharey=True)
if title: fig0.suptitle("Plume integral quantities Q, M, F")
axs[0].plot(Q/(U*L*L), Z, label="Simulation", color='b')
axs[1].plot(M/(U*U*L*L), Z, label="Simulation", color='b')
axs[2].plot(F/(B*U*L*L), Z, label="Simulation", color='b')
axs[3].plot(Fphi/(U*L*L), Z, label="Simulation", color='b')

#TODO change this to just estimate zvirt
analytic_Q = lambda z,a,z0: a*np.power(z-z0, 5/3)
analytic_M = lambda z,a,z0: a*np.power(z-z0, 4/3)
analytic_F = lambda z,a: a
analytic_Fphi = lambda z,a: a

poptQ, _ = optimize.curve_fit(analytic_Q, gzf[plume_indices], Q[plume_indices])
poptM, _ = optimize.curve_fit(analytic_M, gzf[plume_indices], M[plume_indices])
poptF, _ = optimize.curve_fit(analytic_F, gzf[plume_indices], F[plume_indices])
poptFphi, _ = optimize.curve_fit(analytic_Fphi, gzf[plume_indices], Fphi[plume_indices])
print(poptQ, poptM, poptF, poptFphi)

axs[0].plot(1.2*alpha_p * np.power(0.9*alpha_p*poptF[0], 1/3) * np.power(gzf - z_virt, 5/3)/(U*L*L),
        Z, color='r', label="Theory")
axs[1].plot(np.power(0.9*alpha_p*poptF[0], 2/3) * np.power(gzf - z_virt, 4/3) / (U*U*L*L),
        Z, color='r')
axs[2].axvline(poptF[0]/(B*U*L*L), color='r')
axs[3].axvline(poptFphi[0]/(U*L*L), color='r')


axs[0].axhline((z_upper*r_0-md['H'])/L, color='grey', linestyle='--')
axs[0].axhline((z_lower*r_0-md['H'])/L, color='grey', linestyle='--')
axs[1].axhline((z_upper*r_0-md['H'])/L, color='grey', linestyle='--')
axs[1].axhline((z_lower*r_0-md['H'])/L, color='grey', linestyle='--')
axs[2].axhline((z_upper*r_0-md['H'])/L, color='grey', linestyle='--')
axs[2].axhline((z_lower*r_0-md['H'])/L, color='grey', linestyle='--')
axs[3].axhline((z_upper*r_0-md['H'])/L, color='grey', linestyle='--')
axs[3].axhline((z_lower*r_0-md['H'])/L, color='grey', linestyle='--')

#axs[0].plot(Q_full, Z,label="Full",color='b')
#axs[1].plot(M_full, Z,label="Full",color='b')
#axs[2].plot(F_full, Z,label="Full",color='b')
axs[0].set_xlim(0,1.1*max(max(Q),max(Q_full))/(U*L*L))
axs[0].set_ylim(-md['H']/L, 0)
axs[1].set_xlim(0,1.1*max(max(M),max(M_full))/(U*U*L*L))
axs[1].set_ylim(-md['H']/L, 0)
axs[2].set_xlim(0,1.1*max(max(F),max(F_full))/(B*U*L*L))
axs[2].set_ylim(-md['H']/L, 0)
axs[3].set_xlim(0,1.1*max(max(Fphi),max(F_full))/(U*L*L))
axs[3].set_ylim(-md['H']/L, 0)

axs[0].set_ylabel("$z/r_0$")

axs[0].legend()

#axs[1].legend()
#axs[2].legend()
axs[0].set_xlabel("$Q$")
axs[1].set_xlabel("$M$")
axs[2].set_xlabel("$F_b$")
axs[3].set_xlabel("$F_\phi$")

plt.tight_layout()

if save: fig0.savefig(save_loc+'fluxes.png', dpi=300)

##### ----------------------- #####
# estimate flux balance parameter
# analytic alpha (entrainment coefficient)

fig8, ax8 = plt.subplots(1,2, figsize=(width,width*3/10))
if title: fig8.suptitle("Fig 2(a): flux balance parameter $\\Gamma = 5FQ^2 / 8\\beta_g \\alpha_p \\theta_m \
        M^{{5/2}}$ \n (b): theoretical and numerical entrainment coefficient")

ax8[0].plot(Gamma[cont_valid_indices], gzf[cont_valid_indices]/r_0, color='blue')
ax8[0].plot([1]*len(gzf[cont_valid_indices]), gzf[cont_valid_indices]/r_0 ,color='k', linestyle='--',
label="$\\Gamma_p = 1$")

ax8[0].set_ylabel("$z/r_0$")
ax8[0].set_xlabel("$\\Gamma$")
ax8[0].set_xlim(0,2)
ax8[0].set_ylim(0, md['LZ']/r_0)

ax8[1].plot(alpha, gzf[cont_valid_indices]/r_0, color='blue')
ax8[1].plot([alpha_p]*len(gzf[cont_valid_indices]), gzf[cont_valid_indices]/r_0, color='black',
        linestyle='dashed', label="$\\alpha_p = {0:.3f}$".format(alpha_p))
ax8[1].axvline(alpha_avg, linestyle='--', color='blue')
print(alpha_avg)
ax8[1].legend()

ax8[1].set_ylabel("$z/r_0$")
ax8[1].set_xlabel("$\\alpha$")
ax8[1].set_xlim(0,0.2)
ax8[1].set_ylim(0, md['LZ']/r_0)

ax8[0].axhline(z_upper, color='gray', alpha=0.5,linestyle='--')
ax8[0].axhline(z_lower, color='gray', alpha=0.5,linestyle='--')
ax8[1].axhline(z_upper, color='gray', alpha=0.5,linestyle='--')
ax8[1].axhline(z_lower, color='gray', alpha=0.5,linestyle='--')

nticks = 5
ticks = np.linspace(0,0.2,nticks)
ax8[1].set_xticks(ticks)
ax8[1].set_xticklabels(["{0:.2f}".format(i) for i in ticks])

nticks = 5
ticks = np.linspace(0,2,nticks)
ax8[0].set_xticks(ticks)
ax8[0].set_xticklabels(["{0:.1f}".format(i) for i in ticks])

plt.tight_layout()

##### ----------------------- #####

rs = np.linspace(0, 2, 100)

fig23, axs23 = plt.subplots(2,2, figsize=(8, 4), sharex=True)

for i,c in zip(plume_indices,cols):
    if i == plume_indices[0]:
        axs23[0,0].plot(r_points/r_m[i], wbar[i]/w_m[i],color='blue',label="$\overline{w}/w_m$")
        axs23[0,0].plot(r_points/r_m[i], bbar[i]/b_m[i],color='red',linestyle='dashed',label="$\overline{b}/b_m$")
        axs23[0,0].plot(r_points/r_m[i], phibar[i]/b_m[i],color='green',linestyle='dotted',
                label=r"$\overline{\phi}/\phi_m$")
        axs23[0,1].plot(r_points/r_m[i], uflucwflucbar[i]/(w_m[i]*w_m[i]), color='blue',
                label="$\overline{u'w'}/w_m^2$")
        axs23[0,1].plot(r_points/r_m[i], uflucbflucbar[i]/(w_m[i]*b_m[i]), color='red',
                linestyle='dashed', label="$\overline{u'b'}/w_m b_m$")
        axs23[0,1].plot(r_points/r_m[i], uflucphiflucbar[i]/(w_m[i]*b_m[i]), color='green',
                linestyle='dotted', label="$\overline{u'\phi'}/w_m \phi_m$")
    else:
        axs23[0,0].plot(r_points/r_m[i], wbar[i]/w_m[i],color='blue')
        axs23[0,0].plot(r_points/r_m[i], bbar[i]/b_m[i],color='red',linestyle='dashed')
        axs23[0,0].plot(r_points/r_m[i], phibar[i]/b_m[i],color='green',linestyle='dotted')
        axs23[0,1].plot(r_points/r_m[i], uflucwflucbar[i]/(w_m[i]*w_m[i]), color='blue')
        axs23[0,1].plot(r_points/r_m[i], uflucbflucbar[i]/(w_m[i]*b_m[i]), color='red', linestyle='dashed')
        axs23[0,1].plot(r_points/r_m[i], uflucphiflucbar[i]/(w_m[i]*b_m[i]), color='green', linestyle='dotted')

    axs23[0,0].plot(r_points/r_m[plume_indices[0]], 2*np.exp(-2*(r_points/r_m[plume_indices[0]])**2), color='k')

    axs23[1,0].plot(r_points/r_m[i], ubar[i]/w_m[i],color='blue')
    axs23[1,1].plot(r_points/r_m[i], r_points*ubar[i]/(r_m[i]*w_m[i]), color='blue')

axs23[0,0].set_xlim(0, 2)
axs23[0,0].legend()
axs23[0,1].set_xlim(0,2)
axs23[0,1].legend()

axs23[1,1].axhline(-alpha_avg, color='k', linestyle='dashed')
axs23[1,1].text(0.05, 0.005-alpha_avg, "-$\\alpha$", fontsize=14)
axs23[1,0].set_xlim(0, 2)
axs23[1,0].set_xlabel("$r/r_m$")
axs23[1,0].set_ylabel("$\\overline{u}/w_m$")
axs23[1,1].set_xlim(0, 2)
axs23[1,1].set_xlabel("$r/r_m$")
axs23[1,1].set_ylabel("$r\\overline{u}/(r_m w_m)$")

#divider = make_axes_locatable(axs23[1,1])
#cax = divider.append_axes('right', size='5%', pad=0.1)

nticks = 5
ticks = np.linspace(0,2,nticks)
axs23[0,1].set_xticks(ticks)
axs23[0,1].set_xticklabels(["{0:.1f}".format(i) for i in ticks])
axs23[0,0].set_xticks(ticks)
axs23[0,0].set_xticklabels(["{0:.1f}".format(i) for i in ticks])
axs23[1,1].set_xticks(ticks)
axs23[1,1].set_xticklabels(["{0:.1f}".format(i) for i in ticks])
axs23[1,0].set_xticks(ticks)
axs23[1,0].set_xticklabels(["{0:.1f}".format(i) for i in ticks])

for ax, label in zip(axs23.ravel(), ['(a)', '(b)', '(c)', '(d)']):
    ax.text(-0.1, 1.15, label, transform=ax.transAxes,
            va='top', ha='right')

fig23.tight_layout()
fig23.subplots_adjust(right=0.9)
#cbar_ax = fig23.add_axes([0.92, 0.105, 0.02, 0.87])
#fig23.colorbar(sm, cax=cbar_ax, label="$z/r_0$")
#cbar = fig23.colorbar(sm, ax=axs23.ravel().tolist(), shrink=0.95, label="$z/r_0$")
#fig23.colorbar(sm, cax = cax,label="$z/r_0$")
#fig23.savefig('/home/cwp29/Documents/papers/draft/figs/mvr.png', dpi=300)
#fig23.savefig('/home/cwp29/Documents/papers/draft/figs/mvr.pdf')

if save: fig23.savefig(save_loc+'profiles.png', dpi=300)

##### ----------------------- #####

fig13 = plt.figure(constrained_layout=True, figsize=(width,width*4/13))
ax13 = fig13.subplot_mosaic(
        """
        AB
        AC
        """,
        gridspec_kw = {"width_ratios": [1.15, 1], "hspace": 0.1, "wspace":0.1}
        )

ax13["B"].plot(r_regrid, xi)
ax13["B"].set_xlabel("$r/r_m$")
ax13["B"].set_xlim(0,2)
ax13["B"].set_ylabel("$\\xi$")
ax13["B"].set_ylim(-0.2,0.2)

ax13["C"].plot(r_regrid, eta)
ax13["C"].set_xlabel("$r/r_m$")
ax13["C"].set_ylabel("$\\eta$")
ax13["C"].set_xlim(0,2)
ax13["C"].set_ylim(0,1/6)

# 8 lines below add colour gradient to the line to indicate r/r_m value
points = np.array([xi,eta]).T.reshape(-1,1,2)
segments = np.concatenate([points[:-1], points[1:]],axis=1)
norm = plt.Normalize(r_regrid.min(), r_regrid.max())
lc = LineCollection(segments, cmap='jet', norm=norm)
lc.set_array(r_regrid)
lc.set_linewidth(2)
line = ax13["A"].add_collection(lc)
#fig13.colorbar(line, ax=ax13["A"], label="$r/r_m$")

divider = make_axes_locatable(ax13["A"])
cax = divider.append_axes('right', size='5%', pad=0.1)
fig13.colorbar(line, cax = cax, label="$r/r_m$")

#add Lumley triangle
ax13["A"].plot([0, 1/3], [0,1/3], color='k')
ax13["A"].plot([0, -1/6], [0,1/6], color='k')
xis = np.linspace(-1/6,1/3,1000)
ax13["A"].plot(xis, np.sqrt(1/27+2*xis**3), color='k')

ax13["A"].set_xlabel("$\\xi$")
ax13["A"].set_ylabel("$\\eta$")
ax13["A"].set_xlim(-1/6, 1/3)
ax13["A"].set_ylim(0, 1/3)
#if title: plt.title("Fig 8 (a): invariants of the anisotropy tensor in $\\xi-\\eta$ space with Lumley \n \
        #triangle, (b), (c): dependence of $\\xi, \\eta$ on $r/r_m$")

##### ----------------------- #####

if show: plt.show()
