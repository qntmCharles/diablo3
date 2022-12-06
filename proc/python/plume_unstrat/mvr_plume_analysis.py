import h5py, bisect, time, gc, sys
from os import listdir
from os.path import isfile, join
import numpy as np
from math import sqrt
import matplotlib.patheffects as pe
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
save_loc = "/home/cwp29/Documents/4report/figs/mvr/"
save = True
title = False
show = not save

z_upper = 70 # non-dim, scaled by r_0
z_lower = 35

width = 12

eps = 0.02

matplotlib.rcParams.update({'axes.labelsize': 'large'})

##### ----------------------- #####

def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))

def truncate(var, points, ref_var, ref, trunc_indices, display=False):
    # Truncates var at location where ref_var matches ref, computing only at trunc_indices
    # ASSUMES ref_var is decreasing with increasing index
    # Points are the positions of each entry of var
    var_new = np.zeros(shape=(var.shape[0], var.shape[1]+2))
    for i in trunc_indices:
        # Calculate index up to which var is unchanged, after which set to 0
        trunc_idx = ranges(np.where(ref_var[i,:] > ref[i])[0])[0][-1]
        var_new[i, :trunc_idx+1] = var[i, :trunc_idx+1]

        # Calculate exact point to interpolate
        ref_var_interp = ref_var[i, trunc_idx:trunc_idx+2]
        points_interp = points[trunc_idx:trunc_idx+2]
        f = interpolate.interp1d(ref_var_interp, points_interp)
        trunc_pt = f(ref[i])

        # Create interpolation function for var
        points_interp = points[trunc_idx:trunc_idx+2]
        var_interp = var[i,trunc_idx:trunc_idx+2]

        f = interpolate.interp1d(points_interp, var_interp)
        var_new[i, trunc_idx+1] = f(trunc_pt)

    return var_new


##### Get directory locations #####
base_dir, run_dir, save_dir, version = read_params(params_file)
base_dir = '/home/cwp29/diablo3/plume_fawcett/'
save_dir = '/home/cwp29/diablo3/plume_fawcett/'
run_dir = '/home/cwp29/diablo3/plume_fawcett/'
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
    wtrunc = eps*wbar[i,0]
    try:
        rtrunc = ranges(np.where(wbar[i,:] > wtrunc)[0])[0][-1]
    except ValueError:
        cont_valid_indices.remove(i)
        continue

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
print("Q:", Q)
print("M:", M)
print("F:", F)

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

beta_m_avg = 1
beta_u_avg = np.mean(beta_u[plume_indices])
beta_v_avg = np.mean(beta_v[plume_indices])
beta_f_avg = np.mean(beta_f[plume_indices])
beta_p_avg = np.mean(beta_p[plume_indices])
beta_g_avg = np.mean(beta_g[plume_indices])
gamma_m_avg = np.mean(gamma_m[plume_indices])
gamma_f_avg = np.mean(gamma_f[plume_indices])
gamma_p_avg = np.mean(gamma_p[plume_indices])
gamma_g_avg = np.mean(gamma_g[plume_indices])
delta_m_avg = np.mean(delta_m[plume_indices])
delta_f_avg = np.mean(delta_f[plume_indices])
delta_p_avg = np.mean(delta_p[plume_indices])
delta_g_avg = np.mean(delta_g[plume_indices])
theta_m_avg = np.mean(theta_m[plume_indices])
theta_f_avg = np.mean(theta_f[plume_indices])
theta_g_avg = np.mean(theta_g[plume_indices])

print("================ Mean profile coefficients ===============")
print("Beta (f, u, v, p, g)")
print(beta_f_avg, beta_u_avg, beta_v_avg, beta_p_avg, beta_g_avg)
print("Gamma (m, f, p, g)")
print(gamma_m_avg, gamma_f_avg, gamma_p_avg, gamma_g_avg)
print("Delta (m, f, p, g)")
print(delta_m_avg, delta_f_avg, delta_p_avg, delta_g_avg)
print("Theta (m, f, g)")
print(theta_m_avg, theta_f_avg, theta_g_avg)

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

##### Decompose entrainment coefficient #####
alpha_prod = -0.5*delta_g[cont_valid_indices]/gamma_g[cont_valid_indices]
alpha_Ri = alpha_p * (np.power(beta_g[cont_valid_indices],-1) -
        theta_m[cont_valid_indices]/gamma_g[cont_valid_indices]) * 8/5 * beta_g[cont_valid_indices]

shape_operand = np.log(np.power(gamma_g,1/2)/beta_g)[cont_valid_indices]
# original alpha_shape (noisy):
#alpha_shape = r_m[cont_valid_indices]*np.gradient(shape_operand, gzfp[cont_valid_indices])
# smoothed alpha_shape:
nbin = 17
kernel = np.ones(nbin)/nbin
gradient_shape = np.gradient(shape_operand,gzf[cont_valid_indices])
padded_shape = np.pad(gradient_shape, (nbin//2,nbin-1-nbin//2),mode='edge')
alpha_shape = r_m[cont_valid_indices]*np.convolve(padded_shape,kernel,mode='valid')

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

dw_bardr = np.gradient(w_bar, r_regrid)
db_bardr = np.gradient(b_bar, r_regrid)

nu_T = -uw/dw_bardr
D_T = -ub/db_bardr

a_w2 = nu_T/np.abs(dw_bardr)
a_b2 = D_T/np.abs(db_bardr)

l_m = np.sqrt(a_w2)
l_mb = np.sqrt(a_b2)

Pr_T = nu_T/D_T

fb_fw = db_bardr/dw_bardr

fuw_fub = uw/ub

theta_m_avg_an = 1.011
fbfw_an = np.power(2/theta_m_avg_an-1, -2) * np.exp(((2-2*theta_m_avg_an)/(2-theta_m_avg_an)) * \
        np.power(r_regrid, 2))

aw2_an = np.nanmean(np.where(np.abs(r_regrid-0.65) < 0.35, a_w2, np.nan))
ab2_an = np.nanmean(np.where(np.abs(r_regrid-0.65) < 0.35, a_b2, np.nan))
print(aw2_an, ab2_an)

fuwfub_an = np.power(2/theta_m_avg_an-1,2) * (aw2_an/ab2_an) * np.exp(-((2-2*theta_m_avg_an)/ \
        (2-theta_m_avg_an)) * np.power(r_regrid, 2))

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
factor = 0.6

fig0, axs = plt.subplots(1,3,figsize=(width, width*4/10))
if title: fig0.suptitle("Plume integral quantities Q, M, F")
axs[0].plot(Q, factor*gzf/r_0, label="Thresholded", color='b', linestyle='--')
axs[1].plot(M, factor*gzf/r_0, label="Thresholded", color='b', linestyle='--')
axs[2].plot(F, factor*gzf/r_0, label="Thresholded", color='b', linestyle='--')

analytic_Q = lambda z,a,z0: a*(z-z0)**(5/3)
analytic_M = lambda z,a,z0: a*(z-z0)**(4/3)
analytic_F = lambda z,a: a

poptQ, _ = optimize.curve_fit(analytic_Q, gzf[plume_indices], Q[plume_indices])
poptM, _ = optimize.curve_fit(analytic_M, gzf[plume_indices], M[plume_indices])
poptF, _ = optimize.curve_fit(analytic_F, gzf[plume_indices], F[plume_indices])

axs[0].plot(poptQ[0]*np.power(gzf-poptQ[1],5/3), factor*gzf/r_0, color='r')
axs[1].plot(poptM[0]*np.power(gzf-poptM[1],4/3), factor*gzf/r_0, color='r')
axs[2].axvline(poptF[0], color='r')

axs[0].axhline(factor*z_upper, color='grey', linestyle='--')
axs[0].axhline(factor*z_lower, color='grey', linestyle='--')
axs[1].axhline(factor*z_upper, color='grey', linestyle='--')
axs[1].axhline(factor*z_lower, color='grey', linestyle='--')
axs[2].axhline(factor*z_upper, color='grey', linestyle='--')
axs[2].axhline(factor*z_lower, color='grey', linestyle='--')

axs[0].plot(Q_full, factor*gzf/r_0,label="Full",color='b')
axs[1].plot(M_full, factor*gzf/r_0,label="Full",color='b')
axs[2].plot(F_full, factor*gzf/r_0,label="Full",color='b')
axs[0].set_xlim(0,1.1*max(max(Q),max(Q_full)))
axs[0].set_ylim(0,factor*md['LZ']/r_0)
axs[1].set_xlim(0,1.1*max(max(M),max(M_full)))
axs[1].set_ylim(0,factor*md['LZ']/r_0)
axs[2].set_xlim(0,1.1*max(max(F),max(F_full)))
axs[2].set_ylim(0,factor*md['LZ']/r_0)
axs[0].set_ylabel("$z/r_0$")
axs[1].set_ylabel("$z/r_0$")
axs[2].set_ylabel("$z/r_0$")

nticks = 5
ticks = np.linspace(0,md['LZ']/r_0,nticks)
axs[0].set_yticks(factor*ticks)
axs[0].set_yticklabels([str(int(i)) for i in ticks])
axs[1].set_yticks(factor*ticks)
axs[1].set_yticklabels([str(int(i)) for i in ticks])
axs[2].set_yticks(factor*ticks)
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

if save: fig0.savefig(save_loc+'fig0.png', dpi=300)

##### ----------------------- #####
# display threshold radius, theoretical radius w/ MvR et al coefficient, non-dimensionalised radius
factor = 0.6

fig7, ax7 = plt.subplots(1,2,figsize=(width,width*6/12))
if title: fig7.suptitle("Fig 1(a): ensemble and azimuthally \n averaged plume with radii scales")
w_im = ax7[0].imshow(np.flip(wbar,axis=0),cmap='seismic',extent=[0, md['LX']/(r_0), 0, factor*md['LZ']/r_0])
w_im.set_clim(-1,1)
ax7[0].plot([2*i/r_0 for i in r_d], factor*gzf[cont_valid_indices]/r_0, color='r',label="threshold radius")
ax7[0].plot([2*analytic_r(gzf[i],0.105,z_virt)/r_0 for i in cont_valid_indices],
        factor*gzf[cont_valid_indices]/r_0, color='k', linestyle='--', label="analytic radius (MvR)")
ax7[0].plot(2*r_m[cont_valid_indices]/r_0, factor*gzf[cont_valid_indices]/r_0, color='blue',
        label="$r_m$")
ax7[0].plot([2*analytic_r(gzf[i], alpha_p, z_virt)/r_0 for i in cont_valid_indices],
        factor*gzf[cont_valid_indices]/r_0, color='k', label="analytic radius (diablo)")
ax7[0].legend()
ax7[0].set_title("w")
ax7[0].set_xlabel("$r/r_0$")
ax7[0].set_ylabel("$z/r_0$")
ax7[0].set_xlim(0,md['LX']/r_0)
ax7[0].set_ylim(0,factor*md['LZ']/r_0)
ax7[0].axhline(factor*(md['Lyc']+md['Lyp'])/r_0, color='gray', alpha=0.5)
ax7[0].text(35, 0.5+factor*(md['Lyc']+md['Lyp'])/r_0, "Forcing region")
nticks = 5
ticks = np.linspace(0,md['LX']/r_0,nticks)
ax7[0].set_xticks(ticks)
ax7[0].set_xticklabels([str(int(i/2)) for i in ticks])
ticks = np.linspace(0,md['LZ']/r_0,nticks)
ax7[0].set_yticks(factor*ticks)
ax7[0].set_yticklabels([str(int(i)) for i in ticks])
ax7[0].axhline(factor*z_upper, color='gray', alpha=0.5,linestyle='--')
ax7[0].axhline(factor*z_lower, color='gray', alpha=0.5,linestyle='--')

divider0 = make_axes_locatable(ax7[0])
cax0 = divider0.append_axes('right', size='5%', pad=0.1)
cbar0 = fig7.colorbar(w_im, cax = cax0,label="$\\overline{w} \,\,\, (m \, s^{-1})$")

b_im = ax7[1].imshow(np.flip(bbar,axis=0),cmap='seismic',extent=[0, md['LX']/(r_0), 0, factor*md['LZ']/r_0])
b_im.set_clim(-0.15,0.15)
ax7[1].plot([2*i/r_0 for i in r_d], factor*gzf[cont_valid_indices]/r_0, color='r',label="threshold radius")
ax7[1].set_title("b")
ax7[1].plot([2*analytic_r(gzf[i],0.105,z_virt)/r_0 for i in cont_valid_indices],
        factor*gzf[cont_valid_indices]/r_0, color='k', linestyle='--', label="analytic radius (MvR)")
ax7[1].plot(2*r_m[cont_valid_indices]/r_0, factor*gzf[cont_valid_indices]/r_0, color='blue',
        label="$r_m$")
ax7[1].plot([2*analytic_r(gzf[i], alpha_p, z_virt)/r_0 for i in cont_valid_indices],
        factor*gzf[cont_valid_indices]/r_0, color='k', label="analytic radius (diablo)")
ax7[1].legend()
ax7[1].set_xlabel("$r/r_0$")
ax7[1].set_ylabel("$z/r_0$")
ax7[1].set_xlim(0,md['LX']/r_0)
ax7[1].set_ylim(0,factor*md['LZ']/r_0)
ax7[1].axhline(factor*(md['Lyc']+md['Lyp'])/r_0, color='gray', alpha=0.5)
ax7[1].text(35, 0.5+factor*(md['Lyc']+md['Lyp'])/r_0, "Forcing region")
nticks = 5
ticks = np.linspace(0,md['LX']/r_0,nticks)
ax7[1].set_xticks(ticks)
ax7[1].set_xticklabels([str(int(i/2)) for i in ticks])
ticks = np.linspace(0,md['LZ']/r_0,nticks)
ax7[1].set_yticks(factor*ticks)
ax7[1].set_yticklabels([str(int(i)) for i in ticks])
ax7[1].axhline(factor*z_upper, color='gray', alpha=0.5,linestyle='--')
ax7[1].axhline(factor*z_lower, color='gray', alpha=0.5,linestyle='--')

divider1 = make_axes_locatable(ax7[1])
cax1 = divider1.append_axes('right', size='5%', pad=0.1)
cbar1 = fig7.colorbar(b_im, cax = cax1,label="$\\overline{b} \,\,\, (m \, s^{-2})$")

plt.tight_layout()
if save: fig7.savefig(save_loc+'fig7.png', dpi=300)

##### ----------------------- #####

# plot power law scalings for w_m, b_m
fig1 = plt.figure(figsize=(12,6))
if title: fig1.suptitle("Fig 1(d): numerical and analytic characteristic scales $w_m, b_m$")
plt.loglog((gzf-z_virt)/r_0,b_m/b_m[plume_indices[0]], color='b', label="$b_m/b_{{m0}}$")
plt.loglog((gzf-z_virt)/r_0,w_m/w_m[plume_indices[0]], color='r', label="$w_m/w_{{m0}}$")
plt.legend()
yrange = (0.1,2)
xvals = [5,6,7,8,9,20,30,40,50,60,70,80,90]
for val in xvals:
    plt.loglog([val,val],yrange, color='grey',alpha=0.3, linestyle='dotted')
plt.loglog([4,100], [1,1], color='grey', alpha=0.3)
plt.loglog([10,10], yrange, color='grey', alpha=0.3)
plt.ylim(*yrange)
plt.xlim(4,100)
plt.xlabel("$(z - z_0)/r_0$")

F_0 = md['Q0'] * np.pi * r_0**2
#F_0 = md['b0'] * r_0**2
w_m_analytic = 5/6 * 1/alpha_p * np.power(9/10 * alpha_p * F_0/(theta_m_avg * beta_g_avg), 1/3) * \
        np.power(gzf, -1/3)
b_m_analytic = 5/6 * F_0/(alpha_p*theta_m_avg) * np.power(9/10 * alpha_p * F_0/(theta_m_avg * \
        beta_g_avg), -1/3) * np.power(gzf, -5/3)

plt.loglog(gzf[plume_indices]/r_0, b_m_analytic[plume_indices]/b_m[plume_indices[0]],
        color='b',linestyle='dashed')
plt.loglog(gzf[plume_indices]/r_0, w_m_analytic[plume_indices]/w_m[plume_indices[0]],
        color='r',linestyle='dashed')

plt.tight_layout()
if save: fig1.savefig(save_loc+'fig1.png', dpi=300)

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
if save: fig8.savefig(save_loc+'fig8.png', dpi=300)

##### ----------------------- #####

rs = np.linspace(0, 2, 1000)

fig2, axs2 = plt.subplots(1,2, figsize=(11,3.5))#, facecolor=(0.9,0.9,0.9))
if title: fig2.suptitle("Fig 4(c),(f): self-similar profiles of $\overline{w}, \overline{b}, \overline{u'w'}, \overline{u'b'}$")
for i in plume_indices:
    if i == plume_indices[0]:
        axs2[0].plot(r_points/r_m[i], wbar[i]/w_m[i],color='blue',label="$\overline{w}/w_m$")
        axs2[0].plot(r_points/r_m[i], bbar[i]/b_m[i],color='red',linestyle='dashed',label="$\overline{b}/b_m$")
        axs2[1].plot(r_points/r_m[i], uflucwflucbar[i]/(w_m[i]*w_m[i]), color='blue',
                label="$\overline{u'w'}/w_m^2$")
        axs2[1].plot(r_points/r_m[i], uflucbflucbar[i]/(w_m[i]*b_m[i]), color='red',
                linestyle='dashed', label="$\overline{u'b'}/w_m b_m$")
    else:
        axs2[0].plot(r_points/r_m[i], wbar[i]/w_m[i],color='blue')
        axs2[0].plot(r_points/r_m[i], bbar[i]/b_m[i],color='red',linestyle='dashed')
        axs2[1].plot(r_points/r_m[i], uflucwflucbar[i]/(w_m[i]*w_m[i]), color='blue')
        axs2[1].plot(r_points/r_m[i], uflucbflucbar[i]/(w_m[i]*b_m[i]), color='red', linestyle='dashed')

axs2[0].set_xlim(0, 2)
axs2[0].set_xlabel("$r/r_m$")
axs2[0].legend()
axs2[1].set_xlim(0,2)
axs2[1].set_xlabel("$r/r_m$")
axs2[1].legend()

#for text in l1.get_texts():
    #text.set_color(blue)
#for text in l2.get_texts():
    #text.set_color(blue)

nticks = 5
ticks = np.linspace(0,2,nticks)
axs2[1].set_xticks(ticks)
axs2[1].set_xticklabels(["{0:.1f}".format(i) for i in ticks])
axs2[0].set_xticks(ticks)
axs2[0].set_xticklabels(["{0:.1f}".format(i) for i in ticks])
#axs2[0].grid(color='grey', alpha=0.5)
#axs2[1].grid(color='grey', alpha=0.5)

#axs2[0].tick_params(color=blue, labelcolor=blue)
#axs2[1].tick_params(color=blue, labelcolor=blue)

#for spine in axs2[0].spines.values():
    #spine.set_edgecolor(blue)
#for spine in axs2[1].spines.values():
    #spine.set_edgecolor(blue)

fig2.tight_layout()
if save: fig2.savefig(save_loc+'fig2.png', dpi=300)
fig2.savefig(save_loc+'wb_uw_ub.png', dpi=300)

##### ----------------------- #####

rs = np.linspace(0, 2, 1000)

fig23, axs23 = plt.subplots(2,2, figsize=(width,width*6/15), sharex=True)

for i,c in zip(plume_indices,cols):
    if i == plume_indices[0]:
        axs23[0,0].plot(r_points/r_m[i], wbar[i]/w_m[i],color='blue',label="$\overline{w}/w_m$")
        axs23[0,0].plot(r_points/r_m[i], bbar[i]/b_m[i],color='red',linestyle='dashed',label="$\overline{b}/b_m$")
        axs23[0,1].plot(r_points/r_m[i], uflucwflucbar[i]/(w_m[i]*w_m[i]), color='blue',
                label="$\overline{u'w'}/w_m^2$")
        axs23[0,1].plot(r_points/r_m[i], uflucbflucbar[i]/(w_m[i]*b_m[i]), color='red',
                linestyle='dashed', label="$\overline{u'b'}/w_m b_m$")
    else:
        axs23[0,0].plot(r_points/r_m[i], wbar[i]/w_m[i],color='blue')
        axs23[0,0].plot(r_points/r_m[i], bbar[i]/b_m[i],color='red',linestyle='dashed')
        axs23[0,1].plot(r_points/r_m[i], uflucwflucbar[i]/(w_m[i]*w_m[i]), color='blue')
        axs23[0,1].plot(r_points/r_m[i], uflucbflucbar[i]/(w_m[i]*b_m[i]), color='red', linestyle='dashed')

    axs23[1,0].plot(r_points/r_m[i], ubar[i]/w_m[i],color=c)
    axs23[1,1].plot(r_points/r_m[i], r_points*ubar[i]/(r_m[i]*w_m[i]), color=c)

axs23[0,0].set_xlim(0, 2)
axs23[0,0].set_xlabel("$r/r_m$")
axs23[0,0].legend()
axs23[0,1].set_xlim(0,2)
axs23[0,1].set_xlabel("$r/r_m$")
axs23[0,1].legend()

axs23[1,1].axhline(-alpha_p, color='k', linestyle='dashed')
axs23[1,1].text(0.05, 0.005-alpha_p, "-$\\alpha_p$", fontsize=14)
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

fig23.tight_layout()
fig23.subplots_adjust(right=0.9)
cbar_ax = fig23.add_axes([0.92, 0.105, 0.02, 0.87])
fig23.colorbar(sm, cax=cbar_ax, label="$z/r_0$")
#cbar = fig23.colorbar(sm, ax=axs23.ravel().tolist(), shrink=0.95, label="$z/r_0$")
#fig23.colorbar(sm, cax = cax,label="$z/r_0$")
if save: fig23.savefig(save_loc+'fig23.png', dpi=300)

##### ----------------------- #####

fig3, axs3 = plt.subplots(1,2,figsize=(width,width*3.5/12))
if title: fig3.suptitle("Fig 5 (a), (b): self-similar profiles of $\overline{u}, r\overline{u}$")
for i,c in zip(plume_indices,cols):
    axs3[0].plot(r_points/r_m[i], ubar[i]/w_m[i],color=c)
    axs3[1].plot(r_points/r_m[i], r_points*ubar[i]/(r_m[i]*w_m[i]), color=c)

axs3[1].axhline(-alpha_p, color='k', linestyle='dashed')
axs3[1].text(0.05, 0.005-alpha_p, "-$\\alpha_p$", fontsize=14)
axs3[0].set_xlim(0, 2)
axs3[0].set_xlabel("$r/r_m$")
axs3[0].set_ylabel("$\\overline{u}/w_m$")
axs3[1].set_xlim(0, 2)
axs3[1].set_xlabel("$r/r_m$")
axs3[1].set_ylabel("$r\\overline{u}/(r_m w_m)$")

divider = make_axes_locatable(axs3[1])
cax = divider.append_axes('right', size='5%', pad=0.1)
fig3.colorbar(sm, cax = cax,label="$z/r_0$")
fig3.tight_layout()
if save: fig3.savefig(save_loc+'fig3.png', dpi=300)

##### ----------------------- #####

fig5, axs5 = plt.subplots(1,3, figsize=(15,5))
if title: fig5.suptitle("Fig 6 (c), (f), (i): self-similar profiles of $\overline{p}, \overline{u'u'}, \overline{v'v'}, \overline{w'w'}, \overline{w'b'}, \overline{b'b'}$")
for i,c in zip(plume_indices,cols):
    if i == plume_indices[0]:
        axs5[0].plot(r_points/r_m[i], ufluc2bar[i]/(w_m[i]*w_m[i]), color='blue',label="$\\overline{{u'u'}}/w_m^2$")
        axs5[0].plot(r_points/r_m[i], vfluc2bar[i]/(w_m[i]*w_m[i]), color='orange', linestyle='dashed',
                label="$\\overline{{v'v'}}/w_m^2$")
        axs5[0].plot(r_points/r_m[i], pbar[i]/(w_m[i]*w_m[i]), color=c,
                label="$\\overline{{p}}/w_m^2$")
        axs5[1].plot(r_points/r_m[i], wfluc2bar[i]/(w_m[i]*w_m[i]), color='blue',
                label="$\\overline{{w'w'}}/w_m^2$")
        axs5[1].plot(r_points/r_m[i], wflucbflucbar[i]/(w_m[i]*b_m[i]), color='red', linestyle='dashed',
                label="$\\overline{{w'b'}}/w_mb_m$")
        axs5[2].plot(r_points/r_m[i], bfluc2bar[i]/(b_m[i]*b_m[i]), color='blue',
                label="$\\overline{{b'b'}}/b_m^2$")
    else:
        axs5[0].plot(r_points/r_m[i], ufluc2bar[i]/(w_m[i]*w_m[i]), color='blue')
        axs5[0].plot(r_points/r_m[i], vfluc2bar[i]/(w_m[i]*w_m[i]), color='orange', linestyle='dashed')
        axs5[0].plot(r_points/r_m[i], pbar[i]/(w_m[i]*w_m[i]), color=c)
        axs5[1].plot(r_points/r_m[i], wfluc2bar[i]/(w_m[i]*w_m[i]), color='blue')
        axs5[1].plot(r_points/r_m[i], wflucbflucbar[i]/(w_m[i]*b_m[i]), color='red', linestyle='dashed')
        axs5[2].plot(r_points/r_m[i], bfluc2bar[i]/(b_m[i]*b_m[i]), color='blue')

divider = make_axes_locatable(axs5[0])
cax = divider.append_axes('right', size='5%', pad=0.55)
fig5.colorbar(sm, cax = cax, label="$z/r_0$")
cax.yaxis.set_ticks_position('left')
cax.yaxis.set_label_position('left')

nticks = 5
ticks = np.linspace(0,2,nticks)
axs5[0].set_xticks(ticks)
axs5[0].set_xticklabels(["{0:.1f}".format(i) for i in ticks])
axs5[1].set_xticks(ticks)
axs5[1].set_xticklabels(["{0:.1f}".format(i) for i in ticks])
axs5[2].set_xticks(ticks)
axs5[2].set_xticklabels(["{0:.1f}".format(i) for i in ticks])

axs5[0].grid(color='grey', alpha=0.5)
axs5[1].grid(color='grey', alpha=0.5)
axs5[2].grid(color='grey', alpha=0.5)

axs5[0].legend()
axs5[1].legend()
axs5[2].legend()
axs5[0].set_xlim(0,2)
axs5[1].set_xlim(0,2)
axs5[2].set_xlim(0,2)
axs5[0].set_xlabel("$r/r_0$")
axs5[1].set_xlabel("$r/r_0$")
axs5[2].set_xlabel("$r/r_0$")

fig5.tight_layout()
if save: fig5.savefig(save_loc+'fig5.png', dpi=300)

##### ----------------------- #####

fig4 = plt.figure()
if title: fig4.suptitle("Fig 7(f): radial and vertical pressure gradients")
for i in plume_indices:
    if i == plume_indices[0]:
        plt.plot(r_points/r_m[i], dpbardr[i]/(w_m[i]*w_m[i]/r_m[i]), color='blue', label="$\partial \overline{p}/\partial r$")
        plt.plot(r_points/r_m[i], dpbardz[i]/(w_m[i]*w_m[i]/r_m[i]), color='red', linestyle='dashed',
                label="$\partial \overline{p}/\partial z$")
    else:
        plt.plot(r_points/r_m[i], dpbardr[i]/(w_m[i]*w_m[i]/r_m[i]), color='blue')
        plt.plot(r_points/r_m[i], dpbardz[i]/(w_m[i]*w_m[i]/r_m[i]), color='red', linestyle='dashed')
nticks = 5
ticks = np.linspace(0,2,nticks)
plt.xticks(ticks, ["{0:.1f}".format(i) for i in ticks])

plt.grid(color='grey', alpha=0.5)

plt.xlim(0,2)
plt.xlabel("$r/r_m$")
plt.legend()

plt.tight_layout()
if save: fig4.savefig(save_loc+'fig4.png', dpi=300)

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
if save: fig13.savefig(save_loc+'fig13.png', dpi=300)

##### ----------------------- #####
# plots of beta_m, gamma_m, delta_m, theta_m vs z/r_0

fig10, ax10 = plt.subplots(1,3,figsize=(width,width*3/12))
if title: fig10.suptitle("Fig 9(c): mean profile coefficients")
ax10[0].set_xlim(-0.6, 1.4)
ax10[0].set_ylim(np.min(gzf[cont_valid_indices])/r_0, np.max(gzf[cont_valid_indices])/r_0)
ax10[0].set_ylabel("$z/r_0$")
ax10[0].set_title("Mean flow contribution")
ax10[0].plot(theta_m[cont_valid_indices],gzf[cont_valid_indices]/r_0, linestyle="-.",color='purple',
        label="$\\theta_m$")
ax10[0].plot(gamma_m[cont_valid_indices],gzf[cont_valid_indices]/r_0, linestyle="--",color='red',
        label="$\\gamma_m$")
ax10[0].plot(delta_m[cont_valid_indices],gzf[cont_valid_indices]/r_0,color='orange',
        label="$\\delta_m$")
ax10[0].plot(beta_m*np.ones_like(cont_valid_indices),gzf[cont_valid_indices]/r_0, color='b',
        label="$\\beta_m=1$")
ax10[0].legend()
ax10[0].axhline(z_lower, color='grey', alpha=0.5, linestyle='--')
ax10[0].axhline(z_upper, color='grey', alpha=0.5, linestyle='--')

ticks = np.linspace(-0.5,1,4)
ax10[0].set_xticks(ticks)
ax10[1].set_xticks(ticks)
ax10[2].set_xticks(ticks)

ax10[1].set_xlim(-0.6, 1.4)
ax10[1].set_ylim(np.min(gzf[cont_valid_indices])/r_0, np.max(gzf[cont_valid_indices])/r_0)
ax10[1].set_title("Turbulence contribution")
ax10[1].plot(theta_f[cont_valid_indices],gzf[cont_valid_indices]/r_0, linestyle="-.",color='purple',
        label="$\\theta_f$")
ax10[1].plot(gamma_f[cont_valid_indices],gzf[cont_valid_indices]/r_0, linestyle="--",color='red',
        label="$\\gamma_f$")
ax10[1].plot(delta_f[cont_valid_indices],gzf[cont_valid_indices]/r_0,color='orange',
        label="$\\delta_f$")
ax10[1].plot(beta_f[cont_valid_indices],gzf[cont_valid_indices]/r_0, color='b',
        label="$\\beta_f$")
ax10[1].legend()
ax10[1].axhline(z_lower, color='grey', alpha=0.5, linestyle='--')
ax10[1].axhline(z_upper, color='grey', alpha=0.5, linestyle='--')

ax10[2].set_xlim(-0.6, 1.4)
ax10[2].set_title("Pressure contribution")
ax10[2].set_ylim(np.min(gzf[cont_valid_indices])/r_0, np.max(gzf[cont_valid_indices])/r_0)
ax10[2].plot(gamma_p[cont_valid_indices],gzf[cont_valid_indices]/r_0, linestyle="--",color='red',
        label="$\\gamma_p$")
ax10[2].plot(delta_p[cont_valid_indices],gzf[cont_valid_indices]/r_0,color='orange',
        label="$\\delta_p$")
ax10[2].plot(beta_p[cont_valid_indices],gzf[cont_valid_indices]/r_0, color='b',
        label="$\\beta_p$")
ax10[2].legend()
ax10[2].axhline(z_lower, color='grey', alpha=0.5, linestyle='--')
ax10[2].axhline(z_upper, color='grey', alpha=0.5, linestyle='--')

plt.tight_layout()
if save: fig10.savefig(save_loc+'fig10.png', dpi=300)

##### ----------------------- #####

fig11 = plt.figure()
if title: plt.title("Fig 9(f): relative contribution of turbulence and pressure")
ticks = np.linspace(-0.2,0.2,3)
plt.xticks(ticks)
plt.xlim(-0.3, 0.3)
plt.xlabel("$r/r_m$")
plt.ylabel("$z/r_0$")
plt.ylim(0, md['LZ']/r_0)
plt.plot(beta_g[cont_valid_indices]-1, gzf[cont_valid_indices]/r_0, color='b',label="$\\beta_g-1$")
plt.plot(((gamma_g-gamma_m)/gamma_m)[cont_valid_indices], gzf[cont_valid_indices]/r_0, color='red',
        linestyle='dashed',label="$(\\gamma_g-\gamma_m)/\\gamma_m$")
plt.plot(((delta_g-delta_m)/delta_m)[cont_valid_indices], gzf[cont_valid_indices]/r_0, color='orange',
        label="$(\\delta_g-\delta_m)/\\delta_m$")
plt.plot(((theta_g-theta_m)/theta_m)[cont_valid_indices], gzf[cont_valid_indices]/r_0, color='purple',
        linestyle='-.',label="$(\\theta_g-\\theta_m)/\\theta_m$")
plt.axhline(z_lower, color='grey', alpha=0.5, linestyle='--')
plt.axhline(z_upper, color='grey', alpha=0.5, linestyle='--')
plt.legend()
plt.tight_layout()
if save: fig11.savefig(save_loc+'fig11.png', dpi=300)

##### ----------------------- #####

fig12 = plt.figure()
plt.plot(alpha_prod, gzf[cont_valid_indices]/r_0,color="b",label="$\\alpha_{prod}}$")
plt.plot(alpha_Ri, gzf[cont_valid_indices]/r_0,color="red",linestyle="--",label="$\\alpha_\\mathrm{{Ri}}$")
plt.plot(alpha_shape, gzf[cont_valid_indices]/r_0,color="orange",label="$\\alpha_{{shape}}$")
plt.plot(alpha_prod+alpha_Ri+alpha_shape, gzf[cont_valid_indices]/r_0, color="purple", linestyle="-.",
        label="$\\sum \\alpha_\\chi$")
#plt.plot(alpha_prod[-1]+alpha_Ri+alpha_shape, gzfp[cont_valid_indices], color="purple", linestyle=":",
        #alpha=0.5)
plt.ylabel("$z/r_0$")
plt.scatter(alpha, gzf[cont_valid_indices]/r_0, color="purple")
if title: plt.title("Fig 10: contribution to entrainment coefficient from TKE production $\\alpha_{{prod}}$, \n \
buoyancy $\\alpha_\\mathrm{{Ri}}$ and departure from self-similarity $\\alpha_{{shape}}$")
plt.legend()
plt.ylim(0,md['LZ']/r_0)
plt.xlim(-0.01,0.2)
ticks = np.linspace(0,0.2,5)
plt.xticks(ticks)
plt.axhline(z_lower, color='grey', alpha=0.5, linestyle='--')
plt.axhline(z_upper, color='grey', alpha=0.5, linestyle='--')

plt.axvline(alpha_p, color='k', linestyle=':', label="$\\alpha_p$")
plt.axvline(0.6*alpha_p, color='k', linestyle='-.', label="$\\alpha_j$")

if save: fig12.savefig(save_loc+'fig12.png', dpi=300)

##### ----------------------- #####

fig6, ax6 = plt.subplots(1,2, figsize=(12,4))
if title: fig6.suptitle("Fig 12: radial profiles of $\\nu_T$ and $D_T$ and radial mixing lengths")
ax6[0].plot(r_regrid, nu_T, color='b', label="$\\nu_T/(w_m r_m)$")
ax6[0].plot(r_regrid, D_T, linestyle='--', color='r', label="$D_T/(w_m r_m)$")
ax6[0].set_ylim(0,0.1)
ax6[0].set_xlim(0,1.5)
ax6[0].set_xlabel("$r/r_m$")
ax6[0].grid(color='grey', alpha=0.5)
ax6[0].legend()

ax6[1].plot(r_regrid, l_m, color='b', label="$l_m/r_m$")
ax6[1].plot(r_regrid, l_mb, linestyle='--', color='r', label="$l_{mb}/r_m$")
ax6[1].set_ylim(0,0.3)
ax6[1].set_xlim(0,1.5)
ax6[1].set_xlabel("$r/r_m$")
ax6[1].legend()
ax6[1].grid(color='grey', alpha=0.5)

ticks = np.linspace(0, 1.5, 4)
ax6[0].set_xticks(ticks)
ax6[1].set_xticks(ticks)
ticks = np.linspace(0, 0.1, 3)
ax6[0].set_yticks(ticks)
ticks = np.linspace(0, 0.3, 4)
ax6[1].set_yticks(ticks)

plt.tight_layout()
if save: fig6.savefig(save_loc+'fig6.png', dpi=300)

##### ----------------------- #####

fig14 = plt.figure(constrained_layout=True, figsize=(10,8))
ax14 = fig14.subplot_mosaic(
        """
        AA
        BC
        """,
        gridspec_kw = {"hspace": 0.1, "wspace":0.1}
        )

ticks = np.linspace(0, 1.5, 4)
yticks = np.linspace(0, 2, 5)
yticks2 = np.linspace(0, 1.2, 7)

ax14["A"].plot(r_regrid, Pr_T, color='b', label="Data")
ax14["A"].plot(r_regrid, [aw2_an/ab2_an] * len(r_regrid), color='r', linestyle='--', label="Analytic")
ax14["A"].set_xlim(0, 1.5)
ax14["A"].set_xticks(ticks)
ax14["A"].set_ylim(0, 1.2)
ax14["A"].set_yticks(yticks2)
ax14["A"].set_ylabel("$\\mathrm{Pr}_T$")
ax14["A"].set_xlabel("$r/r_m$")

ax14["B"].plot(r_regrid, fb_fw, color='b', label="Data")
ax14["B"].plot(r_regrid, fbfw_an, color='r', linestyle='--', label="Analytic")
ax14["B"].axhline(1, color='k', linestyle=':', alpha=0.5)
ax14["B"].set_xlim(0, 1.5)
ax14["B"].set_ylim(0, 2)
ax14["B"].set_xticks(ticks)
ax14["B"].set_yticks(yticks)
ax14["B"].set_ylabel("$f'_b/f'_w$")
ax14["B"].set_xlabel("$r/r_m$")

ax14["C"].plot(r_regrid, fuw_fub, color='b', label="Data")
ax14["C"].plot(r_regrid, fuwfub_an, color='r', linestyle='--', label="Analytic")
ax14["C"].axhline(1, color='k', linestyle=':', alpha=0.5)
ax14["C"].set_xlim(0, 1.5)
ax14["C"].set_ylim(0, 2)
ax14["C"].set_xticks(ticks)
ax14["C"].set_yticks(yticks)
ax14["C"].set_xlabel("$r/r_m$")
ax14["C"].set_ylabel("$f_{uw}/f_{ub}$")

if save: fig14.savefig(save_loc+'fig14.png', dpi=300)

##### ----------------------- #####

fig15 = plt.figure()
plt.scatter(Gamma[plume_indices], alpha_cont, color='b', label="Data")
gammas = np.linspace(0, 1.5, 50)
plt.plot(gammas, 0.6*alpha_p + 0.4*alpha_p*gammas, linestyle='dashed', color='k', label="PB model")
plt.xlim(0, 1.5)
plt.ylim(0, 0.2)
plt.xlabel("$\\Gamma$")
plt.ylabel("$\\alpha$")
ticks = np.linspace(0, 1.5, 4)
plt.xticks(ticks)
ticks = np.linspace(0, 0.2, 5)
plt.yticks(ticks)

plt.legend()
plt.tight_layout()
if save: fig15.savefig(save_loc+'fig15.png', dpi=300)

if show: plt.show()
