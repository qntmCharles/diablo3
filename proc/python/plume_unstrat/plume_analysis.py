import h5py, bisect, time, gc, sys
from os import listdir
from os.path import isfile, join
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from functions import get_metadata, get_grid, read_params, get_az_data
from scipy import integrate, optimize, interpolate
from matplotlib import cm as cm
from itertools import groupby
from mpl_toolkits.axes_grid1 import make_axes_locatable

##### USER-DEFINED PARAMETERS #####
params_file = "./params.dat"
save_loc = "/home/cwp29/Documents/plume_project/figs/mvr/"
save = False
title = True
show = True

z_upper = 0.19 / 0.002
z_lower = 0.15 / 0.002

eps = 0.02

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
pbar_trunc = truncate(pbar, r_points, wbar, eps*wbar[:,0], plume_indices)
wfluc2bar_trunc = truncate(wfluc2bar, r_points, wbar, eps*wbar[:,0], plume_indices)
uflucwflucbar_trunc = truncate(uflucwflucbar, r_points, wbar, eps*wbar[:,0], plume_indices)
wflucbflucbar_trunc = truncate(wflucbflucbar, r_points, wbar, eps*wbar[:,0], plume_indices)
dwbardz_trunc = truncate(dwbardz, r_points, wbar, eps*wbar[:,0], plume_indices)
dwbardr_trunc = truncate(dwbardr, r_points, wbar, eps*wbar[:,0], plume_indices)

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
beta_p = 2/(w_m*w_m*r_m*r_m)*integrate.trapezoid(pbar_trunc*r_integrate, r_integrate, axis=1)
beta_m = 1
beta_g = 1 + beta_f + beta_p

beta_m_avg = 1
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
print("Beta (f, p, g)")
print(beta_f_avg, beta_p_avg, beta_g_avg)
print("Gamma (m, f, p, g)")
print(gamma_m_avg, gamma_f_avg, gamma_p_avg, gamma_g_avg)
print("Delta (m, f, p, g)")
print(delta_m_avg, delta_f_avg, delta_p_avg, delta_g_avg)
print("Theta (m, f, g)")
print(theta_m_avg, theta_f_avg, theta_g_avg)

##### Estimate alpha_p (plume entrainment coefficient) #####
analytic_r = lambda z,a,z0: 6/5 * a * (z-z0) * r_0
z_comp = np.array([gzf[i]/r_0 for i in plume_indices])
r_comp = np.array([r_m[i] for i in plume_indices])

popt, _ = optimize.curve_fit(analytic_r, z_comp, r_comp)
alpha_p = popt[0]
z_virt = popt[1]

Gamma = 5*F*np.power(Q,2)/(8*alpha_p*beta_g*theta_m*np.power(M,5/2))

##### Calculate alpha (entrainment coefficient) analytically #####
dQdz = np.gradient(Q[plume_indices], gzf[plume_indices])
alpha = 0.5*np.power(M[plume_indices],-1/2)*dQdz
zplot = gzf[plume_indices]

##### Decompose entrainment coefficient #####
alpha_prod = -0.5*delta_g[plume_indices]/gamma_g[plume_indices]
alpha_Ri = alpha_p * (np.power(beta_g[plume_indices],-1) -
        theta_m[plume_indices]/gamma_g[plume_indices]) * 8/5 * beta_g[plume_indices]

shape_operand = np.log(np.power(gamma_g,1/2)/beta_g)[plume_indices]
# original alpha_shape (noisy):
#alpha_shape = r_m[plume_indices]*np.gradient(shape_operand, gzfp[plume_indices])
# smoothed alpha_shape:
nbin = 17
kernel = np.ones(nbin)/nbin
gradient_shape = np.gradient(shape_operand,gzf[plume_indices])
padded_shape = np.pad(gradient_shape, (nbin//2,nbin-1-nbin//2),mode='edge')
alpha_shape = r_m[plume_indices]*np.convolve(padded_shape,kernel,mode='valid')

##### Calculated corrected pressure
dpdz_avg = np.mean([dpbardz[i]/(w_m[i]*w_m[i]/r_m[i]) for i in plume_indices], axis=1)
p_mod = np.zeros_like(dpdz_avg)
dpdz_avg = np.mean(dpdz_avg)
for i in plume_indices[1:]:
    zero_ind = plume_indices[0]
    p_mod[i-zero_ind] = p_mod[i-zero_ind-1] + dpdz_avg*(gzf[i]-gzf[i-1])/r_m[i]

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
        norm=plt.Normalize(vmin=np.min(gzf[plume_indices]),vmax=np.max(gzf[plume_indices])))

##### ----------------------- #####

fig0, axs = plt.subplots(1,3,figsize=(10, 6))
if title: fig0.suptitle("Plume integral quantities Q, M, B")
axs[0].plot(Q, gzfp/r_0, label="Thresholded")
axs[1].plot(M, gzfp/r_0, label="Thresholded")
axs[2].plot(F, gzfp/r_0, label="Thresholded")
axs[0].plot(Q_full, gzfp/r_0,label="Full")
axs[1].plot(M_full, gzfp/r_0,label="Full")
axs[2].plot(F_full, gzfp/r_0,label="Full")
axs[0].set_xlim(0,1.1*max(max(Q),max(Q_full)))
axs[0].set_ylim(0,md['LZ']/r_0)
axs[1].set_xlim(0,1.1*max(max(M),max(M_full)))
axs[1].set_ylim(0,md['LZ']/r_0)
axs[2].set_xlim(0,1.1*max(max(F),max(F_full)))
axs[2].set_ylim(0,md['LZ']/r_0)
axs[0].set_ylabel("$z/r_0$")
axs[1].set_ylabel("$z/r_0$")
axs[2].set_ylabel("$z/r_0$")
nticks = 9
ticks = np.linspace(0,md['LZ']/r_0,nticks)
axs[0].set_yticks(ticks)
axs[0].set_yticklabels([str(int(i)) for i in ticks])
axs[1].set_yticks(ticks)
axs[1].set_yticklabels([str(int(i)) for i in ticks])
axs[2].set_yticks(ticks)
axs[2].set_yticklabels([str(int(i)) for i in ticks])
axs[0].legend()
axs[1].legend()
axs[2].legend()
axs[0].set_title("Q, volume flux")
axs[1].set_title("M, momentum flux")
axs[2].set_title("F, buoyancy flux")
plt.tight_layout()

if save: fig0.savefig(save_loc+'fig0.png', dpi=300)

##### ----------------------- #####

# plot power law scalings for w_m, b_m
fig1 = plt.figure()
if title: fig1.suptitle("Fig 1(d): numerical and analytic characteristic scales $w_m, b_m$")
plt.loglog(gzf/r_0,b_m/np.nanmean(b_m), color='b', label="$b_m/b_{{m0}}$")
plt.loglog(gzf/r_0,w_m/np.nanmean(w_m), color='r', label="$w_m/w_{{m0}}$")
plt.legend()
yrange = plt.gca().get_ylim()
xvals = [5,6,7,8,9,20,30,40,50,60,70,80,90]
for val in xvals:
    plt.loglog([val,val],yrange, color='grey',alpha=0.3, linestyle='dotted')
plt.loglog([4,100], [1,1], color='grey', alpha=0.3)
plt.loglog([10,10], yrange, color='grey', alpha=0.3)
plt.ylim(*yrange)
plt.xlim(4,100)
plt.xlabel("$z/r_0$")

#F_0 = md['Q0'] * np.pi * r_0**2
F_0 = md['b0'] * r_0**2
w_m_analytic = 5/6 * 1/alpha_p * np.power(9/10 * alpha_p * F_0/(theta_m_avg * beta_g_avg), 1/3) * \
        np.power(gzf, -1/3)
b_m_analytic = 5/6 * F_0/(alpha_p*theta_m_avg) * np.power(9/10 * alpha_p * F_0/(theta_m_avg * \
        beta_g_avg), -1/3) * np.power(gzf, -5/3)

plt.loglog(gzf[plume_indices]/r_0, b_m_analytic[plume_indices]/np.nanmean(b_m),
        color='b',linestyle='dashed')
plt.loglog(gzf[plume_indices]/r_0, w_m_analytic[plume_indices]/np.nanmean(w_m),
        color='r',linestyle='dashed')
if save: fig1.savefig(save_loc+'fig1.png', dpi=300)

##### ----------------------- #####

fig2, axs2 = plt.subplots(1,2, figsize=(11,5))
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
fig2.tight_layout()
if save: fig2.savefig(save_loc+'fig2.png', dpi=300)

##### ----------------------- #####

fig3, axs3 = plt.subplots(1,2,figsize=(12,5))
if title: fig3.suptitle("Fig 5 (a), (b): self-similar profiles of $\overline{u}, r\overline{u}$")
for i,c in zip(plume_indices,cols):
    axs3[0].plot(r_points/r_m[i], ubar[i]/w_m[i],color=c)
    axs3[1].plot(r_points/r_m[i], r_points*ubar[i]/(r_m[i]*w_m[i]), color=c)

axs3[1].plot(plt.gca().get_xlim(),[-alpha_p, -alpha_p], color='black', linestyle='dashed',label="$-\\alpha_p$")
axs3[1].legend()
axs3[0].set_xlim(0, 2)
axs3[0].set_xlabel("$r/r_m$")
axs3[0].set_ylabel("$\\overline{u}/w_m$")
axs3[1].set_xlim(0, 2)
axs3[1].set_xlabel("$r/r_m$")
axs3[1].set_ylabel("$r\\overline{u}/(r_m w_m)$")
divider = make_axes_locatable(axs3[1])
cax = divider.append_axes('right', size='5%', pad=0.1)
fig3.colorbar(sm, cax = cax)
fig3.tight_layout()
if save: fig3.savefig(save_loc+'fig3.png', dpi=300)

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
fig4.legend()
plt.xlim(0,2)
plt.xlabel("$r/r_m$")
fig4.tight_layout()
if save: fig4.savefig(save_loc+'fig4.png', dpi=300)

##### ----------------------- #####

fig5, axs5 = plt.subplots(1,3, figsize=(15,5))
if title: fig5.suptitle("Fig 6 (c), (f), (i): self-similar profiles of $\overline{p}, \overline{u'u'}, \overline{v'v'}, \overline{w'w'}, \overline{w'b'}, \overline{b'b'}$")
for i,c in zip(plume_indices,cols):
    if i == plume_indices[0]:
        axs5[0].plot(r_points/r_m[i], ufluc2bar[i]/(w_m[i]*w_m[i]), color='blue',label="$\\overline{{u'u'}}/w_m^2$")
        axs5[0].plot(r_points/r_m[i], vfluc2bar[i]/(w_m[i]*w_m[i]), color='orange', linestyle='dashed',
                label="$\\overline{{v'v'}}/w_m^2$")
        axs5[0].plot(r_points/r_m[i], pbar[i]/(w_m[i]*w_m[i])-p_mod[i-zero_ind], color=c,
                label="corrected $\\overline{{p}}/w_m^2$")
        axs5[1].plot(r_points/r_m[i], wfluc2bar[i]/(w_m[i]*w_m[i]), color='blue',
                label="$\\overline{{w'w'}}/w_m^2$")
        axs5[1].plot(r_points/r_m[i], wflucbflucbar[i]/(w_m[i]*b_m[i]), color='red', linestyle='dashed',
                label="$\\overline{{w'b'}}/w_mb_m$")
        axs5[2].plot(r_points/r_m[i], bfluc2bar[i]/(b_m[i]*b_m[i]), color='blue',
                label="$\\overline{{b'b'}}/b_m^2$")
    else:
        axs5[0].plot(r_points/r_m[i], ufluc2bar[i]/(w_m[i]*w_m[i]), color='blue')
        axs5[0].plot(r_points/r_m[i], vfluc2bar[i]/(w_m[i]*w_m[i]), color='orange', linestyle='dashed')
        axs5[0].plot(r_points/r_m[i], pbar[i]/(w_m[i]*w_m[i])-p_mod[i-zero_ind], color=c)
        axs5[1].plot(r_points/r_m[i], wfluc2bar[i]/(w_m[i]*w_m[i]), color='blue')
        axs5[1].plot(r_points/r_m[i], wflucbflucbar[i]/(w_m[i]*b_m[i]), color='red', linestyle='dashed')
        axs5[2].plot(r_points/r_m[i], bfluc2bar[i]/(b_m[i]*b_m[i]), color='blue')

divider = make_axes_locatable(axs5[0])
cax = divider.append_axes('right', size='5%', pad=0.1)
fig5.colorbar(sm, cax = cax)

axs5[0].legend()
axs5[1].legend()
axs5[2].legend()
axs5[0].set_xlim(0,2)
axs5[1].set_xlim(0,2)
axs5[2].set_xlim(0,2)
fig5.tight_layout()
if save: fig5.savefig(save_loc+'fig5.png', dpi=300)

##### ----------------------- #####

fig6 = plt.figure()
if title: plt.title("Fig 12 (b): radial profiles of $\\nu_T$ and $D_T$")
plt.plot(r_regrid, nu_T, color='b', label="$\\nu_T$")
plt.plot(r_regrid, D_T, linestyle='--', color='r', label="$D_T$")
plt.ylim(0,0.1)
plt.xlim(0,1.5)
plt.ylabel("$\\nu_T, D_T$ normalised")
plt.xlabel("$r/r_m$")
plt.legend()
if save: fig6.savefig(save_loc+'fig6.png', dpi=300)

##### ----------------------- #####
# display threshold radius, theoretical radius w/ MvR et al coefficient, non-dimensionalised radius

wbar_plot = wbar_trunc[plume_indices]
fig7 = plt.figure(figsize=(4,8))
if title: fig7.suptitle("Fig 1(a): ensemble and azimuthally \n averaged plume with radii scales")
plt.imshow(np.flip(wbar_plot,axis=0),cmap='jet',extent=[0, md['LX']/(2*r_0), np.min(gzf[plume_indices])/r_0,
    np.max(gzf[plume_indices])/r_0])
plt.plot([i/r_0 for i in r_d], gzf[plume_indices]/r_0, color='r',label="threshold radius")
plt.plot([analytic_r(gzf[i]/r_0,0.105,z_virt)/r_0 for i in plume_indices],
        gzf[plume_indices]/r_0, color='g', label="analytic radius (MvR)")
plt.plot(r_m[plume_indices]/r_0, gzf[plume_indices]/r_0, color='y',
        label="characteristic radius $r_m$")
plt.plot([analytic_r(gzf[i]/r_0, alpha_p, z_virt)/r_0 for i in plume_indices],
        gzf[plume_indices]/r_0, color='orange', label="analytic radius (diablo)")
plt.legend()
plt.xlabel("$r/r_0$")
plt.ylabel("$z/r_0$")
plt.tight_layout()
if save: fig7.savefig(save_loc+'fig7.png', dpi=300)

##### ----------------------- #####
# estimate flux balance parameter
fig8 = plt.figure()
if title: fig8.suptitle("Fig 2(a): flux balance parameter $\\Gamma = 5FQ^2 / 8\\beta_g \\alpha_p \\theta_m M^{{5/2}}$")
plt.ylabel("$z/r_0$")
plt.xlabel("$\\Gamma$")
plt.plot(Gamma[plume_indices], gzf[plume_indices]/r_0)
plt.xlim(0,2)
plt.plot([1,1],[np.min(gzf[plume_indices])/r_0, np.max(gzf[plume_indices])/r_0],
    linestyle='dashed', color='k')
plt.ylim(np.min(gzf[plume_indices])/r_0, np.max(gzf[plume_indices])/r_0)
plt.tight_layout()
if save: fig8.savefig(save_loc+'fig8.png', dpi=300)

##### ----------------------- #####
# analytic alpha (entrainment coefficient)

fig9 = plt.figure()
if title: fig9.suptitle("Fig 2(b): theoretical and numerical entrainment coefficient")
plt.ylabel("$z/r_0$")
plt.xlabel("$\\alpha$")
plt.plot(alpha, zplot/r_0)
plt.plot([alpha_p]*(len(zplot)), zplot/r_0, color='black', linestyle='dashed',
        label="$\\alpha_p = {0:.3f}$".format(alpha_p))
plt.legend()
plt.ylim(np.min(gzf[plume_indices])/r_0, np.max(gzf[plume_indices])/r_0)
plt.xlim(0,0.2)
if save: fig9.savefig(save_loc+'fig9.png', dpi=300)

##### ----------------------- #####
# plots of beta_m, gamma_m, delta_m, theta_m vs z/r_0

fig10, ax10 = plt.subplots(1,3,figsize=(12,5))
if title: fig10.suptitle("Fig 9(c): mean profile coefficients")
ax10[0].set_xlim(-0.6, 1.4)
ax10[0].set_ylim(np.min(gzf[plume_indices])/r_0, np.max(gzf[plume_indices])/r_0)
ax10[0].set_ylabel("$z/r_0$")
ax10[0].set_xlabel("$r/r_m$")
ax10[0].set_title("Mean flow contribution")
ax10[0].plot(theta_m[plume_indices],gzf[plume_indices]/r_0, linestyle="-.",color='purple',
        label="$\\theta_m$")
ax10[0].plot(gamma_m[plume_indices],gzf[plume_indices]/r_0, linestyle="--",color='red',
        label="$\\gamma_m$")
ax10[0].plot(delta_m[plume_indices],gzf[plume_indices]/r_0,color='orange',
        label="$\\delta_m$")
ax10[0].plot(beta_m*np.ones_like(plume_indices),gzf[plume_indices]/r_0, color='b',
        label="$\\beta_m=1$")
ax10[0].legend()

ax10[1].set_xlim(-0.6, 1.4)
ax10[1].set_ylim(np.min(gzf[plume_indices])/r_0, np.max(gzf[plume_indices])/r_0)
ax10[1].set_xlabel("$r/r_m$")
ax10[1].set_title("Turbulence contribution")
ax10[1].plot(theta_f[plume_indices],gzf[plume_indices]/r_0, linestyle="-.",color='purple',
        label="$\\theta_f$")
ax10[1].plot(gamma_f[plume_indices],gzf[plume_indices]/r_0, linestyle="--",color='red',
        label="$\\gamma_f$")
ax10[1].plot(delta_f[plume_indices],gzf[plume_indices]/r_0,color='orange',
        label="$\\delta_f$")
ax10[1].plot(beta_f[plume_indices],gzf[plume_indices]/r_0, color='b',
        label="$\\beta_f$")
ax10[1].legend()

ax10[2].set_xlim(-0.6, 1.4)
ax10[2].set_title("Pressure contribution")
ax10[2].set_ylim(np.min(gzf[plume_indices])/r_0, np.max(gzf[plume_indices])/r_0)
ax10[2].set_xlabel("$r/r_m$")
ax10[2].plot(gamma_p[plume_indices],gzf[plume_indices]/r_0, linestyle="--",color='red',
        label="$\\gamma_p$")
ax10[2].plot(delta_p[plume_indices],gzf[plume_indices]/r_0,color='orange',
        label="$\\delta_p$")
ax10[2].plot(beta_p[plume_indices],gzf[plume_indices]/r_0, color='b',
        label="$\\beta_p$")
ax10[2].legend()
plt.tight_layout()
if save: fig10.savefig(save_loc+'fig10.png', dpi=300)

##### ----------------------- #####

fig11 = plt.figure()
if title: plt.title("Fig 9(f): relative contribution of turbulence and pressure")
plt.xlim(-0.3, 0.3)
plt.xlabel("$r/r_m$")
plt.ylabel("$z/r_0$")
plt.ylim(np.min(gzf[plume_indices])/r_0, np.max(gzf[plume_indices])/r_0)
plt.plot(beta_g[plume_indices]-1, gzf[plume_indices]/r_0, color='b',label="$\\beta_g-1$")
plt.plot(((gamma_g-gamma_m)/gamma_m)[plume_indices], gzf[plume_indices]/r_0, color='red',
        linestyle='dashed',label="$(\\gamma_g-\gamma_m)/\\gamma_m$")
plt.plot(((delta_g-delta_m)/delta_m)[plume_indices], gzf[plume_indices]/r_0, color='orange',
        label="$(\\delta_g-\delta_m)/\\delta_m$")
plt.plot(((theta_g-theta_m)/theta_m)[plume_indices], gzf[plume_indices]/r_0, color='purple',
        linestyle='-.',label="$(\\theta_g-\\theta_m)/\\theta_m$")
plt.legend()
plt.tight_layout()
if save: fig11.savefig(save_loc+'fig11.png', dpi=300)

##### ----------------------- #####

fig12 = plt.figure()
plt.plot(alpha_prod, gzf[plume_indices],color="b",label="$\\alpha_{prod}}$")
plt.plot(alpha_Ri, gzf[plume_indices],color="red",linestyle="--",label="$\\alpha_\\mathrm{{Ri}}$")
plt.plot(alpha_shape, gzf[plume_indices],color="orange",label="$\\alpha_{{shape}}$")
plt.plot(alpha_prod+alpha_Ri+alpha_shape, gzf[plume_indices], color="purple", linestyle="-.",
        label="$\\sum \\alpha_\\chi$")
#plt.plot(alpha_prod[-1]+alpha_Ri+alpha_shape, gzfp[plume_indices], color="purple", linestyle=":",
        #alpha=0.5)
plt.ylabel("$z/r_0$")
plt.scatter(alpha, zplot, color="purple")
if title: plt.title("Fig 10: contribution to entrainment coefficient from TKE production $\\alpha_{{prod}}$, \n \
buoyancy $\\alpha_\\mathrm{{Ri}}$ and departure from self-similarity $\\alpha_{{shape}}$")
plt.legend()
if save: fig12.savefig(save_loc+'fig12.png', dpi=300)

##### ----------------------- #####

fig13, ax13 = plt.subplots(2,1)
ax13[0].plot(r_regrid, xi)
ax13[0].set_xlabel("$r/r_m$")
ax13[0].set_xlim(0,2)
ax13[0].set_ylabel("$\\xi$")
ax13[0].set_ylim(-0.2,0.2)

ax13[1].plot(r_regrid, eta)
ax13[1].set_xlabel("$r/r_m$")
ax13[1].set_ylabel("$\\eta$")
ax13[1].set_xlim(0,2)
ax13[1].set_ylim(0,1/6)
if title: fig13.suptitle("Fig 8 (b), (c): dependence of $\\xi, \\eta$ on $r/r_m$")
if save: fig13.savefig(save_loc+'fig13.png', dpi=300)

##### ----------------------- #####

fig14, ax14 = plt.subplots()

# 8 lines below add colour gradient to the line to indicate r/r_m value
points = np.array([xi,eta]).T.reshape(-1,1,2)
segments = np.concatenate([points[:-1], points[1:]],axis=1)
norm = plt.Normalize(r_regrid.min(), r_regrid.max())
lc = LineCollection(segments, cmap='jet', norm=norm)
lc.set_array(r_regrid)
lc.set_linewidth(2)
line = ax14.add_collection(lc)
fig14.colorbar(line, ax=ax14, label="$r/r_m$")

#add Lumley triangle
plt.plot([0, 1/3], [0,1/3], color='k')
plt.plot([0, -1/6], [0,1/6], color='k')
xis = np.linspace(-1/6,1/3,1000)
plt.plot(xis, np.sqrt(1/27+2*xis**3), color='k')

plt.xlabel("$\\xi$")
plt.ylabel("$\\eta$")
plt.xlim(-1/6, 1/3)
plt.ylim(0, 1/3)
if title: plt.title("Fig 8 (a): invariants of the anisotropy tensor\n in $\\xi-\\eta$ space with Lumley triangle")
plt.tight_layout()
if save: fig14.savefig(save_loc+'fig14.png', dpi=300)

##### ----------------------- #####
fig15 = plt.figure()
if title: plt.title("Fig 12 (b): radial profiles of radial mixing lengths")
plt.plot(r_regrid, l_m, color='b', label="$l_m$")
plt.plot(r_regrid, l_mb, linestyle='--', color='r', label="$l_{mb}$")
plt.ylim(0,0.3)
plt.xlim(0,1.5)
plt.ylabel("$l_m, l_{mb}$ normalised")
plt.xlabel("$r/r_m$")
plt.legend()
#plt.grid(color='grey', alpha=0.5)
if save: fig15.savefig(save_loc+'fig15.png', dpi=300)

if show: plt.show()
