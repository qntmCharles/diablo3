from scipy.integrate import solve_ivp
from os.path import isfile, join
from matplotlib import pyplot as plt
from functions import get_metadata, get_grid, read_params, get_az_data
from itertools import groupby
from scipy import integrate, optimize, interpolate
import numpy as np

##### USER-DEFINED VARIABLES #####
params_file = "./params.dat"

z_upper = 95
z_lower = 20
#zstar = 0.2

eps = 0.02

##### ---------------------- #####
# Get initial conditions and parameters for equations

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

base_dir, run_dir, save_dir, version = read_params(params_file)
md = get_metadata(run_dir, version)
gxf, gyf, gzf, dz = get_grid(join(run_dir, 'grid.h5'), md)

r_0 = md['r0']
N2 = md['N2']
alpha = md['alpha_e']
zstar = md['H']
dr = md['LX']/md['Nx']
z_upper = md['H']/r_0
z_lower = (md['Lyc']+md['Lyp'])/r_0

nbins = int(md['Nx']/2)
r_bins = np.array([r*dr for r in range(0, nbins+1)])
r_points = np.array([0.5*(r_bins[i]+r_bins[i+1]) for i in range(nbins)])

data = get_az_data(join(save_dir,'az_stats.h5'), md)

wbar = data['w']
bbar = data['b']
pbar = data['p']
wflucbflucbar = data['wflucbfluc']
wfluc2bar = data['wfluc2']

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

wbar_trunc = np.zeros(shape=(wbar.shape[0],wbar.shape[1]+2))
r_integrate = np.zeros(shape=(wbar.shape[0],wbar.shape[1]+2))
for i in cont_valid_indices:
    w_trunc = eps*wbar[i,0]
    r_trunc = max(np.where(wbar[i,:] > wtrunc)[0])

    wbar_trunc[i, :rtrunc+1] = wbar[i, :rtrunc+1]
    wbar_trunc[i, rtrunc+1] = wtrunc

    f = interpolate.interp1d(wbar[i], r_points)
    r_integrate[i, :rtrunc+1] = r_points[:rtrunc+1]
    r_integrate[i, rtrunc+1] = f(wtrunc)
    r_integrate[i, rtrunc+2] = f(wtrunc)

bbar_trunc = truncate(bbar, r_points, wbar, eps*wbar[:,0], cont_valid_indices)
wflucbflucbar_trunc = truncate(wflucbflucbar, r_points, wbar, eps*wbar[:,0], cont_valid_indices)
pbar_trunc = truncate(pbar, r_points, wbar, eps*wbar[:,0], cont_valid_indices)
wfluc2bar_trunc = truncate(wfluc2bar, r_points, wbar, eps*wbar[:,0], cont_valid_indices)

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
print("Beta_g, Theta_m, Theta_g")
print(beta_g_avg, theta_m_avg, theta_g_avg)

analytic_r = lambda z,a,z0: 6/5 * a * (z-z0)
z_comp = np.array([gzf[i] for i in plume_indices])
r_comp = np.array([r_m[i] for i in plume_indices])

popt, _ = optimize.curve_fit(analytic_r, z_comp, r_comp)
alpha_p = popt[0]
z_virt = popt[1]
print("Alpha")
print(alpha_p)

fig0, axs = plt.subplots(1,3,figsize=(10, 6))
axs[0].plot(Q, gzf, label="Thresholded", color='b', linestyle='--')
axs[1].plot(M, gzf, label="Thresholded", color='b', linestyle='--')
axs[2].plot(F, gzf, label="Thresholded", color='b', linestyle='--')

axs[0].axvline(Qf(zstar), linestyle='--', color='grey')
axs[1].axvline(Mf(zstar), linestyle='--', color='grey')
axs[2].axvline(Ff(zstar), linestyle='--', color='grey')

axs[0].set_title("Q, volume flux")
axs[1].set_title("M, momentum flux")
axs[2].set_title("F, buoyancy flux")
axs[0].grid(color='gray',alpha=0.5)
axs[1].grid(color='gray',alpha=0.5)
axs[2].grid(color='gray',alpha=0.5)
plt.tight_layout()

##### ---------------------- #####
# Numerically solve plume equations

def rhs(z, fluxes):
    return [-2*alpha_p*np.power(fluxes[1],1/4),
            2 * fluxes[2] * fluxes[0]/(theta_m_avg*beta_g_avg),
            -N2 * fluxes[0] * theta_m_avg/theta_g_avg]

def max_penetration_height(t, y): return y[1]
def neutral_buoyancy_height(t, y): return y[2]

flux_init = [Qf(zstar), Mf(zstar)**2, Ff(zstar)]
res = solve_ivp(rhs, (zstar, 0.3), flux_init, events=[max_penetration_height,neutral_buoyancy_height])

if res.success:
    print(res.y_events)
    print(res.t_events)
else:
    print("Momentum flux at end of integration: ", res.y[1][-1])
    print("Attained at z = ",res.t[-1])

print("Buoyancy flux reached 0 at z = ",res.t_events[1])
for i in range(3):
    if i != 1:
        axs[i].plot(res.y[i], res.t)
    else:
        axs[i].plot(np.power(res.y[i],1/2), res.t)
#plt.plot(res.t, res.y.T)
#plt.xlabel("z")
#plt.legend(['Q', 'M', 'F'], shadow=True)
plt.show()
