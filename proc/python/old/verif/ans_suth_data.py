import h5py
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from os.path import isfile, join
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from functions import get_metadata, get_grid, read_params, get_az_data
from itertools import groupby
from scipy import integrate, optimize, interpolate
from datetime import datetime

##### USER-DEFINED VARIABLES #####
params_file = "./params.dat"

z_upper = 95
z_lower = 20

eps = 0.02

save_data = True
data_loc = '/home/cwp29/diablo3/strat/high_res/verif/as_data.npy'

##### ---------------------- #####
# Functions

def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))

def compute_pdf(data, ref, bins):
    out_bins = [0 for i in range(len(bins)-1)]

    for i in range(len(bins)-1):
        out_bins[i] = np.sum(np.where(np.logical_and(data >= bins[i],
            data < bins[i+1]), ref, 0))

    out_bins = np.array(out_bins)

    return out_bins

def get_index(z, griddata):
    return int(np.argmin(np.abs(griddata - z)))

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

        if display and np.abs(gzf[i]-0.21) < 0.02:
            fig = plt.figure()
            plt.title("z={0:.5f}".format(gzf[i]))
            plt.plot(r_integrate[i], var_new[i])
            plt.plot(points, var[i])
            plt.axvline(trunc_pt, linestyle='--', color='grey')
            plt.plot(points_interp, var_interp, color='r')
            #plt.axhline(f(trunc_pt), linestyle='--', color='red')
            plt.axhline(0, color='k', alpha=0.5)
            plt.show()

    return var_new


##### ---------------------- #####
# Get initial conditions and parameters for equations

base_dir, run_dir, save_dir, version = read_params(params_file)
md = get_metadata(run_dir, version)
gxf, gyf, gzf, dzf = get_grid(join(run_dir, 'grid.h5'), md)
gx, gy, gz, dz = get_grid(save_dir+"/grid.h5", md, fractional_grid=False)
z_coords, x_coords = np.meshgrid(gzf, gxf, indexing='ij', sparse=True)

print("Complete metadata: ", md)

r_0 = md['r0']
N2 = md['N2']
N = np.sqrt(md['N2'])
alpha = md['alpha_e']
F0 = md['b0'] * r_0**2
print("Theoretical F: ",F0)
zstar = 0.9*md['H']
#zstar = md['H']
dr = md['LX']/md['Nx']
z_upper = md['H']/r_0
z_lower = (md['Lyc']+md['Lyp'])/r_0
TIW = int(md['Nx']/50) # TRUNCATION INTERPOLATION WIDTH

nbins = int(md['Nx']/2)
r_bins = np.array([r*dr for r in range(0, nbins+1)])
r_points = np.array([0.5*(r_bins[i]+r_bins[i+1]) for i in range(nbins)])

data = get_az_data(join(save_dir,'az_stats.h5'), md, 20)

wbar = data['w']
bbar = data['b']
pbar = data['p']
wflucbflucbar = data['wflucbfluc']
wfluc2bar = data['wfluc2']

##### ---------------------- #####

# Get buoyancy data

X, Y = np.meshgrid(gx, gz)
Xf, Yf = np.meshgrid(gxf, gzf)

# Get data
with h5py.File(save_dir+"/movie.h5", 'r') as f:
    #print("Keys: %s" % f.keys())
    time_keys = list(f['th1_xz'])
    #print(time_keys)
    th1_xz = np.array([np.array(f['th1_xz'][t]) for t in time_keys])
    th2_xz = np.array([np.array(f['th2_xz'][t]) for t in time_keys])
    w_xz = np.array([np.array(f['w_xz'][t]) for t in time_keys])
    NSAMP = len(th1_xz)
    times = np.array([float(f['th1_xz'][t].attrs['Time']) for t in time_keys])
    f.close()

print(times)

b_centreline = th1_xz[0, :, int(md['Nx']/2)]
N2data = np.gradient(b_centreline, gzf)
N2f = interpolate.interp1d(gzf, N2data)

for i in range(bbar.shape[1]):
    bbar[:, i] -= b_centreline

##### ---------------------- #####

# find indices where centreline velocity is positive
valid_indices = [j for j in range(wbar.shape[0]) if wbar[j,0] > 0]
cont_valid_indices = list(set(list(sum(max([list(y) for i, y in groupby(zip(valid_indices,
    valid_indices[1:]), key = lambda x: (x[1]-x[0]) == 1)], key=len),()))))
if cont_valid_indices[-1] == md['Nz']-1:
    cont_valid_indices.remove(md['Nz']-1)
if cont_valid_indices[0] == 0:
    cont_valid_indices.remove(0)

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

#print(plume_indices)

##### ---------------------- #####

Q_full = 2*integrate.trapezoid(wbar*r_points, r_points, axis=1)
M_full = 2*integrate.trapezoid(wbar*wbar*r_points, r_points, axis=1)
F_full = 2*integrate.trapezoid(wbar*bbar*r_points, r_points, axis=1)
B_full = 2*integrate.trapezoid(bbar*r_points, r_points, axis=1)

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
    r_integrate[i, :rtrunc+1] = r_points[:rtrunc+1]
    r_integrate[i, rtrunc+1] = f(wtrunc)
    r_integrate[i, rtrunc+2] = f(wtrunc)

"""
    fig = plt.figure()
    plt.title("z={0:.6f}".format(gzf[i]))
    plt.plot(r_points, wbar[i, :])
    plt.plot(r_integrate[i], wbar_trunc[i])
    plt.axvline(f(wtrunc), color='gray', linestyle='--', alpha=0.5)
    plt.axvline(r_points[rtrunc], color='gray', linestyle=':', alpha=0.5)
    plt.axhline(wtrunc, color='red', linestyle='--', alpha=0.5)
    plt.show()
"""

bbar_trunc = truncate(bbar, r_points, wbar, eps*wbar[:,0], cont_valid_indices)
wflucbflucbar_trunc = truncate(wflucbflucbar, r_points, wbar, eps*wbar[:,0], cont_valid_indices)
pbar_trunc = truncate(pbar, r_points, wbar, eps*wbar[:,0], cont_valid_indices)
wfluc2bar_trunc = truncate(wfluc2bar, r_points, wbar, eps*wbar[:,0], cont_valid_indices)

Q = 2*integrate.trapezoid(wbar_trunc*r_integrate, r_integrate, axis=1)
M = 2*integrate.trapezoid(wbar_trunc*wbar_trunc*r_integrate, r_integrate, axis=1)
F = 2*integrate.trapezoid(wbar_trunc*bbar_trunc*r_integrate, r_integrate, axis=1)
B = 2*integrate.trapezoid(bbar_trunc*r_integrate, r_integrate, axis=1)

##### ---------------------- #####

Qf = interpolate.interp1d(gzf, Q)
Mf = interpolate.interp1d(gzf, M)
Ff = interpolate.interp1d(gzf, F)
Bf = interpolate.interp1d(gzf, B)

Qff = interpolate.interp1d(gzf, Q_full)
Mff = interpolate.interp1d(gzf, M_full)
Fff = interpolate.interp1d(gzf, F_full)
Bff = interpolate.interp1d(gzf, B_full)

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

analytic_r = lambda z,a,z0: 6/5 * a * (z-z0)
z_comp = np.array([gzf[i] for i in plume_indices])
r_comp = np.array([r_m[i] for i in plume_indices])

popt, _ = optimize.curve_fit(analytic_r, z_comp, r_comp)
alpha_p = popt[0]
z_virt = popt[1]

print("Initial data at z = ", zstar)
print("Q:", Qf(zstar))
print("M:", Mf(zstar))
print("F:", Ff(zstar))
F0 = Ff(zstar)
print("beta_g = ",beta_g_avg)
print("theta_m = ",theta_m_avg)
print("theta_g = ",theta_g_avg)
print("alpha = ",alpha_p)

z_factor = 1 / (np.power(F0, 1/4) *
        np.power(N, -3/4))

print("Factor to non-dimensionalise length: ",z_factor)

Q_factor = 1 / (np.power(F0, 3/4) *
        np.power(N, -5/4))

F_factor = 1 / F0

B_factor = 1 / (np.power(F0, 3/4) * np.power(N, -1/4))

M_factor = 1 / (F0 * np.power(N, -1))

##### ---------------------- #####

fig0, axs = plt.subplots(1,3,figsize=(13, 5))
"""
axs[0].plot(Q*Q_factor, gzf*z_factor, label="Simulation (thresholded)", color='b', linestyle='--')
axs[1].plot(M*M_factor, gzf*z_factor, label="Thresholded", color='b', linestyle='--')
axs[2].plot(F*F_factor, gzf*z_factor, label="Thresholded", color='b', linestyle='--')

axs[0].plot(Q_full*Q_factor, gzf*z_factor, label="Simulation (full)", color='b', linestyle=':')
axs[1].plot(M_full*M_factor, gzf*z_factor, label="Full", color='b', linestyle=':')
axs[2].plot(F_full*F_factor, gzf*z_factor, label="Full", color='b', linestyle=':')

axs[0].axvline(Qf(zstar)*Q_factor, linestyle='--', color='grey')
axs[1].axvline(Mf(zstar)*M_factor, linestyle='--', color='grey')
axs[2].axvline(Ff(zstar)*F_factor, linestyle='--', color='grey')
"""

axs[0].plot(Q, gzf, label="Simulation (thresholded)", color='b', linestyle='--')
axs[1].plot(M, gzf, label="Thresholded", color='b', linestyle='--')
axs[2].plot(F, gzf, label="Thresholded", color='b', linestyle='--')

axs[0].plot(Q_full, gzf, label="Simulation (full)", color='b', linestyle=':')
axs[1].plot(M_full, gzf, label="Full", color='b', linestyle=':')
axs[2].plot(F_full, gzf, label="Full", color='b', linestyle=':')

axs[0].axvline(Qf(zstar), linestyle='--', color='grey')
axs[1].axvline(Mf(zstar), linestyle='--', color='grey')
axs[2].axvline(Ff(zstar), linestyle='--', color='grey')

axs[0].axvline(0, color='k', alpha=0.5)
axs[1].axvline(0, color='k', alpha=0.5)
axs[2].axvline(0, color='k', alpha=0.5)

axs[0].set_title("Q, volume flux")
axs[1].set_title("M, momentum flux")
axs[2].set_title("F, buoyancy flux")
axs[0].grid(color='gray',alpha=0.5)
axs[1].grid(color='gray',alpha=0.5)
axs[2].grid(color='gray',alpha=0.5)
plt.tight_layout()

axs[0].axvline(0, color='k', alpha=0.5)
axs[1].axvline(0, color='k', alpha=0.5)
axs[2].axvline(0, color='k', alpha=0.5)

axs[0].set_title("Q, volume flux")
axs[1].set_title("M, momentum flux")
axs[2].set_title("F, buoyancy flux")
axs[0].grid(color='gray',alpha=0.5)
axs[1].grid(color='gray',alpha=0.5)
axs[2].grid(color='gray',alpha=0.5)
plt.tight_layout()

##### ---------------------- #####
# Numerically solve plume equations

plot_points = np.linspace(z_factor*zstar, z_factor*0.3, 1000)

factor = 1
def rhs_t(z, fluxes):
    return [2*factor*alpha_p*np.power(fluxes[1],1/4),
            2*fluxes[0]*fluxes[2]/(theta_m_avg*beta_g_avg),
            -(N2f(z/z_factor)/N2)*fluxes[0]*(theta_m_avg/theta_g_avg)]

def rhs_f(z, fluxes):
    return [2*alpha_p*np.power(fluxes[1],1/4),
            2*fluxes[0]*fluxes[2]/(theta_m_avg*theta_g_avg),
            -(N2f(z/z_factor)/N2)*fluxes[0]*(theta_m_avg/theta_g_avg)]

def rhs_Bt(z, fluxes):
    return [2*alpha_p*np.power(fluxes[1],1/2),
            B_factor*Bf(z/z_factor)/(theta_m_avg*beta_g_avg),
            -(N2f(z/z_factor)/N2)*fluxes[0]*(theta_m_avg/theta_g_avg)]

def rhs_Bf(z, fluxes):
    return [2*alpha_p*np.power(fluxes[1],1/2),
            B_factor*Bff(z/z_factor)/(theta_m_avg*beta_g_avg),
            -(N2f(z/z_factor)/N2)*fluxes[0]*(theta_m_avg/theta_g_avg)]

"""
def rhs_Qt(z, fluxes):
    return [2*Q_factor*Qf(z/z_factor)*fluxes[1]/(theta_m_avg*theta_g_avg),
            -(N2f(z/z_factor)/N2)*Q_factor*Qf(z/z_factor)*(theta_m_avg/theta_g_avg)]

def rhs_Qf(z, fluxes):
    return [2*Q_factor*Qff(z/z_factor)*fluxes[1]/(theta_m_avg*theta_g_avg),
            -(N2f(z/z_factor)/N2)*Q_factor*Qff(z/z_factor)*(theta_m_avg/theta_g_avg)]

def rhs_Mt(z, fluxes):
    return [2*alpha_p*np.power(M_factor*Mf(z/z_factor), 1/2)
            -(N2f(z/z_factor)/N2)*fluxes[0]*(theta_m_avg/theta_g_avg)]

def rhs_Mf(z, fluxes):
    return [2*alpha_p*np.power(M_factor*Mff(z/z_factor), 1/2)
            -(N2f(z/z_factor)/N2)*fluxes[0]*(theta_m_avg/theta_g_avg)]

def rhs_Ft(z, fluxes):
    return [2*alpha_p*np.power(fluxes[1], 1/4),
            2*fluxes[0]*F_factor*Ff(z/z_factor)/(theta_m_avg*theta_g_avg)]

def rhs_Ff(z, fluxes):
    return [2*alpha_p*np.power(fluxes[1], 1/4),
            2*fluxes[0]*F_factor*Fff(z/z_factor)/(theta_m_avg*theta_g_avg)]
"""

# Events for numerical solution
def max_penetration_height(t, y): return y[1]
max_penetration_height.terminal = True

def neutral_buoyancy_height(t, y): return y[2]

res_t = solve_ivp(rhs_t, (z_factor*zstar, z_factor*0.3),
        [Q_factor*Qf(zstar), M_factor**2 * Mf(zstar)**2, F_factor*Ff(zstar)],
        events=[max_penetration_height,neutral_buoyancy_height],
        method='Radau')

res_f = solve_ivp(rhs_f, (z_factor*zstar, z_factor*0.3),
        [Q_factor*Qff(zstar), M_factor**2 * Mff(zstar)**2, F_factor*Fff(zstar)],
        events=[max_penetration_height,neutral_buoyancy_height],
        method='Radau')

res_Bf = solve_ivp(rhs_Bf, (z_factor*zstar, z_factor*0.3),
        [Q_factor*Qff(zstar), M_factor * Mff(zstar), F_factor*Fff(zstar)],
        events=[max_penetration_height,neutral_buoyancy_height],
        method='Radau', t_eval=plot_points)

res_Bt = solve_ivp(rhs_Bt, (z_factor*zstar, z_factor*0.3),
        [Q_factor*Qf(zstar), M_factor * Mf(zstar), F_factor*Ff(zstar)],
        events=[max_penetration_height,neutral_buoyancy_height],
        method='Radau', t_eval=plot_points)

"""
res_Qt = solve_ivp(rhs_Qt, (z_factor*zstar, z_factor*0.3),
        [M_factor**2 * Mf(zstar)**2, F_factor*Ff(zstar)],
        method='Radau')

res_Qf = solve_ivp(rhs_Qf, (z_factor*zstar, z_factor*0.3),
        [M_factor**2 * Mff(zstar)**2, F_factor*Fff(zstar)],
        method='Radau')

res_Mt = solve_ivp(rhs_Mt, (z_factor*zstar, z_factor*0.3),
        [Q_factor*Qf(zstar), F_factor*Ff(zstar)],
        method='Radau')

res_Mf = solve_ivp(rhs_Mf, (z_factor*zstar, z_factor*0.3),
        [Q_factor*Qff(zstar), F_factor*Fff(zstar)],
        method='Radau')

res_Ft = solve_ivp(rhs_Ft, (z_factor*zstar, z_factor*0.3),
        [Q_factor*Qf(zstar), M_factor**2 * Mf(zstar)**2],
        method='Radau')

res_Ff = solve_ivp(rhs_Ff, (z_factor*zstar, z_factor*0.3),
        [Q_factor*Qff(zstar), M_factor**2 * Mff(zstar)**2],
        method='Radau')
"""

if res_t.success:
    plot_points_t = np.linspace(zstar*z_factor, res_t.t_events[0][0], 1000)
    #print("(THRESH the) Max penetration height: ",round(res_t.t_events[0][0]/z_factor,5))
elif res_t.y[1][-1] < 1e-9:
    #print("(THRESH the) Max penetration height: ",round(res_t.t[-1]/z_factor,5))
    plot_points_t = np.linspace(zstar*z_factor, res_t.t[-1], 1000)
else:
    #print("THRESH Inconclusive max penetration height.")
    #print("THRESH Momentum flux at end of integration: ", np.sqrt(res_t.y[1][-1]/M_factor))
    #print("THRESH Attained at z = ",res_t.t[-1])
    plot_points_t = np.linspace(zstar*z_factor, res_t.t[-1], 1000)

if res_f.success:
    #print(res_f.t_events[0])
    #print("(the) Max penetration height: ",round(res_f.t_events[0][0]/z_factor,5))
    zmax_theory = res_f.t_events[0][0]/z_factor
    plot_points_f = np.linspace(zstar*z_factor, res_f.t_events[0][0], 1000)
elif res_f.y[1][-1] < 1e-9:
    #print("(the) Max penetration height: ",round(res_f.t[-1]/z_factor,5))
    plot_points_f = np.linspace(zstar*z_factor, res_f.t[-1], 1000)
    zmax_theory = res_f.t[-1]/z_factor
else:
    #print("Inconclusive max penetration height.")
    #print("Momentum flux at end of integration: ", np.sqrt(res_f.y[1][-1]/M_factor))
    #print("Attained at z = ",res_f.t[-1])
    zmax_theory = res_f.t[-1]/z_factor
    plot_points_f = np.linspace(zstar*z_factor, res_f.t[-1], 1000)

print("Non-dimensional max height (theory): ",zmax_theory)
print("Dimensional: ",zmax_theory*z_factor)

#print("(the) Neutral buoyancy height: ",round(res_f.t_events[1][0]/z_factor,5))
zn_theory = res_f.t_events[1][0]/z_factor
print("Non-dimensional neutral buoyancy height (theory): ",zn_theory)
print("Dimensional: ",zn_theory*z_factor)
#print("(THRESH the) Neutral buoyancy height: ",round(res_t.t_events[1][0]/z_factor,5))

res_t = solve_ivp(rhs_t, (z_factor*zstar, z_factor*0.3),
        [Q_factor*Qf(zstar), M_factor**2 * Mf(zstar)**2, F_factor*Ff(zstar)],
        method='Radau', dense_output=True, t_eval=plot_points_t)

res_f = solve_ivp(rhs_f, (z_factor*zstar, z_factor*0.3),
        [Q_factor*Qff(zstar), M_factor**2 * Mff(zstar)**2, F_factor*Fff(zstar)],
        events=[max_penetration_height,neutral_buoyancy_height],
        method='Radau', dense_output=True, t_eval=plot_points_f)


#print("(THRESH the) Mn = ",Mf(zn))
#print("(THRESH the) Qn = ",Qf(zn))
#print("(FULL the) Mn = ",Mff(zn))
#print("(FULL the) Qn = ",Qff(zn))

axs[0].plot(res_t.y[0]/Q_factor, res_t.t/z_factor, label="MTT numerical soln (thresholded)", color='r', linestyle='--')
axs[1].plot(np.sqrt(res_t.y[1])/M_factor, res_t.t/z_factor, color='r', linestyle='--')
axs[2].plot(res_t.y[2]/F_factor, res_t.t/z_factor, color='r', linestyle='--')

axs[0].plot(res_f.y[0]/Q_factor, res_f.t/z_factor, label="MTT numerical soln (full)", color='r', linestyle=':')
axs[1].plot(np.sqrt(res_f.y[1])/M_factor, res_f.t/z_factor, color='r', linestyle=':')
axs[2].plot(res_f.y[2]/F_factor, res_f.t/z_factor, color='r', linestyle=':')

#axs[0].plot(res_Bt.y[0], res_Bt.t, label="Numerical, exp B (non-dim, thresholded)", color='m', linestyle='--')
#axs[1].plot(res_Bt.y[1], res_Bt.t, color='m', linestyle='--')
#axs[2].plot(res_Bt.y[2], res_Bt.t, color='m', linestyle='--')

#axs[0].plot(res_Bf.y[0], res_Bf.t, label="Numerical, exp B (non-dim, full)", color='m', linestyle=':')
#axs[1].plot(res_Bf.y[1], res_Bf.t, color='m', linestyle=':')
#axs[2].plot(res_Bf.y[2], res_Bf.t, color='m', linestyle=':')

"""
axs[1].plot(np.sqrt(res_Qt.y[0]), res_Qt.t, color='m', linestyle='--')
axs[2].plot(res_Qt.y[1], res_Qt.t, color='m', label="Numerical, Q exp (non-dim, thresholded)",
        linestyle='--')

axs[1].plot(np.sqrt(res_Qf.y[0]), res_Qf.t, color='m', linestyle=':')
axs[2].plot(res_Qf.y[1], res_Qf.t, color='m', label="Numerical, Q exp (non-dim, full)", linestyle=':')

axs[0].plot(res_Mt.y[0], res_Mt.t, color='c', linestyle='--')
axs[2].plot(res_Mt.y[1], res_Mt.t, color='c', label="Numerical, M exp (non-dim, thresholded)",
        linestyle='--')

axs[0].plot(res_Mf.y[0], res_Mf.t, color='c', linestyle=':')
axs[2].plot(res_Mf.y[1], res_Mf.t, color='c', label="Numerical, M exp (non-dim, full)", linestyle=':')

axs[0].plot(res_Ft.y[0], res_Ft.t, color='y', linestyle='--')
axs[1].plot(np.sqrt(res_Ft.y[1]), res_Ft.t, color='y', linestyle='--',
        label="Numerical, F exp (non-dim, thresholded)")

axs[0].plot(res_Ff.y[0], res_Ff.t, color='y', linestyle=':')
axs[1].plot(np.sqrt(res_Ff.y[1]), res_Ff.t, color='y', linestyle=':',
        label="Numerical, F exp (non-dim, full)")
"""

##### ---------------------- #####

# Analytic expressions

zstar_index = get_index(zstar, gzf)
ztop_index = get_index(0.3, gzf)
gzf_upper = gzf[zstar_index:]
gzf_lower = gzf[:zstar_index]

factor1 = 1.2 * alpha
factor2 = 0.9 * alpha * F0 / (theta_m_avg*beta_g_avg)

Q_theory = factor1 * np.power(factor2, 1/3) * np.power(gzf, 5/3)
Q_theory_plot = factor1 * np.power(factor2, 1/3) * np.power(gzf[:ztop_index], 5/3)
Qft = interpolate.interp1d(gzf, Q_theory)

F_theory_upper = F0 - 0.25 * N2 * np.power(0.9*alpha, 4/3) * \
        np.power(theta_m_avg**2 * F0 / beta_g_avg, 1/3) * \
        np.power(theta_g_avg, -1) * (np.power(gzf_upper, 8/3) - np.power(zstar, 8/3))

F_theory_lower = F0*np.ones_like(gzf_lower)

#axs[1].plot(M_theory, gzf, label="Full", color='r', linestyle=':')
#axs[2].plot(F_theory_lower, gzf_lower, label="NEW", color='r', linestyle=':')
#axs[2].plot(F_theory_upper, gzf_upper, label="NEW", color='r', linestyle=':')

def rhs_theory(z, fluxes):
    return [2*Q_factor*Qft(z/z_factor)*fluxes[1],
            -(N2f(z/z_factor)/N2)*Q_factor*Qft(z/z_factor)]

res_Qtheory = solve_ivp(rhs_theory, (z_factor*zstar, z_factor*0.3),
        [M_factor**2 * Mf(zstar)**2, F_factor*Ff(zstar)],
        method='Radau', t_eval = plot_points)

#axs[0].plot(Q_theory_plot*Q_factor, gzf[:ztop_index]*z_factor, label="NEW", color='r', linestyle='--')
#axs[1].plot(np.sqrt(res_Qtheory.y[0]), res_Qtheory.t, color='r', linestyle='--')
#axs[2].plot(res_Qtheory.y[1], res_Qtheory.t, color='r', linestyle='--')


axs[0].legend()
#axs[1].legend()
#axs[2].legend()

#axs[0].set_ylim(z_factor*zstar/1.5, z_factor*0.3)
#axs[1].set_ylim(z_factor*zstar/1.5, z_factor*0.3)
#axs[2].set_ylim(z_factor*zstar/1.5, z_factor*0.3)
#axs[0].set_ylim(10, 18)
#axs[1].set_ylim(10, 18)
#axs[2].set_ylim(10, 18)
axs[0].set_ylim(0.8*md['H'], 1.4*md['H'])
axs[1].set_ylim(0.8*md['H'], 1.4*md['H'])
axs[2].set_ylim(0.8*md['H'], 1.4*md['H'])

axs[0].set_ylabel("z (m)")

#axs[0].set_xlim(-1, 6)
#axs[1].set_xlim(-1, 10)
#axs[2].set_xlim(-5, 3)

plt.tight_layout()

##### ---------------------- #####

b_min = 0
b_max = np.max(th1_xz[0,:,int(md['Nx']/2)])/4
nbins = 129

check_data = th1_xz[0,:,int(md['Nx']/2)]
plot_min = -1
plot_max = -1
for j in range(md['Nz']):
    if gzf[j] < md['H'] and gzf[j+1] > md['H']:
        plot_min = j
    if check_data[j-1] <= b_max and check_data[j] >= b_max:
        plot_max = j

if plot_min == -1: print("Warning: plot_min miscalculated")
if plot_max == -1: print("Warning: plot_max miscalculated")

bbins, db = np.linspace(b_min, b_max, nbins, retstep=True)
bins = [0 for i in range(nbins)]

zt_pdf = compute_pdf(z_coords[plot_min:plot_max], th2_xz[-1][plot_min:plot_max], gzf[plot_min:plot_max])

##### ---------------------- #####

avg_tracer = np.mean(th2_xz[:,plot_min:plot_max,int(md['Nx']/2):],axis=0)
avg_tracer_thresh = np.where(avg_tracer > 0.001, 1, 0)
edges = np.argmin(avg_tracer_thresh, axis=1)
peak = np.argwhere(edges == np.max(edges)).flatten()
zn_exp = np.mean(gzf[plot_min:plot_max][peak])
zn_index = get_index(zn_exp, gzf)
print("(exp) Neutral buoyancy height: ", round(zn_exp,5))

##### ---------------------- #####

tracer_thresh = 2e-3
tracer_data_horiz = th2_xz[1:, zn_index, :]
tracer_data_vert = th2_xz[1:, :, int(md['Nx']/2)]

plume_horiz = np.where(tracer_data_horiz > tracer_thresh, 1, 0)
plume_vert = np.where(tracer_data_vert > tracer_thresh, 1, 0)

width_l = []
width_r = []
for i in range(len(plume_horiz)):
    stuff = np.where(plume_horiz[i] == 1)[0]
    if len(stuff) == 0:
        width_r.append(0)
        width_l.append(0)
    else:
        width_l.append(np.max(np.where(plume_horiz[i] == 1)) * md['LX']/md['Nx'] - md['LX']/2)
        width_r.append(md['LX']/2 - np.min(np.where(plume_horiz[i] == 1)) * md['LX']/md['Nx'])

width_l = np.array(width_l)
width_r = np.array(width_r)
mean_data = 0.5*(width_l+width_r)

heights = []
for i in range(len(plume_vert)):
    stuff = np.where(plume_vert[i] == 1)[0]
    if len(stuff) == 0:
        heights.append(0)
    else:
        heights.append(gzf[np.max(stuff)])

##### ---------------------- #####

#mean_data = mean_data[:10]

fig, ax = plt.subplots(1,3)
fig.suptitle("TESTING: horizontal time series at z = {0:.4f}".format(gzf[zn_index]))
X, Y = np.meshgrid(gx, times)
Xf, Yf = np.meshgrid(gxf, times[1:])
ax[0].pcolormesh(X, Y, tracer_data_horiz)
ax[0].contour(Xf, Yf, tracer_data_horiz, levels=[tracer_thresh], colors=['white'])
ax[1].scatter(100*width_l, 0.5*(times[1:]+times[:-1]), color='b', label="left")
ax[1].scatter(100*width_r, 0.5*(times[1:]+times[:-1]), color='r', label="right")
ax[1].scatter(100*mean_data, 0.5*(times[1:]+times[:-1]), color='purple', label="avg")
ax[1].legend()

mean_times = 0.5*(times[1:] + times[:-1])

ax[2].scatter(mean_times[mean_data != 0], 100*mean_data[mean_data != 0])
ax[2].set_yscale('log')
ax[2].set_xscale('log')

R_fit = lambda t, A, k: A*np.power(t,k)

t = mean_times[mean_data != 0][1:]
R = 100*mean_data[mean_data != 0][1:]

early_t = t[:5]
early_R = R[:5]
mid_t = t[15:20]
mid_R = R[15:20]
late_t = t[35:40]
late_R = R[35:40]

popt, _ = curve_fit(R_fit, early_t, early_R)
ax[2].plot(early_t, R_fit(early_t, *popt), color='k', linestyle='--')
ax[2].plot(early_t, early_R)
#print("Early: ",  popt[1])

popt, _ = curve_fit(R_fit, mid_t, mid_R)
ax[2].plot(mid_t, R_fit(mid_t, *popt), color='k', linestyle='--')
ax[2].plot(mid_t, mid_R)
#print("Mid: ",  popt[1])

popt, _ = curve_fit(R_fit, late_t, late_R)
ax[2].plot(late_t, R_fit(late_t, *popt), color='k', linestyle='--')
ax[2].plot(late_t, late_R)
#print("Late: ",  popt[1])

tplot = 0.5*(times[1:]+times[:-1])
v = np.gradient(mean_data, 0.5*(times[1:]+times[:-1]))
t_min = get_index(5.5, tplot)
t_max = get_index(6.5, tplot)

# Use average...
v_intrusion = (mean_data[t_max]-mean_data[t_min])/(tplot[t_max]-tplot[t_min])
ax[1].plot([100*mean_data[t_min], 100*mean_data[t_max]], [tplot[t_min], tplot[t_max]], color='k', linestyle='--')

# If right side is better...
v_intrusion_r = (width_r[t_max]-width_r[t_min])/(tplot[t_max]-tplot[t_min])
ax[1].plot([100*width_r[t_min], 100*width_r[t_max]], [tplot[t_min], tplot[t_max]], color='k', linestyle='--')

# If left side is better...
v_intrusion_l = (width_l[t_max]-width_l[t_min])/(tplot[t_max]-tplot[t_min])
v_intrusion = v_intrusion_l
ax[1].plot([100*width_l[t_min], 100*width_l[t_max]], [tplot[t_min], tplot[t_max]], color='k', linestyle='--')

ax[2].plot(tplot[t_min:t_max], 100*mean_data[t_min:t_max], color='r', linestyle='--')
ax[2].plot(tplot[t_min:t_max], 100*mean_data[t_min] + 100*(tplot[t_min:t_max]-tplot[t_min])*v_intrusion,
        color='r')

ax[1].set_aspect(1)

print("Intrusion speed (mean)", 100*v_intrusion)
print("Intrusion speed (right)", 100*v_intrusion_r)
print("Intrusion speed (left)", 100*v_intrusion_l)

#print("(exp) Initial intrusion speed: ", round(v_intrusion,5))
#print("(THRESH the)", v_intrusion/(Mf(zn)/Qf(zn)))
#print("(FULL the)", v_intrusion/(Mff(zn)/Qff(zn)))

fig1, ax1 = plt.subplots(1,2)
X, Y = np.meshgrid(times[:], gz)
Xf, Yf = np.meshgrid(times[1:], gzf)
fig1.suptitle("TESTING: vertical time series")
ax1[0].pcolormesh(X,Y, np.swapaxes(tracer_data_vert,0,1))
ax1[0].contour(Xf, Yf, np.swapaxes(tracer_data_vert,0,1), levels=[tracer_thresh], colors=['white'])
ax1[1].plot(times[1:], heights)
zmax_exp = np.max(heights[:int(len(times)/4)])
print("(exp) Max penetration height: ", round(zmax_exp,5))

max_loc = np.argwhere(heights == zmax_exp)[-1]
ax1[1].plot(times[max_loc[0]+5:], heights[max_loc[0]+4:])
zss = np.mean(heights[max_loc[0]+4:])
ax1[1].axhline(zss, color='k', linestyle='--')
print("Steady state height", zss)



fig3, ax3 = plt.subplots(1,2)
plt.title("TESTING: neutral buoyancy height")
avg_tracer = np.mean(th2_xz[:,plot_min:plot_max,int(md['Nx']/2):],axis=0)
avg_tracer_thresh = np.where(avg_tracer > 0.001, 1, 0)
ax3[0].imshow(avg_tracer_thresh[::-1])
ax3[1].plot(np.argmin(avg_tracer_thresh, axis=1),gzf[plot_min:plot_max])
edges = np.argmin(avg_tracer_thresh, axis=1)
peak = np.argwhere(edges == np.max(edges)).flatten()
zn = np.mean(gzf[plot_min:plot_max][peak])

ax3[1].axhline(zn, color='k', linestyle='--')

axs[0].axhline(zn_exp*z_factor, color='k', alpha=0.5)
axs[1].axhline(zn_exp*z_factor, color='k', alpha=0.5)
axs[2].axhline(zn_exp*z_factor, color='k', alpha=0.5)

#print("(THRESH exp)", v_intrusion/(Mf(zn_exp)/Qf(zn_exp)))
#print("(FULL exp)", v_intrusion/(Mff(zn_exp)/Qff(zn_exp)))

# Interfacial Froude number

fig4, ax4 = plt.subplots(1, 2)
factor = 0.98
tracer_data_interface = w_xz[max_loc[0]+4:, get_index(factor*md['H'],gzf), :]
thresh = 2e-2

plume_horiz = np.where(tracer_data_interface > thresh, 1, 0)

width_l = []
width_r = []
for i in range(len(plume_horiz)):
    stuff = np.where(plume_horiz[i] == 1)[0]
    if len(stuff) == 0:
        width_r.append(0)
        width_l.append(0)
    else:
        width_l.append(np.max(np.where(plume_horiz[i] == 1)) * md['LX']/md['Nx'] - md['LX']/2)
        width_r.append(md['LX']/2 - np.min(np.where(plume_horiz[i] == 1)) * md['LX']/md['Nx'])

width_l = np.array(width_l)
width_r = np.array(width_r)
mean_data = 0.5*(width_l+width_r)
r_i = np.mean(mean_data)

r_idx = int(r_i * md['Nx']/md['LX'])
w_data = w_xz[max_loc[0]+4:, get_index(factor*md['H'], gzf), int(md['Nx']/2)-r_idx:int(md['Nx']/2)+r_idx]
w_i = np.mean(w_data)
b_data = th1_xz[max_loc[0]+4:, get_index(factor*md['H'], gzf), int(md['Nx']/2)-r_idx:int(md['Nx']/2)+r_idx]
b_i = np.mean(b_data)

ax4[0].imshow(w_data)
ax4[1].imshow(b_data)

Fr_i = w_i/np.sqrt(b_i * r_i)
print("Interfacial Froude number", Fr_i)

# Calculate energetic results for zmax, zn
bm = F0 * np.power(0.9 * alpha * F0, -1/3) * np.power(md['H'], -5/3) / (1.2 * alpha)
zn_en = 2* bm * np.power(N, -2)

wm = np.power(0.9 * alpha * F0, 1/3) * np.power(md['H'], -1/3) / (1.2 * alpha)
zmax_en = wm * 2 * np.power(N, -1)

# Calculate Devenish results for zmax, zn
zn_dev = 1.04 * np.power(alpha, -1/2) * np.power(F0, 1/4) * np.power(N, -3/4)
zmax_dev = (1.36 / 1.04) * zn_dev

if save_data:
    print("Saving data...")
    # Collect together data to save
    data_row = [save_dir, zmax_theory, zmax_exp, zn_theory, zn_exp, zss, Fr_i, v_intrusion, Mf(zn_exp)/Qf(zn_exp),
            Mff(zn_exp)/Qff(zn_exp), zmax_en, zn_en, zmax_dev, zn_dev]
    data_row = np.array(data_row)

    if isfile(data_loc):
        print("Data file found. Saving data to existing file...")
        data_array = np.load(data_loc, allow_pickle=False)
        print(data_array.shape)
        if data_row[0] not in data_array:
            data_array = np.insert(data_array, -1, data_row, axis=0)
            np.save(data_loc, data_array, allow_pickle=False)
        else:
            ow_flag = input("ERROR: Data file contains entry from this directory already. Overwrite? (y/n)")
            if ow_flag == "y":
                print("Overwriting existing data...")
                for i in range(len(data_array)):
                    if data_array[i,0] == save_dir:
                        data_array_new = np.delete(data_array, i, axis=0)
                data_array = np.insert(data_array_new, -1, data_row, axis=0)
                np.save(data_loc, data_array, allow_pickle=False)
    else:
        print("Data file not found. Saving data to new file...")
        data_array = np.array([data_row])
        np.save(data_loc, data_array, allow_pickle=False)

Bfig = plt.figure()
plt.plot(B, gzf, linestyle='--')
plt.plot(B_full, gzf, linestyle=':')

#############################################################################################################

report_fig, axs = plt.subplots(1, 2, figsize=(12, 4))

X_h, Y_h = np.meshgrid(100*(gx[64:-64]-md['LX']/2), times[:])
Xf_h, Yf_h = np.meshgrid(100*(gxf[64:-64]-md['LX']/2), times[1:])
tracer_data_horiz = th2_xz[1:, zn_index, 64:-64]
im0 = axs[0].pcolormesh(X_h, Y_h, tracer_data_horiz, cmap='hot_r')
axs[0].contour(Xf_h, Yf_h, tracer_data_horiz, levels=[tracer_thresh], colors=['blue'], linestyles='--')

im0.set_clim(0, np.max(np.abs(tracer_data_horiz)))

ticks = axs[0].get_xticks()
axs[0].set_xticklabels([round(np.abs(float(i)),0) for i in ticks])
axs[0].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

X_v, Y_v = np.meshgrid(times[:], 100*gz[:get_index(0.4, gz)+1])
Xf_v, Yf_v = np.meshgrid(times[1:], 100*gzf[:get_index(0.4, gzf)+1])
tracer_data_vert = th2_xz[1:, :get_index(0.4, gzf)+1, int(md['Nx']/2)]
im1 = axs[1].pcolormesh(X_v, Y_v, np.swapaxes(tracer_data_vert,0,1), cmap='hot_r')
axs[1].contour(Xf_v, Yf_v, np.swapaxes(tracer_data_vert,0,1), levels=[tracer_thresh], colors=['blue'],
        linestyles='--')

im1.set_clim(0, np.max(np.abs(tracer_data_vert)))

axs[0].set_xlabel("r (cm)")
axs[0].set_ylabel("t (s)")
axs[1].set_xlabel("t (s)")
axs[1].set_ylabel("z (cm)")
plt.tight_layout()
#plt.savefig('/home/cwp29/Documents/4report/figs/timeseries.png', dpi=300)
fig0.savefig('/home/cwp29/Documents/4report/figs/fluxes.png',dpi=300)
plt.show()
