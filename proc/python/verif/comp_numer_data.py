import h5py
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from os.path import isfile, join
from matplotlib import pyplot as plt
from functions import get_metadata, get_grid, read_params, get_az_data
from itertools import groupby
from scipy import integrate, optimize, interpolate
from datetime import datetime

##### USER-DEFINED VARIABLES #####
params_file = "./params.dat"

z_upper = 95
z_lower = 20

eps = 0.02

save_data = False
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
    NSAMP = len(th1_xz)
    times = np.array([float(t)*md['SAVE_MOVIE_DT'] for t in time_keys])
    f.close()

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

print(z_factor)

Q_factor = 1 / (np.power(F0, 3/4) *
        np.power(N, -5/4))

F_factor = 1 / F0

B_factor = 1 / (np.power(F0, 3/4) * np.power(N, -1/4))

M_factor = 1 / (F0 * np.power(N, -1))

##### ---------------------- #####

fig0, axs = plt.subplots(1,3,figsize=(10, 6))
axs[0].plot(Q*Q_factor, gzf*z_factor, label="Thresholded", color='b', linestyle='--')
axs[1].plot(M*M_factor, gzf*z_factor, label="Thresholded", color='b', linestyle='--')
axs[2].plot(F*F_factor, gzf*z_factor, label="Thresholded", color='b', linestyle='--')

axs[0].plot(Q_full*Q_factor, gzf*z_factor, label="Full", color='b', linestyle=':')
axs[1].plot(M_full*M_factor, gzf*z_factor, label="Full", color='b', linestyle=':')
axs[2].plot(F_full*F_factor, gzf*z_factor, label="Full", color='b', linestyle=':')

axs[0].axvline(Qf(zstar)*Q_factor, linestyle='--', color='grey')
axs[1].axvline(Mf(zstar)*M_factor, linestyle='--', color='grey')
axs[2].axvline(Ff(zstar)*F_factor, linestyle='--', color='grey')

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

if res_t.success:
    plot_points_t = np.linspace(zstar*z_factor, res_t.t_events[0][0], 1000)
    print("(THRESH the) Max penetration height: ",round(res_t.t_events[0][0]/z_factor,5))
elif res_t.y[1][-1] < 1e-9:
    print("(THRESH the) Max penetration height: ",round(res_t.t[-1]/z_factor,5))
    plot_points_t = np.linspace(zstar*z_factor, res_t.t[-1], 1000)
else:
    print("THRESH Inconclusive max penetration height.")
    print("THRESH Momentum flux at end of integration: ", np.sqrt(res_t.y[1][-1]/M_factor))
    print("THRESH Attained at z = ",res_t.t[-1])
    plot_points_t = np.linspace(zstar*z_factor, res_t.t[-1], 1000)

if res_f.success:
    print(res_f.t_events[0])
    print("(the) Max penetration height: ",round(res_f.t_events[0][0]/z_factor,5))
    zmax_theory = res_f.t_events[0][0]/z_factor
    plot_points_f = np.linspace(zstar*z_factor, res_f.t_events[0][0], 1000)
elif res_f.y[1][-1] < 1e-9:
    print("(the) Max penetration height: ",round(res_f.t[-1]/z_factor,5))
    plot_points_f = np.linspace(zstar*z_factor, res_f.t[-1], 1000)
    zmax_theory = res_f.t[-1]/z_factor
else:
    print("Inconclusive max penetration height.")
    print("Momentum flux at end of integration: ", np.sqrt(res_f.y[1][-1]/M_factor))
    print("Attained at z = ",res_f.t[-1])
    zmax_theory = res_f.t[-1]/z_factor
    plot_points_f = np.linspace(zstar*z_factor, res_f.t[-1], 1000)

print("(the) Neutral buoyancy height: ",round(res_f.t_events[1][0]/z_factor,5))
zn_theory = res_f.t_events[1][0]/z_factor
print("(THRESH the) Neutral buoyancy height: ",round(res_t.t_events[1][0]/z_factor,5))

res_t = solve_ivp(rhs_t, (z_factor*zstar, z_factor*0.3),
        [Q_factor*Qf(zstar), M_factor**2 * Mf(zstar)**2, F_factor*Ff(zstar)],
        method='Radau', dense_output=True, t_eval=plot_points_t)

res_f = solve_ivp(rhs_f, (z_factor*zstar, z_factor*0.3),
        [Q_factor*Qff(zstar), M_factor**2 * Mff(zstar)**2, F_factor*Fff(zstar)],
        events=[max_penetration_height,neutral_buoyancy_height],
        method='Radau', dense_output=True, t_eval=plot_points_f)


zn = res_f.t_events[1][0]/z_factor
print("(THRESH the) Mn = ",Mf(zn))
print("(THRESH the) Qn = ",Qf(zn))
print("(FULL the) Mn = ",Mff(zn))
print("(FULL the) Qn = ",Qff(zn))
zn_index = get_index(zn, gzf)

axs[1].plot(np.sqrt(res_Qt.y[0]), res_Qt.t, color='m', linestyle='--')
axs[2].plot(res_Qt.y[1], res_Qt.t, color='m', label="Numerical, Q exp (non-dim, thresholded)",
        linestyle='--')

axs[1].plot(np.sqrt(res_Qf.y[0]), res_Qf.t, color='m', linestyle=':')
axs[2].plot(res_Qf.y[1], res_Qf.t, color='m', label="Numerical, Q exp (non-dim, full)", linestyle=':')

axs[0].plot(res_Mt.y[0], res_Mt.t, color='r', linestyle='--')
axs[2].plot(res_Mt.y[1], res_Mt.t, color='r', label="Numerical, M exp (non-dim, thresholded)",
        linestyle='--')

axs[0].plot(res_Mf.y[0], res_Mf.t, color='r', linestyle=':')
axs[2].plot(res_Mf.y[1], res_Mf.t, color='r', label="Numerical, M exp (non-dim, full)", linestyle=':')

axs[0].plot(res_Ft.y[0], res_Ft.t, color='g', linestyle='--')
axs[1].plot(np.sqrt(res_Ft.y[1]), res_Ft.t, color='g', linestyle='--',
        label="Numerical, F exp (non-dim, thresholded)")

axs[0].plot(res_Ff.y[0], res_Ff.t, color='g', linestyle=':')
axs[1].plot(np.sqrt(res_Ff.y[1]), res_Ff.t, color='g', linestyle=':',
        label="Numerical, F exp (non-dim, full)")

##### ---------------------- #####

axs[0].legend()
axs[1].legend()
axs[2].legend()

axs[0].set_ylim(z_factor*zstar/1.5, z_factor*0.3)
axs[1].set_ylim(z_factor*zstar/1.5, z_factor*0.3)
axs[2].set_ylim(z_factor*zstar/1.5, z_factor*0.3)

axs[0].set_xlim(-1, 6)
axs[1].set_xlim(-1, 10)
axs[2].set_xlim(-5, 3)

plt.show()
