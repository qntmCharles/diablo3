import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
from mpl_toolkits import axes_grid1
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from datetime import datetime
from functions import get_metadata, read_params, get_grid, g2gf_1d, compute_F0, get_plotindex, get_index
from os.path import join, isfile
from os import listdir

from scipy.interpolate import griddata, interp1d
from scipy import integrate, ndimage, spatial

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"

save = True
show = not save

dbdphi = 42.2

##### ---------------------- #####

# Get dir locations from param file
base_dir, run_dir, save_dir, version = read_params(params_file)

# Get simulation metadata
md = get_metadata(run_dir, version)
md['kappa'] = md['nu']/0.7

gxf, gyf, gzf, dzf = get_grid(save_dir+"/grid.h5", md)
gx, gy, gz, dz = get_grid(save_dir+"/grid.h5", md, fractional_grid=False)

F0 = compute_F0(save_dir, md, tstart_ind = 6*4, verbose=False, zbot=0.7, ztop=0.95, plot=False)
N = np.sqrt(md['N2'])

T = np.power(N, -1)
L = np.power(F0, 1/4) * np.power(N, -3/4)
B = L * np.power(T, -2)
V = np.power(L, 3)

plot_max = 1.6*md['H']
plot_min = 0.95*md['H']

idx_minf = get_plotindex(plot_min, gzf)-1
idx_maxf = get_plotindex(plot_max, gzf)
idx_min = idx_minf
idx_max = idx_maxf+1

gz -= md['H']
gzf -= md['H']
gz /= L
gzf /= L

with h5py.File(join(save_dir,"mean.h5"), 'r') as f:
    print("Mean keys: %s" % f.keys())
    bbins = np.array(f['PVD_bbins']['0001'])
    phibins = np.array(f['PVD_phibins']['0001'])

    f.close()

bbins /= B

db = bbins[1] - bbins[0]
dphi = phibins[1] - phibins[0]

sx, sy = np.meshgrid(np.append(bbins-db/2, bbins[-1]+db/2),
        np.append(phibins - dphi/2, phibins[-1] + dphi/2))
sxf, syf = np.meshgrid(bbins, phibins)

with h5py.File(join(save_dir, 'movie.h5'), 'r') as f:
    print("Keys: %s" % f.keys())
    time_keys = list(f['th1_xz'])
    W = np.array([np.array(f['td_scatter'][t]) for t in time_keys])
    M = np.array([np.array(f['pvd'][t]) for t in time_keys])
    S = np.array([np.array(f['td_flux'][t]) for t in time_keys])
    F_b = np.array([np.array(f['td_vel_1'][t]) for t in time_keys])
    F_phi = np.array([np.array(f['td_vel_2'][t]) for t in time_keys])

    NSAMP = len(W)

steps = [16, 32, -1]

S = np.where(S == 0, np.NaN, S)
W = np.where(W == 0, np.NaN, W)
W_thresh = np.where(W < 1e-3, np.nan, W)

div_F = np.gradient(F_b, bbins, axis=2) + np.gradient(F_phi, phibins, axis=1)

M_nonthresh = np.where(np.isnan(W), np.NaN, M)
M = np.where(np.isnan(W_thresh), np.NaN, M)
F_b = np.where(np.isnan(W_thresh), np.NaN, F_b)
div_F = np.where(np.isnan(W_thresh), np.NaN, div_F)
F_phi = np.where(np.isnan(W_thresh), np.NaN, F_phi)

W /= V
S /= V
M /= V
F_b /= V*B/T
F_phi /= V/T

##### ---------------------- #####

threshs = [0]

cols = plt.cm.rainbow(np.linspace(0, 1, NSAMP-1))
for i,c in zip(range(NSAMP-1), cols):
    M_bins = np.linspace(0, np.nanmax(M[i]), 200)

    dWdt_int = []
    S_int = []
    divF_int = []

    for m in M_bins[1:]:
        dWdt_int.append(np.nansum(np.where(np.logical_and(sxf > dbdphi * syf, np.logical_and(M[i] >= 0, M[i] < m)),
            (W[i+1]-W[i])/md['SAVE_STATS_DT'], 0)))
        S_int.append(np.nansum(np.where(np.logical_and(sxf > dbdphi*syf, np.logical_and(M[i] >= 0, M[i] < m)),
            S[i], 0)))
        divF_int.append(np.nansum(np.where(np.logical_and(sxf > dbdphi*syf,np.logical_and(M[i]>=0, M[i]<m)), div_F[i], 0)))

    dWdt_int = np.array(dWdt_int)
    S_int = np.array(S_int)
    divF_int = np.array(divF_int)

    threshs.append(M_bins[1:][np.argmin(np.abs(ndimage.uniform_filter1d(divF_int, size=20)))])

threshs = np.array(threshs)
threshs[np.isnan(threshs)] = 0

smooth_threshs = ndimage.uniform_filter1d(threshs, size=10)
print(smooth_threshs)

##### ---------------------- #####

def calculate_partitioned_pdf(field, field_min, field_max, cols, labels, axs):
    h, bins = np.histogram(field.flatten(), bins=256, range = (field_min, field_max))
    bins_plot = 0.5*(bins[1:] + bins[:-1])

    integral = np.sum(h * (bins[1:] - bins[:-1]))
    axs.semilogx(np.power(10, bins_plot), h/integral, color='k', linestyle='--')

    mixing_field = np.where(np.logical_and(pvd < pvd_thresh, pvd > 0), field, np.nan)
    mixed_field = np.where(pvd >= pvd_thresh, field, np.nan)
    plume_field = np.where(pvd <= 0, field, np.nan)

    mixing_h, bins = np.histogram(mixing_field.flatten(), bins=256,
            range = (field_min, field_max))
    axs.semilogx(np.power(10, bins_plot), mixing_h/integral, color=cols['mixing'], label=labels['mixing'])

    mixed_h, bins = np.histogram(mixed_field.flatten(), bins=256,
            range = (field_min, field_max))
    axs.semilogx(np.power(10,bins_plot), mixed_h/integral, color=cols['mixed'], label=labels['mixed'])

    plume_h, bins = np.histogram(plume_field.flatten(), bins=256,
            range = (field_min, field_max))
    axs.semilogx(np.power(10,bins_plot), plume_h/integral, color=cols['plume'], label=labels['plume'])

    axs.set_xlim(np.power(10, field_min), np.power(10, field_max))
    axs.set_ylabel("PDF")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Figure 13: TKED PDFs
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

out_file = 'end.h5'

pvd_lim = 50

with h5py.File(join(save_dir,out_file), 'r') as f:
    print("Keys: %s" % f['Timestep'].keys())

    print(idx_min, idx_max)
    pvd = np.array(f['Timestep']['PVD'][pvd_lim:-pvd_lim, idx_min:idx_max, pvd_lim:-pvd_lim])
    pvd = np.where(pvd == -1e9, np.nan, pvd)
    pvd /= V

    #plt.figure()
    #plt.imshow(pvd[int(len(pvd)/2)])
    #plt.show()
    print("Loaded PVD field")

    phi = np.array(f['Timestep']['TH2'][pvd_lim:-pvd_lim, idx_min:idx_max, pvd_lim:-pvd_lim])

    tked_3d = np.array(f['Timestep']["tked"][pvd_lim:-pvd_lim, idx_min:idx_max, pvd_lim:-pvd_lim])
    tked_3d -= np.log10(np.power(L, 2) * np.power(T, -3))
    tked_3d = np.where(np.isnan(pvd), np.nan, tked_3d)

    nu_t = np.array(f['Timestep']['NU_T'][pvd_lim:-pvd_lim, idx_min:idx_max, pvd_lim:-pvd_lim])
    nu_t /= np.power(L, 2) * np.power(T, -1)
    nu_t = np.where(np.isnan(pvd), np.nan, nu_t) # restrict to plume
    md['nu'] /= np.power(L, 2) * np.power(T, -1)

    chi_3d = np.array(f['Timestep']["chi"][pvd_lim:-pvd_lim, idx_min:idx_max, pvd_lim:-pvd_lim])
    chi_3d -= np.log10(np.power(L, 2) * np.power(T, -3))
    chi_3d = np.where(np.isnan(pvd), np.nan, chi_3d)

    kappa_t = np.array(f['Timestep']['KAPPA_T'][pvd_lim:-pvd_lim, idx_min:idx_max, pvd_lim:-pvd_lim])
    kappa_t = np.where(np.isnan(pvd), np.nan, kappa_t) # restrict to plume
    kappa_t /= np.power(L, 2) * np.power(T, -1)
    md['kappa'] /= np.power(L, 2) * np.power(T, -1)

    Re_b_3d = np.array(f['Timestep']['Re_b'][pvd_lim:-pvd_lim, idx_min:idx_max, pvd_lim:-pvd_lim])
    Re_b_3d = np.where(np.isnan(pvd), np.nan, Re_b_3d) # restrict to plume

    N2_3d = np.array(f['Timestep']['TH1'][pvd_lim:-pvd_lim, idx_min:idx_max, pvd_lim:-pvd_lim])
    N2_3d = np.gradient(N2_3d, gzf[idx_min:idx_max], axis = 1)
    N2_3d = np.where(np.isnan(pvd), np.nan, N2_3d) # restrict to plume
    N2_3d /= B

    f.close()

tked_3d -= np.log10(md['nu'] + nu_t) # remove nu_eff factor
nu_t_zero = np.where(nu_t == 0, nu_t, np.nan)
nu_t_nonzero = np.where(nu_t != 0, nu_t, np.nan)
tked_3d_lower_peak = tked_3d + np.log10(md['nu'] + nu_t_zero)
tked_3d_upper_peak = tked_3d + np.log10(md['nu'] + nu_t_nonzero)
tked_3d += np.log10(md['nu'] + nu_t)

chi_3d -= np.log10(md['kappa'] + kappa_t) # remove kappa_eff factor
kappa_t_zero = np.where(kappa_t == 0, kappa_t, np.nan)
kappa_t_nonzero = np.where(kappa_t != 0, kappa_t, np.nan)
chi_3d_lower_peak = chi_3d + np.log10(md['kappa'] + kappa_t_zero)
chi_3d_upper_peak = chi_3d + np.log10(md['kappa'] + kappa_t_nonzero)
chi_3d += np.log10(md['kappa'] + kappa_t)

pvd_thresh = smooth_threshs[steps[-1]]

lower_cols = {
        'mixed': 'lightcoral',
        'plume': 'lightblue',
        'mixing': 'lightgreen'
        }
lower_labels = {
        'mixed': r'A, $\nu_{\mathrm{SGS}} = 0$',
        'mixing': r'T, $\nu_{\mathrm{SGS}} = 0$',
        'plume': r'U, $\nu_{\mathrm{SGS}} = 0$'
        }
upper_cols = {
        'mixed': 'r',
        'plume': 'b',
        'mixing': 'g'
        }
upper_labels = {
        'mixed': r'A, $\nu_{\mathrm{SGS}} > 0$',
        'mixing': r'T, $\nu_{\mathrm{SGS}} > 0$',
        'plume': r'U, $\nu_{\mathrm{SGS}} > 0$'
        }

tked_min = -8.0
tked_max = 2.0

nu_min = -7.0
nu_max = -1.0

fig13, axs13 = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)

nu_eff = np.log10(md['nu'] + nu_t)
h, bins = np.histogram(nu_eff.flatten(), bins=256, range = (nu_min, nu_max), density=True)
axs13[1].semilogx(np.power(10,0.5*(bins[1:]+bins[:-1])), h, color='b', label=r"$\nu + \nu_T$")
axs13[1].axvline(md['nu'], color='r', label=r"Prescribed $\nu$")
axs13[1].legend()

calculate_partitioned_pdf(tked_3d_lower_peak, tked_min, tked_max, lower_cols, lower_labels, axs13[0])
calculate_partitioned_pdf(tked_3d_upper_peak, tked_min, tked_max, upper_cols, upper_labels, axs13[0])

axs13[0].set_ylim(0, 0.7)
axs13[0].legend()
axs13[0].set_xlabel(r"TKE dissipation rate $\varepsilon$")
axs13[1].set_xlim(np.power(10, nu_min), np.power(10, nu_max))
axs13[1].set_ylim(0, 1)
axs13[1].set_xlabel(r"$\nu + \nu_{\mathrm{SGS}}$")
axs13[1].set_ylabel("PDF")

lower_labels = {
        'mixed': r'A, $\kappa_{\mathrm{SGS}} = 0$',
        'mixing': r'T, $\kappa_{\mathrm{SGS}} = 0$',
        'plume': r'U, $\kappa_{\mathrm{SGS}} = 0$'
        }
upper_labels = {
        'mixed': r'A, $\kappa_{\mathrm{SGS}} > 0$',
        'mixing': r'T, $\kappa_{\mathrm{SGS}} > 0$',
        'plume': r'U, $\kappa_{\mathrm{SGS}} > 0$'
        }

chi_min = -9.0
chi_max = 2.0

kappa_min = -7.0
kappa_max = -1.0

fig14, axs14 = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)

kappa_eff = np.log10(md['kappa'] + kappa_t)
h, bins = np.histogram(kappa_eff.flatten(), bins=256, range = (kappa_min, kappa_max), density=True)
axs14[1].semilogx(np.power(10,0.5*(bins[1:]+bins[:-1])), h, color='b', label=r"$\kappa + \kappa_T$")
axs14[1].axvline(md['kappa'], color='r', alpha=0.5, label=r"Prescribed $\kappa$")
axs14[1].legend()

calculate_partitioned_pdf(chi_3d_lower_peak, chi_min, chi_max, lower_cols, lower_labels, axs14[0])
calculate_partitioned_pdf(chi_3d_upper_peak, chi_min, chi_max, upper_cols, upper_labels, axs14[0])

axs14[0].set_ylim(0, 0.75)
axs14[0].legend()
axs14[0].set_xlabel(r"Buoyancy variance dissipation rate $\chi$")
axs14[1].set_xlim(np.power(10, kappa_min), np.power(10, kappa_max))
axs14[1].set_ylim(0, 1)
axs14[1].set_xlabel(r"$\kappa + \kappa_{\mathrm{SGS}}$")
axs14[1].set_ylabel("PDF")

fields = ["Re_b", "TH1"]
dim_factors = [1, B]
print(B/L)

Re_b_min = -1.0
Re_b_max = 5.0

N2_min = -5.0
N2_max = 10.0

labels = {
        'mixed': r'Class A',
        'mixing': r'Class T',
        'plume': r'Class U'
        }

fig15, axs15 = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)

calculate_partitioned_pdf(Re_b_3d, Re_b_min, Re_b_max, upper_cols, labels, axs15[0])

h, bins = np.histogram(N2_3d.flatten(), bins=256, range = (N2_min, N2_max))
bins_plot = 0.5*(bins[1:] + bins[:-1])
integral = np.sum(h * (bins[1:] - bins[:-1]))

axs15[1].plot(bins_plot, h/integral, color='k', linestyle='--',
    label="Full plume")

mixing_field = np.where(np.logical_and(pvd < pvd_thresh, pvd > 0), N2_3d, np.nan)
mixed_field = np.where(pvd >= pvd_thresh, N2_3d, np.nan)
plume_field = np.where(pvd <= 0, N2_3d, np.nan)

plume_h, bins = np.histogram(plume_field.flatten(), bins=256,
        range = (N2_min, N2_max))
axs15[1].plot(bins_plot, plume_h/integral, color='b', label="U")

mixing_h, bins = np.histogram(mixing_field.flatten(), bins=256,
        range = (N2_min, N2_max))
axs15[1].plot(bins_plot, mixing_h/integral, color='g', label="T")

mixed_h, bins = np.histogram(mixed_field.flatten(), bins=256,
        range = (N2_min, N2_max))
axs15[1].plot(bins_plot, mixed_h/integral, color='r', label="A")


axs15[0].set_ylim(0, 0.7)
axs15[0].legend()
axs15[0].set_xlabel(r"Activity parameter $I$")

axs15[1].set_xlim(N2_min, N2_max)
axs15[1].set_yscale('log')
axs15[1].set_xlabel(r"Vertical buoyancy gradient $\partial_z b$")
axs15[1].set_ylabel("PDF")
axs15[1].legend()

plt.show()
