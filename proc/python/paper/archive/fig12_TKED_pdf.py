import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.animation as animation
from datetime import datetime
from functions import get_metadata, read_params, get_grid, g2gf_1d, get_plotindex, get_index, compute_F0
from scipy import ndimage

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"
out_file = "end.h5"

##### ---------------------- #####

def get_index(z, griddata):
    return int(np.argmin(np.abs(griddata - z)))

##### ---------------------- #####

# Get dir locations from param file
base_dir, run_dir, save_dir, version = read_params(params_file)

# Get simulation metadata
md = get_metadata(run_dir, version)

gxf, gyf, gzf, dzf = get_grid(join(save_dir,'grid.h5'), md)
gx, gy, gz, dz = get_grid(join(save_dir,'grid.h5'), md, fractional_grid=False)

plot_min = 0.95*md['H']
plot_max = 0.3#md['LZ']

idx_minf = get_plotindex(plot_min, gzf)-1
idx_maxf = get_plotindex(plot_max, gzf)

idx_min = idx_minf
idx_max = idx_maxf+1

print(idx_min, idx_max)

#_, dz, _ = np.meshgrid(gxf, dz[idx_min:idx_max], gyf, indexing='ij', sparse=True)
#print(dz.shape)
#volume = (md['LX']/md['Nx'])**2 * dz

print("Complete metadata: ", md)

##### Non-dimensionalisation #####

F0 = compute_F0(save_dir, md, tstart_ind = 6*4, verbose=False, zbot=0.7, ztop=0.95, plot=False)
N = np.sqrt(md['N2'])

T = np.power(N, -1)
L = np.power(F0, 1/4) * np.power(N, -3/4)
B = L * np.power(T, -2)
V = L * L * L

print(F0)
print(md['b0']*md['r0']*md['r0'])

##### ====================== #####

field_min = -8.0
field_max = 2.0

nu_min = -7.0
nu_max = -1.0

pvd_lim = 50

fig, axs = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)

with h5py.File(join(save_dir,out_file), 'r') as f:
    print("Keys: %s" % f['Timestep'].keys())

    pvd = np.array(f['Timestep']['PVD'][pvd_lim:-pvd_lim, idx_min:idx_max, pvd_lim:-pvd_lim])
    pvd = np.where(pvd == -1e9, np.nan, pvd)
    print("Loaded PVD field")

    phi = np.array(f['Timestep']['TH2'][pvd_lim:-pvd_lim, idx_min:idx_max, pvd_lim:-pvd_lim])

    print("Loading field {0}".format("tked"))
    field = np.array(f['Timestep']["tked"][pvd_lim:-pvd_lim, idx_min:idx_max, pvd_lim:-pvd_lim])
    field -= np.log10(np.power(L, 2) * np.power(T, -3))

    if "tked" == "TH1":
        field = np.gradient(field, gzf[idx_min:idx_max], axis = 1)

    # restrict to plume
    field = np.where(np.isnan(pvd), np.nan, field)

    print(np.nanmin(field), np.nanmax(field))
    nu_t = np.array(f['Timestep']['NU_T'][pvd_lim:-pvd_lim, idx_min:idx_max, pvd_lim:-pvd_lim])
    nu_t /= np.power(L, 2) * np.power(T, -1)
    nu_t = np.where(np.isnan(pvd), np.nan, nu_t) # restrict to plume

    md['nu'] /= np.power(L, 2) * np.power(T, -1)

    f.close()

    field -= np.log10(md['nu'] + nu_t) # remove nu_eff factor

    nu_t_zero = np.where(nu_t == 0, nu_t, np.nan)
    nu_t_nonzero = np.where(nu_t != 0, nu_t, np.nan)

    field_lower_peak = field + np.log10(md['nu'] + nu_t_zero)
    field_upper_peak = field + np.log10(md['nu'] + nu_t_nonzero)

    nu_eff = np.log10(md['nu'] + nu_t)
    h, bins = np.histogram(nu_eff.flatten(), bins=256, range = (nu_min, nu_max), density=True)
    axs[1].semilogx(np.power(10,0.5*(bins[1:]+bins[:-1])), h, color='b', label=r"$\nu + \nu_T$")
    axs[1].axvline(md['nu'], color='r', label=r"Prescribed $\nu$")
    axs[1].legend()

    field += np.log10(md['nu'] + nu_t)

    pvd_thresh = 7e-3

    h, bins = np.histogram(field.flatten(), bins=256, range = (field_min, field_max))
    bins_plot = 0.5*(bins[1:] + bins[:-1])

    integral = np.sum(h * (bins[1:] - bins[:-1]))
    axs[0].semilogx(np.power(10, bins_plot), h/integral, color='k', linestyle='--')

    print(integral)

    for j in range(2):
        if j == 0:
            field = field_lower_peak
            cols = {
                    'mixed': 'lightcoral',
                    'plume': 'lightblue',
                    'mixing': 'lightgreen'
                    }
            labels = {
                    'mixed': r'S, $\nu_{\mathrm{SGS}} = 0$',
                    'mixing': r'P, $\nu_{\mathrm{SGS}} = 0$',
                    'plume': r'U, $\nu_{\mathrm{SGS}} = 0$'
                    }
        else:
            field = field_upper_peak
            cols = {
                    'mixed': 'r',
                    'plume': 'b',
                    'mixing': 'g'
                    }
            labels = {
                    'mixed': r'S, $\nu_{\mathrm{SGS}} > 0$',
                    'mixing': r'P, $\nu_{\mathrm{SGS}} > 0$',
                    'plume': r'U, $\nu_{\mathrm{SGS}} > 0$'
                    }

        # restrict to mixing region
        mixing_field = np.where(np.logical_and(pvd < pvd_thresh, pvd > 0), field, np.nan)
        mixed_field = np.where(pvd >= pvd_thresh, field, np.nan)
        plume_field = np.where(pvd <= 0, field, np.nan)
        print(field.shape)

        mixing_h, bins = np.histogram(mixing_field.flatten(), bins=256,
                range = (field_min, field_max))
        axs[0].semilogx(np.power(10, bins_plot), mixing_h/integral, color=cols['mixing'], label=labels['mixing'])

        mixed_h, bins = np.histogram(mixed_field.flatten(), bins=256,
                range = (field_min, field_max))
        axs[0].semilogx(np.power(10,bins_plot), mixed_h/integral, color=cols['mixed'], label=labels['mixed'])

        plume_h, bins = np.histogram(plume_field.flatten(), bins=256,
                range = (field_min, field_max))
        axs[0].semilogx(np.power(10,bins_plot), plume_h/integral, color=cols['plume'], label=labels['plume'])


        print(bins_plot[np.argmax(plume_h)])
        print(bins_plot[np.argmax(mixed_h)])
        print(bins_plot[np.argmax(mixing_h)])

    axs[0].set_xlim(np.power(10, field_min), np.power(10, field_max))
    axs[0].set_ylim(0, 0.4)
    axs[0].legend()
    axs[0].set_xlabel(r"TKE dissipation rate $\varepsilon$")
    axs[0].set_ylabel("PDF")
    axs[1].set_xlim(np.power(10, nu_min), np.power(10, nu_max))
    axs[1].set_ylim(0, 1)
    axs[1].set_xlabel(r"$\nu_{\mathrm{eff}}$")
    axs[1].set_ylabel("PDF")

    #plt.savefig('/home/cwp29/Documents/papers/draft/figs/tked_pdf.png', dpi=300)
    #plt.savefig('/home/cwp29/Documents/papers/draft/figs/tked_pdf.pdf')
    plt.show()
