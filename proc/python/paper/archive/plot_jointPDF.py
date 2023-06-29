import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
from datetime import datetime
from functions import get_metadata, read_params, get_grid, g2gf_1d, get_plotindex, compute_F0
from scipy import ndimage

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"
out_file = "end.h5"

##### ---------------------- #####

def get_index(z, griddata):
    return int(np.argmin(np.abs(griddata - z)))

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

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

print("Complete metadata: ", md)

##### Non-dimensionalisation #####

F0 = compute_F0(save_dir, md, tstart_ind = 6*4, verbose=False, zbot=0.7, ztop=0.95, plot=False)
N = np.sqrt(md['N2'])

T = np.power(N, -1)
L = np.power(F0, 1/4) * np.power(N, -3/4)

##################################

pdf_fields = np.array(
        [["chi", "TH1"], ["chi", "Re_b"], ["chi", "tked"], ["tked", "TH1"], ["Re_b", "Gamma"]]
        )

pdf_fig = plt.figure(constrained_layout=True, figsize=(12, 8))
ax = pdf_fig.subplot_mosaic("ABC;DEF",
        gridspec_kw={'wspace':0.1, 'hspace':0.1})


field_lims = {
        "chi": [-9, 2],
        "tked": [-8, 2],
        "Re_b": [-1, 5],
        "TH1": [-5, 10],
        "Gamma": [-2, 1]
        }

dim_factors = {
        "chi": np.log10(np.power(L, 2) * np.power(T, -3)),
        "tked": np.log10(np.power(L, 2) * np.power(T, -3)),
        "Re_b": 0,
        "TH1": np.log10(np.power(T, -2)),
        "Gamma": 0,
        }

step = 58
idx_lim = 200
pvd_thresh = 7e-3

with h5py.File(join(save_dir,out_file), 'r') as f:
    print("Keys: %s" % f['Timestep'].keys())

    pvd = np.array(f['Timestep']['PVD'][idx_lim:-idx_lim, idx_min:idx_max, idx_lim:-idx_lim])
    pvd = np.where(pvd == -1e9, np.nan, pvd)
    print("Loaded PVD field")

    nu_t = np.array(f['Timestep']['NU_T'][idx_lim:-idx_lim, idx_min:idx_max, idx_lim:-idx_lim])
    kappa_t = np.array(f['Timestep']['KAPPA_T'][idx_lim:-idx_lim, idx_min:idx_max, idx_lim:-idx_lim])
    nu_t = np.where(np.isnan(pvd), np.nan, nu_t)
    nu_t = np.where(nu_t == 0, np.nan, nu_t)
    kappa_t = np.where(np.isnan(pvd), np.nan, kappa_t)
    kappa_t = np.where(kappa_t == 0, np.nan, kappa_t)

    for fields, key in zip(pdf_fields, ["A", "B", "C", "D", "E"]):
        print("Loading field {0}".format(fields[0]))
        field0 = np.array(f['Timestep'][fields[0]][idx_lim:-idx_lim, idx_min:idx_max, idx_lim:-idx_lim])
        field0 = np.where(np.isnan(pvd), np.nan, field0) # restrict to plume

        if fields[0] == "tked":
            field0 = np.where(np.isnan(nu_t), np.nan, field0) # restrict to plume
        if fields[0] == "chi":
            field0 = np.where(np.isnan(kappa_t), np.nan, field0) # restrict to plume

        field0 -= dim_factors[fields[0]]

        field0_mixing = np.where(np.logical_and(pvd < pvd_thresh, pvd > 0), field0, np.nan)
        field0_mixed = np.where(pvd >= pvd_thresh, field0, np.nan)
        field0_plume = np.where(pvd <= 0, field0, np.nan)

        #####

        print("Loading field {0}".format(fields[1]))
        if fields[1] == "Gamma":
            eps = np.array(f['Timestep']["tked"][idx_lim:-idx_lim, idx_min:idx_max, idx_lim:-idx_lim])
            chi = np.array(f['Timestep']["chi"][idx_lim:-idx_lim, idx_min:idx_max, idx_lim:-idx_lim])
            eta = chi / (eps + chi)
            field1 = np.log10(eta/(1-eta))
        else:
            field1 = np.array(f['Timestep'][fields[1]][idx_lim:-idx_lim, idx_min:idx_max, idx_lim:-idx_lim])

        if fields[1] == "TH1":
            field1 = np.gradient(field1, gzf[idx_min:idx_max], axis=1)

        if fields[1] == "tked":
            field1 = np.where(np.isnan(nu_t), np.nan, field1) # restrict to plume
        if fields[1] == "chi":
            field1 = np.where(np.isnan(kappa_t), np.nan, field1) # restrict to plume

        field1 = np.where(np.isnan(pvd), np.nan, field1) # restrict to plume

        field1 -= dim_factors[fields[1]]

        field1_mixing = np.where(np.logical_and(pvd < pvd_thresh, pvd > 0), field1, np.nan)
        field1_mixed = np.where(pvd >= pvd_thresh, field1, np.nan)
        field1_plume = np.where(pvd <= 0, field1, np.nan)

        field_ranges = np.array([field_lims[fields[0]], field_lims[fields[1]]])

        #####

        H, xedges, yedges = np.histogram2d(field0.flatten(), field1.flatten(),
                bins=64, range=field_ranges, density=True)
        H = H.T

        X, Y = np.meshgrid(xedges, yedges)
        Xf, Yf = np.meshgrid(0.5*(xedges[1:]+xedges[:-1]), 0.5*(yedges[1:]+yedges[:-1]))

        #####

        H_mixing, _, _ = np.histogram2d(field0_mixing.flatten(), field1_mixing.flatten(),
                bins=64, range=field_ranges, density=True)
        H_mixing = H_mixing.T
        mixing_levels = np.linspace(np.min(H_mixing), np.max(H_mixing), 5)[1:][::2]
        mixing_colours = plt.cm.Greens(np.linspace(0, 1, len(mixing_levels)+3))[3:]
        ax[key].contour(Xf, Yf, H_mixing, levels = mixing_levels, colors = mixing_colours)

        #####

        H_mixed, _, _ = np.histogram2d(field0_mixed.flatten(), field1_mixed.flatten(),
                bins=64, range=field_ranges, density=True)
        H_mixed = H_mixed.T
        mixed_levels = np.linspace(np.min(H_mixed), np.max(H_mixed), 5)[1:][::2]
        mixed_colours = plt.cm.Reds(np.linspace(0, 1, len(mixed_levels)+3))[3:]
        ax[key].contour(Xf, Yf, H_mixed, levels = mixed_levels, colors = mixed_colours)

        #####

        H_plume, _, _ = np.histogram2d(field0_plume.flatten(), field1_plume.flatten(),
                bins=64, range=field_ranges, density=True)
        H_plume = H_plume.T
        plume_levels = np.linspace(np.min(H_plume), np.max(H_plume), 5)[1:][::2]
        plume_colours = plt.cm.Blues(np.linspace(0, 1, len(plume_levels)+3))[3:]
        ax[key].contour(Xf, Yf, H_plume, levels = plume_levels, colors = plume_colours)

        #####

        ax[key].pcolormesh(X, Y, H, cmap='viridis')#truncate_colormap(plt.cm.Greys, 0.5, 1, 128))

        ax[key].set_xlabel(fields[0])
        ax[key].set_ylabel(fields[1])

    f.close()

    xs = np.linspace(field_lims["chi"][0], field_lims["chi"][1], 10)
    for c in [0.5, 0.1, 0.01]:
        ax["C"].plot(xs, xs + np.log10((1-c)/c), color='white', alpha=0.5)
        loc = np.array((xs[8] - np.log10((1-c)/c), xs[8]))
        angle = ax["C"].transData.transform_angles(np.array((45,)), loc.reshape((1,2)))[0]
        ax["C"].text(loc[0], loc[1]-0.4, r"$\eta = {0:.2f}$".format(c), rotation=angle, color='white')

    for c in [0.9, 0.99]:
        ax["C"].plot(xs, xs + np.log10((1-c)/c), color='white', alpha=0.5)
        loc = np.array((xs[8], xs[8] + np.log10((1-c)/c)))
        angle = ax["C"].transData.transform_angles(np.array((45,)), loc.reshape((1,2)))[0]
        ax["C"].text(loc[0], loc[1]-0.4, r"$\eta = {0:.2f}$".format(c), rotation=angle, color='white')

    ax["C"].set_ylim(field_lims["tked"][0], field_lims["tked"][1])

    #plt.savefig('/home/cwp29/Documents/papers/draft/figs/joint_pdf.png', dpi=300)
    #plt.savefig('/home/cwp29/Documents/papers/draft/figs/joint_pdf.pdf')
    plt.show()
