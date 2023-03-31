import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
from datetime import datetime
from functions import get_metadata, read_params, get_grid, g2gf_1d, get_plotindex
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

pdf_fields = np.array(
        [["chi", "TH1"], ["chi", "Re_b"], ["chi", "Ri"],
            ["chi", "tked"], ["tked", "TH1"], ["tked", "Ri"]]
        )

pdf_fig = plt.figure(constrained_layout=True, figsize=(12, 8))
ax = pdf_fig.subplot_mosaic("ABC;DEF",
        gridspec_kw={'wspace':0.1, 'hspace':0.1})


field_lims = {
        "chi": [-12, -1],
        "tked": [-12, -1],
        "Re_b": [-3, 15],
        "TH1": [-7, 7],
        "Ri": [-1, 2]
        }

step = 58
idx_lim = 200
pvd_thresh = 5e-3

with h5py.File(join(save_dir,out_file), 'r') as f:
    print("Keys: %s" % f['Timestep'].keys())

    pvd = np.array(f['Timestep']['PVD'][idx_lim:-idx_lim, idx_min:idx_max, idx_lim:-idx_lim])
    pvd = np.where(pvd == -1e9, np.nan, pvd)
    print("Loaded PVD field")

    for fields, key in zip(pdf_fields, ["A", "B", "C", "D", "E", "F"]):
        print("Loading field {0}".format(fields[0]))
        field0 = np.array(f['Timestep'][fields[0]][idx_lim:-idx_lim, idx_min:idx_max, idx_lim:-idx_lim])
        field0 = np.where(np.isnan(pvd), np.nan, field0) # restrict to plume

        field0_mixing = np.where(np.logical_and(pvd < pvd_thresh, pvd > 0), field0, np.nan)
        field0_mixed = np.where(pvd >= pvd_thresh, field0, np.nan)
        field0_plume = np.where(pvd <= 0, field0, np.nan)

        #####

        print("Loading field {0}".format(fields[1]))
        field1 = np.array(f['Timestep'][fields[1]][idx_lim:-idx_lim, idx_min:idx_max, idx_lim:-idx_lim])
        if fields[1] == "TH1":
            field1 = np.gradient(field1, gzf[idx_min:idx_max], axis=1)
        field1 = np.where(np.isnan(pvd), np.nan, field1) # restrict to plume

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

    plt.show()
