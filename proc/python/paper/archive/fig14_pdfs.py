import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
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

##### Non-dimensionalisation #####

F0 = compute_F0(save_dir, md, tstart_ind = 6*4, verbose=False, zbot=0.7, ztop=0.95, plot=False)
N = np.sqrt(md['N2'])

T = np.power(N, -1)
L = np.power(F0, 1/4) * np.power(N, -3/4)
B = L * np.power(T, -2)
U = L/T
V = L * L * L

gz /= L
gzf /= L

print(F0)
print(md['b0']*md['r0']*md['r0'])

##### ====================== #####

X, Z, Y = np.meshgrid(gx, gz, gy, indexing='ij', sparse=True)
print(Z.shape)

print("Complete metadata: ", md)

fields = ["Re_b", "TH1"]
dim_factors = [1, B]
print(B/L)
field_mins = [-1.0, -5]
field_maxs = [5.0, 10]

pvd_lim = 50

fig, axs = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)

with h5py.File(join(save_dir,out_file), 'r') as f:
    print("Keys: %s" % f['Timestep'].keys())

    pvd = np.array(f['Timestep']['PVD'][pvd_lim:-pvd_lim, idx_min:idx_max, pvd_lim:-pvd_lim]) # restrict to stratified layer
    pvd = np.where(pvd == -1e9, np.nan, pvd)
    print("Loaded PVD field")

    for i in range(len(fields)):
        print("Loading field {0}".format(fields[i]))
        if fields[i] == "Jb":
            Z = Z[:, idx_min:idx_max, :]
            b = np.array(f['Timestep']["TH1"][pvd_lim:-pvd_lim, idx_min:idx_max, pvd_lim:-pvd_lim])
            b /= B
            w = np.array(f['Timestep']["W"][pvd_lim:-pvd_lim, idx_min:idx_max, pvd_lim:-pvd_lim])
            w /= U
            b -= np.where(Z < md['H'], 0, Z - md['H']/L)
            field = b * w
        else:
            field = np.array(f['Timestep'][fields[i]][pvd_lim:-pvd_lim, idx_min:idx_max, pvd_lim:-pvd_lim])
        field /= dim_factors[i]
        if fields[i] == "TH1":
            field = np.gradient(field, gzf[idx_min:idx_max], axis = 1)
        field = np.where(np.isnan(pvd), np.nan, field) # restrict to plume

        # restrict to mixing region
        pvd_thresh = 7e-3

        h, bins = np.histogram(field.flatten(), bins=256, range = (field_mins[i], field_maxs[i]))
        bins_plot = 0.5*(bins[1:] + bins[:-1])
        integral = np.sum(h * (bins[1:] - bins[:-1]))

        if fields[i] == "Re_b":
            axs[i].semilogx(np.power(10, bins_plot), h/integral, color='k', linestyle='--',
                label="Full plume")
        else:
            axs[i].plot(bins_plot, h/integral, color='k', linestyle='--',
                label="Full plume")

        mixing_field = np.where(np.logical_and(pvd < pvd_thresh, pvd > 0), field, np.nan)
        mixed_field = np.where(pvd >= pvd_thresh, field, np.nan)
        plume_field = np.where(pvd <= 0, field, np.nan)
        print(field.shape)

        plume_h, bins = np.histogram(plume_field.flatten(), bins=256,
                range = (field_mins[i], field_maxs[i]))
        if fields[i] == "Re_b":
            axs[i].semilogx(np.power(10, bins_plot), plume_h/integral, color='b', label="U")
        else:
            axs[i].plot(bins_plot, plume_h/integral, color='b', label="U")

        mixing_h, bins = np.histogram(mixing_field.flatten(), bins=256,
                range = (field_mins[i], field_maxs[i]))
        if fields[i] == "Re_b":
            axs[i].semilogx(np.power(10, bins_plot), mixing_h/integral, color='g', label="P")
        else:
            axs[i].plot(bins_plot, mixing_h/integral, color='g', label="P")

        mixed_h, bins = np.histogram(mixed_field.flatten(), bins=256,
                range = (field_mins[i], field_maxs[i]))
        if fields[i] == "Re_b":
            axs[i].semilogx(np.power(10, bins_plot), mixed_h/integral, color='r', label="S")
        else:
            axs[i].plot(bins_plot, mixed_h/integral, color='r', label="S")

    axs[0].set_xlim(np.power(10, field_mins[0]), np.power(10, field_maxs[0]))
    axs[0].set_ylim(0, 0.7)
    axs[0].legend()
    axs[0].set_xlabel(r"Local buoyancy Reynolds number $\mathrm{Re}_b$")
    axs[0].set_ylabel("PDF")

    #axs[1].axvline(md['N2'] / np.power(T, 2))
    axs[1].set_xlim(field_mins[1], field_maxs[1])
    #axs[1].set_ylim(0, 0.6)
    axs[1].set_yscale('log')
    axs[1].set_xlabel(r"Vertical buoyancy gradient $\partial_z b$")
    axs[1].set_ylabel("PDF")
    axs[1].legend()

    #plt.savefig('/home/cwp29/Documents/papers/draft/figs/reb_dbdz_pdfs.png', dpi=300)
    #plt.savefig('/home/cwp29/Documents/papers/draft/figs/reb_dbdz_pdfs.pdf')
    plt.show()

    f.close()
