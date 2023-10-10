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

fields = ["Re_b", "TH1", "tked", "chi"]
dim_factors = [1, B, np.power(L, 2) * np.power(T, -3), np.power(L, 2) * np.power(T, -3)]
print(B/L)
#field_mins = [-1.0, -5, 1--8, 1e-9]
#field_maxs = [5.0, 10, 1e2, 1e2]

avgs = []

pvd_lim = 50
pvd_thresh = 7e-3

fig, axs = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)

with h5py.File(join(save_dir,out_file), 'r') as f:
    print("Keys: %s" % f['Timestep'].keys())

    pvd = np.array(f['Timestep']['PVD'][pvd_lim:-pvd_lim, idx_min:idx_max, pvd_lim:-pvd_lim]) # restrict to stratified layer
    pvd = np.where(pvd == -1e9, np.nan, pvd)
    print("Loaded PVD field")

    for i in range(len(fields)):
        print("Loading field {0}".format(fields[i]))
        field = np.array(f['Timestep'][fields[i]][pvd_lim:-pvd_lim, idx_min:idx_max, pvd_lim:-pvd_lim])

        if fields[i] == "TH1":
            field = np.abs(np.gradient(field, gzf[idx_min:idx_max], axis = 1))

        if fields[i] == "Re_b":
            tked = np.array(f['Timestep']['tked'][pvd_lim:-pvd_lim, idx_min:idx_max, pvd_lim:-pvd_lim])
            nu_eff = md['nu'] + np.array(f['Timestep']['NU_T'][pvd_lim:-pvd_lim, idx_min:idx_max, pvd_lim:-pvd_lim])
            #tked /= dim_factors[2]
            #nu_eff /= np.power(L, 2) * np.power(T, -1)

            tked -= np.log10(nu_eff)
            tked = np.power(10, tked)
            tked /= np.power(T, -2)

            th1 = np.array(f['Timestep']['TH1'][pvd_lim:-pvd_lim, idx_min:idx_max, pvd_lim:-pvd_lim])
            N2 = np.abs(np.gradient(th1, gzf[idx_min:idx_max], axis = 1))
            N2 /= dim_factors[1]

            N2 = np.where(np.isnan(pvd), np.nan, N2) # restrict to plume
            tked = np.where(np.isnan(pvd), np.nan, tked) # restrict to plume

            mixing_tked = np.where(np.logical_and(pvd < pvd_thresh, pvd > 0), tked, np.nan)
            mixed_tked = np.where(pvd >= pvd_thresh, tked, np.nan)
            plume_tked = np.where(pvd <= 0, tked, np.nan)

            mixing_N2 = np.where(np.logical_and(pvd < pvd_thresh, pvd > 0), N2, np.nan)
            mixed_N2 = np.where(pvd >= pvd_thresh, N2, np.nan)
            plume_N2 = np.where(pvd <= 0, N2, np.nan)

            avgs.append([
                np.nanmean(tked) / np.nanmean(N2),
                np.nanmean(plume_tked) / np.nanmean(plume_N2),
                np.nanmean(mixing_tked) / np.nanmean(mixing_N2),
                np.nanmean(mixed_tked) / np.nanmean(mixed_N2)
                ])

            continue

        field = np.where(np.isnan(pvd), np.nan, field) # restrict to plume

        if fields[i] == "tked":
            field = np.power(10, field)

            nu_t = np.array(f['Timestep']['NU_T'][pvd_lim:-pvd_lim, idx_min:idx_max, pvd_lim:-pvd_lim])
            nu_t /= np.power(L, 2) * np.power(T, -1)
            nu_t = np.where(np.isnan(pvd), np.nan, nu_t) # restrict to plume

            field /= dim_factors[i]

            field_DNS = np.where(nu_t == 0, field, np.nan)
            field_LES = np.where(nu_t > 0, field, np.nan)

            for field in [field, field_DNS, field_LES]:
                mixing_field = np.where(np.logical_and(pvd < pvd_thresh, pvd > 0), field, np.nan)
                mixed_field = np.where(pvd >= pvd_thresh, field, np.nan)
                plume_field = np.where(pvd <= 0, field, np.nan)

                avgs.append([np.nanmean(field), np.nanmean(plume_field), np.nanmean(mixing_field),
                    np.nanmean(mixed_field)])

        elif fields[i] == "chi":
            field = np.power(10, field)

            kappa_t = np.array(f['Timestep']['KAPPA_T'][pvd_lim:-pvd_lim, idx_min:idx_max, pvd_lim:-pvd_lim])
            kappa_t /= np.power(L, 2) * np.power(T, -1)
            kappa_t = np.where(np.isnan(pvd), np.nan, kappa_t) # restrict to plume

            field /= dim_factors[i]

            field_DNS = np.where(kappa_t == 0, field, np.nan)
            field_LES = np.where(kappa_t > 0, field, np.nan)

            for fieldd in [field, field_DNS, field_LES]:
                mixing_field = np.where(np.logical_and(pvd < pvd_thresh, pvd > 0), fieldd, np.nan)
                mixed_field = np.where(pvd >= pvd_thresh, fieldd, np.nan)
                plume_field = np.where(pvd <= 0, fieldd, np.nan)

                avgs.append([np.nanmean(fieldd), np.nanmean(plume_field), np.nanmean(mixing_field),
                    np.nanmean(mixed_field)])
        else:
            field /= dim_factors[i]

            mixing_field = np.where(np.logical_and(pvd < pvd_thresh, pvd > 0), field, np.nan)
            mixed_field = np.where(pvd >= pvd_thresh, field, np.nan)
            plume_field = np.where(pvd <= 0, field, np.nan)

            avgs.append([np.nanmean(field), np.nanmean(plume_field), np.nanmean(mixing_field),
                np.nanmean(mixed_field)])

    f.close()

avgs = np.array(avgs)

for i in [2, 3, 4]:
    nu = np.array([avgs[i+3] / (avgs[i+3] + avgs[i])])
    avgs = np.concatenate((avgs, nu))

print(avgs.T)
#np.savetxt("/home/cwp29/Documents/papers/draft/data.csv", avgs, delimiter=",", fmt="%2.2e")
