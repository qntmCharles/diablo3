import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from functions import get_metadata, read_params, get_grid, g2gf_1d, get_plotindex, get_index
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

fields = ["tked", "chi", "Ri", "Re_b", "TH1"]
field_mins = [-12.5, -12.5, -2, -3, -5]
field_maxs = [0, 0, 4, 15, 20]

pvd_lim = 50

Xf, Yf = np.meshgrid(gx[pvd_lim:-pvd_lim], gz[idx_min:idx_max+1])
X, Y = np.meshgrid(gxf[pvd_lim:-pvd_lim], gz[idx_min:idx_max])

with h5py.File(join(save_dir,out_file), 'r') as f:
    print("Keys: %s" % f['Timestep'].keys())

    pvd = np.array(f['Timestep']['PVD'][pvd_lim:-pvd_lim, idx_min:idx_max, pvd_lim:-pvd_lim])
    pvd = np.where(pvd == -1e9, np.nan, pvd)
    print("Loaded PVD field")

    phi = np.array(f['Timestep']['TH2'][pvd_lim:-pvd_lim, idx_min:idx_max, pvd_lim:-pvd_lim])

    for i in range(len(fields)):
        print("Loading field {0}".format(fields[i]))
        field = np.array(f['Timestep'][fields[i]][pvd_lim:-pvd_lim, idx_min:idx_max, pvd_lim:-pvd_lim])

        if fields[i] == "TH1":
            field = np.gradient(field, gzf[idx_min:idx_max], axis = 1)

        # restrict to plume
        field = np.where(np.isnan(pvd), np.nan, field)

        print(np.nanmin(field), np.nanmax(field))
        if fields[i] == "tked":
            nu_t = np.array(f['Timestep']['NU_T'][pvd_lim:-pvd_lim, idx_min:idx_max, pvd_lim:-pvd_lim])
            nu_t = np.where(np.isnan(pvd), np.nan, nu_t) # restrict to plume

            #field -= np.log10(md['nu'] + nu_t) # remove nu_eff factor

            #nu_t_zero = np.where(nu_t == 0, nu_t, np.nan)
            #nu_t_nonzero = np.where(nu_t != 0, nu_t, np.nan)

            #field_lower_peak = field + np.log10(md['nu'] + nu_t_zero)
            #field_upper_peak = field + np.log10(md['nu'] + nu_t_nonzero)
            field_lower_peak = np.where(nu_t == 0, field, np.nan)
            field_upper_peak = np.where(nu_t > 0, field, np.nan)

            epsilon = np.nanmean(np.power(10, field))
            print(epsilon)
            print("kolm scale", np.power(md['nu']**3 / epsilon, 1/4))
            print("kolm scale lower", np.power(md['nu']**3 /
                np.nanmean(np.power(10, field_lower_peak)), 1/4))
            print("kolm scale upper", np.power(md['nu']**3 /
                np.nanmean(np.power(10, field_upper_peak)), 1/4))
            print(md['LX']/md['Nx'])

            kolm_scale_lower = np.power(md['nu'], 3/4) * \
                    np.power(np.power(10, field_lower_peak), -1/4)
            kolm_scale_upper = np.power(md['nu'], 3/4) * \
                    np.power(np.power(10, field_upper_peak), -1/4)

            plt.figure()
            nu_eff = np.log10(md['nu'] + nu_t)
            h, bins = np.histogram(nu_eff.flatten(), bins=128, range=(np.nanmin(nu_eff),
                np.nanmax(nu_eff)), density=True)
            plt.plot(0.5*(bins[1:]+bins[:-1]), h, color='b')

            plt.figure()
            h, bins = np.histogram(kolm_scale_lower.flatten(), bins=128, range=(np.nanmin(kolm_scale_lower),
                np.nanmax(kolm_scale_lower)))
            plt.plot(0.5*(bins[1:]+bins[:-1]), h, color='b', label='lower')
            h, bins = np.histogram(kolm_scale_upper.flatten(), bins=128, range=(np.nanmin(kolm_scale_upper),
                np.nanmax(kolm_scale_upper)))
            plt.plot(0.5*(bins[1:]+bins[:-1]), h, color='r', label='upper')
            plt.axvline(0.5*np.min(dz))
            plt.legend()

        pvd_thresh = 5e-3

        h, bins = np.histogram(field.flatten(), bins=128, range = (field_mins[i], field_maxs[i]))
        bins_plot = 0.5*(bins[1:] + bins[:-1])

        fig = plt.figure()
        ax = plt.gca()
        plt.title(fields[i])
        integral = np.sum(h * (bins[1:] - bins[:-1]))
        plt.plot(bins_plot, h/integral, color='k', linestyle='--')

        print(integral)

        for j in range(2):
            if j == 0:
                field = field_lower_peak
                cols = {
                        'mixed': 'lightcoral',
                        'plume': 'lightblue',
                        'mixing': 'lightgreen'
                        }
            else:
                field = field_upper_peak
                cols = {
                        'mixed': 'r',
                        'plume': 'b',
                        'mixing': 'g'
                        }

            # restrict to mixing region
            mixing_field = np.where(np.logical_and(pvd < pvd_thresh, pvd > 0), field, np.nan)
            mixed_field = np.where(pvd >= pvd_thresh, field, np.nan)
            plume_field = np.where(pvd <= 0, field, np.nan)
            print(field.shape)

            #h, bins = np.histogram(field.flatten(), bins=128, range = (field_mins[i], field_maxs[i]),
                    #density=True)
                    #weights=volume.flatten(), density=True)
            #bins_plot = 0.5*(bins[1:] + bins[:-1])
            #plt.plot(bins_plot, h, color='k', linestyle='--')

            mixing_h, bins = np.histogram(mixing_field.flatten(), bins=128,
                    range = (field_mins[i], field_maxs[i]))
            plt.plot(bins_plot, mixing_h/integral, color=cols['mixing'])

            mixed_h, bins = np.histogram(mixed_field.flatten(), bins=128,
                    range = (field_mins[i], field_maxs[i]))
            plt.plot(bins_plot, mixed_h/integral, color=cols['mixed'])

            plume_h, bins = np.histogram(plume_field.flatten(), bins=128,
                    range = (field_mins[i], field_maxs[i]))
            plt.plot(bins_plot, plume_h/integral, color=cols['plume'])

            if fields[i] == "chi":
                integral = np.nansum(np.exp(field))
                #print(integral)
                print(np.nansum(np.exp(mixing_field))/integral)
                #print(np.nansum(np.exp(mixed_field))/integral)
                #print(np.nansum(np.exp(plume_field))/integral)

                total_vol = np.count_nonzero(~np.isnan(pvd))
                print(np.count_nonzero(~np.isnan(mixing_field))/total_vol)
                #print(np.nansum(np.where(pvd > pvd_thresh, volume, np.nan))/total_vol)
                #print(np.nansum(np.where(pvd < -pvd_thresh, volume, np.nan))/total_vol)


            print(bins_plot[np.argmax(plume_h)])
            print(bins_plot[np.argmax(mixed_h)])
            print(bins_plot[np.argmax(mixing_h)])

        plt.show()

    f.close()
