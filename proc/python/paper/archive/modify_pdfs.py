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

fields = ["tked", "tked", "chi", "Ri", "Re_b", "TH1"]
field_mins = [-12.5, -12.5, -12.5, -2, -3, -5]
field_maxs = [0, 0, 0, 4, 15, 20]

pvd_lim = 50

Xf, Yf = np.meshgrid(gx[pvd_lim:-pvd_lim], gz[idx_min:idx_max+1])
X, Y = np.meshgrid(gxf[pvd_lim:-pvd_lim], gz[idx_min:idx_max])

flag = True

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
        field = np.where(np.isnan(pvd), np.nan, field) # restrict to plume

        print(np.nanmin(field), np.nanmax(field))
        if fields[i] == "tked" and not flag:
            nu_t = np.array(f['Timestep']['NU_T'][pvd_lim:-pvd_lim, idx_min:idx_max, pvd_lim:-pvd_lim])
            nu_t = np.where(np.isnan(pvd), np.nan, nu_t) # restrict to plume
            print(np.count_nonzero(~np.isnan(nu_t)))

            field = np.power(10, field)
            field /= md['NU_RUN'] + nu_t # remove nu_eff factor

            nu_t_test = np.where(nu_t != 0, np.nan, 0)

            plt.figure()
            plt.pcolormesh(Xf, Yf, np.where(np.isnan(nu_t_test), np.nan, phi)[int(md['Nx']/2), :, :])
            plt.contour(X, Y, phi[int(md['Nx']/2), :, :], levels=[5e-4], colors='g', linestyles='--')

            print(np.count_nonzero(~np.isnan(nu_t_test)))
            nu_t = np.where(nu_t == 0, np.nan, nu_t)
            nu_t = np.log10(nu_t)

            plt.figure()
            h, bins = np.histogram(nu_t.flatten(), bins=128, range= (-10, -5))
            plt.plot(0.5*(bins[1:]+bins[:-1]), h)

            #####
            #nu_t_thresh = np.where(nu_t < np.log10(md['NU_RUN']), 0, nu_t)
            nu_t_thresh = nu_t

            h, bins = np.histogram(nu_t_thresh.flatten(), bins=128, range= (-10, -5))
            plt.plot(0.5*(bins[1:]+bins[:-1]), h)

            field *= md['NU_RUN']+np.power(10, nu_t_thresh)
            field = np.log10(field)

        # restrict to mixing region
        pvd_thresh = 5e-3
        mixing_field = np.where(np.logical_and(pvd < pvd_thresh, pvd > 0), field, np.nan)
        mixed_field = np.where(pvd >= pvd_thresh, field, np.nan)
        plume_field = np.where(pvd <= 0, field, np.nan)
        print(field.shape)

        fig = plt.figure()
        plt.title(fields[i])
        h, bins = np.histogram(field.flatten(), bins=128, range = (field_mins[i], field_maxs[i]),
                density=True)
                #weights=volume.flatten(), density=True)
        bins_plot = 0.5*(bins[1:] + bins[:-1])
        plt.plot(bins_plot, h, color='k', linestyle='--')

        mixing_h, bins = np.histogram(mixing_field.flatten(), bins=128, range = (field_mins[i], field_maxs[i]),
                density=True)
                #weights=volume.flatten(), density=True)
        plt.plot(bins_plot, mixing_h, color='g')

        mixed_h, bins = np.histogram(mixed_field.flatten(), bins=128, range = (field_mins[i], field_maxs[i]),
                density=True)
                #weights=volume.flatten(), density=True)
        plt.plot(bins_plot, mixed_h, color='r')

        plume_h, bins = np.histogram(plume_field.flatten(), bins=128, range = (field_mins[i], field_maxs[i]),
                density=True)
                #weights=volume.flatten(), density=True)
        plt.plot(bins_plot, plume_h, color='b')

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


        #plt.savefig('/home/cwp29/Documents/talks/atmos_group/figs/{0}_pdf.png'.format(fields[i]), dpi=200)
        #plt.clf()
        print(bins_plot[np.argmax(plume_h)])
        print(bins_plot[np.argmax(mixed_h)])
        print(bins_plot[np.argmax(mixing_h)])

        #plt.axvline(bins_plot[np.argmax(plume_h)], color='b')
        #plt.axvline(bins_plot[np.argmax(mixed_h)], color='r')
        #plt.axvline(bins_plot[np.argmax(mixing_h)], color='g')
        if not flag: plt.show()
        flag = False

    f.close()
