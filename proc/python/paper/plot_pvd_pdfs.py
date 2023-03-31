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

fields = ["chi", "tked", "Ri", "Re_b", "TH1"]
field_mins = [-12.5, -12.5, -2, -3, -5]
field_maxs = [0, 0, 4, 15, 20]

with h5py.File(join(save_dir,out_file), 'r') as f:
    print("Keys: %s" % f['Timestep'].keys())

    pvd = np.array(f['Timestep']['PVD'][:, idx_min:idx_max, :]) # restrict to stratified layer
    pvd = np.where(pvd == -1e9, np.nan, pvd)
    print("Loaded PVD field")

    for i in range(len(fields)):
        print("Loading field {0}".format(fields[i]))
        field = np.array(f['Timestep'][fields[i]][:, idx_min:idx_max])
        if fields[i] == "TH1":
            field = np.gradient(field, gzf[idx_min:idx_max], axis = 1)
        field = np.where(np.isnan(pvd), np.nan, field) # restrict to plume

        # restrict to mixing region
        pvd_thresh = 5e-3
        mixing_field = np.where(np.logical_and(pvd < pvd_thresh, pvd > 0), field, np.nan)
        mixed_field = np.where(pvd >= pvd_thresh, field, np.nan)
        plume_field = np.where(pvd <= 0, field, np.nan)
        print(field.shape)

        #im = plt.imshow(field[int(md['Nx']/2), :, :])
        #im.set_clim(field_mins[i], field_maxs[i])
        #plt.show()

        plt.title(fields[i])
        h, bins = np.histogram(field.flatten(), bins=128, range = (field_mins[i], field_maxs[i]),
                density=True)
                #weights=volume.flatten(), density=True)
        plt.plot(0.5*(bins[1:]+bins[:-1]), h, color='k', linestyle='--')

        mixing_h, bins = np.histogram(mixing_field.flatten(), bins=128, range = (field_mins[i], field_maxs[i]),
                density=True)
                #weights=volume.flatten(), density=True)
        plt.plot(0.5*(bins[1:]+bins[:-1]), mixing_h, color='g')

        mixed_h, bins = np.histogram(mixed_field.flatten(), bins=128, range = (field_mins[i], field_maxs[i]),
                density=True)
                #weights=volume.flatten(), density=True)
        plt.plot(0.5*(bins[1:]+bins[:-1]), mixed_h, color='r')

        plume_h, bins = np.histogram(plume_field.flatten(), bins=128, range = (field_mins[i], field_maxs[i]),
                density=True)
                #weights=volume.flatten(), density=True)
        plt.plot(0.5*(bins[1:]+bins[:-1]), plume_h, color='b')

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
        plt.show()

    f.close()
