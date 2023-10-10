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
plot_max = md['LZ']

idx_minf = get_plotindex(plot_min, gzf)-1
idx_maxf = get_plotindex(plot_max, gzf)

idx_min = idx_minf
idx_max = idx_maxf+1

print(idx_min, idx_max)

gx, gy, dz = np.meshgrid(gxf, dz[idx_min:idx_max], gyf, indexing='ij')
volume = (md['LX']/md['Nx'])**2 * dz
print(volume.shape)

print("Complete metadata: ", md)

fields = ["TH1", "chi", "Ri", "Re_b", "tked", "TH1"]
field_mins = [-5, -25, -2, -3, -25, -1]
field_maxs = [20, -5, 4, 15, -5, 1]

with h5py.File(join(save_dir,out_file), 'r') as f:
    print("Keys: %s" % f['Timestep'].keys())

    phi = np.array(f['Timestep']['TH2'])
    phi = phi[:, idx_min:idx_max, :] # restrict to stratified layer

    svd = np.array(f['Timestep']['SVD'])
    svd = svd[:, idx_min:idx_max, :] # restrict to stratified layer
    svd = np.where(svd == -1e9, np.nan, svd)

    for i in range(len(fields)):
        field = np.array(f['Timestep'][fields[i]])
        if fields[i] == "TH1":
            field = np.gradient(field, gzf, axis = 1)
        field = field[:, idx_min:idx_max, :] # restrict to stratified layer
        field = np.where(np.isnan(svd), np.nan, field) # restrict to plume

        # restrict to mixing region
        svd_thresh = 5e-3
        mixing_field = np.where(np.logical_and(svd < svd_thresh, svd > 0), field, np.nan)
        mixed_field = np.where(svd >= svd_thresh, field, np.nan)
        plume_field = np.where(svd <= 0, field, np.nan)
        print(field.shape)

        #im = plt.imshow(field[int(md['Nx']/2), :, :])
        #im.set_clim(field_mins[i], field_maxs[i])
        #plt.show()

        plt.title(fields[i])
        h, bins = np.histogram(field.flatten(), bins=128, range = (field_mins[i], field_maxs[i]),
                weights=volume.flatten(), density=True)
        plt.plot(0.5*(bins[1:]+bins[:-1]), h, color='k', linestyle='--')

        mixing_h, bins = np.histogram(mixing_field.flatten(), bins=128, range = (field_mins[i], field_maxs[i]),
                weights=volume.flatten(), density=True)
        plt.plot(0.5*(bins[1:]+bins[:-1]), mixing_h, color='g')

        mixed_h, bins = np.histogram(mixed_field.flatten(), bins=128, range = (field_mins[i], field_maxs[i]),
                weights=volume.flatten(), density=True)
        plt.plot(0.5*(bins[1:]+bins[:-1]), mixed_h, color='r')

        plume_h, bins = np.histogram(plume_field.flatten(), bins=128, range = (field_mins[i], field_maxs[i]),
                weights=volume.flatten(), density=True)
        plt.plot(0.5*(bins[1:]+bins[:-1]), plume_h, color='b')

        if fields[i] == "chi":
            integral = np.nansum(np.exp(field))
            #print(integral)
            print(np.nansum(np.exp(mixing_field))/integral)
            #print(np.nansum(np.exp(mixed_field))/integral)
            #print(np.nansum(np.exp(plume_field))/integral)

            total_vol = np.nansum(np.where(np.isnan(svd), np.nan, volume))
            print(np.nansum(np.where(np.abs(svd) <= svd_thresh, volume, np.nan))/total_vol)
            #print(np.nansum(np.where(svd > svd_thresh, volume, np.nan))/total_vol)
            #print(np.nansum(np.where(svd < -svd_thresh, volume, np.nan))/total_vol)


        #plt.savefig('/home/cwp29/Documents/talks/atmos_group/figs/{0}_pdf.png'.format(fields[i]), dpi=200)
        #plt.clf()
        plt.show()

    f.close()
