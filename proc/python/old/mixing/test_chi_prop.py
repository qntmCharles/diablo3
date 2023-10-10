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

with h5py.File(join(save_dir,"end.h5"), 'r') as f:
    svd = np.array(f['Timestep']['SVD'])
    svd = svd[:, idx_min:idx_max, :] # restrict to stratified layer

    svd = np.where(svd == -1e9, np.nan, svd)

    plt.imshow(svd[int(md['Nx']/2), :, :])
    plt.show()

field_min = -35
field_max = -15

fig, ax = plt.subplots(1,2)
sum_prop = []
vol_prop = []
svd_threshs = np.linspace(1e-5, 1e-2, 30)
cols = plt.cm.rainbow(np.linspace(0, 1, len(svd_threshs)))

with h5py.File(join(save_dir,"end.h5"), 'r') as f:
    print("Keys: %s" % f['Timestep'].keys())

    phi = np.array(f['Timestep']['TH2'])
    phi = phi[:, idx_min:idx_max, :] # restrict to stratified layer

    field = np.array(f['Timestep']["chi"])
    field = field[:, idx_min:idx_max, :] # restrict to stratified layer
    field = np.where(np.isnan(svd), np.nan, field) # restrict to plume

    integral = np.nansum(np.exp(field))
    total_vol = np.nansum(np.where(np.isnan(svd), np.nan, volume))

    for svd_thresh, c in zip(svd_threshs, cols):
        print(svd_thresh)

        # restrict to mixing region
        mixing_field = np.where(np.abs(svd) <= svd_thresh, field, np.nan)

        ax[1].hist(mixing_field.flatten(), bins=128, range = (field_min, field_max),
                weights=volume.flatten(), density=True, color=c, alpha=0.2)

        ax[0].scatter(svd_thresh, np.nansum(np.where(np.abs(svd) <= svd_thresh, volume, np.nan))/total_vol,
                color=c)
        ax[0].scatter(svd_thresh, np.nansum(np.exp(mixing_field))/integral, color=c)

    f.close()


plt.show()
