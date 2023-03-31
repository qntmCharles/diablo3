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

    eps = np.exp(np.array(f['Timestep']['tked']))
    chi = np.exp(np.array(f['Timestep']['chi']))

    eps = eps[:, idx_min:idx_max, :] # restrict to stratified layer
    eps = np.where(np.isnan(svd), np.nan, eps) # restrict to plume

    chi = chi[:, idx_min:idx_max, :] # restrict to stratified layer
    chi = np.where(np.isnan(svd), np.nan, chi) # restrict to plume

    # restrict to mixing region
    svd_thresh = 3e-3
    mixing_eps = np.nanmean(np.where(np.logical_and(svd < svd_thresh, svd > 0), eps, np.nan))
    mixed_eps = np.nanmean(np.where(svd >= svd_thresh, eps, np.nan))
    plume_eps = np.nanmean(np.where(svd <= 0, eps, np.nan))

    mixing_chi = np.nanmean(np.where(np.logical_and(svd < svd_thresh, svd > 0), chi, np.nan))
    mixed_chi = np.nanmean(np.where(svd >= svd_thresh, chi, np.nan))
    plume_chi = np.nanmean(np.where(svd <= 0, chi, np.nan))

    nu_plume = plume_chi/(plume_chi+plume_eps)
    nu_mixing = mixing_chi/(mixing_chi+mixing_eps)
    nu_mixed = mixed_chi/(mixed_chi+mixed_eps)
    print(nu_plume)
    print(nu_mixed)
    print(nu_mixing)

    f.close()
