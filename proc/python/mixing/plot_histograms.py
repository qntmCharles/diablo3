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

fields = ["Ri", "chi", "Re_b", "tke", "e"]
field_mins = [-1, 0, -4, 0, -2e-3]
field_maxs = [2, 2e-8, 15, 1e-7, 2e-3]

with h5py.File(join(save_dir,"end.h5"), 'r') as f:
    print("Keys: %s" % f['Timestep'].keys())

    phi = np.array(f['Timestep']['TH2'])
    phi = phi[:, idx_min:idx_max, :] # restrict to stratified layer

    for i in range(len(fields)):
        field = np.array(f['Timestep'][fields[i]])
        if fields[i] == "tke": print(np.max(field))
        field = field[:, idx_min:idx_max, :] # restrict to stratified layer
        field = np.where(phi > 0, field, np.nan) # restrict to plume
        print(field.shape)

        im = plt.imshow(field[int(md['Nx']/2), :, :])
        im.set_clim(field_mins[i], field_maxs[i])
        plt.show()

        plt.title(fields[i])
        plt.hist(field.flatten(), bins=128, range = (field_mins[i], field_maxs[i]), weights=volume.flatten(),
                density=True)

        plt.show()

    f.close()
