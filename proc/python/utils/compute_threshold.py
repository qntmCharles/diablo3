import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
from mpl_toolkits import axes_grid1
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from datetime import datetime
from functions import get_metadata, read_params, get_grid, g2gf_1d, compute_F0, get_plotindex, get_index
from os.path import join, isfile
from os import listdir

from scipy.interpolate import griddata, interp1d
from scipy import integrate, ndimage, spatial

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"

save = True
show = not save

fig_save_dir = '/home/cwp29/Documents/papers/conv_pen/draft2/figs'

dbdphi = 42.2

##### ---------------------- #####

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Data acquisition
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Get dir locations from param file
base_dir, run_dir, save_dir, version = read_params(params_file)

# Get simulation metadata
md = get_metadata(run_dir, version)
md['kappa'] = md['nu']/0.7

gxf, gyf, gzf, dzf = get_grid(save_dir+"/grid.h5", md)
gx, gy, gz, dz = get_grid(save_dir+"/grid.h5", md, fractional_grid=False)

X, Y = np.meshgrid(gx, gz)
Xf, Yf = np.meshgrid(gxf, gzf)

print("Complete metadata: ", md)

plot_max = 1.6*md['H']
plot_min = 0.95*md['H']

idx_minf = get_plotindex(plot_min, gzf)-1
idx_maxf = get_plotindex(plot_max, gzf)
idx_min = idx_minf
idx_max = idx_maxf+1

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Figure 13: TKED PDFs
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

out_file = 'end.h5'

pvd_lim = 50

with h5py.File(join(save_dir,out_file), 'r') as f:
    print("Keys: %s" % f['Timestep'].keys())

    print(idx_min, idx_max)
    phi = np.array(f['Timestep']['TH2'][pvd_lim:-pvd_lim, idx_min:idx_max, pvd_lim:-pvd_lim])

    grad_phi = np.abs(np.power(np.gradient(phi, md['LX']/md['Nx'], axis=0), 2) + \
            np.power(np.gradient(phi, md['LZ']/md['Nz'], axis=1), 2) + \
            np.power(np.gradient(phi, md['LY']/md['Ny'], axis=2), 2))

    f.close()

h, bins = np.histogram(np.log10(phi), range=(-7, -2), weights=grad_phi, density=True, bins=256)

plt.plot(0.5*(bins[1:]+bins[:-1]), h)
plt.show()
