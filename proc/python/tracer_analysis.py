import h5py, gc, sys
import numpy as np
from scipy.interpolate import griddata
import matplotlib
from matplotlib import pyplot as plt
from os.path import join, isfile
from os import listdir
from functions import get_metadata, read_params, get_grid

##### USER-DEFINED PARAMETERS #####
params_file = "params.dat"
out_file = "out.002249.h5"
mp = 12

##### ----------------------- #####

##### Get directory locations #####
base_dir, run_dir, save_dir, version = read_params(params_file)
print("Run directory: ", run_dir)
print("Save directory: ", save_dir)

##### Get simulation metadata #####
md = get_metadata(run_dir, version)
print("Complete metadata: ",md)

##### Get grid #####
gxf, gyf, gzf, dz = get_grid(join(run_dir, 'grid.h5'), md)
gzfp = np.flip(gzf)

x_coords, y_coords, z_coords = np.meshgrid(gxf, gyf, gzf, indexing='ij')

##### Get data #####
print("Reading %s..."%out_file)
with h5py.File(join(save_dir,out_file), 'r') as f:
    u = np.array(f['Timestep']['U'])
    v = np.array(f['Timestep']['V'])
    w = np.array(f['Timestep']['W'])
    b = np.array(f['Timestep']['TH1'])
    t = np.array(f['Timestep']['TH2'])
    p = np.array(f['Timestep']['P'])

    u = np.transpose(u, axes=(2,0,1))
    v = np.transpose(v, axes=(2,0,1))
    w = np.transpose(w, axes=(2,0,1))
    b = np.transpose(b, axes=(2,0,1))
    t = np.transpose(t, axes=(2,0,1))
    p = np.transpose(p, axes=(2,0,1))

# b contains values of buoyancy at each point
# z_coords contains z value at each point
# t contains tracer value at each point
## Aim is to bin tracer values wrt buoyancy

# Create buoyancy bins
b_min = 0
#b_max = np.max(b[0]) # this is incorrect
b_max = (md['LZ']-md['H'])*md['N2']/2
nbins = 200

bbins = np.linspace(b_min, b_max, nbins)
bins = [0 for i in range(nbins)]

for i in range(nbins-1):
    bins[i] = np.sum(np.where(np.logical_and(b > bbins[i], b < bbins[i+1]), t, 0))

plot_min = int(md['H']*md['Nz']/md['LZ'])
plot_max = plot_min + int(b_max/(md['N2']*(md['LZ']-md['H'])) * (md['Nz']-plot_min))

fig, ax = plt.subplots(1,3)
ax[0].imshow(np.flip(np.swapaxes(b[:,128,:],0,-1)[plot_min:plot_max],axis=0), cmap='jet')
ax[1].imshow(np.flip(np.swapaxes(t[:,128,:],0,-1)[plot_min:plot_max],axis=0), cmap='jet')
ax[2].plot(np.array(bins)/np.sum(bins), bbins, color='b')
ax[2].set_ylim(b_min, b_max)

asp = np.diff(ax[2].get_xlim())[0]/np.diff(ax[2].get_ylim())[0]
asp /= np.abs(np.diff(ax[1].get_xlim())[0]/np.diff(ax[1].get_ylim())[0])
ax[2].set_aspect(asp)

plt.tight_layout()
plt.show()
