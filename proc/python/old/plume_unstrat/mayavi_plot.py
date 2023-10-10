import h5py, gc, sys
import numpy as np
from scipy.interpolate import griddata
import matplotlib
from matplotlib import pyplot as plt
from mayavi import mlab
from os.path import join, isfile
from os import listdir
from functions import get_metadata, read_params, get_grid, g2gf_1d

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

    u = g2gf_1d(u)
    v = g2gf_1d(v)
    w = g2gf_1d(w)
    b = g2gf_1d(b)
    t = g2gf_1d(t)
    p = g2gf_1d(p)

    u = np.transpose(u, axes=(2,0,1))
    v = np.transpose(v, axes=(2,0,1))
    w = np.transpose(w, axes=(2,0,1))
    b = np.transpose(b, axes=(2,0,1))
    t = np.transpose(t, axes=(2,0,1))
    p = np.transpose(p, axes=(2,0,1))

    # Thin out the data
    u = u[::mp, ::mp]
    v = v[::mp, ::mp]
    w = w[::mp, ::mp]
    p = p[::mp, ::mp]
    b = b[::mp, ::mp]
    t = t[::mp, ::mp]

x_coords = x_coords[::mp, ::mp]
y_coords = y_coords[::mp, ::mp]
z_coords = z_coords[::mp, ::mp]

print(x_coords.shape)
print(y_coords.shape)
print(z_coords.shape)

src = mlab.pipeline.vector_field(x_coords, y_coords, z_coords, u,v,w)
mag = mlab.pipeline.extract_vector_norm(src)

vec = mlab.pipeline.vectors(mag, mask_points = 10, scale_factor=.5)
"""
flow = mlab.pipeline.streamline(mag, seed_visible=True, seed_scale=0.25, seed_resolution=5.,
        integration_direction='both')

#src_p = mlab.pipeline.scalar_field(x_coords, y_coords, z_coords, p)

#plane = mlab.pipeline.image_plane_widget(src_p, plane_orientation='z_axes', vmin = 0, vmax = 0.01)
"""

mlab.contour3d(x_coords, y_coords, z_coords, t, contours=8, opacity=.5, colormap='jet')

mlab.axes()
mlab.show()
