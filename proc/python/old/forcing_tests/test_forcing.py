import h5py, bisect, time, gc, sys
from os import listdir
from os.path import isfile, join
import numpy as np
from math import sqrt, floor
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from functions import get_metadata, get_grid, read_params, get_az_data
from scipy import integrate, optimize, interpolate
from matplotlib import cm as cm
from itertools import groupby

##### USER-DEFINED PARAMETERS #####
params_file = "./params.dat"

##### Get directory locations #####
base_dir, run_dir, save_dir, version = read_params(params_file)
print("Run directory: ", run_dir)
print("Save directory: ", save_dir)

##### Get simulation metadata #####
md = get_metadata(run_dir, version)
print("Complete metadata: ",md)

##### Get grid #####
gxf, gyf, gzf, dz = get_grid(join(run_dir, 'grid.h5'), md)
gx, gy, gz, dz = get_grid(join(run_dir, 'grid.h5'), md, fractional_grid=False)
gzfp = np.flip(gzf)

w_forced = []
xw_forced = []
xb_forced = []
b_forced = []

##### Read output file and extract data #####
with open(join(save_dir, 'output.dat'), 'r') as f:
    text = f.readlines()
    cur_time_step = 0
    for i in range(len(text)):
        strings = text[i].split()
        if len(strings) > 0 and strings[0] == "WDATA":
            strings.extend(text[i+1].split())
            y = float(strings[1])
            z = float(strings[2])
            x = float(strings[3])
            w_forcing_data = float(strings[4])
            if round(z,4) == round(gzf[0],4): # looking at bottom layer
                xw_forced.append(x)
                w_forced.append(w_forcing_data)
        if len(strings) > 0 and strings[0] == "BDATA":
            strings.extend(text[i+1].split())
            y = float(strings[1])
            z = float(strings[2])
            x = float(strings[3])
            b_forcing_data = float(strings[4])
            if round(z,4) == round(gzf[0],4): # looking at bottom layer
                xb_forced.append(x)
                b_forced.append(b_forcing_data)

"""
with h5py.File(join(save_dir, 'movie.h5'), 'r') as f:
    time_keys = list(f['th1_xz'])
    tplot = time_keys[1]
    b_plot = np.array(f['th1_xz'][tplot])
    w_plot = np.array(f['w_xz'][tplot])
"""

alpha = md['alpha_e']
zvirt = -5/6 * md['r0']/alpha
F0 = md['r0']**2 * md['b0']

r_m = np.zeros(shape=(md['Nz'],))
b_m = np.zeros(shape=(md['Nz'],))
w_m = np.zeros(shape=(md['Nz'],))

b_forcing = np.zeros(shape=(md['Nx']+1,md['Ny']+1,md['Nz']))
w_forcing = np.zeros(shape=(md['Nx']+1,md['Ny']+1,md['Nz']))

x, y, z = np.meshgrid(gx, gy, gzf, indexing='ij', sparse=True)

for j in range(md['Nz']):
    r_m[j] = 1.2 * alpha * (gzf[j] - zvirt)
    w_m[j] = (0.9 * alpha * F0)**(1/3) * (gzf[j]-zvirt)**(2/3)/r_m[j]
    b_m[j] = F0/(r_m[j] * r_m[j] * w_m[j])

Lyc = md['Lyc']
Lyp = md['Lyp']

for j in range(md['Nz']):
    w_forcing[:,:,j] = 2*w_m[j]*np.exp(-((x[:,:,0]-md['LX']/2)**2 + (y[:,:,0]-md['LY']/2)**2)/(2*r_m[j]**2)) \
            * (1 - np.tanh((z[:,:,j]-Lyc)/Lyp))/2
    b_forcing[:,:,j] = 2*b_m[j]*np.exp(-((x[:,:,0]-md['LX']/2)**2 + (y[:,:,0]-md['LY']/2)**2)/(2*r_m[j]**2)) \
            * (1 - np.tanh((z[:,:,j]-Lyc)/Lyp))/2

fig, ax = plt.subplots(1,2)
ax[0].scatter(xw_forced, w_forced, color='b')
ax[0].plot(gx, w_forcing[:, int(md['Ny']/2), 0], color='r', linestyle='dashed')
#ax[0].plot(gxf, w_plot[1,:], color='b', linestyle='dashed')

ax[1].scatter(xb_forced, b_forced, color='b')
ax[1].plot(gx, b_forcing[:, int(md['Ny']/2), 0], color='r', linestyle='dashed')
#ax[1].plot(gxf, b_plot[1,:], color='b', linestyle='dashed')

plt.show()
