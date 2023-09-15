import h5py, os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as cm
import matplotlib.animation as animation
from datetime import datetime
from functions import get_metadata, read_params, get_grid, g2gf_1d

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"
out_file = "out.000030.h5"

use_movie = False
if not os.path.isfile(out_file):
    print("Out file not found. Using movie file.")
    use_movie = True

##### ---------------------- #####

# Get dir locations from param file
base_dir, run_dir, save_dir, version = read_params(params_file)

# Get simulation metadata
md = get_metadata(run_dir, version)

print("Complete metadata: ", md)

gxf, gyf, gzf, dz = get_grid(run_dir+"grid.h5",md)


# Get data
if not use_movie:
    with h5py.File(save_dir+out_file,'r') as f:
        u = np.array(f['Timestep']['U'])
        v = np.array(f['Timestep']['V'])
        w = np.array(f['Timestep']['W'])
        b = np.array(f['Timestep']['TH1'])

        u = np.transpose(u, axes=(2,0,1))
        v = np.transpose(v, axes=(2,0,1))
        w = np.transpose(w, axes=(2,0,1))
        b = np.transpose(b, axes=(2,0,1))

        # might need to check the order of the indices....
        u = g2gf_1d(md, u)
        v = g2gf_1d(md, v)
        w = g2gf_1d(md, w)
        b = g2gf_1d(md, b)

        b_plot = b[:, int(md['Ny']/2), :]
        w_plot = w[:, int(md['Ny']/2), :]
else:
    with h5py.File(save_dir+'movie.h5', 'r') as f:
        time_keys = list(f['w_xz'])
        w_plot = np.swapaxes(np.array(f['w_xz'][time_keys[1]]), 0, 1)
        w_plot = g2gf_1d(md, w_plot)
        b_plot = np.swapaxes(np.array(f['th1_xz'][time_keys[1]]), 0, 1)
        b_plot = g2gf_1d(md, b_plot)

nplots=2
cols = plt.cm.rainbow(np.linspace(0,1,nplots))

fig, ax = plt.subplots(1,2)
for i,c in zip(range(nplots),cols):
    ax[0].plot(w_plot[:,i+1], linestyle='--',color=c)
    ax[1].plot(b_plot[:,i+1], linestyle='--',color=c)

r_0 = md['r0']
alpha = md['alpha_e']
Lyc = md['Lyc']
Lyp = md['Lyp']

zvirt = -r_0 / (1.2 * alpha)

if version == "3.5":
    Q_0 = md['Q0']
    F_0 = np.pi*r_0**2 * Q_0
else:
    f_0 = md['b0']
    F_0 = r_0**2 * f_0

r_m = np.zeros(shape=(md['Nz']+1))
w_m = np.zeros(shape=(md['Nz']+1))
b_m = np.zeros(shape=(md['Nz']+1))

for j in range(md['Nz']):
    r_m[j] = 1.2 * alpha * (gzf[j]-zvirt)
    w_m[j] = (0.9 * alpha * F_0)**(1/3) * (gzf[j] - zvirt)**(2/3) / r_m[j]
    b_m[j] = F_0/(r_m[j]**2 * w_m[j])

r_m = np.array(r_m)
w_m = np.array(w_m)
b_m = np.array(b_m)

b_forcing = np.zeros(shape=(md['Nx'],md['Ny'],md['Nz']))
w_forcing = np.zeros(shape=(md['Nx'],md['Ny'],md['Nz']))

x, y, z = np.meshgrid(gxf, gyf, gzf, indexing='ij', sparse=True)

for j in range(md['Nz']):
    w_forcing[:,:,j] = w_m[j]*np.exp(-((x[:,:,0]-md['LX']/2)**2 + (y[:,:,0]-md['LY']/2)**2)/(2*r_m[j]**2)) \
            * (1 - np.tanh((z[:,:,j]-Lyc)/Lyp))/2
    b_forcing[:,:,j] = b_m[j]*np.exp(-((x[:,:,0]-md['LX']/2)**2 + (y[:,:,0]-md['LY']/2)**2)/(2*r_m[j]**2)) \
            * (1 - np.tanh((z[:,:,j]-Lyc)/Lyp))/2

for i,c in zip(range(nplots),cols):
    ax[0].plot(w_forcing[:,int(md['Ny']/2),i],color=c)
    ax[1].plot(b_forcing[:,int(md['Ny']/2),i],color=c)

ax[0].set_title("w")
ax[0].set_xlabel("x")
ax[1].set_title("b")
ax[1].set_xlabel("x")

plt.show()
