import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as cm
import matplotlib.animation as animation
from datetime import datetime
from functions import get_metadata, read_params, get_grid

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"
out_file = "out.000030.h5"

##### ---------------------- #####

# Get dir locations from param file
base_dir, run_dir, save_dir, version = read_params(params_file)

# Get simulation metadata
md = get_metadata(run_dir, version)

print("Complete metadata: ", md)

gx, gy, gz, dz = get_grid(run_dir+"grid.h5",md,fractional_grid=False)


# Get data
with h5py.File(save_dir+out_file,'r') as f:
    u = np.array(f['Timestep']['U'])
    v = np.array(f['Timestep']['V'])
    w = np.array(f['Timestep']['W'])
    b = np.array(f['Timestep']['TH1'])

    u = np.transpose(u, axes=(2,0,1))
    v = np.transpose(v, axes=(2,0,1))
    w = np.transpose(w, axes=(2,0,1))
    b = np.transpose(b, axes=(2,0,1))


nplots=5
cols = plt.cm.rainbow(np.linspace(0,1,nplots))

fig, ax = plt.subplots(1,2)
for i,c in zip(range(nplots),cols):
    ax[0].plot(w[:,128,i], linestyle='--',color=c)
    ax[1].plot(b[:,128,i], linestyle='--',color=c)

r_0 = md['r0']
Q_0 = md['Q0']
alpha = md['alpha_e']
Lyc = md['Lyc']
Lyp = md['Lyp']

zvirt = -r_0 / (1.2 * alpha)
F_0 = np.pi * r_0**2 * Q_0

r_m = np.zeros(shape=(md['Nz']+1))
w_m = np.zeros(shape=(md['Nz']+1))
b_m = np.zeros(shape=(md['Nz']+1))

for j in range(md['Nz']+1):
    r_m[j] = 1.2 * alpha * (gz[j]-zvirt)
    w_m[j] = (0.9 * alpha * F_0)**(1/3) * (gz[j] - zvirt)**(2/3) / r_m[j]
    b_m[j] = F_0/(r_m[j]**2 * w_m[j])

b_forcing = np.zeros(shape=(md['Nx'],md['Ny'],md['Nz']+2))
w_forcing = np.zeros(shape=(md['Nx'],md['Ny'],md['Nz']+2))
for i in range(md['Nx']):
    for j in range(md['Nz']+1):
        for k in range(md['Ny']):
            w_forcing[i,k,j] = w_m[j]*np.exp(-((gx[i]-md['LX']/2)**2 + (gy[k]-md['LY']/2)**2)/(2*r_m[j]**2)) \
                    * (1 - np.tanh((gz[j]-Lyc)/Lyp))/2
            b_forcing[i,k,j] = b_m[j]*np.exp(-((gx[i]-md['LX']/2)**2 + (gy[k]-md['LY']/2)**2)/(2*r_m[j]**2)) \
                    * (1 - np.tanh((gz[j]-Lyc)/Lyp))/2

for i,c in zip(range(nplots),cols):
    ax[0].plot(w_forcing[:,128,i],color=c)
    ax[1].plot(b_forcing[:,128,i],color=c)

plt.show()
