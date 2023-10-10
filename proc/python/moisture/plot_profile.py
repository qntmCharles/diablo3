import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import numpy as np
from matplotlib import pyplot as plt
from functions import get_metadata, read_params, get_grid, g2gf_1d

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"

##### ---------------------- #####

# Get dir locations from param file
base_dir, run_dir, save_dir, version = read_params(params_file)

# Get simulation metadata
md = get_metadata(run_dir, version)

gxf, gyf, gzf, dzf = get_grid(save_dir+'grid.h5', md)
gx, gy, gz, dz = get_grid(save_dir+'grid.h5', md, fractional_grid=False)

##### ENVIRONMENTAL VARIABLES #####

alpha = md['alpha']
beta = md['beta']
q_0 = md['q0']
T0 = 300 # K

##### ---------------------- #####

b = md['N2'] * (gzf - md['H'])
b[gzf < md['H']] = 0

T = 1340 * (b - beta * gzf)

phi_sat = q_0 * np.exp(alpha/1340 * T)

fig, ax = plt.subplots(1, 3)

ax[0].plot(b, gzf, color='r', label=r"$b \, (m s^{-1})$")
ax[1].plot(T0 + T, gzf, color='g', label=r"$T \, (K)$")
ax[2].plot(phi_sat, gzf, color='b', label=r"$\phi_s$")

ax[2].set_xlim(0, 2*q_0)

for a in ax:
    a.legend()
    a.set_ylim(0, md['LZ'])

plt.show()
