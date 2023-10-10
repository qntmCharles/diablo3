import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from functions import get_metadata, read_params, get_grid, g2gf_1d
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

print("Complete metadata: ", md)

# Get data
with h5py.File(join(save_dir,"movie.h5"), 'r') as f:
    print("Keys: %s" % f.keys())
    time_keys = list(f['chi1_xz'])
    print(time_keys)

    Ri_Reb_pdf = np.array([np.array(f['Ri_Reb_pdf'][t]) for t in time_keys])
    chi_Reb_pdf = np.array([np.array(f['chi_Reb_pdf'][t]) for t in time_keys])
    chi_Ri_pdf = np.array([np.array(f['chi_Ri_pdf'][t]) for t in time_keys])
    chi_e_pdf = np.array([np.array(f['chi_e_pdf'][t]) for t in time_keys])
    e_Ri_pdf = np.array([np.array(f['e_Ri_pdf'][t]) for t in time_keys])

    chi_b = np.array([np.array(f['chi1_xz'][t]) for t in time_keys])
    chi_b = g2gf_1d(md,chi_b)
    diapycvel = np.array([np.array(f['diapycvel1_xz'][t]) for t in time_keys])
    diapycvel = g2gf_1d(md,diapycvel)
    NSAMP = len(chi_b)
    times = np.array([f['chi1_xz'][t].attrs['Time'] for t in time_keys])
    f.close()

chi_min = 0
chi_max = 2e-8
Nchi = 128
dchi = (chi_max - chi_min)/Nchi
chi_bins = [chi_min + (i+0.5)*dchi for i in range(Nchi)]

e_min = -2e-3
e_max = 2e-3
Ne = 128
de = (e_max - e_min)/Ne
e_bins = [e_min + (i+0.5)*de for i in range(Ne)]

Ri_min = -1
Ri_max = 2
NRi = 128
dRi = (Ri_max - Ri_min)/NRi
Ri_bins = [Ri_min + (i+0.5)*dRi for i in range(NRi)]

Re_b_min = -4
Re_b_max = 15
NRe_b = 128
dRe_b = (Re_b_max - Re_b_min)/NRe_b
Re_b_bins = [Re_b_min + (i+0.5)*dRe_b for i in range(NRe_b)]


fig, ax = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
fig.suptitle(times[-1])

print(chi_Reb_pdf[0])

sx, sy = np.meshgrid(chi_bins, Re_b_bins)
#chi_Reb_pdf[:, :, 0] = np.nan
chi_Reb_im = ax[0,0].pcolormesh(sx, sy, chi_Reb_pdf[-1], cmap='hot')
plt.colorbar(chi_Reb_im, ax=ax[0,0])
ax[0,0].set_title("$(\chi, \log \mathrm{{Re}}_b)$ joint PDF")
ax[0,0].set_xlabel("$\chi$")
ax[0,0].set_ylabel("$\log \mathrm{{Re}}_b$")

sx, sy = np.meshgrid(chi_bins, e_bins)
#chi_e_pdf[:, 25, :] = np.nan
chi_e_im = ax[0,1].pcolormesh(sx, sy, chi_e_pdf[-1], cmap='hot')
plt.colorbar(chi_e_im, ax=ax[0,1])
ax[0,1].axhline(0, color='gray', alpha=0.7, linestyle='--')
ax[0,1].set_title("$(\chi, e)$ joint PDF")
ax[0,1].set_xlabel("$\chi$")
ax[0,1].set_ylabel("$e$")

sx, sy = np.meshgrid(chi_bins, Ri_bins)
#chi_Ri_pdf[:, :, 0] = np.nan
chi_Ri_im = ax[0,2].pcolormesh(sx, sy, chi_Ri_pdf[-1], cmap='hot')
plt.colorbar(chi_Ri_im, ax=ax[0,2])
ax[0,2].axhline(0, color='gray', alpha=0.7, linestyle='--')
ax[0,2].axhline(0.25, color='gray', alpha=0.7, linestyle='--')
ax[0,2].set_title("$(\chi, \mathrm{{Ri}})$ joint PDF")
ax[0,2].set_xlabel("$\chi$")
ax[0,2].set_ylabel("$\mathrm{{Ri}}$")

sx, sy = np.meshgrid(Ri_bins, Re_b_bins)
#Ri_Reb_pdf[:, :, 0] = np.nan
Ri_Reb_im = ax[1,0].pcolormesh(sx, sy, Ri_Reb_pdf[-1], cmap='hot_r')
plt.colorbar(Ri_Reb_im, ax=ax[1,0])
ax[1,0].axvline(0, color='gray', alpha=0.7, linestyle='--')
ax[1,0].axvline(0.25, color='gray', alpha=0.7, linestyle='--')
ax[1,0].set_title("$(\mathrm{{Ri}}, \log \mathrm{{Re}}_b)$ joint PDF")
ax[1,0].set_xlabel("$\mathrm{{Ri}}$")
ax[1,0].set_ylabel("$\log \mathrm{{Re}}_b$")

sx, sy = np.meshgrid(e_bins, Ri_bins)
#e_Ri_pdf[:, :, 25] = np.nan
e_Ri_im = ax[1,1].pcolormesh(sx, sy, e_Ri_pdf[-1], cmap='hot')
plt.colorbar(e_Ri_im, ax=ax[1,1])
ax[1,1].axvline(0, color='gray', alpha=0.7, linestyle='--')
ax[1,1].axhline(0, color='gray', alpha=0.7, linestyle='--')
ax[1,1].axhline(0.25, color='gray', alpha=0.7, linestyle='--')
ax[1,1].set_title("$(e, \mathrm{{Ri}})$ joint PDF")
ax[1,1].set_xlabel("$e$")
ax[1,1].set_ylabel("$\mathrm{{Ri}}$")

fig.delaxes(ax[1,2])
#plt.savefig('/home/cwp29/Documents/essay/figs/mixPDFs.png', dpi=300)
plt.show()
