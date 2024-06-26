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
    Ri_Reb_pdf_w = np.array([np.array(f['Ri_Reb_pdf_w'][t]) for t in time_keys])
    chi_Reb_pdf = np.array([np.array(f['chi_Reb_pdf'][t]) for t in time_keys])
    chi_Reb_pdf_w = np.array([np.array(f['chi_Reb_pdf_w'][t]) for t in time_keys])
    chi_Ri_pdf = np.array([np.array(f['chi_Ri_pdf'][t]) for t in time_keys])
    chi_Ri_pdf_w = np.array([np.array(f['chi_Ri_pdf_w'][t]) for t in time_keys])
    chi_e_pdf = np.array([np.array(f['chi_e_pdf'][t]) for t in time_keys])
    chi_e_pdf_w = np.array([np.array(f['chi_e_pdf_w'][t]) for t in time_keys])
    e_Ri_pdf = np.array([np.array(f['e_Ri_pdf'][t]) for t in time_keys])
    e_Ri_pdf_w = np.array([np.array(f['e_Ri_pdf_w'][t]) for t in time_keys])
    e_B_pdf = np.array([np.array(f['e_B_pdf'][t]) for t in time_keys])
    e_B_pdf_w = np.array([np.array(f['e_B_pdf_w'][t]) for t in time_keys])
    chi_B_pdf = np.array([np.array(f['chi_B_pdf'][t]) for t in time_keys])
    chi_B_pdf_w = np.array([np.array(f['chi_B_pdf_w'][t]) for t in time_keys])
    Ri_B_pdf = np.array([np.array(f['Ri_B_pdf'][t]) for t in time_keys])
    Ri_B_pdf_w = np.array([np.array(f['Ri_B_pdf_w'][t]) for t in time_keys])

    chi_b = np.array([np.array(f['chi1_xz'][t]) for t in time_keys])
    chi_b = g2gf_1d(md,chi_b)
    diapycvel = np.array([np.array(f['diapycvel1_xz'][t]) for t in time_keys])
    diapycvel = g2gf_1d(md,diapycvel)
    NSAMP = len(chi_b)
    times = np.array([f['chi1_xz'][t].attrs['Time'] for t in time_keys])
    f.close()

Ri_Reb_pdf = np.where(Ri_Reb_pdf == 0, np.nan, Ri_Reb_pdf)
chi_Reb_pdf = np.where(chi_Reb_pdf == 0, np.nan, chi_Reb_pdf)
chi_Ri_pdf = np.where(chi_Ri_pdf == 0, np.nan, chi_Ri_pdf)
chi_e_pdf = np.where(chi_e_pdf == 0, np.nan, chi_e_pdf)
e_Ri_pdf = np.where(e_Ri_pdf == 0, np.nan, e_Ri_pdf)
e_B_pdf = np.where(e_B_pdf == 0, np.nan, e_B_pdf)
chi_B_pdf = np.where(chi_B_pdf == 0, np.nan, chi_B_pdf)
Ri_B_pdf = np.where(Ri_B_pdf == 0, np.nan, Ri_B_pdf)

# Get bins
with h5py.File(join(save_dir, "mean.h5"), 'r') as f:
    print("Keys: %s" % f.keys())
    time_keys = list(f[list(f.keys())[0]])

    chi_bins = np.array(f['chi_pdf_bins'][time_keys[0]])
    chi_min = np.min(chi_bins)
    chi_max = np.max(chi_bins)

    Re_b_bins = np.array(f['Re_b_pdf_bins'][time_keys[0]])
    Re_b_min = np.min(Re_b_bins)
    Re_b_max = np.max(Re_b_bins)

    Ri_bins = np.array(f['Ri_pdf_bins'][time_keys[0]])
    Ri_min = np.min(Ri_bins)
    Ri_max = np.max(Ri_bins)

    e_bins = np.array(f['e_pdf_bins'][time_keys[0]])
    e_min = np.min(e_bins)
    e_max = np.max(e_bins)

    chi_bins = np.array(f['chi_pdf_bins'][time_keys[0]])
    chi_min = np.min(chi_bins)
    chi_max = np.max(chi_bins)

    B_bins = np.array(f['B_pdf_bins'][time_keys[0]])
    B_min = np.min(B_bins)
    B_max = np.max(B_bins)

#############################################################################################################

step = -1

#############################################################################################################

fig, ax = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
fig.suptitle("Joint PDFs at t = {0:.2f}s".format(times[step]))

#############################################################################################################

sx, sy = np.meshgrid(chi_bins, Re_b_bins)
chi_Reb_im = ax[0,0].pcolormesh(sx, sy, chi_Reb_pdf[step], cmap='viridis')
plt.colorbar(chi_Reb_im, ax=ax[0,0])
ax[0,0].set_title("Unnormalised weight {0:.2E}".format(chi_Reb_pdf_w[step,0]))
ax[0,0].set_xlabel("$\chi$")
ax[0,0].set_ylabel("$\log \mathrm{{Re}}_b$")

sx, sy = np.meshgrid(chi_bins, e_bins)
chi_e_im = ax[0,1].pcolormesh(sx, sy, chi_e_pdf[step], cmap='viridis')
plt.colorbar(chi_e_im, ax=ax[0,1])
ax[0,1].set_title("Unnormalised weight {0:.2E}".format(chi_e_pdf_w[step,0]))
ax[0,1].set_xlabel("$\chi$")
ax[0,1].set_ylabel("$e$")

sx, sy = np.meshgrid(chi_bins, Ri_bins)
chi_Ri_im = ax[0,2].pcolormesh(sx, sy, chi_Ri_pdf[step], cmap='viridis')
plt.colorbar(chi_Ri_im, ax=ax[0,2])
ax[0,2].set_title("Unnormalised weight {0:.2E}".format(chi_Ri_pdf_w[step,0]))
ax[0,2].set_xlabel("$\chi$")
ax[0,2].set_ylabel("$\mathrm{{Ri}}$")

sx, sy = np.meshgrid(chi_bins, B_bins)
chi_B_im = ax[0,3].pcolormesh(sx, sy, chi_B_pdf[step], cmap='viridis')
plt.colorbar(chi_B_im, ax=ax[0,3])
ax[0,3].set_title("Unnormalised weight {0:.2E}".format(chi_B_pdf_w[step,0]))
ax[0,3].set_xlabel("$\chi$")
ax[0,3].set_ylabel("$B$")

sx, sy = np.meshgrid(Ri_bins, Re_b_bins)
Ri_Reb_im = ax[1,0].pcolormesh(sx, sy, Ri_Reb_pdf[step], cmap='viridis')
plt.colorbar(Ri_Reb_im, ax=ax[1,0])
ax[1,0].set_title("Unnormalised weight {0:.2E}".format(Ri_Reb_pdf_w[step,0]))
ax[1,0].set_xlabel("$\mathrm{{Ri}}$")
ax[1,0].set_ylabel("$\log \mathrm{{Re}}_b$")

sx, sy = np.meshgrid(e_bins, Ri_bins)
e_Ri_im = ax[1,1].pcolormesh(sx, sy, e_Ri_pdf[step], cmap='viridis')
plt.colorbar(e_Ri_im, ax=ax[1,1])
ax[1,1].set_title("Unnormalised weight {0:.2E}".format(e_Ri_pdf_w[step,0]))
ax[1,1].set_xlabel("$e$")
ax[1,1].set_ylabel("$\mathrm{{Ri}}$")

sx, sy = np.meshgrid(e_bins, B_bins)
e_B_im = ax[1,2].pcolormesh(sx, sy, e_B_pdf[step], cmap='viridis')
plt.colorbar(e_B_im, ax=ax[1,2])
ax[1,2].set_title("Unnormalised weight {0:.2E}".format(e_B_pdf_w[step,0]))
ax[1,2].set_xlabel("$e$")
ax[1,2].set_ylabel("$B$")

sx, sy = np.meshgrid(Ri_bins, B_bins)
Ri_B_im = ax[1,3].pcolormesh(sx, sy, Ri_B_pdf[step], cmap='viridis')
plt.colorbar(Ri_B_im, ax=ax[1,3])
ax[1,3].set_title("Unnormalised weight {0:.2E}".format(Ri_B_pdf_w[step,0]))
ax[1,3].set_xlabel("$\mathrm{{Ri}}$")
ax[1,3].set_ylabel("$B$")

plt.show()
