import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py
import numpy as np
from os.path import join
import matplotlib
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

bmin = 0
bmax = md['b_factor']*md['N2']*(md['LZ']-md['H'])
print(bmax)

F0 = md['b0']*(md['r0']**2)
alpha = md['alpha_e']

tmin = 5e-4
tmax = md['phi_factor']*5*F0 / (3 * alpha) * np.power(0.9*alpha*F0, -1/3) * np.power(
        md['H']+ 5*md['r0']/(6*alpha), -5/3)

Nb = int(md['Nb'])
Nt = int(md['Nphi'])
db = (bmax - bmin)/Nb
dt = (tmax - tmin)/Nt
dx = md['LX']/md['Nx']
dy = md['LY']/md['Ny']
bbins = np.array([bmin + (i+0.5)*db for i in range(Nb)])
tbins = np.array([tmin + (i+0.5)*dt for i in range(Nt)])
sx, sy = np.meshgrid(bbins, tbins)

print("Complete metadata: ", md)

fields = ["Ri", "chi", "Re_b", "tke", "e"]
field_mins = [-1, 0, -4, 0, -2e-3]
field_maxs = [2, 2e-8, 15, 1e-7, 2e-3]

with h5py.File(join(save_dir,"end.h5"), 'r') as f:
    print("Keys: %s" % f['Timestep'].keys())

    phi = np.array(f['Timestep']['TH2'])
    phi = phi[:, idx_min:idx_max, :] # restrict to stratified layer

    b = np.array(f['Timestep']['TH1'])
    b = b[:, idx_min:idx_max, :] # restrict to stratified layer

    f.close()

with h5py.File(join(save_dir, "movie.h5"), 'r') as f:
    print("Keys: %s" % f.keys())
    time_keys = list(f['th1_xz'])
    print(time_keys)

    svd = np.array([np.array(f['svd'][t]) for t in time_keys])

    NSAMP = len(svd)

svd = np.where(svd == 0, np.nan, svd)

#HOW CAN THIS BE MADE FASTER? does it need to be... not exactly using 3D often!
#Maybe add this SVD 'coloured' field to output??
test_array = np.zeros_like(phi)

for k in range(int(md['Nb'])):
    print(k)
    for j in range(int(md['Nphi'])):
        test_array = np.where(np.logical_and(np.logical_and(b > bbins[k] - db/2,
            b <= bbins[k] + db/2),np.logical_and(phi > tbins[j] - dt/2,
            phi <= tbins[j] + dt/2)), svd[-1, j, k], test_array)

test_array = np.where(test_array == 0, np.nan, test_array)

for i in range(len(fields)):
    with h5py.File(join(save_dir,"end.h5"), 'r') as f:
        field = np.array(f['Timestep'][fields[i]])
        field = field[:, idx_min:idx_max, :] # restrict to stratified layer
        field = np.where(phi > 0, field, np.nan) # restrict to plume
        print(field.shape)

        f.close()

    with h5py.File(join(save_dir, "mean.h5"), 'r') as f:
        print("Keys: %s" % f.keys())
        time_keys = list(f[list(f.keys())[0]])
        print(time_keys)

        pdf_plume = np.array([np.array(f[fields[i]+'_pdf_plume'][t]) for t in time_keys])
        pdf_mixed = np.array([np.array(f[fields[i]+'_pdf_mixed'][t]) for t in time_keys])
        pdf_mixing = np.array([np.array(f[fields[i]+'_pdf_mixing'][t]) for t in time_keys])
        print(pdf_plume.shape)

        nbin = 64
        dpdf = (field_maxs[i] - field_mins[i]) / (nbin - 1)
        pdf_bins = np.array([field_mins[i] + j*dpdf for j in range(nbin)])

        f.close()

    field_mixed = np.where(test_array > 0.0005, field, np.nan)
    field_plume = np.where(-test_array > 0.0005, field, np.nan)
    field_mixing = np.where(-np.abs(test_array) > -0.0005, field, np.nan)

    pdf_plot = (pdf_bins[1:] + pdf_bins[:-1])/2

    fig, ax = plt.subplots(1,3)

    ax[0].plot(pdf_plot, pdf_plume[-1, :-1], color='r')
    ax[0].hist(field_plume.flatten(), bins=64, range=(field_mins[i], field_maxs[i]),
            weights = volume.flatten(), density=True)

    ax[1].plot(pdf_plot, pdf_mixed[-1, :-1], color='r')
    ax[1].hist(field_mixed.flatten(), bins=64, range=(field_mins[i], field_maxs[i]),
            weights = volume.flatten(), density=True)

    ax[2].plot(pdf_plot, pdf_mixing[-1, :-1], color='r')
    ax[2].hist(field_mixing.flatten(), bins=64, range=(field_mins[i], field_maxs[i]),
            weights = volume.flatten(), density=True)

    plt.show()
