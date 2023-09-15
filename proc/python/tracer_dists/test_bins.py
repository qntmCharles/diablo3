import h5py, gc, sys, glob
import numpy as np
from scipy.interpolate import griddata
from scipy import integrate
import scipy.special as sc
import matplotlib
from datetime import datetime
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from os.path import join, isfile
from os import listdir
from functions import get_metadata, read_params, get_grid

def get_index(z, griddata):
    return int(np.argmin(np.abs(griddata - z)))

def compute_pdf(data, ref, bins, normalised=False):
    out_bins = [0 for i in range(len(bins)-1)]
    for i in range(len(bins)-1):
        out_bins[i] = np.sum(np.where(np.logical_and(data >= bins[i],
            data < bins[i+1]), ref, 0))

    out_bins = np.array(out_bins)

    if normalised:
        area = integrate.trapezoid(np.abs(out_bins), 0.5*(bins[1:]+bins[:-1]))
        return out_bins/area
    else:
        return out_bins

##### USER-DEFINED PARAMETERS #####
params_file = "params.dat"
#out_file = "out.000129.h5"

save = False

##### ----------------------- #####

##### Get directory locations #####
base_dir, run_dir, save_dir, version = read_params(params_file)
print("Run directory: ", run_dir)
print("Save directory: ", save_dir)

##### Get simulation metadata #####
md = get_metadata(run_dir, version)
print("Complete metadata: ",md)

# Calculate turnover time
F = md['b0'] * (md['r0']**2)
tau = md['r0']**(4/3) * F**(-1/3)
print("Non-dimensional turnover time: {0:.04f}".format(tau))

##### Get grid #####
gxf, gyf, gzf, dzf = get_grid(join(run_dir, 'grid.h5'), md)
gx, gy, gz, dz = get_grid(join(run_dir, 'grid.h5'), md, fractional_grid=False)


##### Get data and time indices #####
times = []
b_3d = []
t_3d = []
for out_file in sorted(glob.glob(join(save_dir,'out.*.h5'))):
    print(out_file)
    with h5py.File(join(save_dir, out_file), 'r') as f:
        times.append(np.array([f['Timestep'].attrs['Time']]))
        b_3d.append(np.array(f['Timestep']['TH1']))
        t_3d.append(np.array(f['Timestep']['TH2']))
        f.close()

b_3d = np.array(b_3d)
t_3d = np.array(t_3d)

with h5py.File(join(save_dir,"mean.h5"), 'r') as f:
    print("Mean keys: %s" % f.keys())
    time_keys = list(f['tb_strat'].keys())
    print(time_keys)
    #bbins_rt = np.array([f['bbins'][tkey] for tkey in time_keys])
    bbins_rt = np.array(f['bbins'][time_keys[-1]])
    source_dist_rt = np.array([f['tb_strat'][tkey] for tkey in time_keys])
    times_rt = np.array([float(f['tb_strat'][tkey].attrs['Time']) for tkey in time_keys])
    print(bbins_rt.shape)
    print(source_dist_rt.shape)

    f.close()

############################################################################################################
# Source tracer vs. buoyancy distribution: calculation and diagnostics
############################################################################################################

# Create buoyancy bins
#nbins = 100
#b_min = 0
#b_max = np.max(centreline_b)/factor

bbins = bbins_rt
print(bbins)
print(source_dist_rt[-1])
bbins_plot = 0.5*(bbins[1:] + bbins[:-1])

##### Compute 'source' PDF #####

source_dist_fig = plt.figure()

# First get data we want to process
#top = md['H']
#depth = 0.05

bottom_idx = get_index(md['H'], gzf)+1

b_source_3d = b_3d[:, :, bottom_idx:, :]
t_source_3d = t_3d[:, :, bottom_idx:, :]

source_dist_3d = []
for i in range(len(times)):
    source_dist_3d.append(compute_pdf(b_source_3d[i], t_source_3d[i], bbins))

cols = plt.cm.rainbow(np.linspace(0, 1, len(times)))
for i in range(len(times)):
    plt.plot(source_dist_3d[i], bbins_plot, color='b', linestyle=':', label=times[i][0])
#for i in range(len(time_keys)):
    plt.plot(source_dist_rt[i+1,:-1], bbins_plot, color='r', linestyle='--', label=times_rt[i+1])
    plt.legend()
    plt.ylim(0, 0.1)
    plt.show()

plt.legend()
#ax[0].set_title("3D")
#ax[1].set_title("Runtime")
#ax[0].set_xlabel("Tracer (normalised)")
#ax[0].set_ylabel("Buoyancy")
#ax[1].set_xlabel("Tracer (normalised)")
#ax[1].set_ylabel("Buoyancy")

##### Plot buoyancy and tracer fields with source region #####

source_bt_fig, ax = plt.subplots(1,2, figsize=(10, 3))
Xplot, Yplot = np.meshgrid(gx, gz)
ax[0].pcolormesh(Xplot, Yplot, np.mean(b_3d[:,int(md['Nx']/2)], axis=0))
ax[1].pcolormesh(Xplot, Yplot, np.mean(t_3d[:,int(md['Nx']/2)], axis=0))

ax[0].axhline(top, color='r', linestyle='--')
ax[0].axhline(top-depth, color='r', linestyle='--')
ax[1].axhline(top, color='r', linestyle='--')
ax[1].axhline(top-depth, color='r', linestyle='--')

plt.tight_layout()
plt.show()
