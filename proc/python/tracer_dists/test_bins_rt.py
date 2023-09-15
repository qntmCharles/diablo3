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
for out_file in [join(save_dir, 'out.000290.h5')]: #sorted(glob.glob(join(save_dir,'out.*.h5'))):
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

#b_data = np.zeros(shape=(md['Nx'], md['Ny'], md['Nz']))
b_data = np.zeros(shape=(md['Nx'], md['Ny'], 136 ))
t_data = np.zeros(shape=(md['Nx'], md['Ny'], 136 ))

xis = []
yis = []
zis = []
zs = []
coords = []

with open(join(save_dir, 'output.dat'), 'r') as f:
    text = f.readlines()
    cur_time_step = 0
    for i in range(len(text)):
        strings = text[i].split()
        try:
            if len(strings) > 2 and strings[0] == "data":
                strings.extend(text[i+1].split())

                xi = int(strings[1])
                zi = int(strings[2])
                yi = int(strings[3])

                z = float(strings[4])

                coords.append(np.array([xi, yi, z]))

                tracer_val = float(strings[5])
                buoy_val = float(strings[6])

                b_data[yi,xi,zi] = buoy_val
                t_data[yi,xi,zi] = tracer_val

                if zi not in zis:
                    zis.append(zi)
                    zs.append(z)

        except Exception as e:
            print(strings)
            print(e)

coords = np.array(coords)

zis = np.array(sorted(zis))
b_source_out = b_data[:,:,zis]
t_source_out = t_data[:,:,zis]

u, c = np.unique(coords, axis=0, return_counts=True)
dups = u[c>1]
for entry in dups:
    print(entry)

print(sorted(zs))

############################################################################################################
# Source tracer vs. buoyancy distribution: calculation and diagnostics
############################################################################################################

# Create buoyancy bins
#nbins = 100
#b_min = 0
#b_max = np.max(centreline_b)/factor

bbins = bbins_rt
#print(bbins)
#print(source_dist_rt[-1])
bbins_plot = 0.5*(bbins[1:] + bbins[:-1])

##### Compute 'source' PDF #####

source_dist_fig = plt.figure()

# First get data we want to process
#top = md['H']
#depth = 0.05

top_idx = get_index(md['LZ'], gzf)
bottom_idx = get_index(md['H'], gzf)+1

b_source_3d = b_3d[:, :, bottom_idx:top_idx, :]
t_source_3d = t_3d[:, :, bottom_idx:top_idx, :]

print(gzf[bottom_idx:top_idx])
input()

source_dist_3d = []
source_dist_out = []

cols = plt.cm.rainbow(np.linspace(0, 1, len(times_rt)))
for i in range(len(times_rt)):
    if round(times_rt[i],1) == 4:
        plt.plot(source_dist_rt[i,:-1], bbins_plot, color='r', linestyle='--', label=times_rt[i])

for i in range(len(times)):
    source_dist_3d.append(compute_pdf(b_source_3d[i], t_source_3d[i], bbins))
    source_dist_out.append(compute_pdf(b_source_out, t_source_out, bbins))

for i in range(len(times)):
    #diff = t_source_out - np.swapaxes(t_source_3d[i],1,2)
    #plt.plot(source_dist_3d[i]+compute_pdf(b_source_out, diff), bbins), color='c', label="diff")
    plt.plot(source_dist_3d[i], bbins_plot, color='b', linestyle='-.', label=times[i][0])
    plt.plot(source_dist_out[i], bbins_plot, color='g', linestyle=':', label=times[i][0])


plt.legend()
plt.ylim(0, np.max(bbins))
plt.show()
