import h5py, gc, sys
import numpy as np
from scipy.interpolate import griddata
from scipy import integrate
import matplotlib
from datetime import datetime
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from os.path import join, isfile
from os import listdir
from functions import get_metadata, read_params, get_grid

def compute_pdf(data, ref, bins, db, normalised=False):
    out_bins = [0 for i in range(len(bins)-1)]

    for i in range(len(bins)-1):
        out_bins[i] = np.sum(np.where(np.logical_and(data >= bins[i],
            data < bins[i+1]), ref, 0))

    out_bins = np.array(out_bins)

    if normalised:
        area = integrate.trapezoid(np.abs(out_bins), dx = db)
        return out_bins/area
    else:
        return out_bins

##### USER-DEFINED PARAMETERS #####
params_file = "params.dat"

save = False

##### ----------------------- #####

##### Get directory locations #####
base_dir, run_dir, save_dir, version = read_params(params_file)
print("Run directory: ", run_dir)
print("Save directory: ", save_dir)

##### Get simulation metadata #####
md = get_metadata(run_dir, version)
print("Complete metadata: ",md)

##### Get grid #####
gxf, gyf, gzf, dzf = get_grid(join(run_dir, 'grid.h5'), md)
gx, gy, gz, dz = get_grid(join(run_dir, 'grid.h5'), md, fractional_grid=False)

z_coords, x_coords = np.meshgrid(gzf, gxf, indexing='ij', sparse=True)


##### Get data #####
with h5py.File(save_dir+"/movie.h5", 'r') as f:
    print("Movie keys: %s" % f.keys())
    time_keys = list(f['th1_xz'])
    print(time_keys)
    # Get buoyancy data
    b = np.array([np.array(f['th1_xz'][tstep]) for tstep in time_keys])
    t = np.array([np.array(f['th2_xz'][tstep]) for tstep in time_keys])
    NSAMP = len(b)
    #times = np.array([float(tstep)*md['SAVE_MOVIE_DT'] for tstep in time_keys])
    times = np.array([float(f['th1_xz'][tstep].attrs['Time']) for tstep in time_keys])
    f.close()

with h5py.File(save_dir+"/mean.h5", 'r') as f:
    print("Mean keys: %s" % f.keys())
    mtime_keys = list(f['thme01'])
    print(mtime_keys)
    # Get data
    bme = np.array([np.array(f['thme01'][tstep]) for tstep in mtime_keys])
    mNSAMP = len(b)
    assert mNSAMP == NSAMP
    f.close()

# b contains values of buoyancy at each point
# z_coords contains z value at each point
# t contains tracer value at each point
## Aim is to bin tracer values wrt buoyancy

# Create buoyancy bins
b_min = 0
b_max = np.max(b[0,:,int(md['Nx']/2)])/4
nbins = 129

check_data = b[0,:,int(md['Nx']/2)]
plot_min = -1
plot_max = -1
for j in range(md['Nz']):
    if gzf[j] < md['H'] and gzf[j+1] > md['H']:
        plot_min = j
    if check_data[j-1] <= b_max and check_data[j] >= b_max:
        plot_max = j

if plot_min == -1: print("Warning: plot_min miscalculated")
if plot_max == -1: print("Warning: plot_max miscalculated")

bbins, db = np.linspace(b_min, b_max, nbins, retstep=True)

print(np.append(times, times[-1]+md['SAVE_STATS_DT']))
X, Y = np.meshgrid(np.append(times, times[-1]+md['SAVE_STATS_DT']), bbins)

bt_pdfs = []
zt_pdfs = []
zb_pdfs = []

dz = gzf[1+plot_min:plot_max] - gzf[plot_min:plot_max-1]

for i in range(NSAMP):
    zb_pdfs.append(compute_pdf(gzf[plot_min:plot_max], bme[i][plot_min:plot_max], gzf[plot_min:plot_max],
        dz))
    bt_pdfs.append(compute_pdf(b[i][plot_min:plot_max], t[i][plot_min:plot_max], bbins, db))
    zt_pdfs.append(compute_pdf(z_coords[plot_min:plot_max], t[i][plot_min:plot_max], gzf[plot_min:plot_max],
        dz))

bt_pdfs = np.array(bt_pdfs)
zt_pdfs = np.array(zt_pdfs)
zb_pdfs = np.array(zb_pdfs)

bt_pdfs = np.swapaxes(bt_pdfs, axis1=1, axis2=0)
zt_pdfs = np.swapaxes(zt_pdfs, axis1=1, axis2=0)

X, Y = np.meshgrid(np.append(times, times[-1]+md['SAVE_STATS_DT']), gz[plot_min:plot_max])
cols = plt.cm.rainbow(np.linspace(0, 1, NSAMP))
fig = plt.figure()
for pdf,c in zip(zb_pdfs, cols):
    plt.plot(pdf, gzf[plot_min:plot_max-1], color=c)

plt.xlim(0, b_max)
plt.ylim(gzf[plot_min], gzf[plot_max])
plt.xlabel("buoyancy")
plt.ylabel("height")

bfig = plt.figure()
plt.plot(b[0,:,0], gzf)

fig, ax = plt.subplots(1,2)

X, Y = np.meshgrid(np.append(times, times[-1]+md['SAVE_STATS_DT']), bbins)
bt_im = ax[0].pcolormesh(X, Y, bt_pdfs, shading='flat', cmap=plt.cm.get_cmap('PuBuGn'))

X, Y = np.meshgrid(np.append(times, times[-1]+md['SAVE_STATS_DT']), gz[plot_min:plot_max])
zt_im = ax[1].pcolormesh(X, Y, zt_pdfs, cmap=plt.cm.get_cmap('PuBuGn'))

t_step = int(round(5.0 / md['SAVE_STATS_DT'], 0))
for i in range(0, NSAMP, t_step):
    ax[0].axvline(times[i], linestyle='--', color='gray', alpha=0.5)
    ax[1].axvline(times[i], linestyle='--', color='gray', alpha=0.5)

ax[0].set_title("PDF of tracer vs. buoyancy")
ax[0].set_xlabel("time (sec)")
ax[0].set_ylabel("buoyancy")
ax[1].set_title("PDF of tracer vs. height")
ax[1].set_xlabel("time (sec)")
ax[1].set_ylabel("height")

plt.show()
