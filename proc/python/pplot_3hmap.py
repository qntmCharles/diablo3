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
    t2 = np.array([np.array(f['th3_xz'][tstep]) for tstep in time_keys])
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
centreline_b = b[0,:,int(md['Nx']/2)]
sim_bins = np.max(centreline_b[1:]-centreline_b[:-1])

b_min = 0
b_max = np.max(b[0,:,int(md['Nx']/2)])/4

nbins = int(np.floor((b_max-b_min)/sim_bins))

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

bbins = np.linspace(b_min, b_max, nbins)
bbins_data = check_data[plot_min:plot_max]
bbins_data = bbins_data[::2]
#bbins_data = np.append(bbins_data, bbins[-1])

print(np.append(times, times[-1]+md['SAVE_STATS_DT']))
X, Y = np.meshgrid(np.append(times, times[-1]+md['SAVE_STATS_DT']), bbins)

bt_pdfs = []
bt2_pdfs = []
zt_pdfs = []
zt2_pdfs = []
zb_pdfs = []

dz = gzf[1+plot_min:plot_max] - gzf[plot_min:plot_max-1]

cols = plt.cm.rainbow(np.linspace(0, 1, NSAMP))
for i,c in zip(range(NSAMP), cols):
    zb_pdfs.append(compute_pdf(gzf[plot_min:plot_max], bme[i][plot_min:plot_max], gzf[plot_min:plot_max]))
    bt_pdfs.append(compute_pdf(b[i][plot_min:plot_max], t[i][plot_min:plot_max], bbins_data))
    bt2_pdfs.append(compute_pdf(b[i][plot_min:plot_max], t2[i][plot_min:plot_max], bbins_data))
    zt_pdfs.append(compute_pdf(z_coords[plot_min:plot_max], t[i][plot_min:plot_max], gzf[plot_min:plot_max]))
    zt2_pdfs.append(compute_pdf(z_coords[plot_min:plot_max], t2[i][plot_min:plot_max],
        gzf[plot_min:plot_max]))
    plt.plot(compute_pdf(b[i][plot_min:plot_max], t2[i][plot_min:plot_max], bbins_data),
            0.5*(bbins_data[1:]+bbins_data[:-1]), color=c, label="t={0:.4f}".format(times[i]))

plt.legend()
plt.show()

bt_pdfs = np.array(bt_pdfs)
bt2_pdfs = np.array(bt2_pdfs)
zt_pdfs = np.array(zt_pdfs)
zt2_pdfs = np.array(zt2_pdfs)
zb_pdfs = np.array(zb_pdfs)

#TODO reduce noise in pdfs

bt_pdfs = np.swapaxes(bt_pdfs, axis1=1, axis2=0)
zt_pdfs = np.swapaxes(zt_pdfs, axis1=1, axis2=0)
bt2_pdfs = np.swapaxes(bt2_pdfs, axis1=1, axis2=0)
zt2_pdfs = np.swapaxes(zt2_pdfs, axis1=1, axis2=0)

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

X, Y = np.meshgrid(np.append(times, times[-1]+md['SAVE_STATS_DT']), bbins_data)
bt_im = ax[0].pcolormesh(X, Y, bt_pdfs, shading='flat', cmap=plt.cm.get_cmap('jet'))

Xz, Yz = np.meshgrid(np.append(times, times[-1]+md['SAVE_STATS_DT']), gz[plot_min:plot_max])
zt_im = ax[1].pcolormesh(Xz, Yz, zt_pdfs, cmap=plt.cm.get_cmap('jet'))

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

fig2, ax2 = plt.subplots(1,2)

tcontours = np.append(times, times[-1]+md['SAVE_STATS_DT'])
tcontours = 0.5*(tcontours[1:] + tcontours[:-1])

X2, Y2 = np.meshgrid(np.append(times, times[-1]+md['SAVE_STATS_DT']), bbins_data)
X2f, Y2f = np.meshgrid(tcontours, 0.5*(bbins_data[1:]+bbins_data[:-1]))
bt2_im = ax2[0].pcolormesh(X2, Y2, bt2_pdfs, shading='flat', cmap=plt.cm.get_cmap('jet'))
zt2_im = ax2[1].pcolormesh(Xz, Yz, zt2_pdfs, cmap=plt.cm.get_cmap('jet'))
ax2[0].contour(X2f, Y2f, bt2_pdfs, 5, colors='white')

t_step = int(round(5.0 / md['SAVE_STATS_DT'], 0))
for i in range(0, NSAMP, t_step):
    ax2[0].axvline(times[i], linestyle='--', color='gray', alpha=0.5)
    ax2[1].axvline(times[i], linestyle='--', color='gray', alpha=0.5)

ax2[0].set_title("PDF of tracer vs. buoyancy")
ax2[0].set_xlabel("time (sec)")
ax2[0].set_ylabel("buoyancy")
ax2[1].set_title("PDF of tracer vs. height")
ax2[1].set_xlabel("time (sec)")
ax2[1].set_ylabel("height")

plt.show()
