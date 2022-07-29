import h5py
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from functions import get_metadata, read_params, get_grid

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"

zn = 0.164790778

##### ---------------------- #####

def compute_pdf(data, ref, bins):
    out_bins = [0 for i in range(len(bins)-1)]

    for i in range(len(bins)-1):
        out_bins[i] = np.sum(np.where(np.logical_and(data >= bins[i],
            data < bins[i+1]), ref, 0))

    out_bins = np.array(out_bins)

    return out_bins

def get_index(z, griddata):
    return int(np.argmin(np.abs(griddata - z)))

# Get dir locations from param file
base_dir, run_dir, save_dir, version = read_params(params_file)

# Get simulation metadata
md = get_metadata(run_dir, version)
gxf, gyf, gzf, dzf = get_grid(save_dir+"/grid.h5", md)
z_coords, x_coords = np.meshgrid(gzf, gxf, indexing='ij', sparse=True)
gx, gy, gz, dz = get_grid(save_dir+"/grid.h5", md, fractional_grid=False)

X, Y = np.meshgrid(gx, gz)
Xf, Yf = np.meshgrid(gxf, gzf)

print("Complete metadata: ", md)

# Get data
with h5py.File(save_dir+"/movie.h5", 'r') as f:
    print("Keys: %s" % f.keys())
    time_keys = list(f['th1_xz'])
    print(time_keys)
    # Get buoyancy data
    th1_xz = np.array([np.array(f['th1_xz'][t]) for t in time_keys])
    th2_xz = np.array([np.array(f['th2_xz'][t]) for t in time_keys])
    NSAMP = len(th1_xz)
    times = np.array([float(t)*md['SAVE_MOVIE_DT'] for t in time_keys])
    f.close()

#contour_lvls_b = np.linspace(0.01, np.max(th1_xz[0]), 30)
#contour_lvls_trace = np.linspace(0.01, 0.1, 8)

#c_b = axs[0].contour(Xf, Yf, th1_xz[step], levels=contour_lvls_b, colors='white', alpha = 0.5)
#c_trace = axs[1].contour(Xf, Yf, th2_xz[step], levels=contour_lvls_trace, colors='white', alpha = 0.5)

tracer_thresh = 5e-3

zn_index = get_index(zn, gz)

tracer_data_horiz = th2_xz[1:, zn_index, :]
tracer_data_vert = th2_xz[1:int(len(times)/3), :, int(md['Nx']/2)]

plume_horiz = np.where(tracer_data_horiz > tracer_thresh, 1, 0)
plume_vert = np.where(tracer_data_vert > tracer_thresh, 1, 0)

width_l = []
width_r = []
for i in range(len(plume_horiz)):
    stuff = np.where(plume_horiz[i] == 1)[0]
    if len(stuff) == 0:
        width_r.append(0)
        width_l.append(0)
    else:
        width_l.append(np.max(np.where(plume_horiz[i] == 1)) * md['LX']/md['Nx'] - md['LX']/2)
        width_r.append(md['LX']/2 - np.min(np.where(plume_horiz[i] == 1)) * md['LX']/md['Nx'])

heights = []
for i in range(len(plume_vert)):
    stuff = np.where(plume_vert[i] == 1)[0]
    if len(stuff) == 0:
        heights.append(0)
    else:
        heights.append(gzf[np.max(stuff)])

##### Compute PDF #####
# Create buoyancy bins
b_min = 0
b_max = np.max(th1_xz[0,:,int(md['Nx']/2)])/4
nbins = 129

check_data = th1_xz[0,:,int(md['Nx']/2)]
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
bins = [0 for i in range(nbins)]

zt_pdf = compute_pdf(z_coords[plot_min:plot_max], th2_xz[-1][plot_min:plot_max], gzf[plot_min:plot_max])


fig, ax = plt.subplots(1,2)
X, Y = np.meshgrid(gx, times)
ax[0].pcolormesh(X, Y, plume_horiz)
ax[1].scatter(width_l, 0.5*(times[1:]+times[:-1]), color='b', label="right")
ax[1].scatter(width_r, 0.5*(times[1:]+times[:-1]), color='r', label="left")
ax[1].legend()
v_l = np.gradient(width_l, 0.5*(times[1:]+times[:-1]))
v_r = np.gradient(width_r, 0.5*(times[1:]+times[:-1]))
t_min = get_index(5, 0.5*(times[1:]+times[:-1]))
t_max = get_index(10, 0.5*(times[1:]+times[:-1]))
v_l_mean = np.mean(v_l[t_min:t_max])
v_r_mean = np.mean(v_r[t_min:t_max])
print("Initial intrusion speed: ", 0.5*(v_l_mean+v_r_mean))

fig1, ax1 = plt.subplots(1,2)
X, Y = np.meshgrid(times[:int(len(times)/3)], gz)
ax1[0].pcolormesh(X,Y, np.swapaxes(plume_vert,0,1))
ax1[1].plot(times[1:int(len(times)/3)], heights)
zmax = np.max(heights)
print("Max penetration height: ", zmax)

fig2 = plt.figure()
plt.plot(zt_pdf, gz[plot_min:plot_max-1])
zn = gz[plot_min:plot_max][np.argmax(zt_pdf)]
print("Neutral buoyancy height: ", zn)

plt.show()
