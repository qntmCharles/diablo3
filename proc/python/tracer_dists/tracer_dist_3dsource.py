import h5py, gc, sys
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

def compute_pdf(data, ref, bins, normalised=False, correction=False, radii=None):
    out_bins = [0 for i in range(len(bins)-1)]
    for i in range(len(bins)-1):
        if correction:
            out_bins[i] = np.sum(np.where(np.logical_and(data >= bins[i],
                data < bins[i+1]), ref*radii, 0))
        else:
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
#out_file = "out.002241.h5"
out_file = "end.h5"

save = False
alpha = 0.1
zv = 0.05

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
with h5py.File(join(save_dir, out_file), 'r') as f:
    times = np.array([f['Timestep'].attrs['Time']])
    b_3d = np.array([f['Timestep']['TH1']])
    t_3d = np.array([f['Timestep']['TH2']])
    b_3d = g2gf_1d(md, b_3d)
    t_3d = g2gf_1d(md, t_3d)
    f.close()

with h5py.File(join(save_dir, 'az_stats.h5'), 'r') as f:
    print("Az keys: %s" % f.keys())
    time_keys = list(f['b_az'])
    #print(time_keys)
    # Get buoyancy data
    b_az = np.array([np.array(f['b_az'][t]) for t in time_keys])
    t_az = np.array([np.array(f['th_az'][t]) for t in time_keys])
    times = np.array([float(f['b_az'][tstep].attrs['Time']) for tstep in time_keys])
    f.close()


with h5py.File(join(save_dir,"movie.h5"), 'r') as f:
    print("Movie keys: %s" % f.keys())
    time_keys = list(f['th1_xz'])
    #print(time_keys)
    # Get buoyancy data
    b_2d = np.array([np.array(f['th1_xz'][tstep]) for tstep in time_keys])
    t_2d = np.array([np.array(f['th2_xz'][tstep]) for tstep in time_keys])
    b_2d = g2gf_1d(md, b_2d)
    t_2d = g2gf_1d(md, t_2d)
    NSAMP = len(b_2d)
    times = np.array([float(f['th1_xz'][tstep].attrs['Time']) for tstep in time_keys])
    f.close()

# Compute time indices
nplot = 3
interval = 300

step = np.round(interval*tau / md['SAVE_STATS_DT'])

t_inds = list(map(int,step*np.array(range(1, nplot+1))))

tplot = [times[i] for i in t_inds]

print("Plotting at times: ",tplot)
print("with interval", step*md['SAVE_STATS_DT'])


############################################################################################################
# Source tracer vs. buoyancy distribution: calculation and diagnostics
############################################################################################################

# Create buoyancy bins
centreline_b = b_2d[0,:,int(md['Nx']/2)]
sim_bins = np.max(centreline_b[1:]-centreline_b[:-1])

b_min = 0
factor = 8
b_max = np.max(centreline_b)/factor

nbins = int(np.floor((b_max-b_min)/sim_bins))

plot_min = -1
plot_max = -1
for j in range(md['Nz']):
    if gzf[j] < md['H'] and gzf[j+1] > md['H']:
        plot_min = j
    if centreline_b[j-1] <= b_max and centreline_b[j] >= b_max:
        plot_max = j

if plot_min == -1: print("Warning: plot_min miscalculated")
if plot_max == -1: print("Warning: plot_max miscalculated")

bbins = np.linspace(b_min, b_max, nbins)
bbins_data = centreline_b[plot_min:plot_max]
bbins = bbins_data

bbins_plot = 0.5*(bbins[1:] + bbins[:-1])

##### Compute 'source' PDF #####

source_dist_fig, ax = plt.subplots(1,4)

# First get data we want to process
top = 0.95*md['H']
depth = 10*md['r0']

top_idx = get_index(top, gzf)
bottom_idx = get_index(top-depth, gzf)

b_source_3d = b_3d[:, :, bottom_idx:top_idx, :]
t_source_3d = t_3d[:, :, bottom_idx:top_idx, :]

b_source_az = b_az[:,bottom_idx:top_idx, :]
t_source_az = t_az[:,bottom_idx:top_idx, :]

b_source_2d = b_2d[:, bottom_idx:top_idx]
t_source_2d = t_2d[:, bottom_idx:top_idx]

z_2d, r_2d = np.meshgrid(gzf[bottom_idx:top_idx], np.abs(gxf-md['LX']/2), indexing='ij')
rms_2d = 1.2 * alpha * (z_2d-zv)
rnorms_2d = r_2d/rms_2d

z_az, r_az = np.meshgrid(gzf[bottom_idx:top_idx], gxf[:int(md['Nx']/2)], indexing='ij')
rms_az = 1.2 * alpha * (z_az-zv)
rnorms_az = r_az/rms_az

# Averaging period (s)
t_start = 10
t_end = 20
start_idx = get_index(t_start, times)
end_idx = get_index(t_end, times)

source_dist_3d = []
source_dist_az = []
source_dist_2d = []

cols = plt.cm.rainbow(np.linspace(0, 1, end_idx-start_idx))
for i, c in zip(range(start_idx, end_idx), cols):
    source_dist_az.append(compute_pdf(b_source_az[i], t_source_az[i], bbins, normalised=True,correction=True,
        radii = r_az))
    source_dist_2d.append(compute_pdf(b_source_2d[i], t_source_2d[i], bbins, normalised=True,correction=True,
        radii = r_2d))

    ax[0].plot(source_dist_2d[i-start_idx], bbins_plot, color=c, alpha=0.5)
    ax[1].plot(source_dist_az[i-start_idx], bbins_plot, color=c, alpha=0.5)

source_dist_az = np.nanmean(source_dist_az, axis=0)
source_dist_2d = np.nanmean(source_dist_2d, axis=0)
source_dist_3d = compute_pdf(b_source_3d[0], t_source_3d[0], bbins, normalised=True)

ax[0].plot(source_dist_2d, bbins_plot, color='k', linestyle='--')
ax[0].set_title("XZ cross-section (2D)")
ax[1].plot(source_dist_az, bbins_plot, color='k', linestyle='--')
ax[1].set_title("Azimuthal average (2D)")
ax[2].plot(source_dist_3d, bbins_plot, color='k', linestyle='--')
ax[2].set_title("Full 3D")

ax[3].plot(source_dist_3d, bbins_plot, color='r', label="3D")
ax[3].plot(source_dist_az, bbins_plot, color='g', label="az")
ax[3].plot(source_dist_2d, bbins_plot, color='b', label="2d")
ax[3].legend()
ax[3].set_title("Comparison")
#plt.xlabel("Tracer (normalised)")
#plt.ylabel("Buoyancy")

##### Plot buoyancy and tracer fields with source region #####

source_bt_fig, ax = plt.subplots(2,3, figsize=(10, 3))
Xplot, Yplot = np.meshgrid(gx, gz[:plot_max+1])
Xplot_az, Yplot_az = np.meshgrid(gx[:int(md['Nx']/2)+1], gz[:plot_max+1])

ax[0,0].pcolormesh(Xplot, Yplot, np.mean(b_2d[:,:plot_max],axis=0))
ax[1,0].pcolormesh(Xplot, Yplot, np.mean(t_2d[:,:plot_max],axis=0))

ax[0,1].pcolormesh(Xplot_az, Yplot_az, np.mean(b_az[:,:plot_max],axis=0))
ax[1,1].pcolormesh(Xplot_az, Yplot_az, np.mean(t_az[:,:plot_max],axis=0))

ax[0,2].pcolormesh(Xplot, Yplot, b_3d[0,int(md['Nx']/2),:plot_max])
ax[1,2].pcolormesh(Xplot, Yplot, t_3d[0,int(md['Nx']/2),:plot_max])

#ax[0].set_title("Buoyancy field")
#ax[1].set_title("Tracer field")
ax[0,0].axhline(top, color='r', linestyle='--')
ax[1,0].axhline(top, color='r', linestyle='--')
ax[0,1].axhline(top, color='r', linestyle='--')
ax[1,1].axhline(top, color='r', linestyle='--')
ax[0,2].axhline(top, color='r', linestyle='--')
ax[1,2].axhline(top, color='r', linestyle='--')

ax[0,0].axhline(top-depth, color='r', linestyle='--')
ax[1,0].axhline(top-depth, color='r', linestyle='--')
ax[0,1].axhline(top-depth, color='r', linestyle='--')
ax[1,1].axhline(top-depth, color='r', linestyle='--')
ax[0,2].axhline(top-depth, color='r', linestyle='--')
ax[1,2].axhline(top-depth, color='r', linestyle='--')
plt.tight_layout()

##### Generate tracer vs. buoyancy heatmap

source_hmap_fig, ax = plt.subplots(2,2, figsize=(15,5))

bt_dists_2d = []
bt_dists_source_2d = []
bt_dists_source_var_2d = []

bt_dists_az = []
bt_dists_source_az = []
bt_dists_source_var_az = []

for i in range(1,NSAMP):
    bt_dists_2d.append(compute_pdf(b_2d[i][plot_min:plot_max], t_2d[i][plot_min:plot_max], bbins, normalised=True))
    bt_dists_az.append(compute_pdf(b_az[i][plot_min:plot_max], t_az[i][plot_min:plot_max], bbins, normalised=True))
    bt_dists_source_2d.append(compute_pdf(b_source_2d[i], t_source_2d[i], bbins, normalised=True))
    bt_dists_source_az.append(compute_pdf(b_source_az[i], t_source_az[i], bbins, normalised=True))

Xhmap, Yhmap = np.meshgrid(np.append(times[1:], times[-1]+md['SAVE_STATS_DT']), bbins)

bt_dists_2d = np.moveaxis(bt_dists_2d, 0, 1)
bt_dists_az = np.moveaxis(bt_dists_az, 0, 1)

for i in range(len(bt_dists_source_2d)):
    bt_dists_source_var_2d.append(bt_dists_source_2d[i] - source_dist_2d)
    bt_dists_source_var_az.append(bt_dists_source_az[i] - source_dist_az)

bt_dists_source_2d = np.moveaxis(bt_dists_source_2d, 0, 1)
bt_dists_source_var_2d = np.moveaxis(bt_dists_source_var_2d, 0, 1)

bt_dists_source_az = np.moveaxis(bt_dists_source_az, 0, 1)
bt_dists_source_var_az = np.moveaxis(bt_dists_source_var_az, 0, 1)


bt_hmap_2d = ax[0,0].pcolormesh(Xhmap, Yhmap, bt_dists_source_2d, shading='flat')
ax[0,0].axvline(t_start, color='r', linestyle='--')
ax[0,0].axvline(t_end, color='r', linestyle='--')
bt_hmap_2d.set_clim(0, 40)
ax[0,0].set_title("Tracer vs. buoyancy heatmap")
ax[0,0].set_xlabel("Time (s)")
ax[0,0].set_ylabel("Buoyancy")

bt_hmap_var_2d = ax[1,0].pcolormesh(Xhmap, Yhmap, bt_dists_source_var_2d, shading='flat')
ax[1,0].axvline(t_start, color='r', linestyle='--')
ax[1,0].axvline(t_end, color='r', linestyle='--')
bt_hmap_var_2d.set_clim(0, 10)
ax[1,0].set_title("Tracer vs. buoyancy heatmap with source distribution subtracted")
ax[1,0].set_xlabel("Time (s)")
ax[1,0].set_ylabel("Buoyancy")

bt_hmap_az = ax[0,1].pcolormesh(Xhmap, Yhmap, bt_dists_source_az, shading='flat')
ax[0,1].axvline(t_start, color='r', linestyle='--')
ax[0,1].axvline(t_end, color='r', linestyle='--')
bt_hmap_az.set_clim(0, 40)
ax[0,1].set_title("Tracer vs. buoyancy heatmap (az)")
ax[0,1].set_xlabel("Time (s)")
ax[0,1].set_ylabel("Buoyancy")

bt_hmap_var_az = ax[1,1].pcolormesh(Xhmap, Yhmap, bt_dists_source_var_az, shading='flat')
ax[1,1].axvline(t_start, color='r', linestyle='--')
ax[1,1].axvline(t_end, color='r', linestyle='--')
bt_hmap_var_az.set_clim(0, 10)
ax[1,1].set_title("Tracer vs. buoyancy heatmap with source distribution subtracted (az)")
ax[1,1].set_xlabel("Time (s)")
ax[1,1].set_ylabel("Buoyancy")

plt.tight_layout()

############################################################################################################
# z_max, z_ss and z_eq calculation
############################################################################################################

zmax = 0.2454
zeq = 0.20778

b_zmax = md['N2']*(zmax-md['H'])
b_zeq = md['N2']*(zeq-md['H'])

############################################################################################################
# Tracer vs. buoyancy distribution plots
############################################################################################################

##### Set up plot #####
t_cont = 0.005
X, Y = np.meshgrid(gx, gz[plot_min:plot_max+1])
Xf, Yf = np.meshgrid(gxf, gzf[plot_min:plot_max])
Xf_az, Yf_az = np.meshgrid(gxf[int(md['Nx']/2):], gzf[plot_min:plot_max])
tcols = plt.cm.OrRd(np.linspace(0,1,nplot+1))[1:]

#fig = plt.figure()
fig, ax = plt.subplots(1,2)

ax[0].pcolormesh(X, Y, b_2d[0][plot_min:plot_max], cmap=plt.cm.get_cmap('jet'), alpha=0.3)
ax[1].plot(source_dist_2d, 0.5*(bbins[1:]+bbins[:-1]), color='k', linestyle='--')
ax[1].plot(source_dist_az, 0.5*(bbins[1:]+bbins[:-1]), color='k', linestyle=':')
ax[1].plot(source_dist_3d, 0.5*(bbins[1:]+bbins[:-1]), color='k', linestyle='-.')

for step,c in zip(t_inds, tcols):
    ax[0].contour(Xf, Yf, t_2d[step][plot_min:plot_max], levels=[t_cont], colors=[c])
    ax[0].contour(Xf_az, Yf_az, t_az[step][plot_min:plot_max], levels=[t_cont], colors=[c], linestyles=['--'])
    ax[0].contour(md['LX']-Xf_az, Yf_az, t_az[step][plot_min:plot_max], levels=[t_cont], colors=[c], linestyles=['--'])
    b_pdf_2d = compute_pdf(b_2d[step][plot_min:plot_max], t_2d[step][plot_min:plot_max], bbins, normalised=True)
    b_pdf_az = compute_pdf(b_az[step][plot_min:plot_max], t_az[step][plot_min:plot_max], bbins, normalised=True)

    ax[1].plot(b_pdf_2d, 0.5*(bbins[1:]+bbins[:-1]), color=c, label = "t={0:.3f} s".format(times[step]))
    ax[1].plot(b_pdf_az, 0.5*(bbins[1:]+bbins[:-1]), color=c, linestyle='--',
            label = "t={0:.3f} s".format(times[step]))

ax[1].axhline(b_zmax, color='r', linestyle='--')
ax[1].axhline(b_zeq, color='r', linestyle='--')

plt.xlabel("tracer (arbitrary units, normalised)")
plt.ylabel("buoyancy ($m \, s^{{-2}}$)")
plt.legend()
plt.tight_layout()
plt.show()
