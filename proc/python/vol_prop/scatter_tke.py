# Script loads in 2D slices and produces a movie of the simulation output import numpy as np import h5py, gc
import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
from functions import get_metadata, read_params, get_grid, g2gf_1d
from scipy import ndimage, interpolate, spatial

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"
convex_hull = True

##### ---------------------- #####

def get_index(z, griddata):
    return int(np.floor(np.argmin(np.abs(griddata - z))))

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
    time_keys = list(f['th1_xz'])
    print(time_keys)
    # Get buoyancy data
    t = np.array([np.array(f['th2_xz'][t]) for t in time_keys])
    t = g2gf_1d(md, t)
    b = np.array([np.array(f['th1_xz'][t]) for t in time_keys])
    b = g2gf_1d(md, b)

    u = np.array([np.array(f['u_xz'][t]) for t in time_keys])
    u = g2gf_1d(md,u)
    v = np.array([np.array(f['v_xz'][t]) for t in time_keys])
    v = g2gf_1d(md,v)
    w = np.array([np.array(f['w_xz'][t]) for t in time_keys])
    w = g2gf_1d(md,w)

    eps = np.array([np.array(f['epsilon_xz'][t]) for t in time_keys])
    eps = g2gf_1d(md,eps)
    kappa = np.array([np.array(f['kappa_t1_xz'][t]) for t in time_keys])
    kappa = g2gf_1d(md,kappa)
    nu_t = np.array([np.array(f['nu_t_xz'][t]) for t in time_keys])
    nu_t = g2gf_1d(md,nu_t)
    diapycvel = np.array([np.array(f['diapycvel1_xz'][t]) for t in time_keys])
    diapycvel = g2gf_1d(md,diapycvel)
    chi_b = np.array([np.array(f['chi1_xz'][t]) for t in time_keys])
    chi_b = g2gf_1d(md,chi_b)
    chi_t = np.array([np.array(f['chi1_xz'][t]) for t in time_keys])
    chi_t = g2gf_1d(md,chi_t)

    scatter = np.array([np.array(f['td_scatter'][t]) for t in time_keys])
    scatter_flux = np.array([np.array(f['td_flux'][t]) for t in time_keys])

    times = np.array([f['th1_xz'][t].attrs['Time'] for t in time_keys])
    NSAMP = len(times)

plot_max = 1.6*md['H']
plot_min = 0.95*md['H']
start_idx = get_index(3.5, times)

idx_max = get_index(plot_max, gz)
idx_min = get_index(plot_min, gz)-1

idx_maxf = get_index(plot_max, gzf)
idx_minf = get_index(plot_min, gzf)

print(idx_min, idx_max)
print(idx_minf, idx_maxf)

X, Y = np.meshgrid(gx, gz[idx_min:idx_max])
Xf, Yf = np.meshgrid(gxf, gzf[idx_minf:idx_maxf])

#########################################################

_,z_coords,_ = np.meshgrid(times, gzf, gxf, indexing='ij', sparse=True)

u_z = np.gradient(u, gzf, axis=1)
v_z = np.gradient(v, gzf, axis=1)
N2 = np.gradient(b, gzf, axis=1)

# Get az data
with h5py.File(join(save_dir,"az_stats.h5"), 'r') as f:
    print("Keys: %s" % f.keys())
    time_keys = list(f['u_az'].keys())
    wbar = np.array([np.array(f['w_az'][t]) for t in time_keys])
    bbar = np.array([np.array(f['b_az'][t]) for t in time_keys])

wbar2 = np.concatenate((np.flip(wbar,axis=2), wbar), axis=2)
wfluc = w-wbar2

bbar2 = np.concatenate((np.flip(bbar,axis=2), bbar), axis=2)
bfluc = b-bbar2

Ri = 2*np.where(np.logical_and(z_coords < md['H'], b<1e-3), np.inf, N2)/(np.power(u_z, 2) + np.power(v_z, 2))
e = kappa * diapycvel/md['N2']
B = bfluc * wfluc
eps *= (md['nu'] + nu_t)/md['nu']
Re_b = eps/((md['nu'] + nu_t)*np.abs(np.where(np.logical_and(z_coords < md['H'], b<1e-3), 1e5, N2)))
Re_b = np.log(Re_b)
eps = np.log(eps)

field = eps
f_thresh = -10
field_str = "epsilon"

#########################################################
# Restrict arrays

bref = b[0, :, int(md['Nx']/2)]
t_orig = t
b = b[start_idx:, idx_minf:idx_maxf, :]
t = t[start_idx:, idx_minf:idx_maxf, :]
field = field[start_idx:, idx_minf:idx_maxf, :]
scatter = scatter[start_idx:]
scatter_flux = scatter_flux[start_idx:]
times = times[start_idx:]
NSAMP = len(b)

#########################################################
# Scatter plot set-up

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
bbins = [bmin + (i+0.5)*db for i in range(Nb)]
tbins = [tmin + (i+0.5)*dt for i in range(Nt)]

print(bmin,bmax)
print(tmin,tmax)

# accumulate fluxes
for i in range(1,NSAMP):
    scatter_flux[i] += scatter_flux[i-1]

scatter_corrected = scatter - scatter_flux

scatter_flux = np.where(scatter_flux == 0, np.nan, scatter_flux)
scatter_corrected = np.where(scatter_corrected == 0, np.nan, scatter_corrected)
scatter = np.where(scatter == 0, np.nan, scatter)

#########################################################
# Compute z_max from simulations

print(t_orig.shape)
tracer_data_vert = t_orig[1:, :, int(md['Nx']/2)]
tracer_thresh = 5e-4
plume_vert = np.where(tracer_data_vert > tracer_thresh, 1, 0)

heights = []
for i in range(len(plume_vert)):
    stuff = np.where(plume_vert[i] == 1)[0]
    if len(stuff) == 0:
        heights.append(0)
    else:
        heights.append(gzf[np.max(stuff)])

zmax_exp = np.max(heights)

f = interpolate.interp1d(gzf, bref)
b_zmax = f(zmax_exp)

# Compute mean tracer value entering computation domain
tracer_data = t[:, 0, int(md['Nx']/2)]

t_rms = np.sqrt(np.mean(np.power(tracer_data,2)))

t_centrelinemax = np.max(tracer_data)
t_bottommax = np.max(t[:,0,:])
t_totmax = np.max(t)
t_max = np.max(t[:,0,int(md['Nx']/2)])

#########################################################
# Figure set-up

fig,ax = plt.subplots(2, 2, figsize=(15,8))
fig.suptitle("time = 0.00 s")

contours_b = np.linspace(0, bmax, 10)

sx, sy = np.meshgrid(bbins, tbins)

#########################################################
# Create colour maps

cvals = [-1e-5, 0, 1e-5]
colors = ["blue","white","red"]

norm=plt.Normalize(min(cvals),max(cvals))
tuples = list(zip(map(norm,cvals), colors))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

s_cvals = np.concatenate((np.array([-1e-5, 0]), np.linspace(0,2e-5,31)[1:]))
s_colors = np.concatenate((np.array([[0.,0.,1.,1.,],[1.,1.,1.,1.]]), plt.cm.hot_r(np.linspace(0,1,30))))

s_norm=plt.Normalize(min(s_cvals),max(s_cvals))
s_tuples = list(zip(map(s_norm,s_cvals), s_colors))
s_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", s_tuples)

#########################################################

test_array = np.zeros_like(t[-1])
for i in range(int(md['Nb'])):
    for j in range(int(md['Nphi'])):
        test_array[np.logical_and(np.logical_and(b[-1] > bbins[i] - db/2,
            b[-1] <= bbins[i] + db/2),np.logical_and(t[-1] > tbins[j] - dt/2,
            t[-1] <= tbins[j] + dt/2))] = scatter_corrected[-1,j,i]

trac_im = ax[0,0].pcolormesh(X, Y, test_array, cmap=s_cmap, norm=s_norm)
b_cont = ax[0,0].contour(Xf, Yf, np.where(t[-1] <= 5e-4, b[-1], np.NaN),
            levels = contours_b, cmap='cool', alpha=0.8)
b_cont_fill = ax[0,0].contourf(b_cont, levels = contours_b, cmap='cool', alpha=0.8)
t_cont = ax[0,0].contour(Xf, Yf, t[-1], levels = [5e-4], colors='green', alpha=0.8)

#########################################################

ax[0,1].set_title("Highlighted regions: (t,b) volume where {0} > {1}".format(field_str, f_thresh))
vol_im_thresh = ax[0,1].pcolormesh(X, Y, np.where(field[-1] > f_thresh, test_array, np.NaN),
    cmap=s_cmap, norm=s_norm)
vol_im_threshc = ax[0,1].pcolormesh(X, Y, np.where(field[-1] <= f_thresh, test_array, np.NaN),
    cmap=s_cmap, norm=s_norm, alpha=0.2)
vol_cont = ax[0,1].contour(Xf, Yf, field[-1], levels=[f_thresh], colors='g', alpha=0.5)

ax[1,1].set_title("Highlighted regions: {0} where cumulative (t,b) volume negative".format(field_str))
field_norm = plt.Normalize(-12,-6)
field_im_thresh = ax[1,1].pcolormesh(X, Y, np.where(test_array < 0, field[-1], np.NaN),
    cmap='hot_r', norm=field_norm)
field_im_threshc = ax[1,1].pcolormesh(X, Y, np.where(test_array > 0, field[-1], np.NaN),
    cmap='hot_r', norm=field_norm, alpha=0.2)
field_cont = ax[1,1].contour(Xf, Yf, np.where(test_array < 0, test_array, 1),
        levels=[0], colors='b', alpha=0.5)

#########################################################

vols = np.linspace(-1e-5, 2e-5, 13)
vols_plot = 0.5*(vols[1:]+vols[:-1])

field_vols = np.zeros_like(vols)[:-1]
field_volsc = np.zeros_like(vols)[:-1]
thresh_array = np.where(np.logical_and(field[-1] > f_thresh, t[-1] >=5e-4), test_array, np.NaN)
threshc_array = np.where(np.logical_and(field[-1] <= f_thresh, t[-1] >= 5e-4), test_array, np.NaN)

dx = gx[1:] - gx[:-1]
dz = gz[1:] - gz[:-1]

dz = dz[idx_minf:idx_maxf]
gdx, gdz = np.meshgrid(dx, dz)
cell_vols = gdx * gdz * np.abs(gxf - md['LX']/2) # last factor corrects for radius
print(cell_vols)

thresh_vol = np.sum(cell_vols[~np.isnan(thresh_array)])
threshc_vol = np.sum(cell_vols[~np.isnan(threshc_array)])

for i in range(1,len(vols)-2):
    field_vols[i] = np.nansum(np.where(np.logical_and(thresh_array >= vols[i], thresh_array < vols[i+1]),
        cell_vols, np.NaN))
field_vols[0] = np.nansum(np.where(thresh_array < vols[1], cell_vols, np.NaN))
field_vols[-1] = np.nansum(np.where(thresh_array >= vols[-2], cell_vols, np.NaN))

#ax[1,0].stairs(field_vols/thresh_vol, vols, color='r', label="{0} > {1}".format(field_str, f_thresh))

for i in range(1,len(vols)-2):
    field_volsc[i] = np.nansum(np.where(np.logical_and(threshc_array >= vols[i], threshc_array < vols[i+1]),
        cell_vols, np.NaN))
field_volsc[0] = np.nansum(np.where(threshc_array < vols[1], cell_vols, np.NaN))
field_volsc[-1] = np.nansum(np.where(threshc_array >= vols[-2], cell_vols, np.NaN))

vol_vols = field_vols + field_volsc
print(vol_vols)

ax[1,0].bar(vols_plot, field_vols/vol_vols, width=vols[1:]-vols[:-1],
        label="{0} > {1}".format(field_str, f_thresh))
ax[1,0].bar(vols_plot, field_volsc/vol_vols, width=vols[1:]-vols[:-1], bottom=field_vols/vol_vols,
        label="{0} <= {1}".format(field_str, f_thresh))

ax[1,0].legend()
ax[1,0].set_ylim(0,1)

#########################################################
# Decorations
im_div = make_axes_locatable(ax[0,0])
im_cax = im_div.append_axes("right", size="5%", pad=0.05)
im_cb = plt.colorbar(trac_im, cax=im_cax, label="volume $(m^3)$")

cont_norm = matplotlib.colors.Normalize(vmin=b_cont.cvalues.min(), vmax=b_cont.cvalues.max())
cont_sm = plt.cm.ScalarMappable(norm=cont_norm, cmap=b_cont.cmap)
im_cont_cax = im_div.append_axes("right", size="5%", pad=0.65)
im_cont_cb = plt.colorbar(b_cont_fill, cax=im_cont_cax, format=lambda x,_ :f"{x:.3f}", label="buoyancy")
im_cont_cb.set_alpha(1)
im_cont_cb.draw_all()


ax[0,0].set_title("Plume cross-section")
#ax[0,0].set_aspect(1)

ax[0,0].set_xlim(0.1, 0.5)
ax[0,1].set_xlim(0.1, 0.5)
ax[1,1].set_xlim(0.1, 0.5)

plt.tight_layout()

#############################################################################################################

fig, ax = plt.subplots(1,2, figsize=(12, 3.5), constrained_layout=True)
fig.set_constrained_layout_pads(wspace=0.05)

#########################################################

vols = np.linspace(-1e-5, 1.75e-5, 13)
vols_plot = 0.5*(vols[1:]+vols[:-1])

field_vols = np.zeros_like(vols)[:-1]
field_volsc = np.zeros_like(vols)[:-1]
thresh_array = np.where(np.logical_and(field[-1] > f_thresh, t[-1] >=5e-4), test_array, np.NaN)
threshc_array = np.where(np.logical_and(field[-1] <= f_thresh, t[-1] >= 5e-4), test_array, np.NaN)

dx = gx[1:] - gx[:-1]
dz = gz[1:] - gz[:-1]

dz = dz[idx_minf:idx_maxf]
gdx, gdz = np.meshgrid(dx, dz)
cell_vols = gdx * gdz * np.abs(gxf - md['LX']/2) # last factor corrects for radius
print(cell_vols)

thresh_vol = np.sum(cell_vols[~np.isnan(thresh_array)])
threshc_vol = np.sum(cell_vols[~np.isnan(threshc_array)])

for i in range(1,len(vols)-2):
    field_vols[i] = np.nansum(np.where(np.logical_and(thresh_array >= vols[i], thresh_array < vols[i+1]),
        cell_vols, np.NaN))
field_vols[0] = np.nansum(np.where(thresh_array < vols[1], cell_vols, np.NaN))
field_vols[-1] = np.nansum(np.where(thresh_array >= vols[-2], cell_vols, np.NaN))

#ax[1].stairs(field_vols/thresh_vol, vols, color='r', label="{0} > {1}".format(field_str, f_thresh))

for i in range(1,len(vols)-2):
    field_volsc[i] = np.nansum(np.where(np.logical_and(threshc_array >= vols[i], threshc_array < vols[i+1]),
        cell_vols, np.NaN))
field_volsc[0] = np.nansum(np.where(threshc_array < vols[1], cell_vols, np.NaN))
field_volsc[-1] = np.nansum(np.where(threshc_array >= vols[-2], cell_vols, np.NaN))

vol_vols = field_vols + field_volsc

ax[1].bar(vols_plot, field_vols/vol_vols, width=vols[1:]-vols[:-1],
        label=r"$\log \varepsilon > -10$")
ax[1].bar(vols_plot, field_volsc/vol_vols, width=vols[1:]-vols[:-1], bottom=field_vols/vol_vols,
        label=r"$\log \varepsilon \leq -10$")

ax[1].legend()
ax[1].set_ylim(0,1)
ax[1].set_xlim(vols[0], vols[-1])
ax[1].set_xlabel("$\Omega\,(m^3)$")
ax[1].set_ylabel("Volume proportion")

#ax[0].set_title("TKE dissipation rate $\\varepsilon$")
field_norm = plt.Normalize(-12, -6)
field_im_thresh = ax[0].pcolormesh(X, Y, np.where(test_array < 0, field[-1], np.NaN),
    cmap='hot_r', norm=field_norm)
field_im_threshc = ax[0].pcolormesh(X, Y, np.where(test_array > 0, field[-1], np.NaN),
    cmap='hot_r', norm=field_norm, alpha=0.3)
field_cont = ax[0].contour(Xf, Yf, np.where(test_array < 0, test_array, 1),
        levels=[0], colors='b', alpha=0.5)

div = make_axes_locatable(ax[0])
cax = div.append_axes("right", size="5%", pad=0.05)
plt.colorbar(field_im_thresh, cax=cax, label="TKE dissipation rate (log)")

#bline = mlines.Line2D([], [], color='b', label="$\Omega < 0$")
bline = mpatches.Patch(facecolor='white', edgecolor='b', label="$\Omega < 0$")
ax[0].legend(handles=[bline])

ax[0].set_aspect(1)
ax[0].set_xlim(0.2, 0.4)
ax[0].set_xlabel(r"$x\,(m)$")
ax[0].set_ylabel(r"$z\,(m)$")

#plt.tight_layout()
plt.savefig('/home/cwp29/Documents/essay/figs/eps_vol.png', dpi=300)
plt.show()
