# Script loads in 2D slices and produces a movie of the simulation output import numpy as np import h5py, gc
import h5py
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.animation as animation
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


_,z_coords,_ = np.meshgrid(times, gzf, gxf, indexing='ij', sparse=True)

u_z = np.gradient(u, gzf, axis=1)
v_z = np.gradient(v, gzf, axis=1)
N2 = np.gradient(b, gzf, axis=1)

Ri = 2*np.where(np.logical_and(z_coords < md['H'], b<1e-3), np.inf, N2)/(np.power(u_z, 2) + np.power(v_z, 2))

#########################################################
# Restrict arrays

bref = b[0, :, int(md['Nx']/2)]
t_orig = t
b = b[start_idx:, idx_minf:idx_maxf, :]
t = t[start_idx:, idx_minf:idx_maxf, :]
Ri = Ri[start_idx:, idx_minf:idx_maxf, :]
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
db = round((bmax - bmin)/Nb,3)
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

ax[0,1].set_title("Highlighted regions: (t,b) volume where Ri < 1/4")
Ri_im = ax[0,1].pcolormesh(X, Y, np.where(Ri[-1] <= 0.25, test_array, np.NaN), cmap=s_cmap, norm=s_norm)
Ri_im = ax[0,1].pcolormesh(X, Y, np.where(Ri[-1] > 0.25, test_array, np.NaN), cmap=s_cmap, norm=s_norm,
        alpha=0.2)
Ri_cont = ax[0,1].contour(Xf, Yf, Ri[-1], levels=[0.25], colors='g', alpha=0.5)

ax[1,1].set_title("Highlighted regions: Ri where cumulative (t,b) volume negative")
Ri_norm = plt.Normalize(0, 0.25)
Ri_im = ax[1,1].pcolormesh(X, Y, np.where(test_array < 0, Ri[-1], np.NaN), cmap='hot_r', norm=Ri_norm)
Ri_im = ax[1,1].pcolormesh(X, Y, np.where(test_array > 0, Ri[-1], np.NaN), cmap='hot_r', norm=Ri_norm,
        alpha=0.2)
Ri_cont = ax[1,1].contour(Xf, Yf, np.where(test_array < 0, test_array, 1),
        levels=[0], colors='b', alpha=0.5)

"""
Ri_im = ax[1,1].pcolormesh(X, Y, np.where(Ri[-1] > 0.25, test_array, np.NaN), cmap=s_cmap, norm=s_norm)
Ri_im = ax[1,1].pcolormesh(X, Y, np.where(Ri[-1] <= 0.25, test_array, np.NaN), cmap=s_cmap, norm=s_norm,
        alpha=0.2)
Ri_cont = ax[1,1].contour(Xf, Yf, Ri[-1], levels=[0.25], colors='g', alpha=0.5)
"""

#########################################################

vols = np.linspace(-1e-5, 2e-5, 20)
vols_plot = 0.5*(vols[1:]+vols[:-1])

Ri_vols = np.zeros_like(vols)[:-1]
stable_array = np.where(np.logical_and(Ri[-1] > 0.25, t[-1] >=5e-4), test_array, np.NaN)
unstable_array = np.where(np.logical_and(Ri[-1] <= 0.25, t[-1] >= 5e-4), test_array, np.NaN)

dx = gx[1:] - gx[:-1]
dz = gz[1:] - gz[:-1]

dz = dz[idx_minf:idx_maxf]
gdx, gdz = np.meshgrid(dx, dz)
cell_vols = gdx * gdz
print(cell_vols)

stable_vol = np.sum(cell_vols[~np.isnan(stable_array)])
unstable_vol = np.sum(cell_vols[~np.isnan(unstable_array)])

for i in range(1,len(vols)-2):
    Ri_vols[i] = np.nansum(np.where(np.logical_and(stable_array >= vols[i], stable_array < vols[i+1]), cell_vols,
        np.NaN))
Ri_vols[0] = np.nansum(np.where(stable_array < vols[1], cell_vols, np.NaN))
Ri_vols[-1] = np.nansum(np.where(stable_array >= vols[-2], cell_vols, np.NaN))

ax[1,0].stairs(Ri_vols/stable_vol, vols, color='r', label="stable")

for i in range(1,len(vols)-2):
    Ri_vols[i] = np.nansum(np.where(np.logical_and(unstable_array >= vols[i], unstable_array < vols[i+1]), cell_vols,
        np.NaN))
Ri_vols[0] = np.nansum(np.where(unstable_array < vols[1], cell_vols, np.NaN))
Ri_vols[-1] = np.nansum(np.where(unstable_array >= vols[-2], cell_vols, np.NaN))

ax[1,0].stairs(Ri_vols/unstable_vol, vols, color='b', label="unstable")
#ax[1,0].plot(vols_plot, Ri_vols/unstable_vol, color='b', label="unstable")
ax[1,0].legend()

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

#plt.tight_layout()
#plt.show()

#############################################################################################################

def animate(step):
    ax[0,0].clear()
    ax[0,1].clear()
    ax[1,0].clear()
    ax[1,1].clear()

    test_array = np.zeros_like(t[step])
    for i in range(int(md['Nb'])):
        for j in range(int(md['Nphi'])):
            test_array[np.logical_and(np.logical_and(b[step] > bbins[i] - db/2,
                b[step] <= bbins[i] + db/2),np.logical_and(t[step] > tbins[j] - dt/2,
                t[step] <= tbins[j] + dt/2))] = scatter_corrected[step,j,i]

    trac_im = ax[0,0].pcolormesh(X, Y, test_array, cmap=s_cmap, norm=s_norm)
    b_cont = ax[0,0].contour(Xf, Yf, np.where(t[step] <= 5e-4, b[step], np.NaN),
                levels = contours_b, cmap='cool', alpha=0.8)
    b_cont_fill = ax[0,0].contourf(b_cont, levels = contours_b, cmap='cool', alpha=0.8)
    t_cont = ax[0,0].contour(Xf, Yf, t[step], levels = [5e-4], colors='green', alpha=0.8)

    #########################################################

    ax[0,1].set_title("Highlighted regions: (t,b) volume where Ri < 1/4")
    Ri_im_unstab = ax[0,1].pcolormesh(X, Y, np.where(Ri[step] <= 0.25, test_array, np.NaN), cmap=s_cmap,
            norm=s_norm)
    Ri_im_unstab2 = ax[0,1].pcolormesh(X, Y, np.where(Ri[step] > 0.25, test_array, np.NaN), cmap=s_cmap,
            norm=s_norm, alpha=0.2)
    Ri_cont_unstab = ax[0,1].contour(Xf, Yf, Ri[step], levels=[0.25], colors='g', alpha=0.5)

    #Ri_im_stab = ax[1,1].pcolormesh(X, Y, np.where(Ri[step] > 0.25, test_array, np.NaN), cmap=s_cmap,
            #norm=s_norm)
    #Ri_im_stab2 = ax[1,1].pcolormesh(X, Y, np.where(Ri[step] <= 0.25, test_array, np.NaN), cmap=s_cmap,
            #norm=s_norm, alpha=0.2)
    #Ri_cont_stab = ax[1,1].contour(Xf, Yf, Ri[step], levels=[0.25], colors='g', alpha=0.5)


    ax[1,1].set_title("Highlighted regions: Ri where cumulative (t,b) volume negative")
    Ri_im_stab = ax[1,1].pcolormesh(X, Y, np.where(test_array < 0, Ri[step], np.NaN), cmap='hot',
            norm=Ri_norm)
    Ri_im_stab2 = ax[1,1].pcolormesh(X, Y, np.where(test_array > 0, Ri[step], np.NaN), cmap='hot',
            norm=Ri_norm, alpha=0.2)
    Ri_cont_stab = ax[1,1].contour(Xf, Yf, np.where(test_array < 0, test_array, 1),
            levels=[0], colors='b', alpha=0.5)

    #########################################################

    Ri_vols = np.zeros_like(vols)[:-1]
    stable_array = np.where(np.logical_and(Ri[step] > 0.25, t[step] >=5e-4), test_array, np.NaN)
    unstable_array = np.where(np.logical_and(Ri[step] <= 0.25, t[step] >= 5e-4), test_array, np.NaN)

    stable_vol = np.sum(cell_vols[~np.isnan(stable_array)])
    unstable_vol = np.sum(cell_vols[~np.isnan(unstable_array)])

    for i in range(1,len(vols)-2):
        Ri_vols[i] = np.nansum(np.where(np.logical_and(stable_array >= vols[i], stable_array < vols[i+1]), cell_vols,
            np.NaN))
    Ri_vols[0] = np.nansum(np.where(stable_array < vols[1], cell_vols, np.NaN))
    Ri_vols[-1] = np.nansum(np.where(stable_array >= vols[-2], cell_vols, np.NaN))

    stab_plot = ax[1,0].stairs(Ri_vols/stable_vol, vols, color='r', label="stable")

    for i in range(1,len(vols)-2):
        Ri_vols[i] = np.nansum(np.where(np.logical_and(unstable_array >= vols[i], unstable_array < vols[i+1]), cell_vols,
            np.NaN))
    Ri_vols[0] = np.nansum(np.where(unstable_array < vols[1], cell_vols, np.NaN))
    Ri_vols[-1] = np.nansum(np.where(unstable_array >= vols[-2], cell_vols, np.NaN))

    #unstab_plot = ax[1,0].plot(vols_plot, Ri_vols/unstable_vol, color='b', label="unstable")
    unstab_plot = ax[1,0].stairs(Ri_vols/unstable_vol, vols, color='b', label="unstable")

    ax[1,0].set_ylim(0, 0.5)
    ax[1,0].legend()

    ax[0,0].set_xlim(0.1, 0.5)
    ax[0,1].set_xlim(0.1, 0.5)
    ax[1,1].set_xlim(0.1, 0.5)

    fig.suptitle("time = {0:.2f} s".format(times[step]))

    return stab_plot, unstab_plot, trac_im, Ri_im_stab, Ri_im_stab2, Ri_im_unstab, Ri_im_unstab2,


Writer = animation.writers['ffmpeg']
writer = Writer(fps=4, bitrate=-1)

anim = animation.FuncAnimation(fig, animate, interval=250, frames=NSAMP)
now = datetime.now()
#anim.save(save_dir+'scatter_Ri_%s.mp4'%now.strftime("%d-%m-%Y"),writer=writer)
plt.tight_layout()
plt.show()
