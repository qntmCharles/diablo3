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
    b = np.array([np.array(f['th1_xz'][t]) for t in time_keys])
    t = g2gf_1d(md, t)
    b = g2gf_1d(md, b)
    scatter = np.array([np.array(f['td_scatter'][t]) for t in time_keys])
    scatter_flux = np.array([np.array(f['td_flux'][t]) for t in time_keys])

    times = np.array([f['th1_xz'][t].attrs['Time'] for t in time_keys])
    NSAMP = len(times)

plot_max = 1.6*md['H']
plot_min = 0.95*md['H']
start_idx = get_index(3.5, times)

idx_max = get_index(plot_max, gz)+1
idx_min = get_index(plot_min, gz)

idx_maxf = get_index(plot_max, gzf)
idx_minf = get_index(plot_min, gzf)

print(idx_min, idx_max)
print(idx_minf, idx_maxf)

X, Y = np.meshgrid(gx, gz[idx_min:idx_max])
Xf, Yf = np.meshgrid(gxf, gzf[idx_minf:idx_maxf])

bref = b[0, :, int(md['Nx']/2)]
t_orig = t
b = b[start_idx:, idx_minf:idx_maxf, :]
t = t[start_idx:, idx_minf:idx_maxf, :]
scatter = scatter[start_idx:]
scatter_flux = scatter_flux[start_idx:]
times = times[start_idx:]
NSAMP = len(b)

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

scatter_corrected = np.gradient(scatter_corrected, times, axis=0)

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

contours_b = np.linspace(0, np.max(b), 10)

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

dvdt_cvals = [np.nanmin(scatter_corrected), 0, np.nanmax(scatter_corrected)]
dvdt_colors = ["blue", "white", "red"]

dvdt_norm=plt.Normalize(min(dvdt_cvals),max(dvdt_cvals))
dvdt_tuples = list(zip(map(dvdt_norm,dvdt_cvals), dvdt_colors))
dvdt_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", dvdt_tuples)

#########################################################

im_scatter = ax[0,1].pcolormesh(sx, sy, scatter[-1], cmap='jet')
im_flux = ax[1,0].pcolormesh(sx, sy, scatter_flux[-1], cmap=cmap, norm=norm)
im_corrected = ax[1,1].pcolormesh(sx, sy, scatter_corrected[-1], cmap=dvdt_cmap, norm=dvdt_norm)

test_array = np.zeros_like(t[-1])
for i in range(int(md['Nb'])):
    for j in range(int(md['Nphi'])):
        test_array[np.logical_and(np.logical_and(b[-1] > bbins[i] - db/2,
            b[-1] <= bbins[i] + db/2),np.logical_and(t[-1] > tbins[j] - dt/2,
            t[-1] <= tbins[j] + dt/2))] = scatter_corrected[-1,j,i]

trac_im = ax[0,0].pcolormesh(X, Y, test_array, cmap=dvdt_cmap, norm=dvdt_norm)
b_cont = ax[0,0].contour(Xf, Yf, np.where(t[-1] <= 5e-4, b[-1], np.NaN),
            levels = contours_b, cmap='cool', alpha=0.8)
b_cont_fill = ax[0,0].contourf(b_cont, levels = contours_b, cmap='cool', alpha=0.8)
t_cont = ax[0,0].contour(Xf, Yf, t[-1], levels = [5e-4], colors='green', alpha=0.8)

#########################################################

sx_nan = sx[~np.isnan(scatter_corrected[-1])].flatten()
sy_nan = sy[~np.isnan(scatter_corrected[-1])].flatten()
points = np.array(list(zip(sx_nan,sy_nan)))

if len(points) > 0:
    hull = spatial.ConvexHull(points)

    ax[1,1].plot(points[hull.simplices[0],0], points[hull.simplices[0],1], 'r--', label="convex hull")
    for simplex in hull.simplices:
        ax[1,1].plot(points[simplex,0], points[simplex,1], 'r--')

#########################################################

bgrid, tgrid = np.meshgrid(bbins, tbins)

b_com = np.nansum(bgrid * np.where(scatter_corrected[-1] > 0, scatter_corrected[-1], 0)) \
        / np.nansum(np.where(scatter_corrected[-1]>0, scatter_corrected[-1],0))
t_com = np.nansum(tgrid * np.where(scatter_corrected[-1] > 0, scatter_corrected[-1], 0)) \
        / np.nansum(np.where(scatter_corrected[-1]>0, scatter_corrected[-1],0))

ax[0,1].scatter(b_com, t_com, color='green', edgecolor='white', marker='^', s=200)
ax[1,0].scatter(b_com, t_com, color='green', edgecolor='white', marker='^', s=200)
ax[1,1].scatter(b_com, t_com, color='green', edgecolor='white', marker='^', s=200)

#########################################################
# Decorations
im_div = make_axes_locatable(ax[0,0])
im_cax = im_div.append_axes("right", size="5%", pad=0.05)
im_cb = plt.colorbar(trac_im, cax=im_cax, label="volume $(m^3)$")
im_scatter.set_clim(0, np.nanmax(scatter[-1]))
#trac_im.set_clim(0, np.max(t[-1]))

cont_norm = matplotlib.colors.Normalize(vmin=b_cont.cvalues.min(), vmax=b_cont.cvalues.max())
cont_sm = plt.cm.ScalarMappable(norm=cont_norm, cmap=b_cont.cmap)
im_cont_cax = im_div.append_axes("right", size="5%", pad=0.65)
im_cont_cb = plt.colorbar(b_cont_fill, cax=im_cont_cax, format=lambda x,_ :f"{x:.3f}", label="buoyancy")
im_cont_cb.set_alpha(1)
im_cont_cb.draw_all()

sc_div = make_axes_locatable(ax[0,1])
sc_cax = sc_div.append_axes("right", size="5%", pad=0.05)
sc_cb = plt.colorbar(im_scatter, cax=sc_cax, label="volume")

flux_div = make_axes_locatable(ax[1,0])
flux_cax = flux_div.append_axes("right", size="5%", pad=0.05)
flux_cb = plt.colorbar(im_flux, cax=flux_cax, label="volume")

corr_div = make_axes_locatable(ax[1,1])
corr_cax = corr_div.append_axes("right", size="5%", pad=0.05)
corr_cb = plt.colorbar(im_corrected, cax=corr_cax, label="volume")

ax[0,0].set_title("Plume cross-section")
ax[0,1].set_title("Stratified region $(b,\\phi)$ distribution")
ax[1,0].set_title("Accumulated $(b,\\phi)$ distribution input to stratified region")
ax[1,1].set_title("Total - accumulated input $(b,\\phi)$ distribution")

ax[0,1].set_xlabel("buoyancy")
ax[0,1].set_ylabel("tracer")
ax[0,1].set_xlim(bmin, bmax)
ax[0,1].set_ylim(tmin, tmax)
ax[1,0].set_xlabel("buoyancy")
ax[1,0].set_ylabel("tracer")
ax[1,0].set_xlim(bmin, bmax)
ax[1,0].set_ylim(tmin, tmax)
ax[1,1].set_xlabel("buoyancy")
ax[1,1].set_ylabel("tracer")
ax[1,1].set_xlim(bmin, bmax)
ax[1,1].set_ylim(tmin, tmax)

ax[0,1].axvline(b_zmax, color='k', linestyle='--')
ax[1,0].axvline(b_zmax, color='k', linestyle='--')
ax[1,1].axvline(b_zmax, color='k', linestyle='--')

ax[0,1].axhline(t_rms, color='k', linestyle='--')
ax[1,0].axhline(t_rms, color='k', linestyle='--')
ax[1,1].axhline(t_rms, color='k', linestyle='--')

ax[0,0].set_aspect(1)

ax[1,1].legend()

#plt.tight_layout()
#plt.show()
#########################################################

def animate(step):
    #global b_cont, t_cont, b_cont_fill

    ax[0,0].clear()
    ax[0,1].clear()
    ax[1,0].clear()
    ax[1,1].clear()

    ax[0,0].set_title("Plume cross-section")
    ax[0,1].set_title("Stratified region $(b,\\phi)$ distribution")
    ax[1,0].set_title("Accumulated $(b,\\phi)$ distribution input to stratified region")
    ax[1,1].set_title("Total - accumulated input $(b,\\phi)$ distribution")

    ax[0,1].axvline(b_zmax, color='k', linestyle='--')
    ax[1,0].axvline(b_zmax, color='k', linestyle='--')
    ax[1,1].axvline(b_zmax, color='k', linestyle='--')

    ax[0,1].axhline(t_rms, color='k', linestyle='--')
    ax[1,0].axhline(t_rms, color='k', linestyle='--')
    ax[1,1].axhline(t_rms, color='k', linestyle='--')

    b_com = np.nansum(bgrid * np.where(scatter_corrected[step] > 0, scatter_corrected[step], 0)) \
            / np.nansum(np.where(scatter_corrected[step]>0, scatter_corrected[step],0))
    t_com = np.nansum(tgrid * np.where(scatter_corrected[step] > 0, scatter_corrected[step], 0)) \
            / np.nansum(np.where(scatter_corrected[step]>0, scatter_corrected[step],0))

    im_scatter = ax[0,1].pcolormesh(sx, sy, scatter[step], cmap='jet')
    im_scatter.set_clim(0, np.nanmax(scatter[-1]))
    im_flux = ax[1,0].pcolormesh(sx, sy, scatter_flux[step], cmap=cmap, norm=norm)
    im_corrected = ax[1,1].pcolormesh(sx, sy, scatter_corrected[step], cmap=dvdt_cmap, norm=dvdt_norm)

    ax[0,1].scatter(b_com, t_com, color='green', edgecolor='white', marker='^', s=200)
    ax[1,0].scatter(b_com, t_com, color='green', edgecolor='white', marker='^', s=200)
    ax[1,1].scatter(b_com, t_com, color='green', edgecolor='white', marker='^', s=200)

    sx_nan = sx[~np.isnan(scatter_corrected[step])].flatten()
    sy_nan = sy[~np.isnan(scatter_corrected[step])].flatten()
    points = np.array(list(zip(sx_nan,sy_nan)))

    if len(points) > 0:
        hull = spatial.ConvexHull(points)

        ax[1,1].plot(points[hull.simplices[0],0], points[hull.simplices[0],1], 'r--', label="convex hull")
        for simplex in hull.simplices:
            ax[1,1].plot(points[simplex,0], points[simplex,1], 'r--')

    fig.suptitle("time = {0:.2f} s".format(times[step]))

    #for coll in b_cont.collections:
        #coll.remove()
    #for coll in b_cont_fill.collections:
        #coll.remove()
    #for coll in t_cont.collections:
        #coll.remove()

    test_array = np.zeros_like(t[step])
    for i in range(int(md['Nb'])):
        for j in range(int(md['Nphi'])):
            test_array[np.logical_and(np.logical_and(b[step] > bbins[i] - db/2,
                b[step] <= bbins[i] + db/2),np.logical_and(t[step] > tbins[j] - dt/2,
                t[step] <= tbins[j] + dt/2))] = scatter_corrected[step,j,i]

    trac_im = ax[0,0].pcolormesh(X, Y, test_array, cmap=dvdt_cmap, norm=dvdt_norm)


    b_cont = ax[0,0].contour(Xf, Yf, np.where(t[step] <= 5e-4, b[step], np.NaN),
            levels = contours_b, cmap='cool', alpha=0.8)
    b_cont_fill = ax[0,0].contourf(b_cont, levels = contours_b, cmap='cool', alpha=0.8)
    t_cont = ax[0,0].contour(Xf, Yf, t[step], levels = [5e-4], colors='green', alpha=0.8)

    ax[0,0].set_xlim(0.2, 0.4)

    ax[0,1].set_xlabel("buoyancy")
    ax[0,1].set_ylabel("tracer")
    ax[0,1].set_xlim(bmin, bmax)
    ax[0,1].set_ylim(tmin, tmax)
    ax[1,0].set_xlabel("buoyancy")
    ax[1,0].set_ylabel("tracer")
    ax[1,0].set_xlim(bmin, bmax)
    ax[1,0].set_ylim(tmin, tmax)
    ax[1,1].set_xlabel("buoyancy")
    ax[1,1].set_ylabel("tracer")
    ax[1,1].set_xlim(bmin, bmax)
    ax[1,1].set_ylim(tmin, tmax)

    ax[1,1].legend()

    return im_scatter, im_flux, im_corrected, trac_im

Writer = animation.writers['ffmpeg']
writer = Writer(fps=4, bitrate=-1)

anim = animation.FuncAnimation(fig, animate, interval=250, frames=NSAMP)
now = datetime.now()
#anim.save(save_dir+'td_scatter_%s.mp4'%now.strftime("%d-%m-%Y-%H"),writer=writer)
plt.tight_layout()
plt.show()
