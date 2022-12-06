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
from scipy import ndimage, interpolate

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"

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
    b = g2gf_1d(md, b)
    t = g2gf_1d(md, t)
    b = np.array([np.array(f['th1_xz'][t]) for t in time_keys])
    scatter = np.array([np.array(f['td_scatter'][t]) for t in time_keys])
    scatter_flux = np.array([np.array(f['td_flux'][t]) for t in time_keys])
    scatter_corrected = np.array([np.array(f['td_corrected'][t]) for t in time_keys])
    NSAMP = len(scatter)
    times = np.array([f['th1_xz'][t].attrs['Time'] for t in time_keys])

plot_max = md['LZ']
plot_min = 0.95*md['H']
start_idx = get_index(3, times)

idx_max = get_index(plot_max, gz)+1
idx_min = get_index(plot_min, gz)-1

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
scatter_corrected = scatter_corrected[start_idx:]
times = times[start_idx:]
NSAMP = len(b)

#bmin = -0.5*md['N2']*(md['LZ']-md['H'])
#bmax = 0.5*md['N2']*(md['LZ']-md['H'])
bmin = 0
bmax = md['b_factor']*md['N2']*(md['LZ']-md['H'])

F0 = md['b0']*(md['r0']**2)
alpha = md['alpha_e']

tmin = 1e-7
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

scatter_corrected = scatter - scatter_flux

scatter_flux = np.where(scatter_flux == 0, np.nan, scatter_flux)
#scatter_flux = np.log(scatter_flux)

scatter_corrected_og = scatter_corrected
scatter_corrected = np.where(scatter_corrected == 0, np.nan, scatter_corrected)
#scatter_corrected = np.log(scatter_corrected)

scatter = np.where(scatter == 0, np.nan, scatter)
#scatter = np.log10(scatter)

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
print(b_zmax)

"""
anotherfig = plt.figure()
plt.plot(times[1:], heights)

testfig = plt.figure()
Xt, Yt = np.meshgrid(times[:], gz)
Xft, Yft = np.meshgrid(times[1:], gzf)
plt.pcolormesh(Xt,Yt, np.swapaxes(tracer_data_vert,0,1),cmap='hot_r')
plt.contour(Xft, Yft, np.swapaxes(tracer_data_vert,0,1), levels=[tracer_thresh], colors=['k'])
plt.axhline(zmax_exp)

bfig = plt.figure()

plt.plot(bref, gzf)
plt.axhline(zmax_exp)

plt.show()
"""

# Compute mean tracer value entering computation domain
tracer_data = t[:, 0, int(md['Nx']/2)]

t_rms = np.sqrt(np.mean(np.power(tracer_data,2)))

t_centrelinemax = np.max(tracer_data)
t_bottommax = np.max(t[:,0,:])
t_totmax = np.max(t)
t_max = np.max(t[:,0,int(md['Nx']/2)])


#########################################################

fig,ax = plt.subplots(2, 2, figsize=(15,10))
fig.suptitle("time = 0.00 s")

contours_b = np.linspace(0, bmax, 10)

#scatter[:, 0, 0] = np.NaN
#scatter_flux[:, 0, 0] = np.NaN
#scatter_corrected[:, 0, 0] = np.NaN

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

trac_im = ax[0,0].pcolormesh(X, Y, t[-1], cmap='hot_r')
b_cont = ax[0,0].contour(Xf, Yf, np.where(t[-1] <= 5e-4, b[-1], np.NaN),
            levels = contours_b, cmap='cool', alpha=0.8)
b_cont_fill = ax[0,0].contourf(b_cont, levels = contours_b, cmap='cool', alpha=0.8)
t_cont = ax[0,0].contour(Xf, Yf, t[-1], levels = [5e-4], colors='white', alpha=0.8)

im_scatter = ax[0,1].scatter(sx, sy, c=scatter[-1], cmap='jet')
im_flux = ax[1,0].scatter(sx, sy, c=scatter_flux[-1], cmap=cmap, norm=norm)
im_corrected = ax[1,1].scatter(sx, sy, c=scatter_corrected[-1], cmap=s_cmap, norm=s_norm)

bgrid, tgrid = np.meshgrid(bbins, tbins, indexing='ij')

b_com = np.sum(bgrid * np.where(scatter_corrected_og[-1] > 0, scatter_corrected_og[-1], 0)) \
        / np.sum(scatter_corrected_og[-1])
t_com = np.sum(tgrid * np.where(scatter_corrected_og[-1] > 0, scatter_corrected_og[-1], 0)) \
        / np.sum(scatter_corrected_og[-1])

ax[0,1].scatter(b_com, t_com, color='green', edgecolor='white', marker='^', s=200)
ax[1,0].scatter(b_com, t_com, color='green', edgecolor='white', marker='^', s=200)
ax[1,1].scatter(b_com, t_com, color='green', edgecolor='white', marker='^', s=200)

#########################################################
# Decorations
im_div = make_axes_locatable(ax[0,0])
im_cax = im_div.append_axes("right", size="5%", pad=0.05)
im_cb = plt.colorbar(trac_im, cax=im_cax, format=lambda x,_ :f"{x:.4f}")
trac_im.set_clim(0, np.max(t[-1]))

#cont_norm = matplotlib.colors.Normalize(vmin=b_cont.cvalues.min(), vmax=b_cont.cvalues.max())
#cont_sm = plt.cm.ScalarMappable(norm=cont_norm, cmap=b_cont.cmap)
im_cont_cax = im_div.append_axes("right", size="5%", pad=0.65)
im_cont_cb = plt.colorbar(b_cont_fill, cax=im_cont_cax, format=lambda x,_ :f"{x:.3f}")
im_cont_cb.set_alpha(1)
im_cont_cb.draw_all()

sc_div = make_axes_locatable(ax[0,1])
sc_cax = sc_div.append_axes("right", size="5%", pad=0.05)
sc_cb = plt.colorbar(im_scatter, cax=sc_cax)

flux_div = make_axes_locatable(ax[1,0])
flux_cax = flux_div.append_axes("right", size="5%", pad=0.05)
flux_cb = plt.colorbar(im_flux, cax=flux_cax)

corr_div = make_axes_locatable(ax[1,1])
corr_cax = corr_div.append_axes("right", size="5%", pad=0.05)
corr_cb = plt.colorbar(im_corrected, cax=corr_cax)

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

#plt.tight_layout()
#plt.show()
#########################################################


testfig = plt.figure()
#plt.pcolormesh(X, Y, t[-1], cmap='hot_r')

test_cmap = matplotlib.colors.ListedColormap(['white','r'])

for i in range(int(md['Nb'])):
    for j in range(int(md['Nphi'])):
        if scatter_corrected[-1,i,j] > 0:
            #print(im_corrected.to_rgba(scatter_corrected[-1,i,j]))
            test_cmap = matplotlib.colors.ListedColormap(['white',
                im_corrected.to_rgba(scatter_corrected[-1,i,j])])
            test_im = plt.pcolormesh(X, Y, np.where(np.logical_and(np.logical_and(b[-1] > bbins[i] - db/2,
                b[-1] <= bbins[i] + db/2),np.logical_and(t[-1] > tbins[j] - dt/2,
                t[-1] <= tbins[j] + dt/2)), 1, np.NaN),
                cmap = test_cmap)
                #color=im_corrected.to_rgba(scatter_corrected[-1,i,j]))
            test_im.set_clim(0, np.max(t[-1]))

plt.show()

#########################################################

def animate(step):
    global b_cont, t_cont, b_cont_fill

    ax[0,1].clear()
    ax[1,0].clear()
    ax[1,1].clear()

    ax[0,1].axvline(b_zmax, color='k', linestyle='--')
    ax[1,0].axvline(b_zmax, color='k', linestyle='--')
    ax[1,1].axvline(b_zmax, color='k', linestyle='--')

    ax[0,1].axhline(t_rms, color='k', linestyle='--')
    ax[1,0].axhline(t_rms, color='k', linestyle='--')
    ax[1,1].axhline(t_rms, color='k', linestyle='--')

    b_com = np.sum(bgrid * np.where(scatter_corrected_og[step] > 0, scatter_corrected_og[step], 0)) \
            / np.sum(np.where(scatter_corrected_og[step]>0, scatter_corrected_og[step],0))
    t_com = np.sum(tgrid * np.where(scatter_corrected_og[step] > 0, scatter_corrected_og[step], 0)) \
            / np.sum(np.where(scatter_corrected_og[step]>0, scatter_corrected_og[step],0))

    im = ax[0,1].scatter(sx, sy, c=scatter[step], cmap='jet')
    im_flux = ax[1,0].scatter(sx, sy, c=scatter_flux[step], cmap=cmap, norm=norm)
    im_corrected = ax[1,1].scatter(sx, sy, c=scatter_corrected[step], cmap=s_cmap, norm=s_norm)

    ax[0,1].scatter(b_com, t_com, color='green', edgecolor='white', marker='^', s=200)
    ax[1,0].scatter(b_com, t_com, color='green', edgecolor='white', marker='^', s=200)
    ax[1,1].scatter(b_com, t_com, color='green', edgecolor='white', marker='^', s=200)

    trac_im.set_array(t[step].ravel())

    fig.suptitle("time = {0:.2f} s".format(times[step]))

    for coll in b_cont.collections:
        coll.remove()
    for coll in b_cont_fill.collections:
        coll.remove()
    for coll in t_cont.collections:
        coll.remove()

    b_cont = ax[0,0].contour(Xf, Yf, np.where(t[step] <= 5e-4, b[step], np.NaN),
            levels = contours_b, cmap='cool', alpha=0.8)
    b_cont_fill = ax[0,0].contourf(b_cont, levels = contours_b, cmap='cool', alpha=0.8)
    t_cont = ax[0,0].contour(Xf, Yf, t[step], levels = [5e-4], colors='white', alpha=0.8)

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

    return im, im_flux, im_corrected, trac_im

Writer = animation.writers['ffmpeg']
writer = Writer(fps=4, bitrate=1800)

anim = animation.FuncAnimation(fig, animate, interval=250, frames=NSAMP)
now = datetime.now()
#anim.save(save_dir+'td_scatter_wtracer_%s.mp4'%now.strftime("%d-%m-%Y-%H"),writer=writer)
plt.tight_layout()
plt.show()
