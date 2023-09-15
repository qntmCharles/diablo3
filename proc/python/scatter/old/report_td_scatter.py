# Script loads in 2D slices and produces a movie of the simulation output import numpy as np import h5py, gc
import h5py
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from functions import get_metadata, read_params, get_grid, g2gf_1d
from scipy import ndimage

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
    b = np.array([np.array(f['th1_xz'][t]) for t in time_keys])
    b = g2gf_1d(md, b)
    t = g2gf_1d(md, t)
    scatter = np.array([np.array(f['td_scatter'][t]) for t in time_keys])
    NSAMP = len(scatter)
    times = np.array([f['th1_xz'][t].attrs['Time'] for t in time_keys])
    f.close()


plot_max = md['LZ']
plot_min = md['H']

idx_max = get_index(plot_max, gz)
idx_min = get_index(plot_min, gz)

idx_maxf = get_index(plot_max, gzf)+1
idx_minf = get_index(plot_min, gzf)

X, Y = np.meshgrid(gx, gz[idx_min:idx_max+1])
Xf, Yf = np.meshgrid(gxf, gzf[idx_minf:idx_maxf])

b = b[:, idx_minf:idx_maxf, :]
t = t[:, idx_minf:idx_maxf, :]

#bmin = -0.5*md['N2']*(md['LZ']-md['H'])
#bmax = 0.5*md['N2']*(md['LZ']-md['H'])
bmin = 0
bmax = md['N2']*(md['LZ']-md['H'])

F0 = md['b0']*(md['r0']**2)
alpha = md['alpha_e']

tmin = 0
tmax = 5*F0 / (3 * alpha) * np.power(0.9*alpha*F0, -1/3) * np.power(md['H']+ 5*md['r0']/(6*alpha), -5/3)

Nb = 51
Nt = 51
db = round((bmax - bmin)/Nb,3)
dt = (tmax - tmin)/Nt
bbins = [bmin + (i+0.5)*db for i in range(Nb)]
tbins = [tmin + (i+0.5)*dt for i in range(Nt)]

print(bmin,bmax)
print(tmin,tmax)

scatter = np.where(scatter == 0, np.nan, scatter)
scatter = np.log(scatter)

fig,ax = plt.subplots(1,2, figsize=(15,5))
fig.suptitle("time = 0.00 s")

contours_b = np.linspace(0, bmax, 8)

scatter[:, 0, 0] = np.NaN

sx, sy = np.meshgrid(bbins, tbins)

im = ax[1].scatter(sx, sy, c=scatter[-1], cmap='jet')
trac_im = ax[0].pcolormesh(X, Y, t[-1], cmap='jet')
b_cont = ax[0].contour(Xf, Yf, b[-1], levels = contours_b, colors='white', alpha=0.8)

ax[1].set_xlabel("buoyancy")
ax[1].set_ylabel("tracer")
ax[1].set_xlim(bmin, bmax)
ax[1].set_ylim(tmin, tmax)

def animate(step):
    global b_cont

    ax[1].clear()

    im = ax[1].scatter(sx, sy, c=scatter[step], cmap='jet')
    trac_im.set_array(t[step].ravel())

    fig.suptitle("time = {0:.2f} s".format(times[step]))

    for coll in b_cont.collections:
        coll.remove()

    b_cont = ax[0].contour(Xf, Yf, b[step], levels = contours_b, colors='white', alpha=0.8)

    ax[1].set_xlabel("buoyancy")
    ax[1].set_ylabel("tracer")
    ax[1].set_xlim(bmin, bmax)
    ax[1].set_ylim(tmin, tmax)

    #im.set_clim(0, 0.5*np.nanmax(scatter[step]))

    return im, trac_im

Writer = animation.writers['ffmpeg']
writer = Writer(fps=4, bitrate=1800)

anim = animation.FuncAnimation(fig, animate, interval=250, frames=NSAMP)
now = datetime.now()
#anim.save(save_dir+'td_scatter_wtracer_%s.mp4'%now.strftime("%d-%m-%Y-%H"),writer=writer)
plt.show()

#############################################################################################################
# plotting for report
#############################################################################################################
pfig, pax = plt.subplots(2,3, figsize=(12,5))

plot_step1 = 15
plot_step2 = 30
plot_step3 = 60

sx, sy = np.meshgrid(bbins, tbins)

t = t[:,:80,200:-200]
b = b[:,:80,200:-200]
X = X[:81,200:-200]
Y = Y[:81,200:-200]
Xf = Xf[:80,200:-200]
Yf = Yf[:80,200:-200]

im1 = pax[0,0].scatter(sx, sy, c=scatter[plot_step1], cmap='jet',s=10)
im2 = pax[0,1].scatter(sx, sy, c=scatter[plot_step2], cmap='jet',s=10)
im3 = pax[0,2].scatter(sx, sy, c=scatter[plot_step3], cmap='jet',s=10)

trac_im1 = pax[1,0].pcolormesh(X, Y, t[plot_step1], cmap='hot_r')
b_cont1 = pax[1,0].contour(Xf, Yf, b[plot_step1], levels = contours_b, colors='grey', alpha=0.8)

trac_im2 = pax[1,1].pcolormesh(X, Y, t[plot_step2], cmap='hot_r')
b_cont2 = pax[1,1].contour(Xf, Yf, b[plot_step2], levels = contours_b, colors='grey', alpha=0.8)

trac_im3 = pax[1,2].pcolormesh(X, Y, t[plot_step3], cmap='hot_r')
b_cont3 = pax[1,2].contour(Xf, Yf, b[plot_step3], levels = contours_b, colors='grey', alpha=0.8)

pax[0,0].set_xlabel("buoyancy")
pax[0,0].set_ylabel("tracer")
pax[0,0].set_xlim(bmin, bbins[20])
pax[0,0].set_ylim(tmin, tmax)
pax[0,1].set_xlabel("buoyancy")
pax[0,1].set_ylabel("tracer")
pax[0,1].set_xlim(bmin, bbins[20])
pax[0,1].set_ylim(tmin, tmax)
pax[0,2].set_xlabel("buoyancy")
pax[0,2].set_ylabel("tracer")
pax[0,2].set_xlim(bmin, bbins[20])
pax[0,2].set_ylim(tmin, tmax)

def fmt(x): return f"{x:.3f}"

pax[0,0].clabel(b_cont1, b_cont1.levels, inline=True, fmt=fmt)
pax[0,1].clabel(b_cont2, b_cont2.levels, inline=True, fmt=fmt)
pax[0,2].clabel(b_cont3, b_cont3.levels, inline=True, fmt=fmt)

pax[1,0].set_xlabel("x (m)")
pax[1,0].set_ylabel("z (m)")
pax[1,1].set_xlabel("x (m)")
pax[1,1].set_ylabel("z (m)")
pax[1,2].set_xlabel("x (m)")
pax[1,2].set_ylabel("z (m)")

pax[0,0].set_title("(a) t = {0:.2f} s".format(times[plot_step1]))
pax[0,1].set_title("(b) t = {0:.2f} s".format(times[plot_step2]))
pax[0,2].set_title("(c) t = {0:.2f} s".format(times[plot_step3]))

plt.tight_layout()
#plt.savefig('/home/cwp29/Documents/4report/figs/tb_scatter.png',dpi=200)
plt.show()
