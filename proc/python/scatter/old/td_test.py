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
    t = g2gf_1d(md, t)
    b = g2gf_1d(md, b)
    t_2d = np.array([np.array(f['th2_xy'][t]) for t in time_keys])
    b_2d = np.array([np.array(f['th1_xy'][t]) for t in time_keys])
    w_2d = np.array([np.array(f['w_xy'][t]) for t in time_keys])
    b_2d = g2gf_1d(md, b_2d)
    t_2d = g2gf_1d(md, t_2d)
    w_2d = g2gf_1d(md, w_2d)
    scatter = np.array([np.array(f['td_source'][t]) for t in time_keys])
    NSAMP = len(scatter)
    times = np.array([f['th1_xz'][t].attrs['Time'] for t in time_keys])


xs = []
ys = []
bs = []
ts = []
with open(join(save_dir, "output1.dat"), "r") as f:
    text = f.readlines()
    for i in range(len(text)):
        strings = text[i].split()
        try:
            if strings[0] == "DATA":
                strings.extend(text[i+1].split())

                x = float(strings[1])
                y = float(strings[2])
                z = float(strings[3])
                print(z)

                b = float(strings[4])
                t = float(strings[5])

                xs.append(x)
                ys.append(y)
                bs.append(b)
                ts.append(t)

        except:
            continue

plt.scatter(xs, ys, c = ts)
plt.show()


plot_max = md['LZ']
plot_min = md['H']

idx_max = get_index(plot_max, gz)+1
idx_min = get_index(plot_min, gz)-1

idx_maxf = get_index(plot_max, gzf)
idx_minf = get_index(plot_min, gzf)

print(idx_min, idx_max)
print(idx_minf, idx_maxf)

X, Y = np.meshgrid(gx, gz[idx_min:idx_max])
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
dx = md['LX']/md['Nx']
dy = md['LY']/md['Ny']
bbins = [bmin + (i+0.5)*db for i in range(Nb)]
tbins = [tmin + (i+0.5)*dt for i in range(Nt)]

print(bmin,bmax)
print(tmin,tmax)

weights = np.zeros(shape=(NSAMP,Nb,Nt))
volume = np.zeros(shape=(NSAMP,))
for i in range(NSAMP):
    for j in range(md['Nx']):
        for k in range(md['Ny']):
            tbin = -1
            bbin = -1

            for l in range(Nb):
                if (b_2d[i,j,k] - bbins[l] >= -0.5*db) and (b_2d[i,j,k] - bbins[l] < 0.5*db):
                    bbin = l

            for l in range(Nt):
                if (t_2d[i,j,k] - tbins[l] >= -0.5*dt) and (t_2d[i,j,k] - tbins[l] < 0.5*dt):
                    tbin = l

            if (bbin >= 0) and (tbin >= 0):
                weights[i,bbin, tbin] += dz[get_index(md['H'],gzf)] * dx * dy
                volume[i] += dz[get_index(md['H'],gzf)] * dx * dy

weights = np.moveaxis(weights,1,2)
weights = np.where(weights == 0, np.nan, weights)
weights = np.log(weights)

scatter = np.where(scatter == 0, np.nan, scatter)
scatter = np.log(scatter)

fig,ax = plt.subplots(1,3, figsize=(15,5))
fig.suptitle("time = 0.00 s")

contours_b = np.linspace(0, 0.5*bmax, 8)

scatter[:, 0, 0] = np.NaN
weights[:, 0, 0] = np.NaN

sx, sy = np.meshgrid(bbins, tbins)

im = ax[1].scatter(sx, sy, c=scatter[-1], cmap='jet')
im_2d = ax[2].scatter(sx, sy, c=weights[-1], cmap='jet')
trac_im = ax[0].pcolormesh(X, Y, t[-1], cmap='jet')
b_cont = ax[0].contour(Xf, Yf, b[-1], levels = contours_b, colors='white', alpha=0.8)

ax[1].set_xlabel("buoyancy")
ax[1].set_ylabel("tracer")
ax[1].set_xlim(bmin, bbins[20])
ax[1].set_ylim(tmin, tmax)
ax[2].set_xlabel("buoyancy")
ax[2].set_ylabel("tracer")
ax[2].set_xlim(bmin, bbins[20])
ax[2].set_ylim(tmin, tmax)

def animate(step):
    global b_cont

    ax[1].clear()
    ax[2].clear()

    im = ax[1].scatter(sx, sy, c=scatter[step], cmap='jet')
    im_2d = ax[2].scatter(sx, sy, c=weights[step], cmap='jet')
    trac_im.set_array(t[step].ravel())

    fig.suptitle("time = {0:.2f} s".format(times[step]))

    for coll in b_cont.collections:
        coll.remove()

    b_cont = ax[0].contour(Xf, Yf, b[step], levels = contours_b, colors='white', alpha=0.8)

    ax[1].set_xlabel("buoyancy")
    ax[1].set_ylabel("tracer")
    ax[1].set_xlim(bmin, bbins[20])
    ax[1].set_ylim(tmin, tmax)
    ax[2].set_xlabel("buoyancy")
    ax[2].set_ylabel("tracer")
    ax[2].set_xlim(bmin, bbins[20])
    ax[2].set_ylim(tmin, tmax)

    #im.set_clim(0, 0.5*np.nanmax(scatter[step]))

    return im, trac_im

Writer = animation.writers['ffmpeg']
writer = Writer(fps=4, bitrate=1800)

anim = animation.FuncAnimation(fig, animate, interval=250, frames=NSAMP)
now = datetime.now()
#anim.save(save_dir+'td_scatter_wtracer_%s.mp4'%now.strftime("%d-%m-%Y-%H"),writer=writer)
plt.show()
