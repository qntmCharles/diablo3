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
out_file = "out.000343.h5"

##### ---------------------- #####

def get_index(z, griddata):
    return int(np.argmin(np.abs(griddata - z)))

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
    scatter = np.array([np.array(f['td_scatter'][t]) for t in time_keys])
    b = np.array([np.array(f['th1_xz'][t]) for t in time_keys])
    t = np.array([np.array(f['th2_xz'][t]) for t in time_keys])

    b = g2gf_1d(md, b)
    t = g2gf_1d(md, t)
    NSAMP = len(b)
    times = np.array([f['th1_xz'][t].attrs['Time'] for t in time_keys])
    f.close()

with h5py.File(join(save_dir, out_file), 'r') as f:
    b_3d = np.array(f['Timestep']['TH1'])
    t_3d = np.array(f['Timestep']['TH2'])
    f.close()

b -= b[0]

plot_max = md['LZ']
plot_min = md['H']

idx_max = get_index(plot_max, gz)
idx_min = get_index(plot_min, gz)+1

idx_maxf = get_index(plot_max, gzf)
idx_minf = get_index(plot_min, gzf)+1

X, Y = np.meshgrid(gx, gz[idx_min:idx_max+1])
Xf, Yf = np.meshgrid(gxf, gzf[idx_minf:idx_maxf])

b = b[:, idx_minf:idx_maxf, :]
t = t[:, idx_minf:idx_maxf, :]

bmin = -0.5*md['N2']*(md['LZ']-md['H'])
bmax = 0.5*md['N2']*(md['LZ']-md['H'])

F0 = md['b0']*(md['r0']**2)
alpha = md['alpha_e']

tmin = 0
tmax = 5*F0 / (3 * alpha) * np.power(0.9*alpha*F0, -1/3) * np.power(md['H']+ 5*md['r0']/(6*alpha), -5/3)

Nb = 50
Nt = 50
db = round((bmax - bmin)/Nb,3)
dt = (tmax - tmin)/Nt

bbins = [bmin + (i+0.5)*db for i in range(Nb)]
tbins = [tmin + (i+0.5)*dt for i in range(Nt)]
print(bmin,bmax)
print(tmin,tmax)
print(bbins)
print(tbins)

dx = md['LX']/md['Nx']
dy = md['LY']/md['Ny']

weights = np.zeros(shape=(NSAMP,Nb,Nt))
volume = np.zeros(shape=(NSAMP,))

for i in range(NSAMP):
    for j in range(idx_maxf-idx_minf):
        for k in range(md['Nx']):
            bbin = -1
            if b[i, j, k] < bmin:
                bbin = 0
            elif b[i,j,k] >= bmax:
                bbin = Nb-1
            else:
                for l in range(Nb):
                    if (b[i,j,k] - bbins[l] >= -0.5*db) and (b[i,j,k] - bbins[l] < 0.5*db):
                        bbin = l

            if t[i, j, k] < tmin:
                tbin = 0
            elif t[i,j,k] >= tmax:
                tbin = Nt-1
            else:
                for l in range(Nt):
                    if (t[i,j,k] - tbins[l] >= -0.5*dt) and (t[i,j,k] - tbins[l] < 0.5*dt):
                        tbin = l

            if (bbin < 0) or (tbin < 0):
                print("error")
                print(bmin, bmax, tmin, tmax)
                print(b[i,j,k], t[i,j,k], bbin, tbin)
                input()
            else:
                weights[i,bbin, tbin] += dz[j+idx_minf] * dx * dy
                volume[i] += dz[j+idx_minf] * dx * dy

weights_3d = np.zeros(shape=(Nb,Nt))
volume_3d = 0
"""
for i in range(md['Ny']):
    for j in range(idx_maxf-idx_minf):
        for k in range(md['Nx']):
            bbin = -1
            if b_3d[i, j, k] < bmin:
                bbin = 0
            elif b_3d[i,j,k] >= bmax:
                bbin = Nb-1
            else:
                for l in range(Nb):
                    if (b_3d[i,j,k] - bbins[l] >= -0.5*db) and (b_3d[i,j,k] - bbins[l] < 0.5*db):
                        bbin = l

            if t_3d[i, j, k] < tmin:
                tbin = 0
            elif t_3d[i,j,k] >= tmax:
                tbin = Nt-1
            else:
                for l in range(Nt):
                    if (t_3d[i,j,k] - tbins[l] >= -0.5*dt) and (t_3d[i,j,k] - tbins[l] < 0.5*dt):
                        tbin = l

            if (bbin < 0) or (tbin < 0):
                print("error")
                print(bmin, bmax, tmin, tmax)
                print(b_3d[i,j,k], t_3d[i,j,k], bbin, tbin)
                input()
            else:
                weights_3d[bbin, tbin] += dz[j+idx_minf] * dx * dy
                volume_3d += dz[j+idx_minf] * dx * dy
"""


weights = np.moveaxis(weights,1,2)
weights = np.where(weights == 0, np.nan, weights)
weights = np.log(weights)

scatter = np.where(scatter == 0, np.nan, scatter)
scatter = np.log(scatter)

"""
weights_3d = np.moveaxis(weights_3d,0,1)
weights_3d = np.where(weights_3d == 0, np.nan, weights_3d)
weights_3d = np.log(weights_3d)

fig, ax = plt.subplots(1,2)
fig.suptitle("time = 0.00 s")
sx, sy = np.meshgrid(bbins, tbins)

im_post = ax[0].scatter(sx, sy, c=weights_3d, cmap='jet')
im_rt = ax[1].scatter(sx, sy, c=scatter[20], cmap='jet')

ax[0].set_xlabel("b anomaly")
ax[0].set_ylabel("tracer")
ax[0].set_xlim(bmin, bmax)
ax[0].set_ylim(tmin, tmax)

ax[1].set_xlabel("b anomaly")
ax[1].set_ylabel("tracer")
ax[1].set_xlim(bmin, bmax)
ax[1].set_ylim(tmin, tmax)

plt.show()
"""

fig, ax = plt.subplots(1,2)
fig.suptitle("time = 0.00 s")

sx, sy = np.meshgrid(bbins, tbins)

im_post = ax[0].scatter(sx, sy, c=weights[-1], cmap='jet')
im_rt = ax[1].scatter(sx, sy, c=scatter[-1], cmap='jet')

ax[0].set_xlabel("b anomaly")
ax[0].set_ylabel("tracer")
ax[0].set_xlim(bmin, bmax)
ax[0].set_ylim(tmin, tmax)

ax[1].set_xlabel("b anomaly")
ax[1].set_ylabel("tracer")
ax[1].set_xlim(bmin, bmax)
ax[1].set_ylim(tmin, tmax)

def animate(step):
    ax[0].clear()
    ax[1].clear()

    im_post = ax[0].scatter(sx, sy, c=weights[step], cmap='jet')
    im_rt = ax[1].scatter(sx, sy, c=scatter[step], cmap='jet')

    fig.suptitle("time = {0:.2f} s".format(times[step]))

    ax[0].set_xlabel("b anomaly")
    ax[0].set_ylabel("tracer")
    ax[0].set_xlim(bmin, bmax)
    ax[0].set_ylim(tmin, tmax)

    ax[1].set_xlabel("b anomaly")
    ax[1].set_ylabel("tracer")
    ax[1].set_xlim(bmin, bmax)
    ax[1].set_ylim(tmin, tmax)

    return im_post, im_rt,

Writer = animation.writers['ffmpeg']
writer = Writer(fps=4, bitrate=1800)

anim = animation.FuncAnimation(fig, animate, interval=250, frames=NSAMP)
now = datetime.now()
#anim.save(save_dir+'td_scatter_%s.mp4'%now.strftime("%d-%m-%Y-%H"),writer=writer)
plt.show()
