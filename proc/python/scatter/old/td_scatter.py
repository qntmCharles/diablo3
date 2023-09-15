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
    NSAMP = len(scatter)
    times = np.array([f['th1_xz'][t].attrs['Time'] for t in time_keys])
    f.close()


plot_max = md['LZ']
plot_min = md['H']

idx_max = get_index(plot_max, gz)
idx_min = get_index(plot_min, gz)+1

idx_maxf = get_index(plot_max, gzf)
idx_minf = get_index(plot_min, gzf)+1

X, Y = np.meshgrid(gx, gz[idx_min:idx_max+1])
Xf, Yf = np.meshgrid(gxf, gzf[idx_minf:idx_maxf])

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
db = (bmax - bmin)/Nb
dt = (tmax - tmin)/Nt
bbins = [bmin + (i+0.5)*db for i in range(Nb)]
tbins = [tmin + (i+0.5)*dt for i in range(Nt)]

print(bmin,bmax)
print(tmin,tmax)

scatter = np.where(scatter == 0, np.nan, scatter)
scatter = np.log(scatter)

fig = plt.figure()
ax = plt.gca()
fig.suptitle("time = 0.00 s")

sx, sy = np.meshgrid(bbins, tbins)

im = plt.scatter(sx, sy, c=scatter[-1], cmap='jet')

#plt.xlabel("b anomaly")
plt.xlabel("buoyancy")
plt.ylabel("tracer")
plt.xlim(bmin, bmax)
plt.ylim(tmin, tmax)

def animate(step):
    ax.clear()

    im = plt.scatter(sx, sy, c=scatter[step], cmap='jet')

    fig.suptitle("time = {0:.2f} s".format(times[step]))

    #plt.xlabel("b anomaly")
    plt.xlabel("buoyancy")
    plt.ylabel("tracer")
    plt.xlim(bmin, bmax)
    plt.ylim(tmin, tmax)

    return im,

Writer = animation.writers['ffmpeg']
writer = Writer(fps=4, bitrate=1800)

anim = animation.FuncAnimation(fig, animate, interval=250, frames=NSAMP)
now = datetime.now()
anim.save(save_dir+'td_scatter_%s.mp4'%now.strftime("%d-%m-%Y-%H"),writer=writer)
plt.show()
