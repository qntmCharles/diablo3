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

var1_key = 'th1_xz'
var2_key = 'diapycvel1_xz'

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
    time_keys = list(f[var1_key])
    print(time_keys)
    # Get buoyancy data
    var1 = np.array([np.array(f[var1_key][t]) for t in time_keys])
    var2 = np.array([np.array(f[var2_key][t]) for t in time_keys])
    b = np.array([np.array(f['th1_xz'][t]) for t in time_keys])
    t = np.array([np.array(f['th2_xz'][t]) for t in time_keys])
    w = np.array([np.array(f['w_xz'][t]) for t in time_keys])
    v = np.array([np.array(f['v_xz'][t]) for t in time_keys])
    u = np.array([np.array(f['u_xz'][t]) for t in time_keys])

    var1 = g2gf_1d(var1)
    var2 = g2gf_1d(var2)
    b = g2gf_1d(b)
    t = g2gf_1d(t)
    w = g2gf_1d(w)
    u = g2gf_1d(u)
    v = g2gf_1d(v)

    NSAMP = len(var1)
    times = np.array([f[var1_key][t].attrs['Time'] for t in time_keys])
    f.close()

plot_max = 0.15 + md['H']
plot_min = 0.9 * md['H']

idx_max = get_index(plot_max, gz)
idx_min = get_index(plot_min, gz)

idx_maxf = get_index(plot_max, gzf)
idx_minf = get_index(plot_min, gzf)

X, Y = np.meshgrid(gx, gz[idx_min:idx_max+1])
Xf, Yf = np.meshgrid(gxf, gzf[idx_minf:idx_maxf])

dtdz = np.gradient(t, gzf, axis=1)
dtdx = np.gradient(t, gxf, axis=2)

#var2 = 2*md['nu']*0.7*(np.power(dtdz,2) + np.power(dtdx,2))/md['N2']

var1 = var1[:, idx_min:idx_max, :]
var2 = var2[:, idx_min:idx_max, :]
b = b[:, idx_minf:idx_maxf, :]
t = t[:, idx_minf:idx_maxf, :]

print("Dimensional times: ",times)

print("Setting up data arrays...")
contour_lvls_b = [1e-2, 5e-2, 1e-1, 1.5e-1, 2e-1, 2.5e-1]
contour_lvls_t = [1e-3]

fig, axs = plt.subplots(1,2,figsize=(10,5))
ims = np.array([None,None])
cb = np.array([None,None])

ims[0] = axs[0].pcolormesh(X, Y, var1[-1], cmap='jet')
#c_b = axs[1].contour(Xf, Yf, b[-1], levels= contour_lvls_b, colors='grey', alpha=0.5)
#c_t = axs[1].contour(Xf, Yf, t[-1], levels= contour_lvls_t, colors='green', linestyles='--')
ims[1] = axs[1].pcolormesh(X, Y, var2[-1], cmap='jet')#cmap='hot_r')

# Add forcing level
axs[0].axhline(md['Lyc']+md['Lyp'],color='white', linestyle=':')
axs[1].axhline(md['Lyc']+md['Lyp'],color='grey', linestyle=':')

cb[0] = plt.colorbar(ims[0],ax=axs[0])
cb[1] = plt.colorbar(ims[1],ax=axs[1])
ims[0].set_clim(0, np.max(np.abs(var1[-1])))
ims[1].set_clim(-np.max(np.abs(var2[-1])), np.max(np.abs(var2[-1])))
#ims[1].set_clim(0,1e-7)

fig.suptitle("time = 0.00 s")
axs[0].set_xlabel("$x$")
axs[1].set_xlabel("$y$")
axs[0].set_xlabel("$x$")
axs[1].set_xlabel("$y$")

axs[0].set_ylim(gz[idx_min], gz[idx_max])
axs[1].set_ylim(gz[idx_min], gz[idx_max])

def animate(step):
    #global c_b, c_t
    ims[0].set_array(var1[step].ravel())
    ims[1].set_array(var2[step].ravel())
    fig.suptitle("time = {0:.2f} s".format(times[step]))

    #for coll in c_b.collections:
        #coll.remove()
    #for coll in c_t.collections:
        #coll.remove()

    #c_b = axs[1].contour(Xf, Yf, b[step], levels= contour_lvls_b, colors='grey', alpha=0.5)
    #c_t = axs[1].contour(Xf, Yf, t[step], levels= contour_lvls_t, colors='green', linestyles='--')

    return ims.flatten(),

print("Initialising mp4 writer...")
Writer = animation.writers['ffmpeg']
writer = Writer(fps=4, bitrate=1800)

print("Starting plot...")
anim = animation.FuncAnimation(fig, animate, interval=250, frames=NSAMP)
now = datetime.now()
#anim.save(save_dir+'fountain%s.mp4'%now.strftime("%d-%m-%Y-%H-%m"),writer=writer)
plt.show()
