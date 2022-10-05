# Script loads in 2D slices and produces a movie of the simulation output import numpy as np import h5py, gc
import h5py
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from functions import get_metadata, read_params, get_grid

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"

##### ---------------------- #####

# Get dir locations from param file
base_dir, run_dir, save_dir, version = read_params(params_file)

# Get simulation metadata
md = get_metadata(run_dir, version)

gxf, gyf, gzf, dzf = get_grid(save_dir+'grid.h5', md)
gx, gy, gz, dz = get_grid(save_dir+'grid.h5', md, fractional_grid=False)

X, Y = np.meshgrid(gx, gz)
Xf, Yf = np.meshgrid(gxf, gzf)

print("Complete metadata: ", md)

# Get data
with h5py.File(save_dir+"/movie.h5", 'r') as f:
    print("Keys: %s" % f.keys())
    time_keys = list(f['th1_xy'])
    print(time_keys)
    # Get buoyancy data
    th1_xy = np.array([np.array(f['th1_xz'][t]) for t in time_keys])
    th1_zy = np.array([np.array(f['th2_xz'][t]) for t in time_keys])
    NSAMP = len(th1_xy)
    times = np.array([float(f['th1_xz'][tstep].attrs['Time']) for tstep in time_keys])
    f.close()

th1_xy -= th1_xy[0]
print("Total time steps: %s"%NSAMP)
print("Dimensional times: ",times)

print("Setting up data arrays...")
fig, axs = plt.subplots(1,2,figsize=(15, 10))
ims = np.array([None,None])
cb = np.array([None,None])

print("Setting up initial plot...")
ims[0] = axs[0].pcolormesh(X, Y, th1_xy[-1], cmap='jet')
ims[1] = axs[1].pcolormesh(X, Y, th1_zy[-1], cmap='jet')

# Add forcing level
axs[0].axhline(md['Lyc']+md['Lyp'],color='white', linestyle=':')
axs[1].axhline(md['Lyc']+md['Lyp'],color='white', linestyle=':')

cb[0] = plt.colorbar(ims[0],ax=axs[0])
cb[1] = plt.colorbar(ims[1],ax=axs[1])
#ims[0].set_clim(0, md['N2']*(md['LY']-md['H']))

r0 = md['r0']
F0 = md['b0'] * r0**2
alpha = md['alpha_e']

t_max = 5 * F0 * np.power(0.9 * alpha * F0, -1/3) * np.power(md['H'] + 5*r0/(6*alpha), -5/3) / \
        (3 * alpha)
print(t_max)
ims[1].set_clim(0, t_max)

fig.suptitle("$\\theta_1$, time = 0 secs")
axs[0].set_ylabel("$z$")
axs[1].set_ylabel("$z$")
axs[0].set_xlabel("$x$")
axs[1].set_xlabel("$y$")

axs[0].set_ylim(md['H'], md['H']+0.15)
axs[1].set_ylim(md['H'], md['H']+0.15)

def animate(step):
    ims[0].set_array(th1_xy[step].ravel())
    ims[1].set_array(th1_zy[step].ravel())
    fig.suptitle("$\\theta_1$, time = {0:.2f} secs".format(times[step]))

    return ims.flatten(),

print("Initialising mp4 writer...")
Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, bitrate=1800)

print("Starting plot...")
anim = animation.FuncAnimation(fig, animate, interval=500, frames=NSAMP)
now = datetime.now()
#anim.save(save_dir+'jet_%s.mp4'%now.strftime("%d-%m-%Y-%H-%m"),writer=writer)
plt.show()
