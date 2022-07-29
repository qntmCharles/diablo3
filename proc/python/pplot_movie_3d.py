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
    th1_zy = np.array([np.array(f['th1_xz'][t]) for t in time_keys])
    NSAMP = len(th1_xy)
    times = np.array([float(f['th1_xz'][tstep].attrs['Time']) for tstep in time_keys])
    f.close()

print("Total time steps: %s"%NSAMP)
print("Dimensional times: ",times)

print("Setting up data arrays...")
fig, axs = plt.subplots(1,2,figsize=(15, 10))
ims = np.array([None,None])
cb = np.array([None,None])

print("Setting up initial plot...")
ims[0] = axs[0].pcolormesh(X, Y, th1_xy[0], cmap='jet')
ims[1] = axs[1].pcolormesh(X, Y, th1_zy[0], cmap='jet')

# Add forcing level
axs[0].axhline(md['Lyc']+md['Lyp'],color='white', linestyle=':')
axs[1].axhline(md['Lyc']+md['Lyp'],color='white', linestyle=':')

cb[0] = plt.colorbar(ims[0],ax=axs[0])
cb[1] = plt.colorbar(ims[1],ax=axs[1])
ims[0].set_clim(0, 0.5)
#ims[1].set_clim(0, 1e-5)

fig.suptitle("$\\theta_1$, time = 0 secs")
axs[0].set_ylabel("$z$")
axs[1].set_ylabel("$z$")
axs[0].set_xlabel("$x$")
axs[1].set_xlabel("$y$")

def animate(step):
    ims[0].set_array(th1_xy[step].ravel())
    ims[1].set_array(th1_zy[step].ravel())
    fig.suptitle("$\\theta_1$, time = {0:.2f} secs".format(times[step]))

    return ims.flatten(),

print("Initialising mp4 writer...")
Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, bitrate=1800)

print("Starting plot...")
anim = animation.FuncAnimation(fig, animate, interval=20, frames=NSAMP)
now = datetime.now()
#anim.save(save_dir+'fountain%s.mp4'%now.strftime("%d-%m-%Y-%H-%m"),writer=writer)
plt.show()
