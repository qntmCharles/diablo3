# Script loads in 2D slices and produces a movie of the simulation output import numpy as np import h5py, gc
import h5py
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from functions import get_metadata, read_params, g2gf_1d

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"

##### ---------------------- #####

# Get dir locations from param file
base_dir, run_dir, save_dir, version = read_params(params_file)

# Get simulation metadata
md = get_metadata(run_dir, version)

print("Complete metadata: ", md)

# Get data
with h5py.File(save_dir+"/movie.h5", 'r') as f:
    print("Keys: %s" % f.keys())
    time_keys = list(f['th1_xy'])
    print(time_keys)
    # Get buoyancy data
    th1_xz = np.array([np.array(f['th1_xz'][t]) for t in time_keys])
    th2_xz = np.array([np.array(f['th2_xz'][t]) for t in time_keys])
    th1_xz = g2gf_1d(th1_xz)
    th2_xz = g2gf_1d(th2_xz)
    NSAMP = len(th1_xz)
    times = np.array([float(t)*md['SAVE_MOVIE_DT'] for t in time_keys])
    f.close()

th1_xz = np.flip(th1_xz,axis=1)
th2_xz = np.flip(th2_xz,axis=1)

print("Total time steps: %s"%NSAMP)
print("Dimensional times: ",times)

print("Setting up data arrays...")
fig, axs = plt.subplots(1,2,figsize=(15, 10))
ims = np.array([None,None])
cb = np.array([None,None])

contour_lvls_b = np.linspace(0.01, np.max(th1_xz[0]), 30)
contour_lvls_trace = np.linspace(0.005, 0.1, 8)

print("Setting up initial plot...")
ims[0] = axs[0].imshow(th1_xz[0], cmap='jet', interpolation='bicubic', animated=True,
        extent=[0,md['LX'],0,md['LZ']])
ims[1] = axs[1].imshow(th2_xz[0], cmap='jet', interpolation='bicubic', animated=True,
        extent=[0,md['LY'],0,md['LZ']])
c_b = axs[0].contour(np.flip(th1_xz[0],axis=0), extent=[0,md['LY'],0,md['LZ']], levels=contour_lvls_b,
        colors='white', alpha = 0.5)
c_trace = axs[1].contour(np.flip(th2_xz[0],axis=0), extent=[0,md['LY'],0,md['LZ']], levels=contour_lvls_trace,
        colors='white', alpha = 0.5)

# Add forcing level
axs[0].axhline(md['Lyc']+md['Lyp'],color='white', linestyle=':')
axs[1].axhline(md['Lyc']+md['Lyp'],color='white', linestyle=':')

cb[0] = plt.colorbar(ims[0],ax=axs[0])
cb[1] = plt.colorbar(ims[1],ax=axs[1])
ims[0].set_clim(0, np.max(th1_xz[0]))
ims[1].set_clim(0, 0.2)

fig.suptitle("$\\theta_1$, time = 0 hours")
axs[0].set_ylabel("$z$")
axs[1].set_ylabel("$z$")
axs[0].set_xlabel("$x$")
axs[1].set_xlabel("$y$")

def animate(step):
    global c_b, c_trace

    ims[0].set_data(th1_xz[step])
    ims[1].set_data(th2_xz[step])
    fig.suptitle("$\\theta_1$, time = {0:.4f} hours".format(times[step]/3600))

    for coll in c_b.collections:
        coll.remove()
    for coll in c_trace.collections:
        coll.remove()

    c_b = axs[0].contour(np.flip(th1_xz[step],axis=0), extent=[0,md['LY'],0,md['LZ']], levels=contour_lvls_b,
            colors='white', alpha = 0.5)
    c_trace = axs[1].contour(np.flip(th2_xz[step],axis=0), extent=[0,md['LY'],0,md['LZ']], levels=contour_lvls_trace,
            colors='white', alpha = 0.5)

    return ims.flatten(),

print("Initialising mp4 writer...")
Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, bitrate=1800)

print("Starting plot...")
anim = animation.FuncAnimation(fig, animate, interval=20, frames=NSAMP)
now = datetime.now()
#anim.save(save_dir+'fountain%s.mp4'%now.strftime("%d-%m-%Y-%H-%m"),writer=writer)
plt.show()
