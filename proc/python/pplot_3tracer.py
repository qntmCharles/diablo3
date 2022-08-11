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
gxf, gyf, gzf, dzf = get_grid(save_dir+"/grid.h5", md)
gx, gy, gz, dz = get_grid(save_dir+"/grid.h5", md, fractional_grid=False)

X, Y = np.meshgrid(gx, gz)
Xf, Yf = np.meshgrid(gxf, gzf)

print("Complete metadata: ", md)

# Get data
with h5py.File(save_dir+"/movie.h5", 'r') as f:
    print("Keys: %s" % f.keys())
    time_keys = list(f['th1_xz'])
    print(time_keys)
    # Get buoyancy data
    th1_xz = np.array([np.array(f['th1_xz'][t]) for t in time_keys])
    th2_xz = np.array([np.array(f['th2_xz'][t]) for t in time_keys])
    th3_xz = np.array([np.array(f['th3_xz'][t]) for t in time_keys])
    NSAMP = len(th1_xz)
    times = np.array([float(t)*md['SAVE_MOVIE_DT'] for t in time_keys])
    f.close()

print("Total time steps: %s"%NSAMP)
print("Dimensional times: ",times)

print("Setting up data arrays...")
fig, axs = plt.subplots(1,3,figsize=(18, 8))
ims = np.array([None,None,None])
cb = np.array([None,None,None])

contour_lvls_b = np.linspace(0.01, np.max(th1_xz[0]), 30)
contour_lvls_trace = np.linspace(0.005, 0.1, 8)
contour_lvls_trace2 = np.linspace(0.01, 0.14, 8)

print("Setting up initial plot...")
ims[0] = axs[0].pcolormesh(X,Y,th1_xz[0], cmap='jet')
ims[1] = axs[1].pcolormesh(X,Y,th2_xz[0], cmap='jet')
ims[2] = axs[2].pcolormesh(X,Y,th3_xz[0], cmap='jet')
c_b = axs[0].contour(Xf, Yf, th1_xz[0], levels=contour_lvls_b, colors='white', alpha = 0.5)
c_trace = axs[1].contour(Xf, Yf, th2_xz[0], levels=contour_lvls_trace, colors='white', alpha = 0.5)
c_trace2 = axs[2].contour(Xf, Yf, th3_xz[0], levels=contour_lvls_trace, colors='white', alpha = 0.5)

# Add forcing level
axs[0].axhline(md['Lyc']+md['Lyp'],color='white', linestyle=':')
axs[1].axhline(md['Lyc']+md['Lyp'],color='white', linestyle=':')
axs[2].axhline(md['Lyc']+md['Lyp'],color='white', linestyle=':')

ims[0].set_clim(0, np.max(th1_xz[1]))
ims[1].set_clim(0, np.max(th2_xz[1]))
ims[2].set_clim(0, np.max(th3_xz[1]))
cb[0] = plt.colorbar(ims[0],ax=axs[0])
cb[1] = plt.colorbar(ims[1],ax=axs[1])
cb[2] = plt.colorbar(ims[2],ax=axs[2])

fig.suptitle("time = 0 seconds")
axs[0].set_ylabel("$z$")
axs[1].set_ylabel("$z$")
axs[2].set_ylabel("$z$")
axs[0].set_xlabel("$x$")
axs[1].set_xlabel("$y$")
axs[2].set_xlabel("$y$")

def animate(step):
    global c_b, c_trace, c_trace2

    ims[0].set_array(th1_xz[step].ravel())
    ims[1].set_array(th2_xz[step].ravel())
    ims[2].set_array(th3_xz[step].ravel())
    fig.suptitle("time = {0:.2f} seconds".format(times[step]))

    for coll in c_b.collections:
        coll.remove()
    for coll in c_trace.collections:
        coll.remove()
    for coll in c_trace2.collections:
        coll.remove()

    c_b = axs[0].contour(Xf, Yf, th1_xz[step], levels=contour_lvls_b, colors='white', alpha = 0.5)
    c_trace = axs[1].contour(Xf, Yf, th2_xz[step], levels=contour_lvls_trace, colors='white', alpha = 0.5)
    c_trace2 = axs[2].contour(Xf, Yf, th3_xz[step], levels=contour_lvls_trace2, colors='white', alpha = 0.5)

    return ims.flatten(),

print("Initialising mp4 writer...")
Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, bitrate=1800)

print("Starting plot...")
anim = animation.FuncAnimation(fig, animate, interval=20, frames=NSAMP)
now = datetime.now()
#anim.save(save_dir+'tracer2_%s.mp4'%now.strftime("%d-%m-%Y-%H-%m"),writer=writer)
plt.show()
