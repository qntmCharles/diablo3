import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
# Script loads in 2D slices and produces a movie of the simulation output import numpy as np import h5py, gc
import h5py
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from functions import get_metadata, read_params, get_grid, g2gf_1d

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"
var1 = 'b_az'
var2 = 'w_az'
var3 = 'u_az'
save = False

##### ---------------------- #####

# Get dir locations from param file
base_dir, run_dir, save_dir, version = read_params(params_file)
print(run_dir)
print(save_dir)

# Get simulation metadata
md = get_metadata(run_dir, version)

gxf, gyf, gzf, dzf = get_grid(save_dir+'grid.h5', md)
gx, gy, gz, dz = get_grid(save_dir+'grid.h5', md, fractional_grid=False)

X, Y = np.meshgrid(gx[:int(md['Nx']/2)+1], gz)

print("Complete metadata: ", md)

# Get data
with h5py.File(save_dir+"/az_stats.h5", 'r') as f:
    print("Keys: %s" % f.keys())
    time_keys = list(f[var1])
    print(time_keys)
    # Get buoyancy data
    data1 = np.array([np.array(f[var1][t]) for t in time_keys])
    data2 = np.array([np.array(f[var2][t]) for t in time_keys])
    data3 = np.array([np.array(f[var3][t]) for t in time_keys])

    data1 = g2gf_1d(md, data1)
    data2 = g2gf_1d(md, data2)
    data3 = g2gf_1d(md, data3)
    print("Data has shape ",data1.shape)
    NSAMP = len(data1)
    times = np.array([(float(t)-1)*md['SAVE_MOVIE_DT'] for t in time_keys])
    f.close()

print("Total time steps: %s"%NSAMP)
print("Dimensional times: ",times)

fig, ax = plt.subplots(1,3, figsize=(12, 5))
init_clim = 0.02
ims = np.array([None,None,None])
cb = np.array([None,None,None])

print("Setting up initial plot...")
ims[0] = ax[0].pcolormesh(X, Y, data1[0], cmap='jet')
ims[1] = ax[1].pcolormesh(X, Y, data2[0], cmap='jet')
ims[2] = ax[2].pcolormesh(X, Y, data3[0], cmap='jet')

# Add forcing level
ax[0].axhline(md['Lyc']+md['Lyp'],color='white')
ax[0].axhline(md['Lyc']+md['Lyp'],color='white')
ax[1].axhline(md['Lyc']+md['Lyp'],color='white')
ax[1].axhline(md['Lyc']+md['Lyp'],color='white')
ax[2].axhline(md['Lyc']+md['Lyp'],color='white')
ax[2].axhline(md['Lyc']+md['Lyp'],color='white')

cb[0] = plt.colorbar(ims[0],ax=ax[0])
cb[1] = plt.colorbar(ims[1],ax=ax[1])
cb[2] = plt.colorbar(ims[2],ax=ax[2])
ims[0].set_clim(0, 0.1)
ims[1].set_clim(-0.2, 0.2)
ims[2].set_clim(-0.01, 0.01)
#ims[1].set_clim(-1e-9, 1e-9)
#ims[2].set_clim(-1e-9, 1e-9)

fig.suptitle("time = 0 mins")
ax[0].set_ylabel("$z$")
ax[0].set_xlabel("$x$")
ax[0].set_title(var1)
ax[1].set_ylabel("$z$")
ax[1].set_xlabel("$x$")
ax[1].set_title(var2)
ax[2].set_ylabel("$z$")
ax[2].set_xlabel("$x$")
ax[2].set_title(var3)

def animate(step):
    ims[0].set_array(data1[step].ravel())
    ims[1].set_array(data2[step].ravel())
    ims[2].set_array(data3[step].ravel())
    fig.suptitle("time = {0:.3f} mins".format(times[step]/60))

    return ims.flatten(),

print("Initialising mp4 writer...")
Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, bitrate=1800)

print("Starting plot...")
anim = animation.FuncAnimation(fig, animate, interval=50, frames=NSAMP)
plt.tight_layout()
plt.subplots_adjust(wspace=0.3,hspace=0)
now = datetime.now()

if save:
    anim.save(base_dir+save_dir+'plume%s.mp4'%now.strftime("%d-%m-%Y-%H"),writer=writer)
else:
    plt.show()
