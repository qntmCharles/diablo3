# Script loads in 2D slices and produces a movie of the simulation output import numpy as np import h5py, gc
import h5py
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from functions import get_metadata, read_params

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
    print("Data has shape ",data1.shape)
    NSAMP = len(data1)
    times = np.array([(float(t)-1)*md['SAVE_MOVIE_DT'] for t in time_keys])
    f.close()

data1 = np.flip(data1,axis=1)
data2 = np.flip(data2,axis=1)
data3 = np.flip(data3,axis=1)

print("Total time steps: %s"%NSAMP)
print("Dimensional times: ",times)

fig, ax = plt.subplots(1,3)
init_clim = 0.02
ims = np.array([None,None,None])
cb = np.array([None,None,None])

print("Setting up initial plot...")
ims[0] = ax[0].imshow(data1[0], cmap='jet', interpolation='bicubic', animated=True,
        extent=[0,md['LX']/2,0,md['LZ']])
ims[1] = ax[1].imshow(data2[0], cmap='jet', interpolation='bicubic', animated=True,
        extent=[0,md['LX']/2,0,md['LZ']])
ims[2] = ax[2].imshow(data3[0], cmap='jet', interpolation='bicubic', animated=True,
        extent=[0,md['LX']/2,0,md['LZ']])

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
ims[1].set_clim(0, 0.1)
ims[2].set_clim(0, 0.1)

fig.suptitle("time = 0 mins")
ax[0].set_ylabel("$z$")
ax[0].set_xlabel("$x$")
ax[1].set_ylabel("$z$")
ax[1].set_xlabel("$x$")
ax[2].set_ylabel("$z$")
ax[2].set_xlabel("$x$")
plt.tight_layout()

def animate(step):
    ims[0].set_data(data1[step])
    ims[0].set_data(data1[step])
    ims[1].set_data(data2[step])
    ims[1].set_data(data2[step])
    ims[2].set_data(data3[step])
    ims[2].set_data(data3[step])
    fig.suptitle("time = {0:.3f} mins".format(times[step]/60))

    return ims.flatten(),

print("Initialising mp4 writer...")
Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, bitrate=1800)

print("Starting plot...")
anim = animation.FuncAnimation(fig, animate, interval=500, frames=NSAMP)
now = datetime.now()

if save:
    anim.save(base_dir+save_dir+'plume%s.mp4'%now.strftime("%d-%m-%Y-%H"),writer=writer)
else:
    plt.show()
